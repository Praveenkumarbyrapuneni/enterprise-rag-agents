"""
scripts/export_finetune_data.py — Export feedback data as fine-tuning datasets.

Exports three JSONL files from accumulated feedback + query logs:

  1. data/sft_synthesizer.jsonl  — Supervised fine-tuning (SFT) for synthesizer
     Format: Qwen3 chat template  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
     Source: helpful=True answers with faithfulness >= MIN_FAITH and relevance >= MIN_REL
     Use: fine-tune Qwen3 (or any chat model) to replace Claude Sonnet for routine queries

  2. data/dpo_pairs.jsonl  — Direct Preference Optimization (DPO) pairs
     Format: {"prompt": q, "chosen": good_answer, "rejected": bad_answer}
     Source: questions with both helpful=True and helpful=False responses in the audit log
     Use: preference fine-tuning to steer model away from answer patterns users disliked

  3. data/reranker_train.jsonl  — Reranker fine-tuning pairs
     Format: {"query": q, "pos": ["cited chunk..."], "neg": ["uncited chunk..."]}
     Source: query_chunks table (cited=True are positives, cited=False are negatives)
     Use: fine-tune a cross-encoder or Cohere custom reranker on your financial domain

Usage:
  python -m scripts.export_finetune_data                     # all tenants
  python -m scripts.export_finetune_data --tenant goldman    # one tenant
  python -m scripts.export_finetune_data --min-feedback 100  # skip small tenants

Output is written to data/ directory. Existing files are overwritten.
Each run prints a summary of what was exported and skipped.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_FAITH       = float(os.getenv("EXPORT_MIN_FAITHFULNESS", "0.85"))
MIN_REL         = float(os.getenv("EXPORT_MIN_RELEVANCE",    "0.80"))
MIN_FEEDBACK    = 10    # skip tenants with fewer feedbacks than this

# ── DB connection ─────────────────────────────────────────────────────────────

def _conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set in .env", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(url, connect_timeout=10)


# ── Qwen3 chat template ───────────────────────────────────────────────────────

def _to_chat(question: str, answer: str) -> dict:
    """Format as Qwen3 chat template for SFT training."""
    return {
        "messages": [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# ── Export 1: SFT synthesizer data ───────────────────────────────────────────

def export_sft(conn, tenant_filter: str | None) -> list[dict]:
    """
    Fetch helpful=True answers that passed quality thresholds.
    These are (question, answer) pairs the model should learn to replicate.
    SQL/hybrid queries excluded — those answers contain live transaction data
    that changes and shouldn't be baked into a model's weights.
    """
    query = """
        SELECT al.question, al.answer
        FROM query_feedback   f
        JOIN query_audit_log  al ON al.id = f.query_id
        WHERE f.helpful = true
          AND al.faithfulness >= %(faith)s
          AND al.relevance    >= %(rel)s
          AND al.data_source  = 'rag'
          AND al.answer IS NOT NULL
          AND al.answer != ''
    """
    params: dict = {"faith": MIN_FAITH, "rel": MIN_REL}
    if tenant_filter:
        query += " AND f.tenant_id = %(tenant)s"
        params["tenant"] = tenant_filter
    query += " ORDER BY al.created_at DESC"

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [_to_chat(r["question"], r["answer"]) for r in rows]


# ── Export 2: DPO preference pairs ───────────────────────────────────────────

def export_dpo(conn, tenant_filter: str | None) -> list[dict]:
    """
    Find questions that have both a helpful=True and a helpful=False response.
    Pairs them as (chosen=good_answer, rejected=bad_answer).
    Exact question match only — no embedding similarity needed.
    """
    query = """
        SELECT
            al.question,
            MAX(al.answer) FILTER (WHERE f.helpful = true)  AS chosen,
            MAX(al.answer) FILTER (WHERE f.helpful = false) AS rejected
        FROM query_feedback   f
        JOIN query_audit_log  al ON al.id = f.query_id
        WHERE al.data_source = 'rag'
          AND al.answer IS NOT NULL
    """
    params: dict = {}
    if tenant_filter:
        query += " AND f.tenant_id = %(tenant)s"
        params["tenant"] = tenant_filter
    query += """
        GROUP BY al.question
        HAVING
            COUNT(*) FILTER (WHERE f.helpful = true)  > 0
            AND COUNT(*) FILTER (WHERE f.helpful = false) > 0
    """

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [
        {"prompt": r["question"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in rows
        if r["chosen"] and r["rejected"] and r["chosen"] != r["rejected"]
    ]


# ── Export 3: Reranker training pairs ────────────────────────────────────────

def export_reranker(conn, tenant_filter: str | None) -> list[dict]:
    """
    Build (query, positive_chunks, negative_chunks) triples for reranker training.

    Positive = chunks marked cited=True (synthesizer used them to write the answer).
    Negative = chunks retrieved but NOT cited (retrieved but not useful enough).
    Only uses queries where the user marked helpful=True (good retrieval signal).

    Requires query_chunks table to be populated (happens automatically during queries).
    """
    query = """
        SELECT
            al.question,
            array_agg(qc.chunk_text) FILTER (WHERE qc.cited = true)  AS pos_chunks,
            array_agg(qc.chunk_text) FILTER (WHERE qc.cited = false) AS neg_chunks
        FROM query_feedback   f
        JOIN query_audit_log  al ON al.id = f.query_id
        JOIN query_chunks     qc ON qc.query_id = f.query_id
        WHERE f.helpful = true
          AND al.data_source = 'rag'
    """
    params: dict = {}
    if tenant_filter:
        query += " AND f.tenant_id = %(tenant)s"
        params["tenant"] = tenant_filter
    query += """
        GROUP BY al.question
        HAVING
            array_agg(qc.chunk_text) FILTER (WHERE qc.cited = true)  IS NOT NULL
            AND array_agg(qc.chunk_text) FILTER (WHERE qc.cited = false) IS NOT NULL
    """

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [
        {
            "query": r["question"],
            "pos":   [t for t in r["pos_chunks"] if t],
            "neg":   [t for t in r["neg_chunks"] if t],
        }
        for r in rows
        if r["pos_chunks"] and r["neg_chunks"]
    ]


# ── Write JSONL ───────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export fine-tuning datasets from feedback data")
    parser.add_argument("--tenant",       help="Export only this tenant (default: all)")
    parser.add_argument("--min-feedback", type=int, default=MIN_FEEDBACK,
                        help=f"Skip tenants with fewer than N feedbacks (default: {MIN_FEEDBACK})")
    parser.add_argument("--out-dir",      default="data",
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    conn = _conn()

    print(f"\nExporting fine-tuning data{'  tenant='+args.tenant if args.tenant else ' (all tenants)'}")
    print(f"Thresholds: faithfulness>={MIN_FAITH}  relevance>={MIN_REL}  min_feedback={args.min_feedback}\n")

    # ── SFT ───────────────────────────────────────────────────────────────────
    sft = export_sft(conn, args.tenant)
    sft_path = out / "sft_synthesizer.jsonl"
    _write_jsonl(sft_path, sft)
    print(f"SFT synthesizer:   {len(sft):>5} examples  →  {sft_path}")
    if len(sft) < 100:
        print(f"  ⚠  < 100 examples — collect more feedback before training (target: 500+)")

    # ── DPO ───────────────────────────────────────────────────────────────────
    dpo = export_dpo(conn, args.tenant)
    dpo_path = out / "dpo_pairs.jsonl"
    _write_jsonl(dpo_path, dpo)
    print(f"DPO pairs:         {len(dpo):>5} pairs     →  {dpo_path}")
    if len(dpo) < 50:
        print(f"  ⚠  < 50 pairs — need more questions with mixed feedback (helpful + unhelpful)")

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker = export_reranker(conn, args.tenant)
    reranker_path = out / "reranker_train.jsonl"
    _write_jsonl(reranker_path, reranker)
    print(f"Reranker pairs:    {len(reranker):>5} triples   →  {reranker_path}")
    if len(reranker) < 200:
        print(f"  ⚠  < 200 triples — query_chunks table may not be populated yet (queries log chunks automatically)")

    conn.close()

    print(f"\nDone. Next steps:")
    if len(sft) >= 100:
        print(f"  SFT ready  → fine-tune Qwen3 on {sft_path}")
    if len(dpo) >= 50:
        print(f"  DPO ready  → run DPO training on {dpo_path}")
    if len(reranker) >= 200:
        print(f"  Reranker ready → fine-tune cross-encoder on {reranker_path}")


if __name__ == "__main__":
    main()
