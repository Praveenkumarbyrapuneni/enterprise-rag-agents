"""
agents/retriever.py — Agent 2: Retrieval

Search strategy:
  1. Embed question (and hyde_query if set) via Cohere Embed v3 on Bedrock
     using input_type="search_query" — critical: wrong type degrades accuracy 5-15%.
  2. Hybrid search in Qdrant — Prefetch (dense + BM25 sparse) fused via RRF.
     Dense captures semantic meaning. BM25 captures exact financial terms
     ("SOFR rate 2024-06-30", "Form 10-Q Schedule 14A", specific dollar amounts).
     RRF (Reciprocal Rank Fusion) combines both without parameter tuning.
  3. Optional external Cohere rerank: disabled by default so document data stays
     inside AWS. Set ALLOW_EXTERNAL_RERANK=true + COHERE_API_KEY only if your
     deployment explicitly permits sending candidate text to Cohere.
  4. MMR: remove near-duplicate chunks while preserving relevance.
     Uses stored dense vectors — no extra API calls.
  5. Parent context expansion: chunks with a parent_id get swapped for their
     full parent paragraph from PostgreSQL. Wider context = better synthesis.

Production failure handling:
  - Empty question          → set error, return immediately
  - Qdrant timeout          → retry 3x with exponential backoff; set error on exhaustion
  - Zero results            → set error, return cleanly
  - Rerank disabled/failure → return raw RRF order (non-fatal)
  - PostgreSQL parent fetch  → log warning, return child text only (non-fatal)
  - Missing COHERE_API_KEY  → skip opt-in external reranking, log warning
"""

import concurrent.futures
import json
import os
import time
from typing import Optional

import boto3
import numpy as np
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    FieldCondition, Filter, FusionQuery, Fusion, MatchValue, Prefetch, SparseVector
)

from ingestion.bm25 import bm25_sparse_vector

from .logger import get_logger
from .state import RAGState, RetrievedChunk

load_dotenv()
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_QDRANT_TOP_K   = 20     # candidates per query vector → reranker
_RERANK_TOP_N   = 8      # reranker output → MMR input
_MMR_FINAL_K    = 6      # chunks delivered to synthesizer
_MMR_LAMBDA     = 0.6    # 0=max diversity, 1=max relevance; 0.6 favors precision
_COHERE_TIMEOUT = 12     # seconds; Cohere p50=100-400ms, spikes under load
_QDRANT_RETRIES = 3
_RETRY_DELAYS   = [1, 2, 4]   # seconds between retries
_EMBED_MODEL    = "cohere.embed-english-v3"
_RERANK_MODEL   = "rerank-english-v3.0"
_COLLECTION     = os.getenv("QDRANT_COLLECTION_NAME", "documents")

# Process-level singletons — one client per worker, not per request
import threading as _threading
_bedrock_client: Optional[object] = None
_qdrant_client:  Optional[QdrantClient] = None
_bedrock_lock = _threading.Lock()
_qdrant_lock  = _threading.Lock()


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        with _bedrock_lock:
            if _bedrock_client is None:
                _bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=os.getenv("AWS_REGION", "us-east-1"),
                )
    return _bedrock_client


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(
                    host=os.getenv("QDRANT_HOST", "localhost"),
                    port=int(os.getenv("QDRANT_PORT", "6333")),
                    timeout=30.0,
                )
    return _qdrant_client


# ── Step 1: Embed query text ──────────────────────────────────────────────────


def _embed_query(text: str) -> list[float]:
    """
    Embed a query string via Cohere Embed v3 on AWS Bedrock.

    input_type="search_query" is mandatory — ingestion used "search_document".
    Mixing the two input types degrades asymmetric retrieval accuracy by 5-15%.
    """
    body = json.dumps({
        "texts": [text[:2048]],   # Bedrock validates input length before Cohere sees it
        "input_type": "search_query",
        "truncate": "END",
    })
    response = _get_bedrock().invoke_model(
        modelId=_EMBED_MODEL,
        body=body,
        contentType="application/json",
        accept="*/*",
    )
    result = json.loads(response["body"].read())
    return result["embeddings"][0]


# ── Step 2: Qdrant hybrid search (dense + BM25 sparse, RRF fusion) ────────────


def _build_tenant_filter(tenant_id: str) -> Filter | None:
    """
    Build a Qdrant payload filter that restricts search to one tenant.
    Returns None only when tenant_id is empty (dev/test without isolation).
    In production, tenant_id is always set — an empty tenant_id is a bug.
    """
    if not tenant_id:
        logger.warning(
            "[retriever] tenant_id is empty — searching ALL tenants. "
            "This must never happen in production."
        )
        return None
    return Filter(
        must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    )


def _extract_dense_vector(raw) -> list[float]:
    """Extract dense vector from named-vector dict or plain list (backward compat)."""
    if isinstance(raw, dict):
        return raw.get("dense") or []
    if isinstance(raw, list):
        return raw
    return []


def _search_qdrant(
    dense_vecs: list[list[float]],
    sparse_vecs: list[tuple[list[int], list[float]]],
    top_k: int,
    tenant_filter: Filter | None = None,
) -> list[dict]:
    """
    Hybrid dense + BM25 sparse search with RRF (Reciprocal Rank Fusion).

    dense_vecs and sparse_vecs are parallel lists — one entry per query vector
    (question, HyDE hypothesis, or sub-question).

    Each query vector produces two Prefetch entries: one for "dense" (semantic)
    and one for "sparse" (BM25 exact terms). Qdrant fuses all Prefetch results
    using RRF, which combines rankings without needing a tuning parameter.

    with_vectors=True fetches stored dense vectors for MMR cosine similarity —
    no second API call needed.

    Retries on connection errors and Qdrant 5xx. Raises RuntimeError after
    all retries are exhausted so the caller can set state["error"].
    """
    # Build Prefetch list: one dense + one sparse entry per query vector.
    # Sparse Prefetch is skipped when BM25 produced no terms (empty text, all stopwords).
    prefetches = []
    for d_vec, (s_idx, s_val) in zip(dense_vecs, sparse_vecs):
        prefetches.append(Prefetch(
            query=d_vec,
            using="dense",
            limit=top_k,
            filter=tenant_filter,
        ))
        if s_idx:
            prefetches.append(Prefetch(
                query=SparseVector(indices=s_idx, values=s_val),
                using="sparse",
                limit=top_k,
                filter=tenant_filter,
            ))

    last_exc: Optional[Exception] = None

    for attempt in range(_QDRANT_RETRIES + 1):
        if attempt > 0:
            delay = _RETRY_DELAYS[attempt - 1]
            logger.warning(
                f"[retriever] Qdrant retry {attempt}/{_QDRANT_RETRIES} in {delay}s"
            )
            time.sleep(delay)

        try:
            hits = _get_qdrant().query_points(
                collection_name=_COLLECTION,
                prefetch=prefetches,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
                with_vectors=True,
            ).points
            return [
                {
                    "id":      str(hit.id),
                    "score":   hit.score,
                    "vector":  _extract_dense_vector(hit.vector),
                    "payload": hit.payload or {},
                }
                for hit in hits
            ]

        except (ConnectionError, TimeoutError, OSError) as e:
            last_exc = e

        except UnexpectedResponse as e:
            if e.status_code >= 500:
                last_exc = e
            else:
                raise   # 4xx = caller error, retrying won't help

    raise RuntimeError(
        f"[retriever] Qdrant search failed after {_QDRANT_RETRIES} retries: {last_exc}"
    ) from last_exc


# ── Step 3: Optional external reranking ───────────────────────────────────────


def _rerank(question: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    Optionally re-score candidates with Cohere rerank.

    Disabled by default: this project promises that financial document data stays
    in AWS. Direct Cohere rerank sends candidate text to Cohere, so it only runs
    when ALLOW_EXTERNAL_RERANK=true is set deliberately. Non-fatal: returns raw
    top-k on any failure or when disabled.

    Score gotcha: Cohere scores are ordinal, not arithmetic.
    0.91 is not twice as relevant as 0.45 — use them for ranking only.

    ThreadPoolExecutor enforces _COHERE_TIMEOUT — the SDK itself has no timeout param.
    """
    if not candidates:
        return candidates

    allow_external = os.getenv("ALLOW_EXTERNAL_RERANK", "false").lower() in {
        "1", "true", "yes"
    }
    if not allow_external:
        logger.info(
            "[retriever] external reranking disabled — using raw RRF order "
            "(set ALLOW_EXTERNAL_RERANK=true to opt in)"
        )
        return candidates[:top_n]

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.warning("[retriever] ALLOW_EXTERNAL_RERANK=true but COHERE_API_KEY is not set")
        return candidates[:top_n]

    def _call_cohere() -> list[dict]:
        import cohere
        co = cohere.Client(api_key=api_key)
        docs = [c["payload"].get("text", "") for c in candidates]
        results = co.rerank(
            model=_RERANK_MODEL,
            query=question,
            documents=docs,
            top_n=top_n,
        )
        reranked = []
        for r in results.results:
            c = dict(candidates[r.index])   # copy so we don't mutate the original
            c["score"] = r.relevance_score
            reranked.append(c)
        return reranked

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_cohere)
            return future.result(timeout=_COHERE_TIMEOUT)

    except concurrent.futures.TimeoutError:
        logger.warning(
            f"[retriever] Cohere rerank timed out after {_COHERE_TIMEOUT}s — "
            "falling back to raw vector order"
        )
        return candidates[:top_n]

    except Exception as e:
        logger.warning(
            f"[retriever] Cohere rerank failed: {type(e).__name__}: {e} — "
            "falling back to raw vector order"
        )
        return candidates[:top_n]


# ── Step 4: MMR deduplication ─────────────────────────────────────────────────


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0.0 else 0.0


def _mmr(candidates: list[dict], final_k: int) -> list[dict]:
    """
    Maximal Marginal Relevance: pick chunks that are relevant but not copies of
    each other. Uses stored Qdrant vectors for similarity — no extra API calls.

    lambda=0.6: slightly favors relevance over diversity. At Discover scale,
    a wrong-but-diverse chunk is worse than a slightly redundant correct one.
    """
    if len(candidates) <= final_k:
        return candidates

    selected: list[int] = [0]            # always take the top-ranked chunk first
    remaining = list(range(1, len(candidates)))

    while remaining and len(selected) < final_k:
        best_score, best_idx = -float("inf"), remaining[0]

        for i in remaining:
            relevance = candidates[i]["score"]
            vec_i = candidates[i].get("vector") or []

            if vec_i:
                max_sim = max(
                    _cosine_sim(vec_i, candidates[j].get("vector") or [])
                    for j in selected
                )
            else:
                max_sim = 0.0   # no vector = assume dissimilar, don't penalize

            mmr_score = _MMR_LAMBDA * relevance - (1.0 - _MMR_LAMBDA) * max_sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


# ── Step 5: Fetch parent context from PostgreSQL ──────────────────────────────


def _fetch_parents(parent_ids: list[str]) -> dict[str, str]:
    """
    Batch-fetch parent chunk texts so the synthesizer gets full paragraphs,
    not just the child sentences that happened to be closest to the query.
    Non-fatal: returns {} on any failure; caller falls back to child text.
    """
    if not parent_ids:
        return {}
    url = os.getenv("DATABASE_URL")
    if not url:
        return {}
    try:
        import psycopg2
        conn = psycopg2.connect(url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT parent_id::text, text FROM parent_chunks "
                    "WHERE parent_id = ANY(%s::uuid[])",
                    (parent_ids,),
                )
                return {str(row[0]): row[1] for row in cur.fetchall()}
        finally:
            conn.close()
    except Exception as e:
        logger.warning(
            f"[retriever] Parent fetch failed: {type(e).__name__}: {e}. "
            "Returning child text only."
        )
        return {}


# ── Step 6: Assemble RetrievedChunk output ───────────────────────────────────


def _to_chunks(candidates: list[dict], parent_texts: dict[str, str]) -> list[RetrievedChunk]:
    chunks = []
    for c in candidates:
        p = c["payload"]
        parent_id = p.get("parent_id")
        text = (
            parent_texts.get(str(parent_id), p.get("text", ""))
            if parent_id
            else p.get("text", "")
        )
        chunks.append(RetrievedChunk(
            text=text,
            score=round(c["score"], 4),
            source=p.get("file_name", "unknown"),
            page=int(p.get("page") or 0),
            doc_id=c["id"],
        ))
    return chunks


# ── LangGraph node ────────────────────────────────────────────────────────────


def retrieve(state: RAGState) -> RAGState:
    """
    LangGraph node: fills state["chunks"] from Qdrant, filtered to tenant_id.

    Reads:  state["question"], state["hyde_query"], state["tenant_id"]
    Writes: state["chunks"], state["error"]

    Tenant isolation: every Qdrant search includes a payload filter on tenant_id.
    A missing or empty tenant_id logs a critical warning — it means isolation is broken.
    """
    question  = (state.get("question")  or "").strip()
    tenant_id = (state.get("tenant_id") or "").strip()

    if not question:
        return {**state, "error": "retrieve: question is empty"}

    # Build tenant filter — applied to EVERY Qdrant search in this call
    tenant_filter = _build_tenant_filter(tenant_id)

    try:
        # ── Build dense + sparse vectors for all query texts ──────────────────
        dense_vecs: list[list[float]] = [_embed_query(question)]
        sparse_vecs: list[tuple[list[int], list[float]]] = [bm25_sparse_vector(question)]

        hyde_query = (state.get("hyde_query") or "").strip()
        if hyde_query:
            dense_vecs.append(_embed_query(hyde_query))
            sparse_vecs.append(bm25_sparse_vector(hyde_query))

        # For comparative/multi_company: search with each sub-question independently.
        for sq in (state.get("sub_questions") or []):
            if sq.strip():
                dense_vecs.append(_embed_query(sq.strip()))
                sparse_vecs.append(bm25_sparse_vector(sq.strip()))

        # ── Single hybrid search call: all query vectors fused via RRF ───────
        # Qdrant's RRF naturally deduplicates — no manual seen_ids needed.
        candidates = _search_qdrant(dense_vecs, sparse_vecs, _QDRANT_TOP_K, tenant_filter)

        if not candidates:
            return {**state, "chunks": [], "error": "retrieve: no results found in Qdrant"}

        # ── Rerank → MMR → Parent expansion ───────────────────────────────────
        reranked = _rerank(question, candidates, _RERANK_TOP_N)
        diverse  = _mmr(reranked, _MMR_FINAL_K)

        parent_ids = [
            c["payload"].get("parent_id")
            for c in diverse
            if c["payload"].get("parent_id")
        ]
        parent_texts = _fetch_parents(parent_ids)
        chunks = _to_chunks(diverse, parent_texts)

        logger.info(
            f"[retriever] done — {len(chunks)} chunks "
            f"(searched {len(candidates)}, reranked {len(reranked)}, MMR {len(diverse)})"
        )
        return {**state, "chunks": chunks, "error": None}

    except RuntimeError as e:
        logger.error(f"[retriever] fatal: {e}")
        return {**state, "chunks": [], "error": str(e)}

    except ClientError as e:
        msg = f"retrieve: Bedrock {e.response['Error']['Code']}: {e}"
        logger.error(f"[retriever] {msg}")
        return {**state, "chunks": [], "error": msg}


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tests non-network logic only. Run: python3 -m agents.retriever

    # 1. cosine similarity — identical vectors = 1.0
    assert abs(_cosine_sim([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-5

    # 2. cosine similarity — orthogonal vectors = 0.0
    assert abs(_cosine_sim([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])) < 1e-5

    # 3. cosine similarity — empty vector = 0.0 (no crash)
    assert _cosine_sim([], [1.0, 0.0]) == 0.0

    # 4. MMR: must pick diverse chunks, not near-duplicates
    fake = [
        {"score": 0.9, "vector": [1.0, 0.0, 0.0], "payload": {"text": "a"}},
        {"score": 0.8, "vector": [0.98, 0.02, 0.0], "payload": {"text": "b"}},  # near-dup of a
        {"score": 0.7, "vector": [0.0, 1.0, 0.0],  "payload": {"text": "c"}},  # diverse
    ]
    result = _mmr(fake, 2)
    assert len(result) == 2
    texts = {r["payload"]["text"] for r in result}
    assert "a" in texts, "MMR must select highest-scored chunk"
    assert "c" in texts, "MMR must prefer diverse chunk over near-duplicate"

    # 5. MMR with fewer candidates than final_k returns all
    assert len(_mmr(fake, 10)) == len(fake)

    # 6. _to_chunks falls back to child text when parent_id absent
    cands = [{"score": 0.9, "vector": [], "id": "x", "payload": {"text": "child", "file_name": "f.pdf", "page": 1}}]
    chunks = _to_chunks(cands, {})
    assert chunks[0]["text"] == "child"

    print("retriever.py self-check passed — cosine sim, MMR diversity, fallback text all correct")
