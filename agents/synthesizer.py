"""
agents/synthesizer.py — Agent 3: Answer Synthesis

Reads the chunks from the retriever and writes a grounded, cited answer.

Research basis:
  - Lost-in-the-middle (ICLR 2025): LLMs have a U-shaped attention curve —
    they read the beginning and end carefully but ignore the middle.
    Fix: reorder chunks so best is FIRST, second-best is LAST, rest in middle.
    (arxiv 2410.05983, ICLR 2025 proceedings)

  - Citation hallucination (FACTUM, arxiv 2601.05866): models cite sources
    that don't support the claim. Root cause: "attributional drift" — the model
    loses track of which chunk said what during long-form generation.
    Fix: number each chunk explicitly, require chunk number in every inline citation.

  - Financial-specific hallucination (arxiv 2602.05723):
    55% temporal confusion (Q3 figure attributed to Q4)
    28% fiscal/calendar year confusion
    17% rounding errors (exact value becomes approximation)
    Fix: prompt explicitly requires time period + exact values + no paraphrasing.

  - Post-rationalization: model generates from its own memory then finds a
    chunk to justify it. Fix: "use ONLY the chunks below, nothing else."

  - Empty context: never call Claude with no chunks — return "cannot answer"
    immediately and let the evaluator route to a retry.

Failure handling:
  - Empty chunks          → immediate "cannot answer", no Claude call
  - All chunks empty text → same
  - Bedrock timeout       → set error in state
  - Bedrock ClientError   → set error in state
  - "Cannot answer" reply → valid answer, not an error (evaluator handles quality)
  - No citations in answer → log warning (evaluator will flag faithfulness)
"""

import json
import logging
import os
import re
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from .logger import get_logger
from .state import RAGState, RetrievedChunk

load_dotenv()
logger = get_logger(__name__)

_SONNET_MODEL  = os.getenv(
    "BEDROCK_SONNET_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
)
_MAX_TOKENS    = 1024     # 200-500 word answer = 300-700 tokens; 1024 gives headroom
_TIMEOUT_S     = 45       # Sonnet + long context is slower than Haiku
_MAX_CHUNK_LEN = 3000     # characters per chunk in the prompt (parent chunks can be long)
_CANNOT_ANSWER     = "I cannot find this information in the available documents."
_CANNOT_ANSWER_SQL = "I cannot find this information in your transaction records."

_bedrock_client: Optional[object] = None


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    return _bedrock_client


# ── Step 1: Lost-in-the-middle reordering ────────────────────────────────────


def _reorder_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Reorder chunks to fight the lost-in-the-middle attention problem.

    LLMs attend strongly to beginning and end of context, weakly to the middle.
    Strategy: best chunk first, second-best chunk last, everything else in between.

    Input (sorted best→worst): [0, 1, 2, 3, 4, 5]
    Output:                    [0, 2, 3, 4, 5, 1]
                                ↑ best first        ↑ second-best last
    """
    if len(chunks) <= 2:
        return chunks
    return [chunks[0]] + chunks[2:] + [chunks[1]]


# ── Step 2: Build the prompt ─────────────────────────────────────────────────


_SYSTEM = """You are a financial analyst answering questions from retrieved document chunks.

STRICT RULES — follow all of them exactly:
1. Answer ONLY using the numbered chunks provided below. Use no other knowledge.
2. After EVERY factual claim, add an inline citation in this exact format:
   [Source: filename, page N, Chunk #K]
3. Always include the exact time period (quarter, year, date) when citing any financial figure.
4. Never paraphrase, estimate, or round numbers. Use the exact value written in the chunk.
5. Never say "based on my knowledge" or "I know that" — only reference the chunks.
6. If the answer is NOT present in any chunk, respond with exactly this sentence:
   I cannot find this information in the available documents.
7. Do not combine information from different companies without making it explicit which company each figure belongs to."""

_CANNOT_ANSWER_LOWER = _CANNOT_ANSWER.lower()


def _build_hybrid_prompt(question: str, sql_result: str, chunks: list[RetrievedChunk]) -> str:
    """
    For hybrid queries: combine live transaction data + document chunks into one prompt.
    Claude synthesizes both sources into one grounded answer.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = (chunk["text"] or "").strip()
        if len(text) > _MAX_CHUNK_LEN:
            text = text[:_MAX_CHUNK_LEN] + "... [truncated]"
        context_parts.append(
            f"[Document Chunk #{i}] Source: {chunk['source']}, page {chunk['page']}\n{text}"
        )
    doc_context = "\n\n".join(context_parts)
    return (
        f"LIVE TRANSACTION DATA (from account database):\n{sql_result}\n\n"
        f"POLICY / DOCUMENT CONTEXT:\n{doc_context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (cite document chunks inline using [Source: filename, page N, Chunk #K]):"
    )


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Assemble the context + question into the user message.
    Chunks are already reordered for lost-in-the-middle.
    Each chunk is truncated to _MAX_CHUNK_LEN chars to keep the prompt manageable.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = (chunk["text"] or "").strip()
        if len(text) > _MAX_CHUNK_LEN:
            text = text[:_MAX_CHUNK_LEN] + "... [truncated]"
        context_parts.append(
            f"[Chunk #{i}] Source: {chunk['source']}, page {chunk['page']}\n{text}"
        )

    context = "\n\n".join(context_parts)
    return (
        f"CONTEXT CHUNKS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER (cite every claim inline using [Source: filename, page N, Chunk #K]):"
    )


# ── Step 3: Call Claude Sonnet ────────────────────────────────────────────────


def _call_sonnet(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    One Bedrock call → grounded answer string.
    Raises ClientError on Bedrock failure so caller can set state["error"].
    """
    user_message = _build_prompt(question, chunks)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": _MAX_TOKENS,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user_message}],
    })

    response = _get_bedrock().invoke_model(
        modelId=_SONNET_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


# ── Step 4: Extract sources from inline citations ─────────────────────────────


def _extract_sources(answer: str) -> list[str]:
    """
    Pull every [Source: ...] citation out of the answer text.
    Returns a deduplicated list preserving first-occurrence order.
    """
    found = re.findall(r'\[Source:\s*([^\]]+)\]', answer)
    # dict.fromkeys deduplicates while preserving insertion order
    return list(dict.fromkeys(s.strip() for s in found))


# ── Step 5: Validate the answer ───────────────────────────────────────────────


def _validate_answer(answer: str) -> None:
    """Log warnings for quality issues. Does not raise — evaluator handles scoring."""
    if answer.lower().strip() == _CANNOT_ANSWER_LOWER:
        return   # valid "no answer" response, nothing to warn about

    sources = _extract_sources(answer)
    if not sources:
        logger.warning(
            "[synthesizer] Answer contains no inline citations — "
            "evaluator faithfulness score will likely fail"
        )

    if len(answer.strip()) < 50:
        logger.warning(
            f"[synthesizer] Answer very short ({len(answer)} chars) — "
            "may indicate Claude returned an incomplete response"
        )


# ── LangGraph node ────────────────────────────────────────────────────────────


def synthesize(state: RAGState) -> RAGState:
    """
    LangGraph node: write a grounded answer from chunks and/or sql_result.

    Reads:  state["question"], state["chunks"], state["sql_result"], state["data_source"]
    Writes: state["answer"], state["sources"], state["error"]

    Three paths:
      sql    → format sql_result directly (no Claude call, no hallucination possible)
      rag    → existing behavior: Claude synthesizes from document chunks
      hybrid → Claude synthesizes from BOTH sql_result and document chunks
    """
    question    = (state.get("question")    or "").strip()
    data_source = (state.get("data_source") or "rag")
    sql_result  = state.get("sql_result")
    chunks      = state.get("chunks") or []

    # ── SQL-only: format database result directly — no Claude needed ──────────
    if data_source == "sql":
        if not sql_result:
            logger.warning("[synthesizer] sql data_source but sql_result is empty")
            return {
                **state,
                "answer":  _CANNOT_ANSWER_SQL,
                "sources": [],
                "error":   None,
            }
        logger.info(f"[synthesizer] sql path — {len(sql_result)} chars from DB")
        return {
            **state,
            "answer":  sql_result,
            "sources": ["Live transaction database"],
            "error":   None,
        }

    # ── Hybrid: Claude combines sql_result + document chunks ──────────────────
    if data_source == "hybrid":
        usable = [c for c in chunks if (c.get("text") or "").strip()]
        if not sql_result and not usable:
            return {
                **state,
                "answer":  _CANNOT_ANSWER,
                "sources": [],
                "error":   None,
            }
        # If only sql_result (no chunks retrieved), treat as sql path
        if not usable and sql_result:
            return {
                **state,
                "answer":  sql_result,
                "sources": ["Live transaction database"],
                "error":   None,
            }
        # Both available — Claude synthesizes
        if not sql_result:
            sql_result = "No live transaction data available for this query."

        try:
            ordered = _reorder_chunks(usable)
            user_message = _build_hybrid_prompt(question, sql_result, ordered)
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": _MAX_TOKENS,
                "system": _SYSTEM,
                "messages": [{"role": "user", "content": user_message}],
            })
            response = _get_bedrock().invoke_model(
                modelId=_SONNET_MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            answer  = result["content"][0]["text"].strip()
            sources = _extract_sources(answer)
            if "Live transaction database" not in sources:
                sources = ["Live transaction database"] + sources
            _validate_answer(answer)
            logger.info(f"[synthesizer] hybrid path — {len(answer)} chars, {len(sources)} sources")
            return {**state, "answer": answer, "sources": sources, "error": None}

        except ClientError as e:
            msg = f"synthesize: Bedrock {e.response['Error']['Code']}: {e}"
            logger.error(f"[synthesizer] hybrid failed: {msg}")
            # Graceful degradation: return just the sql_result
            return {
                **state,
                "answer":  f"{sql_result}\n\n(Document context unavailable due to service error.)",
                "sources": ["Live transaction database"],
                "error":   msg,
            }

        except Exception as e:
            msg = f"synthesize: unexpected error: {type(e).__name__}: {e}"
            logger.error(f"[synthesizer] {msg}")
            return {**state, "answer": sql_result, "sources": ["Live transaction database"], "error": msg}

    # ── RAG-only: existing behavior (document chunks → Claude answer) ─────────
    usable = [c for c in chunks if (c.get("text") or "").strip()]
    if not usable:
        logger.warning("[synthesizer] No usable chunks — returning cannot-answer without LLM call")
        return {
            **state,
            "answer":  _CANNOT_ANSWER,
            "sources": [],
            "error":   None,
        }

    try:
        ordered = _reorder_chunks(usable)
        answer  = _call_sonnet(question, ordered)
        sources = _extract_sources(answer)
        _validate_answer(answer)

        logger.info(
            f"[synthesizer] rag path — {len(answer)} chars "
            f"sources={len(sources)} "
            f"cannot_answer={'yes' if _CANNOT_ANSWER_LOWER in answer.lower() else 'no'}"
        )
        return {
            **state,
            "answer":  answer,
            "sources": sources,
            "error":   None,
        }

    except ClientError as e:
        msg = f"synthesize: Bedrock {e.response['Error']['Code']}: {e}"
        logger.error(f"[synthesizer] {msg}")
        return {**state, "answer": "", "sources": [], "error": msg}

    except Exception as e:
        msg = f"synthesize: unexpected error: {type(e).__name__}: {e}"
        logger.error(f"[synthesizer] {msg}")
        return {**state, "answer": "", "sources": [], "error": msg}


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tests all non-LLM logic. No Bedrock calls needed.

    # 1. Lost-in-the-middle reorder: best first, second-best last
    chunks = [
        RetrievedChunk(text="a", score=0.9, source="f1.pdf", page=1, doc_id="1"),
        RetrievedChunk(text="b", score=0.8, source="f2.pdf", page=2, doc_id="2"),
        RetrievedChunk(text="c", score=0.7, source="f3.pdf", page=3, doc_id="3"),
        RetrievedChunk(text="d", score=0.6, source="f4.pdf", page=4, doc_id="4"),
    ]
    reordered = _reorder_chunks(chunks)
    assert reordered[0]["text"] == "a",  "Best chunk must be first"
    assert reordered[-1]["text"] == "b", "Second-best chunk must be last"
    assert reordered[0]["text"] not in [r["text"] for r in reordered[1:-1]], \
        "Best chunk must not appear in middle"

    # 2. Reorder with ≤2 chunks returns unchanged
    two = chunks[:2]
    assert _reorder_chunks(two) == two

    # 3. Source extraction — inline citations parsed correctly
    answer = (
        "Apple iPhone revenue was $46.2B [Source: Apple.htm, page 14, Chunk #1]. "
        "Goldman trading revenue was $8.1B [Source: Goldman.htm, page 7, Chunk #3]."
    )
    sources = _extract_sources(answer)
    assert len(sources) == 2
    assert "Apple.htm, page 14, Chunk #1" in sources
    assert "Goldman.htm, page 7, Chunk #3" in sources

    # 4. Source extraction deduplicates
    dup_answer = (
        "Revenue grew [Source: Apple.htm, page 5, Chunk #2]. "
        "Revenue grew further [Source: Apple.htm, page 5, Chunk #2]."
    )
    assert len(_extract_sources(dup_answer)) == 1

    # 5. Empty chunks → cannot-answer returned without LLM call
    from agents.state import initial_state
    s = initial_state("What was Apple revenue?")
    result = synthesize(s)
    assert result["answer"] == _CANNOT_ANSWER
    assert result["sources"] == []
    assert result["error"] is None

    # 6. Chunks with only whitespace text treated as empty
    s2 = {**initial_state("test"), "chunks": [
        RetrievedChunk(text="   ", score=0.9, source="f.pdf", page=1, doc_id="x")
    ]}
    result2 = synthesize(s2)
    assert result2["answer"] == _CANNOT_ANSWER

    # 7. Prompt contains chunk numbers and source filenames
    prompt = _build_prompt("test question", chunks[:2])
    assert "[Chunk #1]" in prompt
    assert "[Chunk #2]" in prompt
    assert "f1.pdf" in prompt
    assert "test question" in prompt

    print("synthesizer.py self-check passed — reorder, citations, empty-context, prompt format all correct")
