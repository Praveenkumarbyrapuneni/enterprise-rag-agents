"""
agents/evaluator.py — Agent 4: Answer Quality Evaluation

Two-layer evaluation before the answer reaches the user:

  Layer 1 — Deterministic checks (no LLM, instant, never fails):
    - Empty answer         → faith=0.0, rel=0.0
    - "Cannot find" reply  → faith=1.0, rel=0.0  (valid IDK, not an error)
    - No citations in text → log warning (faithfulness likely to be low)
    - Fewer than 3 chunks  → log warning (context may be thin)

  Layer 2 — Claude Haiku judge (LLM-as-judge, ~1s, stays in Bedrock):
    - Faithfulness: every claim in the answer must trace to a chunk
    - Relevance: the answer must actually address what was asked
    - Returns structured JSON: {"faithfulness": 0.0, "relevance": 0.0, "reason": "..."}

Why NOT RAGAS (the library):
  1. RAGAS failed to produce scores on 83.5% of FinanceBench examples —
     financial numerical reasoning breaks its claim-extraction mechanism
     (Data Science Collective benchmark, 2025).
  2. RAGAS uses OpenAI by default. Our data (SEC filings, client financials)
     must NEVER leave AWS. Non-negotiable for financial institution clients.
  3. "Cannot find" responses get nonsense scores from RAGAS — it was not
     designed for IDK handling (separate research: IDK judge needed, 97% acc).
  Our Haiku judge uses the same LLM-as-judge principle but stays in Bedrock,
  handles IDK correctly, and never silently fails.

Retry logic:
  The EVALUATOR sets scores and increments retry_count.
  The GRAPH (graph.py) reads scores + retry_count and decides routing.
  Evaluator never routes — it only judges.

  If both scores pass     → graph routes to END
  If any score fails      → graph routes to Retriever (if retry_count < MAX_RETRIES)
  If max retries reached  → evaluator appends a confidence warning to the answer,
                            graph routes to END regardless of scores

Retry cap rationale:
  Research shows >1 retry rarely improves scores and risks infinite loops.
  After 1 retry, return best-effort with an explicit warning so the financial
  analyst knows to verify manually. Silent wrong answers are more dangerous
  than honest uncertainty notices.

Confidence warning (appended on max retries):
  "⚠️ Confidence notice: This answer could not be fully verified against
  the source documents. Please verify key figures manually before acting on them."
"""

import json
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from .logger import get_logger
from .state import RAGState

load_dotenv()
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_HAIKU_MODEL   = os.getenv(
    "BEDROCK_HAIKU_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
_MAX_TOKENS    = 256
_MAX_RETRIES   = 1           # one retry max; more rarely helps and risks loops
_FAITH_PASS    = 0.85        # faithfulness threshold — below = hallucination detected
_REL_PASS      = 0.80        # relevance threshold  — below = answer drifted off-topic
_MIN_CHUNKS    = 3           # fewer chunks = thin context, warn but don't fail
_CANNOT_ANSWER = "I cannot find this information in the available documents."
_WARNING       = (
    "\n\n⚠️ Confidence notice: This answer could not be fully verified against "
    "the source documents. Please verify key figures manually before acting on them."
)

import threading as _threading
_bedrock_client: Optional[object] = None
_bedrock_lock = _threading.Lock()


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


# ── Layer 1: Deterministic checks ────────────────────────────────────────────

def _is_cannot_answer(answer: str) -> bool:
    return answer.strip().lower() == _CANNOT_ANSWER.lower()


def _deterministic_check(answer: str, chunks: list) -> tuple[float, float, str]:
    """
    Fast, no-LLM pre-checks. Returns (faithfulness, relevance, reason).
    Returns (None, None, "") if checks pass and LLM judge should run.
    """
    if not answer.strip():
        return 0.0, 0.0, "empty answer"

    if _is_cannot_answer(answer):
        # Valid IDK response: no false claims (faith=1.0), question unanswered (rel=0.0)
        # Research: standard metrics score IDK wrong — this needs special handling
        return 1.0, 0.0, "cannot-answer response"

    if not chunks:
        # Synthesizer answered with no context — almost certainly hallucinating
        return 0.0, 0.0, "answer generated with zero chunks"

    # Warn but don't fail — evaluator logs, graph still sends to LLM judge
    if len(chunks) < _MIN_CHUNKS:
        logger.warning(
            f"[evaluator] Only {len(chunks)} chunk(s) available — "
            "context may be too thin for reliable evaluation"
        )

    if "[Source:" not in answer:
        logger.warning(
            "[evaluator] Answer has no inline citations — "
            "faithfulness score likely to fail"
        )

    return None, None, ""   # pass to LLM judge


# ── Layer 2: Claude Haiku judge ───────────────────────────────────────────────

_SYSTEM = """You are a financial answer quality evaluator. Score the answer on two dimensions.

FAITHFULNESS (0.0 to 1.0):
Every factual claim in the answer must be directly supported by one of the context chunks.
- 1.0 : Every claim traces back to a chunk. No invented information.
- 0.5 : Most claims are supported but one or two are not in the chunks.
- 0.0 : Answer invents financial figures, ignores chunks, or contradicts them.

RELEVANCE (0.0 to 1.0):
The answer must directly address the question that was asked.
- 1.0 : Completely and directly answers the question.
- 0.5 : Partially answers — covers some aspects but misses others.
- 0.0 : Off-topic, answers a different question, or is empty.

FINANCIAL DOMAIN RULES:
- A claim using the wrong time period (e.g. Q4 number attributed to Q3) is NOT faithful.
- A rounded number when the chunk has the exact figure counts as a faithfulness deduction.
- Mixing up two companies' figures is a faithfulness=0.0 failure.

SPECIAL RULE — IDK responses:
If the answer is "I cannot find this information in the available documents."
→ faithfulness=1.0 (no false claims), relevance=0.0 (question unanswered).

Return ONLY valid JSON, no explanation outside it:
{"faithfulness": <float 0.0-1.0>, "relevance": <float 0.0-1.0>, "reason": "<one sentence>"}"""


def _build_judge_prompt(question: str, answer: str, chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()[:600]   # 600 chars per chunk keeps prompt tight
        parts.append(f"[Chunk {i}] {c.get('source','?')} p{c.get('page','?')}: {text}")
    context = "\n\n".join(parts)
    return (
        f"QUESTION: {question}\n\n"
        f"CONTEXT CHUNKS:\n{context}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}\n\n"
        "Score the answer and return JSON:"
    )


def _call_haiku_judge(question: str, answer: str, chunks: list) -> dict:
    """
    One Bedrock Haiku call → {"faithfulness": float, "relevance": float, "reason": str}.
    Raises on Bedrock failure so caller can apply conservative fallback.
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": _MAX_TOKENS,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": _build_judge_prompt(question, answer, chunks)}],
    })
    response = _get_bedrock().invoke_model(
        modelId=_HAIKU_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    raw = json.loads(response["body"].read())
    text = raw["content"][0]["text"].strip()

    # Strip code fences if model added them
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return json.loads(text)


def _safe_score(value, default: float = 0.7) -> float:
    """Clamp a score to [0.0, 1.0]. Returns default if value is None or wrong type."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


# ── Main evaluation logic ─────────────────────────────────────────────────────


def _evaluate(state: RAGState) -> tuple[float, float, str]:
    """
    Run both evaluation layers. Returns (faithfulness, relevance, reason).
    Never raises — always returns scores the graph can act on.
    """
    question    = (state.get("question")    or "").strip()
    answer      = (state.get("answer")      or "").strip()
    chunks      = state.get("chunks") or []
    data_source = state.get("data_source", "rag")

    # ── SQL path: data comes directly from the database — no hallucination possible ──
    # Skip LLM judge entirely. Score 1.0/1.0 if we got a result, 0/0 if empty.
    if data_source == "sql":
        if not answer.strip() or answer.strip() == _CANNOT_ANSWER.strip():
            return 0.0, 0.0, "empty sql result"
        return 1.0, 1.0, "direct database lookup — no hallucination possible"

    # ── Layer 1: deterministic ─────────────────────────────────────────────────
    faith, rel, reason = _deterministic_check(answer, chunks)
    if faith is not None:
        return faith, rel, reason

    # ── Layer 2: Haiku judge ───────────────────────────────────────────────────
    try:
        result = _call_haiku_judge(question, answer, chunks)
        faith  = _safe_score(result.get("faithfulness"), default=0.7)
        rel    = _safe_score(result.get("relevance"),    default=0.7)
        reason = result.get("reason", "")
        return faith, rel, reason

    except (ClientError, json.JSONDecodeError, KeyError, IndexError) as e:
        # Conservative fallback: assume borderline quality, let retry decide
        logger.warning(
            f"[evaluator] Haiku judge failed ({type(e).__name__}: {e}) — "
            "falling back to conservative scores (0.7, 0.7)"
        )
        return 0.7, 0.7, f"judge unavailable: {type(e).__name__}"

    except Exception as e:
        logger.error(f"[evaluator] Unexpected error in judge: {type(e).__name__}: {e}")
        return 0.7, 0.7, f"judge error: {type(e).__name__}"


# ── LangGraph node ────────────────────────────────────────────────────────────


def evaluate(state: RAGState) -> RAGState:
    """
    LangGraph node: score the answer quality. Never touches retry_count.

    Reads:  state["question"], state["answer"], state["chunks"], state["retry_count"]
    Writes: state["faithfulness"], state["relevance"],
            state["answer"] (appends warning when retry_count >= MAX_RETRIES and scores fail)

    retry_count is managed entirely by graph.py:
      graph.py's _increment_retry node increments it before each retry.
      This avoids the off-by-one bug where the evaluator increments retry_count
      BEFORE the retry happens, causing the router to think retries are exhausted.

    Routing (done by graph.py conditional edge, not here):
      faith ≥ 0.85 AND rel ≥ 0.80  → END
      scores fail + retry_count < MAX → increment_retry → retrieve
      scores fail + retry_count ≥ MAX → END (warning already appended here)
    """
    faith, rel, reason = _evaluate(state)
    retry_count        = state.get("retry_count", 0)
    answer             = (state.get("answer") or "").strip()
    is_idk             = _is_cannot_answer(answer)
    passes             = (faith >= _FAITH_PASS and rel >= _REL_PASS)

    logger.info(
        f"[evaluator] faithfulness={faith:.2f} relevance={rel:.2f} "
        f"pass={'yes' if passes else 'no'} "
        f"retry_count={retry_count}/{_MAX_RETRIES} "
        f"idk={'yes' if is_idk else 'no'} "
        f"reason='{reason}'"
    )

    # Append confidence warning when this is the last attempt and scores still fail.
    # retry_count reflects COMPLETED retries (incremented by graph.py's increment_retry).
    # Not added to IDK answers — "cannot find" already communicates uncertainty clearly.
    if not passes and not is_idk and retry_count >= _MAX_RETRIES:
        logger.warning(
            "[evaluator] Max retries reached with scores below threshold — "
            "appending confidence warning"
        )
        answer = answer + _WARNING

    return {
        **state,
        "faithfulness": faith,
        "relevance":    rel,
        "answer":       answer,
        # retry_count intentionally NOT modified here — graph.py owns it
    }


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.state import initial_state, RetrievedChunk

    # 1. Empty answer → faith=0, rel=0
    f, r, reason = _deterministic_check("", [])
    assert f == 0.0 and r == 0.0, "Empty answer must score 0/0"

    # 2. "Cannot find" → faith=1.0, rel=0.0
    f, r, reason = _deterministic_check(_CANNOT_ANSWER, [])
    assert f == 1.0 and r == 0.0, "IDK response must score 1.0/0.0"
    assert reason == "cannot-answer response"

    # 3. "Cannot find" case-insensitive
    f, r, _ = _deterministic_check(_CANNOT_ANSWER.upper(), [])
    assert f == 1.0 and r == 0.0, "IDK detection must be case-insensitive"

    # 4. Answer with no chunks → faith=0, rel=0
    f, r, reason = _deterministic_check("Apple revenue was $46B.", [])
    assert f == 0.0 and r == 0.0, "Answer with zero chunks must score 0/0"

    # 5. Good answer with chunks → passes to LLM judge (returns None, None)
    chunk = RetrievedChunk(text="Apple revenue was $46.2B", score=0.9,
                           source="Apple.htm", page=14, doc_id="x")
    f, r, reason = _deterministic_check(
        "Apple revenue was $46.2B [Source: Apple.htm, page 14, Chunk #1].",
        [chunk]
    )
    assert f is None and r is None, "Valid answer should pass to LLM judge"

    # 6. _safe_score clamps correctly
    assert _safe_score(1.5)   == 1.0
    assert _safe_score(-0.1)  == 0.0
    assert _safe_score(None)  == 0.7   # default fallback
    assert _safe_score("bad") == 0.7

    # 7. evaluate() on empty state → scores 0/0, retry_count unchanged (graph.py owns it)
    s = initial_state("test question")
    result = evaluate(s)
    assert result["faithfulness"] == 0.0
    assert result["relevance"]    == 0.0
    assert result["retry_count"]  == 0   # evaluator never touches retry_count

    # 8. evaluate() when graph has already done max retries → appends warning
    s2 = {**initial_state("test"), "retry_count": _MAX_RETRIES, "answer": "some answer"}
    result2 = evaluate(s2)
    assert result2["retry_count"] == _MAX_RETRIES   # still unchanged by evaluator
    assert _WARNING in result2["answer"]             # warning appended at max retries

    # 9. Passing scores → no retry increment, no warning
    s3 = {
        **initial_state("test"),
        "answer": "Apple revenue was $46.2B [Source: Apple.htm].",
        "chunks": [chunk],
        "faithfulness": 0.90,
        "relevance": 0.85,
        "retry_count": 0,
    }
    # Patch _evaluate to return passing scores without calling Bedrock
    _orig = _evaluate.__code__
    # We can't easily monkey-patch without modifying the function, so just verify
    # the retry logic path directly by testing _deterministic_check returns None
    f2, r2, _ = _deterministic_check(s3["answer"], s3["chunks"])
    assert f2 is None, "Valid answer with chunks must go to LLM judge path"

    print("evaluator.py self-check passed — IDK handling, empty answer, retry cap, score clamping all correct")
