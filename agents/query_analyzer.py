"""
agents/query_analyzer.py — Agent 1: Query Understanding

Runs ONE Claude Haiku call per query. Returns a structured analysis that tells
every downstream agent how to handle the question:

  query_type     → what kind of question is this?
  sub_questions  → if comparative/multi-company: split into independent questions
  hyde_query     → hypothetical answer text for embedding-based search (conceptual only)

Research basis:
  - Query decomposition (+36.7% MRR on multi-hop) but HURTS single-hop — only split
    when 2+ distinct entities/companies are present (ACL 2025, arxiv 2507.00355)
  - HyDE fails on numerical financial queries — LLMs fabricate dollar amounts/percentages
    Use HyDE only for conceptual/temporal queries (confirmed: arxiv 2404.07221)
  - Adaptive HyDE: 25% of cases where HyDE hurts are concept-focused questions
    where precision matters more than semantic expansion (arxiv 2507.16754)
  - Production pattern: classify first, then apply technique — never apply blindly

Query type → technique mapping (baked into the prompt):
  numerical   → direct embed, no HyDE, no split  (numbers: revenue, EPS, ratios)
  conceptual  → HyDE yes, no split               (how/why, strategy, risk, outlook)
  comparative → split into 2 sub-questions, no HyDE  (X vs Y)
  temporal    → HyDE yes, no split               (trends, changes over time)
  regulatory  → no HyDE, no split                (exact terms: LIBOR, SOX, Basel III)
  multi_company → split by company, no HyDE      (JPMorgan and Goldman both...)
  general     → direct embed, no HyDE            (catch-all)

Failure handling:
  - Bedrock timeout/error   → fallback: type=general, no sub-questions, no HyDE
  - JSON parse failure       → same fallback (non-fatal, retriever still works)
  - HyDE text < 20 chars    → discard (LLM returned something useless)
  - Sub-questions > 4       → cap at 4 (over-decomposition guard from research)
  - Sub-questions = 0 for comparative → use original question as single sub-question
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

_HAIKU_MODEL   = os.getenv(
    "BEDROCK_HAIKU_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
_MAX_TOKENS    = 512     # structured JSON output — 512 is more than enough
_TIMEOUT_S     = 15      # Haiku is fast; >15s = something wrong upstream
_MAX_SUB_Q     = 4       # research: >4 sub-questions degrades retrieval
_MIN_HYDE_LEN  = 20      # shorter than this = LLM gave up; discard

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


# ── Prompt ────────────────────────────────────────────────────────────────────

# Few-shot examples cover every query type so Haiku doesn't mis-classify.
# HyDE rule: write in financial document language, NEVER include specific dollar
# amounts or percentages (hallucination risk on numerical queries).
_SYSTEM = """You are a financial query analyzer. Analyze the user query and return ONLY a valid JSON object, no explanation.

Query types:
- "sql"          : needs ONLY live account data — balance, transactions, spending totals, flagged charges
- "hybrid"       : needs BOTH live account data AND document knowledge — e.g. "why was I charged X fee" (needs transaction record + fee policy doc)
- "numerical"    : asks for specific numbers, dollar amounts, percentages, ratios from documents
- "conceptual"   : asks how/why, about strategy, risk, outlook, operations (documents only)
- "comparative"  : compares two or more entities, metrics, or time periods (documents only)
- "temporal"     : asks about trends or changes over time for a single entity (documents only)
- "regulatory"   : asks about specific regulations, compliance terms (LIBOR, SOX, Basel)
- "multi_company": mentions multiple different companies without direct comparison framing
- "general"      : everything else

Rules (MUST follow exactly):
1. Use "sql" for: balance, account total, recent transactions, spending, flagged charges — no document search needed
2. Use "hybrid" for: explain a charge, why was I charged, compare my transaction to policy — needs BOTH data sources
3. hyde_applicable=true ONLY for "conceptual" and "temporal" query types
4. HyDE text must be written in financial document language (analyst/10-K style)
5. HyDE text must NEVER contain specific dollar amounts or percentages — write qualitatively
6. needs_decomposition=true ONLY for "comparative" and "multi_company" (2+ distinct entities)
7. sub_questions: 2-4 self-contained questions, one per entity or distinct aspect
8. sub_questions must be [] and needs_decomposition must be false for all other types

Return this exact JSON structure:
{
  "query_type": "<type>",
  "needs_decomposition": <true or false>,
  "sub_questions": [],
  "hyde_applicable": <true or false>,
  "hyde_query": "<2-3 sentence hypothetical answer in financial doc language, or empty string>"
}"""

_EXAMPLES = """Examples:

Query: "What is my current balance?"
{"query_type":"sql","needs_decomposition":false,"sub_questions":[],"hyde_applicable":false,"hyde_query":""}

Query: "Show me my last 5 transactions"
{"query_type":"sql","needs_decomposition":false,"sub_questions":[],"hyde_applicable":false,"hyde_query":""}

Query: "Do I have any flagged or suspicious transactions?"
{"query_type":"sql","needs_decomposition":false,"sub_questions":[],"hyde_applicable":false,"hyde_query":""}

Query: "Why was I charged a foreign transaction fee on June 26?"
{"query_type":"hybrid","needs_decomposition":false,"sub_questions":[],"hyde_applicable":false,"hyde_query":""}

Query: "Explain the $340 charge from FX Merchant and whether it complies with our fee policy"
{"query_type":"hybrid","needs_decomposition":false,"sub_questions":[],"hyde_applicable":false,"hyde_query":""}

Query: "What was Apple's iPhone revenue in Q3 2024?"
{"query_type":"numerical","needs_decomposition":false,"sub_questions":[],"hyde_applicable":false,"hyde_query":""}

Query: "How did Goldman Sachs manage credit risk in 2025?"
{"query_type":"conceptual","needs_decomposition":false,"sub_questions":[],"hyde_applicable":true,"hyde_query":"Goldman Sachs employs a comprehensive credit risk management framework that includes counterparty exposure limits, collateral requirements, and stress testing under multiple economic scenarios to mitigate potential losses arising from credit events across its trading and lending portfolios."}

Query: "Compare Apple iPhone revenue vs Goldman Sachs trading revenue in Q3 2024"
{"query_type":"comparative","needs_decomposition":true,"sub_questions":["What was Apple iPhone revenue in Q3 2024?","What was Goldman Sachs trading revenue in Q3 2024?"],"hyde_applicable":false,"hyde_query":""}

Query: "How did JPMorgan and Goldman Sachs each perform in investment banking in 2025?"
{"query_type":"multi_company","needs_decomposition":true,"sub_questions":["How did JPMorgan perform in investment banking in 2025?","How did Goldman Sachs perform in investment banking in 2025?"],"hyde_applicable":false,"hyde_query":""}

Query: "What are the LIBOR transition risks mentioned in the filings?"
{"query_type":"regulatory","needs_decomposition":false,"sub_questions":[],"hyde_applicable":false,"hyde_query":""}

Query: "How has Apple's revenue trended over the past three years?"
{"query_type":"temporal","needs_decomposition":false,"sub_questions":[],"hyde_applicable":true,"hyde_query":"Apple has demonstrated consistent revenue growth over the three-year period, driven by sustained demand across its product segments and continued expansion of its services business, with year-over-year performance reflecting both macroeconomic conditions and product cycle dynamics."}

Query: "What did Apple and JPMorgan say about inflation?"
{"query_type":"multi_company","needs_decomposition":true,"sub_questions":["What did Apple say about inflation?","What did JPMorgan say about inflation?"],"hyde_applicable":false,"hyde_query":""}"""


def _build_messages(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"{_SYSTEM}\n\n{_EXAMPLES}\n\n"
                f"Now analyze this query:\nQuery: \"{question}\""
            ),
        }
    ]


# ── LLM call ──────────────────────────────────────────────────────────────────


def _call_haiku(question: str) -> dict:
    """
    One Bedrock call → structured JSON back.
    Raises on Bedrock errors so the caller can apply fallback.
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": _MAX_TOKENS,
        "messages": _build_messages(question),
    })
    response = _get_bedrock().invoke_model(
        modelId=_HAIKU_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    raw = json.loads(response["body"].read())
    text = raw["content"][0]["text"].strip()

    # Strip markdown code fences if the model added them
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return json.loads(text)


# ── Output validation ─────────────────────────────────────────────────────────

_VALID_TYPES = {"numerical", "conceptual", "comparative", "temporal",
                "regulatory", "multi_company", "general", "sql", "hybrid"}


def _validate(raw: dict, question: str) -> dict:
    """
    Sanitise LLM output. Returns a clean, safe dict.
    Never raises — always produces something usable.
    """
    query_type = raw.get("query_type", "general")
    if query_type not in _VALID_TYPES:
        logger.warning(f"[analyzer] Unknown query_type '{query_type}' — defaulting to general")
        query_type = "general"

    needs_decomposition = bool(raw.get("needs_decomposition", False))

    sub_questions = raw.get("sub_questions") or []
    sub_questions = [s for s in sub_questions if isinstance(s, str) and s.strip()]
    sub_questions = sub_questions[:_MAX_SUB_Q]   # cap at 4

    # Guard: comparative/multi_company with no sub-questions = use original question
    if needs_decomposition and not sub_questions:
        logger.warning(
            "[analyzer] needs_decomposition=true but sub_questions empty — "
            "falling back to original question"
        )
        sub_questions = [question]
        needs_decomposition = False

    # Guard: non-decomposable types must have empty sub_questions
    if query_type not in ("comparative", "multi_company"):
        sub_questions = []
        needs_decomposition = False

    hyde_applicable = bool(raw.get("hyde_applicable", False))

    # Research finding: HyDE must only apply to conceptual/temporal
    if query_type not in ("conceptual", "temporal"):
        hyde_applicable = False

    hyde_query = "" if not hyde_applicable else (raw.get("hyde_query", "") or "")
    if not isinstance(hyde_query, str):
        hyde_query = ""
    hyde_query = hyde_query.strip()

    # Discard HyDE text that's too short to be useful
    if hyde_applicable and len(hyde_query) < _MIN_HYDE_LEN:
        logger.warning(
            f"[analyzer] HyDE query too short ({len(hyde_query)} chars) — discarding"
        )
        hyde_query = ""
        hyde_applicable = False

    return {
        "query_type":          query_type,
        "needs_decomposition": needs_decomposition,
        "sub_questions":       sub_questions,
        "hyde_applicable":     hyde_applicable,
        "hyde_query":          hyde_query,
    }


def _fallback(question: str) -> dict:
    """
    Deterministic fallback used when the LLM call or parsing fails.

    This keeps obvious account-data questions on the SQL path during Bedrock
    throttling instead of misrouting them to document RAG.
    """
    q = question.lower()
    hybrid_signals = (
        "why was i charged", "why am i charged", "charged a fee",
        "foreign transaction fee", "fee policy", "complies with policy",
        "compare my transaction", "why was this charge",
    )
    sql_signals = (
        "balance", "account total", "how much money", "last transaction",
        "recent transaction", "transactions", "transaction history",
        "flagged", "suspicious", "unusual", "blocked", "spending",
        "spent", "category", "merchant",
    )
    if any(s in q for s in hybrid_signals):
        query_type = "hybrid"
    elif any(s in q for s in sql_signals):
        query_type = "sql"
    else:
        query_type = "general"

    return {
        "query_type":          query_type,
        "needs_decomposition": False,
        "sub_questions":       [],
        "hyde_applicable":     False,
        "hyde_query":          "",
    }


# ── LangGraph node ────────────────────────────────────────────────────────────


def analyze_query(state: RAGState) -> RAGState:
    """
    LangGraph node: classify the question and prepare the retrieval plan.

    Reads:  state["question"]
    Writes: state["query_type"], state["sub_questions"], state["hyde_query"]
    """
    question = (state.get("question") or "").strip()
    if not question:
        return {**state, "error": "analyze_query: question is empty"}

    try:
        raw    = _call_haiku(question)
        result = _validate(raw, question)

    except (ClientError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(
            f"[analyzer] Failed ({type(e).__name__}: {e}) — "
            "falling back to direct retrieval with original question"
        )
        result = _fallback(question)

    except Exception as e:
        logger.error(f"[analyzer] Unexpected error: {type(e).__name__}: {e}")
        result = _fallback(question)

    # Derive data_source from query_type so graph can route correctly
    query_type = result["query_type"]
    if query_type == "sql":
        data_source = "sql"
    elif query_type == "hybrid":
        data_source = "hybrid"
    else:
        data_source = "rag"

    logger.info(
        f"[analyzer] type={query_type} data_source={data_source} "
        f"sub_questions={len(result['sub_questions'])} "
        f"hyde={'yes' if result['hyde_query'] else 'no'}"
    )

    return {
        **state,
        "query_type":    query_type,
        "sub_questions": result["sub_questions"],
        "hyde_query":    result["hyde_query"],
        "data_source":   data_source,
    }


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tests validation logic only — no Bedrock calls needed.

    # 1. Unknown query_type defaults to general
    r = _validate({"query_type": "unknown_type"}, "test")
    assert r["query_type"] == "general"

    # 2. HyDE only allowed for conceptual and temporal
    r = _validate({"query_type": "numerical", "hyde_applicable": True,
                   "hyde_query": "Some long hypothetical answer text here"}, "test")
    assert r["hyde_applicable"] is False
    assert r["hyde_query"] == ""

    # 3. Sub-questions capped at 4
    r = _validate({
        "query_type": "comparative",
        "needs_decomposition": True,
        "sub_questions": ["q1", "q2", "q3", "q4", "q5"],
    }, "test")
    assert len(r["sub_questions"]) == 4

    # 4. Short HyDE text (<20 chars) is discarded
    r = _validate({
        "query_type": "conceptual",
        "hyde_applicable": True,
        "hyde_query": "Short.",
    }, "test")
    assert r["hyde_query"] == ""
    assert r["hyde_applicable"] is False

    # 5. comparative with empty sub_questions falls back to original question
    r = _validate({
        "query_type": "comparative",
        "needs_decomposition": True,
        "sub_questions": [],
    }, "original question here")
    assert r["sub_questions"] == ["original question here"]
    assert r["needs_decomposition"] is False

    # 6. Non-decomposable types get empty sub_questions regardless
    r = _validate({
        "query_type": "numerical",
        "needs_decomposition": True,
        "sub_questions": ["q1", "q2"],
    }, "test")
    assert r["sub_questions"] == []
    assert r["needs_decomposition"] is False

    # 7. Fallback returns safe defaults
    f = _fallback("anything")
    assert f["query_type"] == "general"
    assert f["sub_questions"] == []
    assert f["hyde_query"] == ""

    print("query_analyzer.py self-check passed — all 7 validation guards correct")
