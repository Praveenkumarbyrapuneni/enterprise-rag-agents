"""
agents/graph.py — LangGraph pipeline wiring all agents into one runnable graph.

Phase 3 pipeline — three data paths based on query type:

  ┌─────────────────────────────────────────────────────────────────┐
  │                        START                                    │
  │                          │                                      │
  │                    analyze_query                                │
  │                          │                                      │
  │          ┌───────────────┼───────────────┐                     │
  │         sql            hybrid            rag                   │
  │          │               │               │                     │
  │       db_lookup       db_lookup       retrieve                 │
  │          │               │               │                     │
  │          │            retrieve            │                     │
  │          │               │               │                     │
  │          └───────────────┴───────────────┘                     │
  │                          │                                      │
  │                      synthesize                                 │
  │                          │                                      │
  │                       evaluate                                  │
  │                          │                                      │
  │            ┌─────────────┴─────────────┐                       │
  │           pass                        fail                      │
  │            │                           │                        │
  │           END              (retry_count < MAX)                  │
  │                                        │                        │
  │                              increment_retry → retrieve         │
  │                                             → synthesize        │
  │                                             → evaluate → END    │
  └─────────────────────────────────────────────────────────────────┘

Routing rules:
  After analyze_query:
    data_source == "sql"    → db_lookup → synthesize
    data_source == "hybrid" → db_lookup → retrieve → synthesize
    else                    → retrieve  → synthesize

  After db_lookup:
    data_source == "hybrid" → retrieve
    else                    → synthesize

  After evaluate:
    scores pass OR max retries → END
    scores fail, retries left  → increment_retry → retrieve (re-run RAG path)

Note on SQL retry:
  If a SQL query fails (DB down), evaluator scores 0/0, graph tries one RAG retry.
  The RAG retry is unlikely to help for balance questions, but it's bounded (max 1 retry)
  and ends with the confidence warning — correct graceful degradation.

Entry point: run_query(question, tenant_id, customer_id) → dict
"""

import time
import uuid
from typing import Optional

from langgraph.graph import END, START, StateGraph

from .db_lookup import db_lookup
from .evaluator import _FAITH_PASS, _MAX_RETRIES, _REL_PASS, _WARNING, _is_cannot_answer
from .logger import get_logger
from .query_analyzer import analyze_query
from .retriever import retrieve
from .semantic_cache import get_cached, set_cache
from .state import RAGState, initial_state
from .synthesizer import synthesize
from .evaluator import evaluate

logger = get_logger(__name__)


# ── Retry counter node ────────────────────────────────────────────────────────


def _increment_retry(state: RAGState) -> RAGState:
    """
    Increment retry_count. Runs ONLY on the evaluate→retrieve retry path.
    This is the single place retry_count changes — no off-by-one bugs possible.
    """
    completed = state.get("retry_count", 0)
    new_count  = completed + 1
    logger.info(
        f"[graph] Retry {new_count}/{_MAX_RETRIES} — "
        "re-running retriever with same query vectors"
    )
    return {**state, "retry_count": new_count}


# ── Conditional routers ───────────────────────────────────────────────────────


def _route_after_analyze(state: RAGState) -> str:
    """
    Route after analyze_query based on data_source.
      sql / hybrid → db_lookup first
      everything else → retrieve directly
    """
    data_source = state.get("data_source", "rag")
    if data_source in ("sql", "hybrid"):
        logger.info(f"[graph] analyze_query → db_lookup (data_source={data_source})")
        return "db_lookup"
    logger.info(f"[graph] analyze_query → retrieve (data_source={data_source})")
    return "retrieve"


def _route_after_db_lookup(state: RAGState) -> str:
    """
    Route after db_lookup.
      hybrid → retrieve (need document chunks too)
      sql    → synthesize (db result is the only data needed)
    """
    data_source = state.get("data_source", "rag")
    if data_source == "hybrid":
        logger.info("[graph] db_lookup → retrieve (hybrid: need document chunks too)")
        return "retrieve"
    logger.info("[graph] db_lookup → synthesize (sql: db result is sufficient)")
    return "synthesize"


def _route_after_evaluate(state: RAGState) -> str:
    """
    Decide: END the pipeline or route back to retrieve for a retry.

    Rules (in order):
      1. Both scores pass                          → "end"
      2. retry_count has reached MAX_RETRIES       → "end"  (evaluator added warning)
      3. IDK answer and retries exhausted          → "end"  (cannot improve further)
      4. Otherwise                                 → "retry" (→ increment_retry → retrieve)

    retry_count here = COMPLETED retries (incremented by _increment_retry node).
    """
    faith       = state.get("faithfulness", 0.0)
    rel         = state.get("relevance", 0.0)
    retry_count = state.get("retry_count", 0)
    answer      = state.get("answer") or ""

    if faith >= _FAITH_PASS and rel >= _REL_PASS:
        logger.info(f"[graph] Scores passed (faith={faith:.2f} rel={rel:.2f}) → END")
        return "end"

    if retry_count >= _MAX_RETRIES:
        logger.info(
            f"[graph] Max retries reached ({retry_count}/{_MAX_RETRIES}), "
            f"scores faith={faith:.2f} rel={rel:.2f} → END"
        )
        return "end"

    if _is_cannot_answer(answer) and retry_count >= _MAX_RETRIES:
        logger.info("[graph] IDK answer, retries exhausted → END")
        return "end"

    logger.info(
        f"[graph] Scores failed (faith={faith:.2f} rel={rel:.2f}), "
        f"retry_count={retry_count} < {_MAX_RETRIES} → retry"
    )
    return "retry"


# ── Graph construction ────────────────────────────────────────────────────────


def _build_graph() -> StateGraph:
    """Assemble the LangGraph StateGraph. Called once at module load."""
    builder = StateGraph(RAGState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("analyze_query",   analyze_query)
    builder.add_node("db_lookup",       db_lookup)
    builder.add_node("retrieve",        retrieve)
    builder.add_node("synthesize",      synthesize)
    builder.add_node("evaluate",        evaluate)
    builder.add_node("increment_retry", _increment_retry)

    # ── Entry ──────────────────────────────────────────────────────────────────
    builder.add_edge(START, "analyze_query")

    # ── Route after analyze_query: sql/hybrid → db_lookup, else → retrieve ────
    builder.add_conditional_edges(
        "analyze_query",
        _route_after_analyze,
        {"db_lookup": "db_lookup", "retrieve": "retrieve"},
    )

    # ── Route after db_lookup: hybrid → retrieve, sql → synthesize ────────────
    builder.add_conditional_edges(
        "db_lookup",
        _route_after_db_lookup,
        {"retrieve": "retrieve", "synthesize": "synthesize"},
    )

    # ── Happy path ─────────────────────────────────────────────────────────────
    builder.add_edge("retrieve",   "synthesize")
    builder.add_edge("synthesize", "evaluate")

    # ── Retry or END ───────────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"retry": "increment_retry", "end": END},
    )

    # ── Retry path: after increment, go back to retrieve (skip analyze+db_lookup) ─
    builder.add_edge("increment_retry", "retrieve")

    return builder.compile()


# Build once at import time
_graph = _build_graph()
logger.info("[graph] LangGraph pipeline compiled and ready (Phase 3: sql + hybrid + rag)")


# ── Public entry point ────────────────────────────────────────────────────────


def run_query(
    question:    str,
    tenant_id:   str = "demo",
    customer_id: str = "demo_user_001",
) -> dict:
    """
    Main entry point for the RAG pipeline.

    Checks the semantic cache first. On a hit, returns the cached answer tagged
    with cached=True — the full pipeline is skipped. On a miss, runs the pipeline
    and stores the result in the cache for future hits.

    Returns:
        {
            "query_id":     str   — UUID for feedback submission
            "answer":       str   — grounded answer with inline citations
            "sources":      list  — list of cited sources
            "query_type":   str   — classification (sql, hybrid, numerical, etc.)
            "data_source":  str   — "rag" | "sql" | "hybrid"
            "faithfulness": float — score 0.0–1.0
            "relevance":    float — score 0.0–1.0
            "retry_count":  int   — how many retries were needed (0 or 1)
            "cached":       bool  — True if served from semantic cache
            "error":        str | None
        }
    """
    question = (question or "").strip()
    query_id = str(uuid.uuid4())

    if not question:
        logger.warning("[graph] run_query called with empty question")
        return {
            "query_id":     query_id,
            "answer":       "Please provide a question.",
            "sources":      [],
            "query_type":   "general",
            "data_source":  "rag",
            "faithfulness": 0.0,
            "relevance":    0.0,
            "retry_count":  0,
            "cached":       False,
            "error":        "empty question",
        }

    logger.info(
        f"[graph] START — tenant={tenant_id} customer={customer_id} "
        f"question='{question[:100]}{'...' if len(question) > 100 else ''}'"
    )
    start = time.time()

    # ── Cache check (skip for SQL — live data must always be fresh) ───────────
    # We pass data_source="rag" as a hint; get_cached skips SQL internally too.
    cached = get_cached(question, tenant_id=tenant_id, data_source="rag")
    if cached:
        elapsed = time.time() - start
        logger.info(f"[graph] CACHE HIT — {elapsed*1000:.0f}ms saved")
        return {
            "query_id":     query_id,
            "retry_count":  0,
            **cached,
        }

    try:
        final   = _graph.invoke(initial_state(question, tenant_id=tenant_id, customer_id=customer_id, query_id=query_id))
        elapsed = time.time() - start

        faith   = final.get("faithfulness", 0.0)
        rel     = final.get("relevance", 0.0)
        retries = final.get("retry_count", 0)
        error   = final.get("error")

        logger.info(
            f"[graph] END — elapsed={elapsed:.2f}s "
            f"faith={faith:.2f} rel={rel:.2f} "
            f"retries={retries} data_source={final.get('data_source','rag')} "
            f"error={'none' if not error else error[:60]}"
        )

        result = {
            "query_id":     query_id,
            "answer":       final.get("answer", ""),
            "sources":      final.get("sources", []),
            "query_type":   final.get("query_type", "general"),
            "data_source":  final.get("data_source", "rag"),
            "faithfulness": faith,
            "relevance":    rel,
            "retry_count":  retries,
            "cached":       False,
            "error":        error,
        }

        # Store in cache only if quality is good enough to be worth caching
        if not error and faith >= 0.8 and rel >= 0.75:
            set_cache(question, tenant_id=tenant_id, result=result)

        return result

    except Exception as e:
        elapsed = time.time() - start
        msg     = f"{type(e).__name__}: {e}"
        logger.error(f"[graph] PIPELINE CRASH after {elapsed:.2f}s — {msg}")
        return {
            "query_id":     query_id,
            "answer":       (
                "The system encountered an unexpected error. "
                "Please try again or contact support."
            ),
            "sources":      [],
            "query_type":   "general",
            "data_source":  "rag",
            "faithfulness": 0.0,
            "relevance":    0.0,
            "retry_count":  0,
            "cached":       False,
            "error":        msg,
        }


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.state import initial_state as _initial_state
    from agents.evaluator import _CANNOT_ANSWER, _WARNING

    # 1. Graph compiles without error
    assert _graph is not None

    # 2. Graph has all 6 expected nodes
    node_names = set(_graph.get_graph().nodes.keys())
    for expected in ["analyze_query", "db_lookup", "retrieve", "synthesize", "evaluate", "increment_retry"]:
        assert expected in node_names, f"Missing node: {expected}"

    # 3. After analyze: sql/hybrid → db_lookup, rag → retrieve
    assert _route_after_analyze({**_initial_state("test"), "data_source": "sql"})    == "db_lookup"
    assert _route_after_analyze({**_initial_state("test"), "data_source": "hybrid"}) == "db_lookup"
    assert _route_after_analyze({**_initial_state("test"), "data_source": "rag"})    == "retrieve"
    assert _route_after_analyze({**_initial_state("test"), "data_source": ""})       == "retrieve"

    # 4. After db_lookup: hybrid → retrieve, sql → synthesize
    assert _route_after_db_lookup({**_initial_state("test"), "data_source": "hybrid"}) == "retrieve"
    assert _route_after_db_lookup({**_initial_state("test"), "data_source": "sql"})    == "synthesize"

    # 5. Evaluate router: passing scores → "end"
    s_pass = {**_initial_state("test"), "faithfulness": 0.90, "relevance": 0.85,
               "retry_count": 0, "answer": "good answer [Source: f.pdf, page 1, Chunk #1]"}
    assert _route_after_evaluate(s_pass) == "end"

    # 6. Evaluate router: failing scores, no retries used → "retry"
    s_retry = {**_initial_state("test"), "faithfulness": 0.70, "relevance": 0.75,
                "retry_count": 0, "answer": "answer without citations"}
    assert _route_after_evaluate(s_retry) == "retry"

    # 7. Evaluate router: max retries → "end"
    s_maxed = {**_initial_state("test"), "faithfulness": 0.70, "relevance": 0.75,
                "retry_count": _MAX_RETRIES, "answer": "answer" + _WARNING}
    assert _route_after_evaluate(s_maxed) == "end"

    # 8. increment_retry increments correctly
    s_inc = {**_initial_state("test"), "retry_count": 0}
    assert _increment_retry(s_inc)["retry_count"] == 1

    # 9. run_query with empty string → graceful error, no crash
    result = run_query("")
    assert result["error"] == "empty question"
    assert result["answer"] == "Please provide a question."

    # 10. run_query returns data_source field
    assert "data_source" in result

    print("graph.py self-check passed — all 6 nodes, all routers, empty-query, data_source field all correct")
