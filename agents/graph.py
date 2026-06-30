"""
agents/graph.py — LangGraph pipeline: wires all 4 agents into a single runnable graph

Pipeline flow:

  START
    │
    ▼
  analyze_query   ← classifies question, generates hyde_query + sub_questions
    │
    ▼
  retrieve        ← dense vector search → Cohere rerank → MMR → parent expansion
    │
    ▼
  synthesize      ← lost-in-middle reorder → grounded Claude answer with citations
    │
    ▼
  evaluate        ← deterministic checks + Haiku judge → faithfulness + relevance
    │
    ├── PASS (faith ≥ 0.85, rel ≥ 0.80) ─────────────────────────────────► END
    │
    └── FAIL + retries remaining ──► increment_retry ──► retrieve (retry path)
                                                             │
                                              synthesize ◄──┘
                                                  │
                                              evaluate
                                                  │
                                        FAIL + no retries ──► END
                                        (warning already appended by evaluator)

Why analyze_query is SKIPPED on retry:
  Query classification, sub_questions, and hyde_query are set correctly first time.
  Re-analyzing the same question produces the same output — wasted Bedrock call.
  The retry targets the RETRIEVER: re-run Qdrant search (may get different chunks
  if Qdrant had transient ranking issues) then re-synthesize from new context.

Why a dedicated increment_retry node (not done in evaluator):
  If the evaluator incremented retry_count to signal "I want a retry", the router
  would immediately see the incremented count and think "max retries done → END".
  Off-by-one bug. Clean solution: evaluator only scores, graph owns retry_count.
  increment_retry runs ONLY on the retry path, so retry_count = completed retries.

Error propagation:
  Every agent degrades gracefully — none crash the pipeline:
    - retrieve error    → chunks=[], synthesizer returns "cannot find"
    - synthesize error  → answer="", evaluator scores 0/0 → retry or END
    - evaluate error    → conservative scores (0.7, 0.7) → router decides
  Max 1 retry total. After that the evaluator appends a confidence warning and
  the pipeline ends cleanly with the best answer it managed to produce.

Entry point: run_query(question) → dict with answer, sources, scores, error
"""

import time
from typing import Optional

from langgraph.graph import END, START, StateGraph

from .evaluator import _FAITH_PASS, _MAX_RETRIES, _REL_PASS, _WARNING, _is_cannot_answer
from .logger import get_logger
from .query_analyzer import analyze_query
from .retriever import retrieve
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


# ── Conditional router ────────────────────────────────────────────────────────


def _route_after_evaluate(state: RAGState) -> str:
    """
    Decide: END the pipeline or route back to retrieve for a retry.

    Rules (in order):
      1. Both scores pass                          → "end"
      2. retry_count has reached MAX_RETRIES       → "end"  (evaluator added warning)
      3. IDK answer and retries exhausted          → "end"  (cannot improve further)
      4. Otherwise                                 → "retry" (→ increment_retry → retrieve)

    retry_count here = COMPLETED retries (incremented by _increment_retry node).
    So retry_count=0 on first evaluation → retry allowed.
        retry_count=1 after one retry    → retries exhausted → END.
    """
    faith       = state.get("faithfulness", 0.0)
    rel         = state.get("relevance", 0.0)
    retry_count = state.get("retry_count", 0)
    answer      = state.get("answer") or ""

    if faith >= _FAITH_PASS and rel >= _REL_PASS:
        logger.info(
            f"[graph] Scores passed (faith={faith:.2f} rel={rel:.2f}) → END"
        )
        return "end"

    if retry_count >= _MAX_RETRIES:
        logger.info(
            f"[graph] Max retries reached ({retry_count}/{_MAX_RETRIES}), "
            f"scores faith={faith:.2f} rel={rel:.2f} → END"
        )
        return "end"

    # IDK answer at max retries — evaluator didn't append warning (by design,
    # "cannot find" already communicates uncertainty). End cleanly anyway.
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
    """
    Assemble the LangGraph StateGraph. Called once at module load.
    Returns a compiled graph ready for invoke().
    """
    builder = StateGraph(RAGState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("analyze_query",   analyze_query)
    builder.add_node("retrieve",        retrieve)
    builder.add_node("synthesize",      synthesize)
    builder.add_node("evaluate",        evaluate)
    builder.add_node("increment_retry", _increment_retry)

    # ── Edges — happy path ─────────────────────────────────────────────────────
    builder.add_edge(START,            "analyze_query")
    builder.add_edge("analyze_query",  "retrieve")
    builder.add_edge("retrieve",       "synthesize")
    builder.add_edge("synthesize",     "evaluate")

    # ── Conditional edge — retry or END ───────────────────────────────────────
    builder.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "retry": "increment_retry",   # → increment counter → back to retrieve
            "end":   END,
        },
    )

    # ── Retry path ─────────────────────────────────────────────────────────────
    # After incrementing, go straight to retrieve (skip analyze_query — same question)
    builder.add_edge("increment_retry", "retrieve")

    return builder.compile()


# Build once at import time — reused for every query
_graph = _build_graph()
logger.info("[graph] LangGraph pipeline compiled and ready")


# ── Public entry point ────────────────────────────────────────────────────────


def run_query(question: str) -> dict:
    """
    Main entry point for the RAG pipeline.

    Args:
        question: The user's question in plain English.

    Returns:
        {
            "answer":       str   — grounded answer with inline citations
            "sources":      list  — list of cited sources
            "query_type":   str   — classification (numerical, conceptual, etc.)
            "faithfulness": float — RAGAS-style faithfulness score (0.0–1.0)
            "relevance":    float — answer relevance score (0.0–1.0)
            "retry_count":  int   — how many retries were needed (0 or 1)
            "error":        str | None — pipeline error if any
        }
    """
    question = (question or "").strip()

    if not question:
        logger.warning("[graph] run_query called with empty question")
        return {
            "answer":       "Please provide a question.",
            "sources":      [],
            "query_type":   "general",
            "faithfulness": 0.0,
            "relevance":    0.0,
            "retry_count":  0,
            "error":        "empty question",
        }

    logger.info(f"[graph] START — question='{question[:100]}{'...' if len(question) > 100 else ''}'")
    start = time.time()

    try:
        final   = _graph.invoke(initial_state(question))
        elapsed = time.time() - start

        faith   = final.get("faithfulness", 0.0)
        rel     = final.get("relevance", 0.0)
        retries = final.get("retry_count", 0)
        error   = final.get("error")

        logger.info(
            f"[graph] END — elapsed={elapsed:.2f}s "
            f"faith={faith:.2f} rel={rel:.2f} "
            f"retries={retries} "
            f"error={'none' if not error else error[:60]}"
        )

        return {
            "answer":       final.get("answer", ""),
            "sources":      final.get("sources", []),
            "query_type":   final.get("query_type", "general"),
            "faithfulness": faith,
            "relevance":    rel,
            "retry_count":  retries,
            "error":        error,
        }

    except Exception as e:
        elapsed = time.time() - start
        msg     = f"{type(e).__name__}: {e}"
        logger.error(f"[graph] PIPELINE CRASH after {elapsed:.2f}s — {msg}")
        return {
            "answer":       (
                "The system encountered an unexpected error. "
                "Please try again or contact support."
            ),
            "sources":      [],
            "query_type":   "general",
            "faithfulness": 0.0,
            "relevance":    0.0,
            "retry_count":  0,
            "error":        msg,
        }


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.state import initial_state as _initial_state

    # 1. Graph compiles without error (already done at import — just verify object)
    assert _graph is not None, "Graph must compile at import"

    # 2. Graph has all 5 expected nodes
    node_names = set(_graph.get_graph().nodes.keys())
    for expected in ["analyze_query", "retrieve", "synthesize", "evaluate", "increment_retry"]:
        assert expected in node_names, f"Missing node: {expected}"

    # 3. Router: passing scores → "end"
    s_pass = {
        **_initial_state("test"),
        "faithfulness": 0.90,
        "relevance":    0.85,
        "retry_count":  0,
        "answer":       "good answer [Source: f.pdf, page 1, Chunk #1]",
    }
    assert _route_after_evaluate(s_pass) == "end", "Passing scores must route to end"

    # 4. Router: failing scores, no retries used → "retry"
    s_retry = {
        **_initial_state("test"),
        "faithfulness": 0.70,
        "relevance":    0.75,
        "retry_count":  0,
        "answer":       "answer without citations",
    }
    assert _route_after_evaluate(s_retry) == "retry", "Failing scores with retries left must retry"

    # 5. Router: failing scores, max retries used → "end"
    s_maxed = {
        **_initial_state("test"),
        "faithfulness": 0.70,
        "relevance":    0.75,
        "retry_count":  _MAX_RETRIES,
        "answer":       "answer" + _WARNING,
    }
    assert _route_after_evaluate(s_maxed) == "end", "Max retries exhausted must route to end"

    # 6. Router: IDK answer at max retries → "end"
    from agents.evaluator import _CANNOT_ANSWER
    s_idk_max = {
        **_initial_state("test"),
        "faithfulness": 1.0,
        "relevance":    0.0,
        "retry_count":  _MAX_RETRIES,
        "answer":       _CANNOT_ANSWER,
    }
    assert _route_after_evaluate(s_idk_max) == "end", "IDK at max retries must route to end"

    # 7. Router: IDK answer with retries remaining → "retry"
    s_idk_first = {
        **_initial_state("test"),
        "faithfulness": 1.0,
        "relevance":    0.0,
        "retry_count":  0,
        "answer":       _CANNOT_ANSWER,
    }
    assert _route_after_evaluate(s_idk_first) == "retry", "IDK with retries remaining must retry"

    # 8. increment_retry increments correctly
    s_inc = {**_initial_state("test"), "retry_count": 0}
    result_inc = _increment_retry(s_inc)
    assert result_inc["retry_count"] == 1, "increment_retry must add 1 to retry_count"

    # 9. run_query with empty string returns graceful error (no crash)
    result_empty = run_query("")
    assert result_empty["error"] == "empty question"
    assert result_empty["answer"] == "Please provide a question."

    print("graph.py self-check passed — compilation, nodes, router (all 5 cases), increment, empty-query all correct")
