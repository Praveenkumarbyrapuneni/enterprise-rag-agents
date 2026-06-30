from typing import Optional, TypedDict


class RetrievedChunk(TypedDict):
    """One chunk returned from Qdrant, after reranking and MMR."""
    text: str
    score: float
    source: str   # e.g. "Apple_10K_2024.pdf"
    page: int
    doc_id: str


class RAGState(TypedDict):
    """
    Shared state that flows through every agent node in the LangGraph pipeline.

    Query Analyzer  → fills: query_type, sub_questions, hyde_query
    Retriever       → fills: chunks
    Synthesizer     → fills: answer, sources
    Evaluator       → fills: faithfulness, relevance, retry_count
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    question: str

    # ── Query Analyzer ─────────────────────────────────────────────────────────
    query_type: str              # "risk" | "revenue" | "comparison" | "regulatory" | "general"
    sub_questions: list[str]     # split sub-questions for multi-part queries
    hyde_query: str              # hypothetical answer text used as the Qdrant search vector

    # ── Retriever ──────────────────────────────────────────────────────────────
    chunks: list[RetrievedChunk]

    # ── Synthesizer ────────────────────────────────────────────────────────────
    answer: str
    sources: list[str]           # e.g. ["Apple 10-K 2024, page 14"]

    # ── Evaluator ──────────────────────────────────────────────────────────────
    faithfulness: float          # RAGAS score — threshold 0.85
    relevance: float             # RAGAS score — threshold 0.80
    retry_count: int             # max 1 retry before returning best-effort answer

    # ── Error handling ─────────────────────────────────────────────────────────
    error: Optional[str]


def initial_state(question: str) -> RAGState:
    """
    Build a blank state with only the question set.
    Every agent fills in its own fields; this ensures no KeyError on first access.
    """
    return RAGState(
        question=question,
        query_type="",
        sub_questions=[],
        hyde_query="",
        chunks=[],
        answer="",
        sources=[],
        faithfulness=0.0,
        relevance=0.0,
        retry_count=0,
        error=None,
    )


if __name__ == "__main__":
    s = initial_state("What was Apple's iPhone revenue in Q3 2024?")
    assert s["question"] == "What was Apple's iPhone revenue in Q3 2024?"
    assert s["retry_count"] == 0
    assert s["chunks"] == []
    assert s["error"] is None
    print("state.py OK —", list(s.keys()))
