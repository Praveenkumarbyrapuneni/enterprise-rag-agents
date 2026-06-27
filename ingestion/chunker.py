"""
Chunker — splits parsed document pages into chunks ready for embedding.

Two strategies, selected by content-aware routing:

  Strategy A (fixed):        Sentence-aware fixed-size windows.
                             512 tokens, 50 token overlap.
                             Never cuts mid-sentence.
                             Used for non-financial documents.

  Strategy B (hierarchical): Parent-child two-level chunking.
                             Parent: 1024 tokens → stored in PostgreSQL.
                             Child:  256 tokens, 30 overlap → stored in Qdrant.
                             Search finds the child. LLM receives the parent.
                             Used for financial documents.

Content classifier modes (set via CHUNK_CLASSIFIER env var):
  keyword  — regex + keyword scoring, zero API cost
  haiku    — Claude Haiku API call, near-perfect accuracy (~$0.00025/1K tokens)
  local    — local open-source model via Ollama, full data privacy for
             regulated environments where data must never leave the institution

Single entry point: chunk_document(pages, file_name, file_hash)

Returns:
  {
    "chunks":  List[dict]  → sent to embedder → Qdrant
    "parents": List[dict]  → stored directly in PostgreSQL (empty for Strategy A)
  }
"""

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import tiktoken

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── Token helpers ─────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences on . ! ? boundaries.
    Keeps abbreviations like 'U.S.' from triggering false splits by requiring
    the character after the punctuation to be whitespace followed by a capital.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", text.strip())
    return [s.strip() for s in parts if s.strip()]


# ── Content classifier ────────────────────────────────────────────────────────


_FINANCIAL_KEYWORDS = [
    "revenue", "net income", "ebitda", "gross margin", "earnings per share",
    "cash flow", "balance sheet", "liabilities", "equity", "fiscal year",
    "quarterly results", "annual report", "10-k", "10-q", "operating income",
    "diluted shares", "free cash flow", "guidance", "basis points",
    "net profit", "operating margin", "return on equity", "interest income",
    "loan portfolio", "credit loss", "provision", "tier 1 capital",
    "dividend", "market capitalization", "working capital", "debt ratio",
]

_CURRENCY_RE = re.compile(r"[\$\€\£\¥]\s*[\d,.]+\s*[BMKbmk]?")
_PERCENT_RE  = re.compile(r"\d+\.?\d*\s*%")


def _keyword_score(text: str) -> bool:
    """Score text for financial signals. Returns True if financial."""
    lower = text.lower()
    keyword_hits = sum(1 for kw in _FINANCIAL_KEYWORDS if kw in lower)
    currency_hits = 1 if len(_CURRENCY_RE.findall(text)) >= 3 else 0
    percent_hits  = 1 if len(_PERCENT_RE.findall(text)) >= 3 else 0
    return (keyword_hits + currency_hits + percent_hits) >= 3


def _haiku_classify(text: str) -> bool:
    """
    Ask Claude Haiku to classify the document.
    Haiku understands context — distinguishes a passing mention of revenue
    from a full earnings report. Cost: ~$0.00025 per 1K input tokens.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{
            "role": "user",
            "content": (
                "Is the following document a financial document containing "
                "earnings data, balance sheets, revenue figures, financial "
                "analysis, or transaction records? "
                "Answer with YES or NO only.\n\n"
                f"{text[:2000]}"
            ),
        }],
    )
    return response.content[0].text.strip().upper().startswith("YES")


def _local_classify(text: str) -> bool:
    """
    Ask a local open-source model via Ollama to classify the document.

    Used in regulated financial environments where document data must never
    leave the institution's infrastructure. A small model (Gemma 3 1B,
    Phi-3 Mini, GLM-4) is sufficient for binary YES/NO classification.
    Set LOCAL_CLASSIFIER_MODEL env var to choose the model.

    Falls back to keyword scoring if Ollama is not running.
    """
    try:
        import ollama
        model = os.getenv("LOCAL_CLASSIFIER_MODEL", "gemma3:1b")
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    "Is the following document a financial document containing "
                    "earnings data, balance sheets, revenue figures, financial "
                    "analysis, or transaction records? "
                    "Answer with YES or NO only.\n\n"
                    f"{text[:2000]}"
                ),
            }],
        )
        return response["message"]["content"].strip().upper().startswith("YES")
    except Exception as e:
        logger.warning(f"Local classifier unavailable: {e}. Falling back to keyword scoring.")
        return _keyword_score(text)


def _is_financial(pages: List[Dict[str, Any]]) -> bool:
    """
    Route to the correct classifier based on CHUNK_CLASSIFIER env var.

    keyword → zero cost, regex-based, fast
    haiku   → Claude Haiku API, near-perfect accuracy (default)
    local   → local model via Ollama, full data privacy
    """
    full_text = " ".join(p["text"] for p in pages)
    mode = os.getenv("CHUNK_CLASSIFIER", "haiku").lower()

    if mode == "keyword":
        return _keyword_score(full_text)
    elif mode == "haiku":
        return _haiku_classify(full_text)
    elif mode == "local":
        return _local_classify(full_text)
    else:
        logger.warning(f"Unknown CHUNK_CLASSIFIER='{mode}'. Falling back to keyword scoring.")
        return _keyword_score(full_text)


# ── Sentence-aware chunking engine ────────────────────────────────────────────


def _sentences_with_pages(pages: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """
    Flatten pages into (sentence, page_number) tuples.
    Tracks which page each sentence came from so metadata is accurate.
    """
    result: List[Tuple[str, int]] = []
    for page in pages:
        for sentence in _split_sentences(page["text"]):
            result.append((sentence, page["page"]))
    return result


def _build_chunks(
    sentences: List[Tuple[str, int]],
    chunk_size: int,
    overlap_tokens: int,
) -> List[Tuple[str, int]]:
    """
    Build chunks from (sentence, page_number) tuples.

    Accumulates sentences until the next sentence would exceed chunk_size.
    At that point the current chunk is saved and the last overlap_tokens
    worth of sentences are carried forward into the next chunk.

    Never cuts mid-sentence. Returns List of (chunk_text, first_page_in_chunk).
    """
    chunks: List[Tuple[str, int]] = []
    current: List[Tuple[str, int]] = []
    current_tokens = 0

    for sentence, page in sentences:
        s_tokens = _count_tokens(sentence)

        if current and current_tokens + s_tokens > chunk_size:
            # Save current chunk
            chunks.append((" ".join(s for s, _ in current), current[0][1]))

            # Build overlap: carry sentences from the end of the current chunk
            # until we have enough tokens to fill the overlap window
            overlap: List[Tuple[str, int]] = []
            overlap_count = 0
            for s, p in reversed(current):
                t = _count_tokens(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap.insert(0, (s, p))
                overlap_count += t

            current = overlap
            current_tokens = overlap_count

        current.append((sentence, page))
        current_tokens += s_tokens

    if current:
        chunks.append((" ".join(s for s, _ in current), current[0][1]))

    return chunks


# ── Strategy A — Sentence-aware fixed-size ────────────────────────────────────


def _strategy_fixed(
    pages: List[Dict[str, Any]],
    file_name: str,
    file_hash: str,
) -> Dict[str, List[Dict]]:
    """
    512 tokens per chunk, 50 token overlap, never mid-sentence.
    All chunks go to the embedder → Qdrant. No parents.
    """
    sentences = _sentences_with_pages(pages)
    raw_chunks = _build_chunks(sentences, chunk_size=512, overlap_tokens=50)
    total = len(raw_chunks)
    ts = datetime.now(timezone.utc).isoformat()

    chunks = [
        {
            "text": text,
            "metadata": {
                "file_name": file_name,
                "file_hash": file_hash,
                "page": page,
                "chunk_index": idx,
                "total_chunks": total,
                "strategy": "fixed",
                "parent_id": None,
                "timestamp": ts,
            },
        }
        for idx, (text, page) in enumerate(raw_chunks)
    ]

    return {"chunks": chunks, "parents": []}


# ── Strategy B — Hierarchical parent-child ────────────────────────────────────


def _strategy_hierarchical(
    pages: List[Dict[str, Any]],
    file_name: str,
    file_hash: str,
) -> Dict[str, List[Dict]]:
    """
    Two-level chunking for high-value financial documents.

    Parents (1024 tokens):
      Stored in PostgreSQL. Returned to the LLM at query time for full context.
      Not embedded — never searched directly.

    Children (256 tokens, 30 overlap):
      Embedded and stored in Qdrant. What gets searched.
      Each child carries parent_id so the retriever can fetch the parent.

    When a user asks a question:
      1. Qdrant finds the right child (small = precise match).
      2. Retriever fetches the parent from PostgreSQL via parent_id.
      3. LLM receives the full parent (large = full context).
    """
    sentences = _sentences_with_pages(pages)
    ts = datetime.now(timezone.utc).isoformat()

    raw_parents = _build_chunks(sentences, chunk_size=1024, overlap_tokens=0)
    parents: List[Dict] = []
    all_children: List[Dict] = []

    for parent_text, parent_page in raw_parents:
        parent_id = str(uuid.uuid4())
        parents.append({
            "id": parent_id,
            "text": parent_text,
            "file_name": file_name,
            "file_hash": file_hash,
            "page": parent_page,
            "timestamp": ts,
        })

        # Build children from this parent's text only (not the full document)
        # so children never span across parent boundaries
        parent_sentences = [(s, parent_page) for s in _split_sentences(parent_text)]
        raw_children = _build_chunks(parent_sentences, chunk_size=256, overlap_tokens=30)

        for child_text, child_page in raw_children:
            all_children.append({
                "text": child_text,
                "metadata": {
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "page": child_page,
                    "chunk_index": len(all_children),
                    "total_chunks": None,   # filled after all children collected
                    "strategy": "hierarchical_child",
                    "parent_id": parent_id,
                    "timestamp": ts,
                },
            })

    # Fill in total_chunks now that we know the final count
    total = len(all_children)
    for idx, child in enumerate(all_children):
        child["metadata"]["chunk_index"] = idx
        child["metadata"]["total_chunks"] = total

    return {"chunks": all_children, "parents": parents}


# ── Router ────────────────────────────────────────────────────────────────────


def chunk_document(
    pages: List[Dict[str, Any]],
    file_name: str,
    file_hash: str,
) -> Dict[str, List[Dict]]:
    """
    Single entry point. Classifies document content, routes to strategy.

    Args:
        pages:     Output from parse_document() — List[{"page": int, "text": str}]
        file_name: Original filename, carried in every chunk's metadata for citation.
        file_hash: SHA-256 hash of the original file, used for deduplication tracing.

    Returns:
        {
            "chunks":  List of chunk dicts — sent to embedder → Qdrant
            "parents": List of parent dicts — stored directly in PostgreSQL
                       Empty list when strategy is fixed.
        }

    Environment variables:
        CHUNK_CLASSIFIER: "keyword" | "haiku" (default) | "local"
    """
    if not pages:
        logger.warning(f"chunk_document called with empty pages for {file_name}")
        return {"chunks": [], "parents": []}

    financial = _is_financial(pages)
    strategy = "hierarchical" if financial else "fixed"
    logger.info(
        f"{file_name}: classified as {'financial' if financial else 'standard'} "
        f"→ strategy={strategy} via CHUNK_CLASSIFIER={os.getenv('CHUNK_CLASSIFIER', 'haiku')}"
    )

    if strategy == "hierarchical":
        return _strategy_hierarchical(pages, file_name, file_hash)
    return _strategy_fixed(pages, file_name, file_hash)
