"""
BM25 sparse vector generator — zero new dependencies, pure Python.

Converts text to a sparse vector (parallel index + value lists) compatible
with Qdrant's SparseVector format.

Used in two places:
  ingestion/embedder.py  — generates sparse vector for every chunk at ingest time
  agents/retriever.py    — generates sparse vector for every query at search time

Why BM25 alongside dense vectors:
  Cohere Embed v3 is strong at semantic meaning ("profit trends" ≈ "earnings growth").
  It is weak at exact term retrieval: "SOFR rate 2024-06-30", "Form 10-Q Schedule 14A",
  specific dollar amounts, ticker symbols. BM25 finds exact financial terms that
  dense search misses. Hybrid (dense + BM25 fused via RRF) beats either alone.

Approach:
  - Tokenize: lowercase, strip punctuation, remove stopwords
  - Term frequency: count each unique token in the text
  - Sublinear TF scaling: 1 + log(tf) reduces dominance of repeated terms
  - Token → index: deterministic polynomial hash mod 2^16 (65,536 slots)
    Cross-process stable — does NOT use Python's PYTHONHASHSEED-dependent hash()

Why no IDF:
  IDF requires corpus-level statistics (how many docs contain each term).
  At streaming ingestion time, the full corpus is not yet known.
  For financial documents (narrow, homogeneous domain), stopword removal
  removes most noise. Domain-specific terms (EBITDA, SOFR, Tier 1) are rare
  enough across documents that TF alone scores them correctly.
"""

import math
import re
from typing import List, Tuple

# ── Stopwords ─────────────────────────────────────────────────────────────────
# Common English stopwords — removes high-frequency noise without a library.
# Financial terms (revenue, income, earnings) are intentionally NOT stopwords.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "we", "you", "i",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "how", "when", "where", "why", "all", "any",
    "each", "both", "few", "more", "most", "other", "such", "than", "too",
    "very", "not", "no", "as", "if", "so", "up", "out", "about", "also",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "same", "then", "once", "there", "here",
})

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE  = re.compile(r"\s+")
_VOCAB_BITS = 16
_VOCAB_SIZE  = 1 << _VOCAB_BITS   # 65,536 slots


def _term_index(term: str) -> int:
    """
    Map a term string to a stable integer index [0, _VOCAB_SIZE).

    Uses a djb2-style polynomial hash over the UTF-8 bytes of the term.
    Deterministic across processes and Python versions — unlike Python's
    built-in hash() which varies with PYTHONHASHSEED.
    """
    h = 5381
    for byte in term.encode():
        h = ((h << 5) + h + byte) & 0xFFFFFFFF   # h * 33 + byte, 32-bit
    return h % _VOCAB_SIZE


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stopwords."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return [
        t for t in _SPACE_RE.split(cleaned.strip())
        if t and len(t) > 1 and t not in _STOPWORDS
    ]


def bm25_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
    """
    Convert text to a BM25-style sparse vector.

    Args:
        text: Any document or query text.

    Returns:
        (indices, values) — parallel lists for SparseVector(indices, values).
        Indices are sorted ascending (Qdrant requirement).
        Returns ([], []) for empty or stopword-only text.

    Example:
        indices, values = bm25_sparse_vector("Goldman revenue Q3 2024")
        # → indices like [1234, 5678, 9012, 43210]
        # → values like [1.0, 1.0, 1.0, 1.0]
    """
    if not text or not text.strip():
        return [], []

    tokens = _tokenize(text)
    if not tokens:
        return [], []

    # Term frequency count
    tf: dict[int, int] = {}
    for token in tokens:
        idx = _term_index(token)
        tf[idx] = tf.get(idx, 0) + 1

    # Sublinear TF: 1 + log(tf) — same formula as Lucene/Elasticsearch.
    # Prevents a term appearing 10× from being 10× more important than once.
    indices = sorted(tf.keys())
    values  = [1.0 + math.log(tf[i]) for i in indices]
    return indices, values


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Empty text returns empty vectors
    assert bm25_sparse_vector("") == ([], [])
    assert bm25_sparse_vector("   ") == ([], [])

    # 2. Stopword-only returns empty vectors
    assert bm25_sparse_vector("the and or") == ([], [])

    # 3. Financial terms produce non-empty vectors
    idx, val = bm25_sparse_vector("Goldman Sachs Q3 2024 revenue EBITDA")
    assert len(idx) > 0
    assert len(idx) == len(val)
    assert all(isinstance(i, int) for i in idx)
    assert all(isinstance(v, float) for v in val)

    # 4. Indices are sorted ascending (Qdrant requirement)
    assert idx == sorted(idx)

    # 5. All values are >= 1.0 (TF >= 1 → 1 + log(1) = 1.0)
    assert all(v >= 1.0 for v in val)

    # 6. Repeated term increases value
    idx1, val1 = bm25_sparse_vector("revenue")
    idx2, val2 = bm25_sparse_vector("revenue revenue revenue")
    # The revenue index should be in both and value should be higher in val2
    assert len(idx1) == 1 and len(idx2) == 1
    assert idx1[0] == idx2[0]              # same index
    assert val2[0] > val1[0]              # higher TF → higher value

    # 7. _term_index is deterministic and bounded
    for term in ["revenue", "EBITDA", "Goldman", "Q3", "2024", "sofr"]:
        i = _term_index(term)
        assert 0 <= i < _VOCAB_SIZE, f"index out of range for '{term}': {i}"
        assert _term_index(term) == _term_index(term)  # stable

    print("✅ bm25.py self-check passed")
