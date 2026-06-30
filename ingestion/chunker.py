"""
Chunker — splits parsed document pages into chunks ready for embedding.

Two strategies, selected by content-aware routing:

  Strategy A (fixed):        Sentence-aware fixed-size windows.
                             512 tokens, 50 token overlap.
                             Never cuts mid-sentence unless a single sentence
                             exceeds the chunk limit (last-resort token split).
                             Used for non-financial documents.

  Strategy B (hierarchical): Parent-child two-level chunking.
                             Parent: 1024 tokens → stored in PostgreSQL.
                             Child:  256 tokens, 30 overlap → stored in Qdrant.
                             Search finds the child. LLM receives the parent.
                             Child page numbers reflect the actual page the
                             child content came from, not the parent's first page.
                             Used for financial documents.

Content classifier modes (set via CHUNK_CLASSIFIER env var):
  keyword  — regex + keyword scoring, zero API cost
  haiku    — Claude Haiku API call, near-perfect accuracy (~$0.00025/1K tokens)
             Falls back to keyword scoring if the API call fails.
  local    — local open-source model via Ollama, full data privacy for
             regulated environments where data must never leave the institution.
             Falls back to keyword scoring if Ollama is not running.

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
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import tiktoken

logger = logging.getLogger(__name__)

# Bedrock client singleton — all AI calls route through AWS, data never leaves VPC.
_bedrock_client      = None
_bedrock_client_lock = threading.Lock()
_HAIKU_MODEL_ID      = os.getenv(
    "BEDROCK_HAIKU_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        with _bedrock_client_lock:
            if _bedrock_client is None:
                import boto3
                _bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=os.getenv("AWS_REGION", "us-east-1"),
                )
    return _bedrock_client

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Hard ceiling from Cohere Embed v3 on Bedrock — no chunk may exceed this
_EMBEDDING_TOKEN_LIMIT = 512


# ── Token helpers ─────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    # disallowed_special=() prevents ValueError on special tokens like <|endoftext|>
    # which appear in SEC filings and corrupted PDFs (same fix applied in embedder).
    return len(_TOKENIZER.encode(text, disallowed_special=()))


_ABBREVS_RE = re.compile(
    r"(?:^|\s)(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|Corp|Inc|Ltd|Co|Dept|U\.S|e\.g|i\.e|etc)\.",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences on . ! ? boundaries.

    Protects known abbreviations (Dr., Corp., U.S., etc.) from triggering
    false splits. Without this, 'Dr. Smith reviewed revenue.' splits into
    ['Dr.', 'Smith reviewed revenue.'] — breaking the sentence and producing
    a 1-token chunk that carries no useful meaning.

    Strategy: temporarily replace periods inside known abbreviations with a
    placeholder, split on real sentence boundaries, then restore.
    """
    if not text.strip():
        return []

    protected = _ABBREVS_RE.sub(
        lambda m: m.group().replace(".", "\x00DOT\x00"), text.strip()
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", protected)
    return [s.replace("\x00DOT\x00", ".").strip() for s in parts if s.strip()]


def _split_at_token_limit(text: str, limit: int) -> List[str]:
    """
    Last-resort token-level split for a single sentence that exceeds the limit.
    Used only when a sentence alone is longer than the chunk size — rare but
    possible in legal disclaimers and dense financial tables formatted as prose.
    """
    tokens = _TOKENIZER.encode(text, disallowed_special=())
    parts = []
    for start in range(0, len(tokens), limit):
        parts.append(_TOKENIZER.decode(tokens[start:start + limit]))
    return parts


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


def _haiku_classify(text: str, file_name: str) -> bool:
    """
    Ask Claude Haiku via AWS Bedrock to classify the document.
    Data never leaves AWS — no external API calls.
    Falls back to keyword scoring on any API failure.
    """
    import json
    try:
        client = _get_bedrock()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 5,
            "messages": [{
                "role": "user",
                "content": (
                    "Is the following document a financial document containing "
                    "earnings data, balance sheets, revenue figures, financial "
                    "analysis, or transaction records? "
                    "Answer with YES or NO only.\n\n"
                    f"{text[:2000]}"
                ),
            }],
        })
        response = client.invoke_model(
            modelId=_HAIKU_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip().upper().startswith("YES")
    except Exception as e:
        logger.warning(
            f"[{file_name}] Haiku classifier failed: {type(e).__name__}: {e}. "
            "Falling back to keyword scoring."
        )
        return _keyword_score(text)


def _local_classify(text: str, file_name: str) -> bool:
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
        logger.warning(
            f"[{file_name}] Local classifier unavailable: {type(e).__name__}: {e}. "
            "Falling back to keyword scoring."
        )
        return _keyword_score(text)


def _is_financial(pages: List[Dict[str, Any]], file_name: str) -> bool:
    """
    Route to the correct classifier based on CHUNK_CLASSIFIER env var.

    keyword → zero cost, regex-based, fast
    haiku   → Claude Haiku API, near-perfect accuracy (default)
    local   → local model via Ollama, full data privacy

    Only samples the first 3 pages for classification — enough signal for any
    financial document. Avoids building a 500K+ character string from a
    500-page document when only the first 2000 characters are ever used.
    """
    sample_text = " ".join(
        p.get("text", "") for p in pages[:3]
    )
    mode = os.getenv("CHUNK_CLASSIFIER", "haiku").lower()

    if mode == "keyword":
        return _keyword_score(sample_text)
    elif mode == "haiku":
        return _haiku_classify(sample_text, file_name)
    elif mode == "local":
        return _local_classify(sample_text, file_name)
    else:
        logger.warning(
            f"[{file_name}] Unknown CHUNK_CLASSIFIER='{mode}'. "
            "Falling back to keyword scoring."
        )
        return _keyword_score(sample_text)


# ── Sentence-aware chunking engine ────────────────────────────────────────────


def _sentences_with_pages(pages: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """
    Flatten pages into (sentence, page_number) tuples.
    Tracks which page each sentence came from so page metadata is accurate
    at both the parent and child level.

    If a single sentence exceeds the chunk size limit, it is split at the
    token level as a last resort. Each token-split piece inherits the same
    page number as the original sentence.
    """
    result: List[Tuple[str, int]] = []
    for page in pages:
        page_num = page.get("page", 0)
        text = page.get("text")

        if text is None:
            logger.warning(f"Page {page_num} is missing 'text' field — skipped")
            continue
        if not isinstance(text, str):
            logger.warning(
                f"Page {page_num} 'text' is {type(text).__name__}, expected str — skipped"
            )
            continue

        for sentence in _split_sentences(text):
            if _count_tokens(sentence) > _EMBEDDING_TOKEN_LIMIT:
                # Last resort: split a monster sentence at token boundaries
                for piece in _split_at_token_limit(sentence, 512):
                    result.append((piece, page_num))
            else:
                result.append((sentence, page_num))
    return result


def _build_chunk_groups(
    sentences: List[Tuple[str, int]],
    chunk_size: int,
    overlap_tokens: int,
) -> List[List[Tuple[str, int]]]:
    """
    Group (sentence, page_number) tuples into chunks.

    Returns List of groups, where each group is a List[Tuple[str, int]].
    Preserving the full (sentence, page) tuples — not just the text — means
    callers retain accurate page numbers for every sentence in every chunk.
    This is what allows child chunks to carry their own correct page number
    rather than inheriting the parent's first page.

    Accumulates sentences until the next sentence would exceed chunk_size.
    Carries overlap_tokens worth of sentences forward into the next group.
    """
    groups: List[List[Tuple[str, int]]] = []
    current: List[Tuple[str, int]] = []
    current_tokens = 0

    for sentence, page in sentences:
        s_tokens = _count_tokens(sentence)

        if current and current_tokens + s_tokens > chunk_size:
            groups.append(list(current))

            # Carry overlap: sentences from the end of the current group
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
        groups.append(list(current))

    return groups


def _group_to_text_and_page(group: List[Tuple[str, int]]) -> Tuple[str, int]:
    """Join a sentence group into (text, first_page_in_group)."""
    return " ".join(s for s, _ in group), group[0][1]


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
    groups = _build_chunk_groups(sentences, chunk_size=512, overlap_tokens=50)
    total = len(groups)
    ts = datetime.now(timezone.utc).isoformat()
    file_ext = os.path.splitext(file_name)[1].lower()

    chunks = []
    for idx, group in enumerate(groups):
        text, page = _group_to_text_and_page(group)
        chunks.append({
            "text": text,
            "metadata": {
                "file_name": file_name,
                "file_extension": file_ext,
                "file_hash": file_hash,
                "page": page,
                "chunk_index": idx,
                "total_chunks": total,
                "strategy": "fixed",
                "parent_id": None,
                "timestamp": ts,
            },
        })

    return {"chunks": chunks, "parents": []}


# ── Strategy B — Hierarchical parent-child ────────────────────────────────────


def _strategy_hierarchical(
    pages: List[Dict[str, Any]],
    file_name: str,
    file_hash: str,
) -> Dict[str, List[Dict]]:
    """
    Two-level chunking for high-value financial documents.

    Parents (1024 tokens, no overlap):
      Stored in PostgreSQL. Returned to the LLM at query time for full context.
      Not embedded — never searched directly.

    Children (256 tokens, 30 overlap):
      Embedded and stored in Qdrant. What gets searched.
      Each child carries parent_id so the retriever can fetch the parent.
      Each child's page number reflects the actual page its content came from,
      not the parent's first page — ensuring citation accuracy.

    When a user asks a question:
      1. Qdrant finds the right child (small = precise match).
      2. Retriever fetches the parent from PostgreSQL via parent_id.
      3. LLM receives the full parent (large = full context).
    """
    sentences = _sentences_with_pages(pages)
    ts = datetime.now(timezone.utc).isoformat()
    file_ext = os.path.splitext(file_name)[1].lower()

    # Build parent groups — each group is the sentences that form one parent
    parent_groups = _build_chunk_groups(sentences, chunk_size=1024, overlap_tokens=0)

    parents: List[Dict] = []
    all_children: List[Dict] = []

    for parent_group in parent_groups:
        parent_text, parent_page = _group_to_text_and_page(parent_group)
        parent_id = str(uuid.uuid4())

        parents.append({
            "id": parent_id,
            "text": parent_text,
            "file_name": file_name,
            "file_extension": file_ext,
            "file_hash": file_hash,
            "page": parent_page,
            "timestamp": ts,
        })

        # Build children from the SAME sentence groups used to build this parent.
        # This preserves per-sentence page numbers so each child knows its exact page.
        child_groups = _build_chunk_groups(
            parent_group, chunk_size=256, overlap_tokens=30
        )

        for child_group in child_groups:
            child_text, child_page = _group_to_text_and_page(child_group)
            all_children.append({
                "text": child_text,
                "metadata": {
                    "file_name": file_name,
                    "file_extension": file_ext,
                    "file_hash": file_hash,
                    "page": child_page,         # actual page, not parent's first page
                    "chunk_index": len(all_children),
                    "total_chunks": None,        # filled after all children collected
                    "strategy": "hierarchical_child",
                    "parent_id": parent_id,
                    "timestamp": ts,
                },
            })

    # Fill in total_chunks now that the full count is known
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
        CHUNK_CLASSIFIER:       "keyword" | "haiku" (default) | "local"
        LOCAL_CLASSIFIER_MODEL: Ollama model name when using local mode
                                (default: "gemma3:1b")
    """
    if not file_name:
        raise ValueError(
            "file_name is required — chunks cannot be cited or traced without a source filename"
        )
    if not file_hash:
        raise ValueError(
            "file_hash is required — chunks cannot be deduplicated without a source hash"
        )

    if not pages:
        logger.warning(
            f"[{file_name}] CHUNK WARNING: called with 0 pages — "
            "parser may have returned empty result, check parse logs for this file"
        )
        return {"chunks": [], "parents": []}

    try:
        financial = _is_financial(pages, file_name)
        strategy = "hierarchical" if financial else "fixed"
        logger.info(
            f"[{file_name}] chunking — classified={'financial' if financial else 'standard'} "
            f"strategy={strategy} "
            f"classifier={os.getenv('CHUNK_CLASSIFIER', 'haiku')}"
        )

        result = (
            _strategy_hierarchical(pages, file_name, file_hash)
            if strategy == "hierarchical"
            else _strategy_fixed(pages, file_name, file_hash)
        )

        if not result["chunks"] and pages:
            logger.warning(
                f"[{file_name}] CHUNK WARNING: produced 0 chunks from {len(pages)} pages — "
                "all pages may contain only whitespace or unparseable content"
            )
        else:
            logger.info(
                f"[{file_name}] CHUNK SUCCESS: {len(result['chunks'])} chunks, "
                f"{len(result['parents'])} parents — strategy={strategy}"
            )
        return result

    except Exception as e:
        logger.error(
            f"[{file_name}] CHUNK FAILED at stage=chunker: "
            f"{type(e).__name__}: {e}"
        )
        raise
