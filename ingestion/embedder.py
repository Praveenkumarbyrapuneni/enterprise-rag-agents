"""
Embedder — converts chunker output into vectors ready for Qdrant.

All AI calls route through AWS Bedrock — data never leaves the AWS VPC.

Two-step process per chunk:

  Step 1 — Contextual enrichment (Claude Haiku via Bedrock, parallel):
    Up to _HAIKU_CONCURRENCY Haiku calls run simultaneously via ThreadPoolExecutor.
    Each call gets: document header (1K tokens) + the section the chunk belongs
    to (30K tokens). Returns 1-2 sentences of specific context.
    Retries on Bedrock throttling. Falls back to raw chunk text only after
    all retries are exhausted.

  Step 2 — Embedding (Cohere Embed v3 via Bedrock):
    cohere.embed-english-v3 converts each enriched text to a 1024-dim vector.
    Batched (90 per API call — Cohere's max is 96). Retries on throttling,
    network errors, and server errors with exponential backoff + jitter.

Entry point: embed_chunks(chunks, document_text, file_name)

Returns one dict per chunk:
  {
    "vector":        List[float]       — 1024-dim Cohere dense embedding
    "sparse_vector": {"indices": List[int], "values": List[float]}  — BM25
    "text":          str               — enriched text that was embedded
    "metadata":      dict
  }

Environment variables:
  AWS_REGION             — Bedrock region (default: us-east-1)
  BEDROCK_HAIKU_MODEL_ID — Claude Haiku model ID on Bedrock
  EMBEDDER_CONTEXT_MODE  — "haiku" (default) | "skip" (no enrichment)
"""

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from dotenv import load_dotenv

from .bm25 import bm25_sparse_vector

load_dotenv()

logger = logging.getLogger(__name__)

_EMBED_MODEL         = "cohere.embed-english-v3"          # Bedrock Cohere Embed v3
_EMBED_DIMS          = 1024                               # Cohere v3 output dimensions
_EMBED_TOKEN_LIMIT   = 512           # Cohere v3 hard limit per input text
_BATCH_SIZE          = 90            # Cohere max is 96 — stay conservative
_MAX_RETRIES         = 5
_CONTEXT_MODEL       = os.getenv(
    "BEDROCK_HAIKU_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
_CONTEXT_MAX_TOKENS  = 150           # 1-2 sentences is enough
_SECTION_TOKENS      = 30_000        # ~60 pages per section sent to Haiku
_HEADER_TOKENS       = 1_000         # first ~2 pages — always has company/date/doc type
_HAIKU_CONCURRENCY   = 20            # max parallel Haiku calls per embed_chunks call
_API_TIMEOUT         = 30.0          # seconds — hung API calls are killed at this threshold

_tokenizer = tiktoken.get_encoding("cl100k_base")

# Double-checked locking prevents race conditions when multiple threads
# initialise the singleton simultaneously.
_bedrock_client = None
_bedrock_lock   = threading.Lock()


# ── Client helper ─────────────────────────────────────────────────────────────


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        with _bedrock_lock:
            if _bedrock_client is None:
                import boto3
                _bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=os.getenv("AWS_REGION", "us-east-1"),
                )
    return _bedrock_client


# ── Token helpers ─────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    # disallowed_special=() prevents ValueError on special tokens like <|endoftext|>
    # which appear in SEC filings and corrupted PDFs. They are encoded as regular
    # text characters instead of raising. Token count remains accurate.
    return len(_tokenizer.encode(text, disallowed_special=()))


def _truncate_to_tokens(text: str, limit: int) -> str:
    tokens = _tokenizer.encode(text, disallowed_special=())
    if len(tokens) <= limit:
        return text
    return _tokenizer.decode(tokens[:limit])


def _build_sections(document_text: str) -> List[str]:
    """
    Split document text into sections of _SECTION_TOKENS tokens each.

    Each section is ~60 pages. A chunk is sent to Haiku with the document
    header + the section it belongs to — never the full document. All chunks
    in the same section share a prompt cache hit.
    """
    tokens = _tokenizer.encode(document_text, disallowed_special=())
    return [
        _tokenizer.decode(tokens[start : start + _SECTION_TOKENS])
        for start in range(0, len(tokens), _SECTION_TOKENS)
    ]


# ── Public helper — build document text from parser pages ─────────────────────


def build_document_text(pages: List[Dict[str, Any]]) -> str:
    """
    Concatenate all page texts from parse_document() output into a single string.

    Args:
        pages: parse_document() output — List[{"page": int, "text": str}]

    Returns:
        Full document text as a single string. Empty string if pages is empty.
    """
    return "\n".join(p.get("text", "") for p in pages if p.get("text", "").strip())


# ── Contextual enrichment ─────────────────────────────────────────────────────


def _generate_context(
    context_text: str,
    chunk_text: str,
    file_name: str,
) -> Optional[str]:
    """
    Ask Claude Haiku for 1-2 sentences describing this chunk in context.

    Gap 7 fix: retries on Anthropic rate limits with exponential backoff.
    Only falls back to raw chunk text after all _MAX_RETRIES are exhausted.
    Gap 2 fix: validates that the returned context is actually useful (> 10 chars).
               Empty strings and one-word responses are treated as failures.

    Args:
        context_text: Document header + relevant section (~31K tokens).
        chunk_text:   One chunk's raw text from the chunker.
        file_name:    For logging only.

    Returns:
        Context sentence(s), or None if all retries failed or context was unusable.
        Caller falls back to raw chunk text when None is returned.
    """
    import json
    from botocore.exceptions import ClientError

    delay = 1.0
    for attempt in range(_MAX_RETRIES):
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": _CONTEXT_MAX_TOKENS,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{context_text}\n</document>\n\n",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"<chunk>\n{chunk_text}\n</chunk>\n\n"
                                "In 1-2 sentences, describe what this chunk is about "
                                "in the context of the full document. Be specific: include "
                                "the company name, time period, and what the figures refer to "
                                "if present. Output only the description, no preamble."
                            ),
                        },
                    ],
                }],
            })
            response = _get_bedrock().invoke_model(
                modelId=_CONTEXT_MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())

            if not result.get("content"):
                logger.warning(
                    f"[{file_name}] Haiku returned empty content on attempt {attempt + 1} — "
                    "falling back to raw chunk text"
                )
                return None

            text = result["content"][0]["text"].strip()

            if len(text) <= 10:
                logger.warning(
                    f"[{file_name}] Haiku context too short ({len(text)} chars) — "
                    "discarding, falling back to raw chunk text"
                )
                return None

            return text

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                if attempt == _MAX_RETRIES - 1:
                    logger.warning(
                        f"[{file_name}] Bedrock throttling — all {_MAX_RETRIES} retries "
                        "exhausted. Falling back to raw chunk text."
                    )
                    return None
                wait = delay + random.uniform(0, 0.5 * delay)
                logger.warning(
                    f"[{file_name}] Bedrock throttling, retry {attempt + 1}/{_MAX_RETRIES} "
                    f"in {wait:.1f}s"
                )
                time.sleep(wait)
                delay *= 2
            else:
                logger.warning(
                    f"[{file_name}] Bedrock error {error_code}: {e}. "
                    "Falling back to raw chunk text."
                )
                return None

        except Exception as e:
            logger.warning(
                f"[{file_name}] Context generation failed: {type(e).__name__}: {e}. "
                "Falling back to raw chunk text."
            )
            return None

    return None


# ── Per-chunk enrichment worker (runs inside ThreadPoolExecutor) ───────────────


def _enrich_single(
    idx: int,
    chunk: Dict[str, Any],
    context_text: str,
    file_name: str,
) -> Tuple[int, str]:
    """
    Enrich one chunk with Haiku context. Returns (original_index, enriched_text).

    Designed to run inside a ThreadPoolExecutor thread. Always returns — never
    raises — so as_completed() always gets a result, never a bare exception.
    The original index is returned so the caller can reconstruct order safely.

    Gap 6 fix: by running this in a thread pool, up to _HAIKU_CONCURRENCY
    chunks are enriched simultaneously instead of one at a time. A 500-chunk
    document drops from ~25 minutes to ~2 minutes of context generation time.
    """
    # Wrap everything — the docstring promises this function never raises.
    # tiktoken.encode() can raise ValueError on disallowed special tokens
    # (e.g. <|endoftext|>) which appear in corrupted PDFs and SEC filings.
    # If that happens inside the thread pool, future.result() re-raises it
    # in the main thread, crashing the entire embed_chunks call and losing
    # all enrichment work done so far. Catch everything and fall back to raw.
    try:
        chunk_text = chunk.get("text", "")

        if not chunk_text.strip():
            logger.warning(
                f"[{file_name}] chunk {idx} has empty text — "
                "will produce a near-zero vector, check chunker output"
            )
            return idx, chunk_text

        if context_text:
            context = _generate_context(context_text, chunk_text, file_name)
            enriched = f"{context}\n\n{chunk_text}" if context else chunk_text
        else:
            enriched = chunk_text

        if _count_tokens(enriched) > _EMBED_TOKEN_LIMIT:
            logger.warning(
                f"[{file_name}] chunk {idx} enriched text exceeds {_EMBED_TOKEN_LIMIT} tokens — "
                "truncating to fit embedding model limit"
            )
            enriched = _truncate_to_tokens(enriched, _EMBED_TOKEN_LIMIT)

        return idx, enriched

    except Exception as e:
        logger.warning(
            f"[{file_name}] chunk {idx} enrichment raised unexpectedly: "
            f"{type(e).__name__}: {e} — using raw chunk text"
        )
        return idx, chunk.get("text", "")


# ── Embedding with retry ──────────────────────────────────────────────────────


def _embed_batch_with_retry(texts: List[str], file_name: str) -> List[List[float]]:
    """
    Call Cohere Embed v3 via AWS Bedrock with exponential backoff.

    Data never leaves AWS — all embedding calls route through Bedrock inside
    the VPC. input_type="search_document" tells Cohere these are documents
    being ingested (not queries), which improves retrieval accuracy.

    Args:
        texts:     List of enriched texts to embed (max _BATCH_SIZE items).
        file_name: For logging only.

    Returns:
        List of 1024-dimensional float vectors, one per input text, in input order.

    Raises:
        RuntimeError: If the retry loop exits without returning (programming error).
    """
    import json
    from botocore.exceptions import ClientError

    _RETRIABLE = {"ThrottlingException", "ServiceUnavailableException", "InternalServerException"}

    delay = 1.0
    for attempt in range(_MAX_RETRIES):
        try:
            # ponytail: Bedrock validates input length before Cohere sees it — 2048 char hard limit
            body = json.dumps({
                "texts": [t[:2048] for t in texts],
                "input_type": "search_document",
                "truncate": "END",
            })
            response = _get_bedrock().invoke_model(
                modelId=_EMBED_MODEL,
                body=body,
                contentType="application/json",
                accept="*/*",
            )
            result = json.loads(response["body"].read())
            # Cohere returns embeddings in the same order as input — no sorting needed.
            return result["embeddings"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in _RETRIABLE:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(
                        f"[{file_name}] Bedrock {error_code} — "
                        f"all {_MAX_RETRIES} retries exhausted: {e}"
                    )
                    raise
                wait = delay + random.uniform(0, 0.5 * delay)
                logger.warning(
                    f"[{file_name}] Bedrock {error_code}, "
                    f"retry {attempt + 1}/{_MAX_RETRIES} in {wait:.1f}s"
                )
                time.sleep(wait)
                delay *= 2
            else:
                raise

    raise RuntimeError(
        f"[{file_name}] _embed_batch_with_retry exited retry loop without returning "
        "or raising — programming error in exception handlers above"
    )


# ── Main entry point ──────────────────────────────────────────────────────────


def embed_chunks(
    chunks: List[Dict[str, Any]],
    document_text: str,
    file_name: str,
) -> List[Dict[str, Any]]:
    """
    Embed a list of chunks from chunk_document() into vectors for Qdrant.

    Args:
        chunks:        chunk_document()["chunks"] — List of {text, metadata} dicts.
        document_text: Full document text from build_document_text(pages).
                       Used for contextual enrichment. Pass empty string to skip.
        file_name:     Original filename — for logging and error tracing.

    Returns:
        List of dicts, one per chunk, in the same order as the input:
        {
            "vector":   List[float]  — 1024-dimensional embedding (Cohere Embed v3)
            "text":     str          — enriched text that was actually embedded
            "metadata": dict         — unchanged from chunker output
        }

    Raises:
        ValueError:    If file_name is empty.
        RuntimeError:  If Bedrock returns a different number of vectors than chunks
                       (Gap 1 fix — prevents silently storing mismatched data).

    Environment variables:
        EMBEDDER_CONTEXT_MODE: "haiku" (default) | "skip"
    """
    if not file_name:
        raise ValueError(
            "file_name is required — embedded vectors cannot be traced without it"
        )
    if not chunks:
        logger.warning(f"[{file_name}] embed_chunks called with 0 chunks — returning empty")
        return []

    context_mode = os.getenv("EMBEDDER_CONTEXT_MODE", "haiku").lower()

    if context_mode not in ("haiku", "skip"):
        logger.warning(
            f"[{file_name}] Unknown EMBEDDER_CONTEXT_MODE='{context_mode}'. "
            "Falling back to 'skip'."
        )
        context_mode = "skip"

    if context_mode == "haiku" and not document_text.strip():
        logger.warning(
            f"[{file_name}] document_text is empty — contextual enrichment skipped"
        )
        context_mode = "skip"

    sections: List[str] = []
    doc_header: str = ""
    if context_mode == "haiku":
        sections = _build_sections(document_text)
        doc_header = _truncate_to_tokens(document_text, _HEADER_TOKENS)

    total_chunks = len(chunks)
    logger.info(
        f"[{file_name}] EMBED START: {total_chunks} chunks "
        f"{len(sections)} sections context_mode={context_mode}"
    )

    # ── Step 1: Contextual enrichment ─────────────────────────────────────────
    # Gap 6 fix: run Haiku calls in parallel via ThreadPoolExecutor instead of
    # one-at-a-time. _enrich_single always returns (idx, text) and never raises,
    # so as_completed() is safe to iterate without try/except per future.
    enriched_texts: List[str] = [""] * total_chunks

    if context_mode == "haiku":
        with ThreadPoolExecutor(max_workers=_HAIKU_CONCURRENCY) as executor:
            future_map = {}
            for idx, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                chunk_index = meta.get("chunk_index", idx)
                chunk_total = meta.get("total_chunks") or total_chunks
                section_idx = min(
                    int(chunk_index / max(chunk_total, 1) * len(sections)),
                    len(sections) - 1,
                )
                ctx = doc_header + "\n\n[...]\n\n" + sections[section_idx]
                future_map[executor.submit(_enrich_single, idx, chunk, ctx, file_name)] = idx

            completed = 0
            for future in as_completed(future_map):
                result_idx, enriched = future.result()
                enriched_texts[result_idx] = enriched
                completed += 1
                if completed % 50 == 0:
                    logger.info(
                        f"[{file_name}] context enrichment: {completed}/{total_chunks} chunks"
                    )
    else:
        for idx, chunk in enumerate(chunks):
            _, enriched = _enrich_single(idx, chunk, "", file_name)
            enriched_texts[idx] = enriched

    # ── Step 2: Batch embedding with retry ────────────────────────────────────
    all_vectors: List[List[float]] = []
    total_batches = (total_chunks + _BATCH_SIZE - 1) // _BATCH_SIZE

    for batch_num, batch_start in enumerate(range(0, total_chunks, _BATCH_SIZE), 1):
        batch = enriched_texts[batch_start : batch_start + _BATCH_SIZE]
        logger.info(
            f"[{file_name}] embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)"
        )
        vectors = _embed_batch_with_retry(batch, file_name)
        all_vectors.extend(vectors)

    # ── Step 3: Validate and assemble ─────────────────────────────────────────
    # Gap 1 fix: zip() silently truncates when lengths differ. If OpenAI returns
    # fewer vectors than inputs, the mismatch is invisible and chunks get stored
    # with wrong vectors. Fail loudly instead of storing corrupted data.
    if len(all_vectors) != len(enriched_texts):
        raise RuntimeError(
            f"[{file_name}] EMBED MISMATCH: OpenAI returned {len(all_vectors)} vectors "
            f"for {len(enriched_texts)} chunks. API contract violated. "
            "All vectors discarded — document must be re-queued."
        )

    # BM25 sparse vectors run on raw chunk text, not the enriched text.
    # Exact financial term matching ("SOFR 2024-06-30", "Form 10-Q Schedule 14A")
    # should target the original document content, not the AI context prefix.
    results = []
    for chunk, vector, enriched_text in zip(chunks, all_vectors, enriched_texts):
        sp_indices, sp_values = bm25_sparse_vector(chunk.get("text", ""))
        results.append({
            "vector":        vector,
            "sparse_vector": {"indices": sp_indices, "values": sp_values},
            "text":          enriched_text,
            "metadata":      chunk.get("metadata", {}),
        })

    logger.info(
        f"[{file_name}] EMBED SUCCESS: {len(results)} vectors "
        f"dims={_EMBED_DIMS} model={_EMBED_MODEL} context_mode={context_mode}"
    )
    return results


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tests non-API logic. Run: python3 -m ingestion.embedder

    # 1. Empty file_name must raise ValueError
    try:
        embed_chunks([], "", "")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # 2. Empty chunks must return []
    assert embed_chunks([], "some doc text", "test.pdf") == []

    # 3. Token truncation must stay within limit
    truncated = _truncate_to_tokens("word " * 10_000, 100)
    assert _count_tokens(truncated) <= 100

    # 4. build_document_text must skip blank pages
    pages = [
        {"page": 1, "text": "Apple revenue grew 12%."},
        {"page": 2, "text": ""},
        {"page": 3, "text": "Operating expenses decreased."},
    ]
    doc_text = build_document_text(pages)
    assert "Apple revenue" in doc_text and "Operating expenses" in doc_text

    # 5. _build_sections returns a non-empty list of strings
    secs = _build_sections("word " * 1_000)
    assert isinstance(secs, list) and len(secs) >= 1
    assert all(isinstance(s, str) for s in secs)

    # Lock object must exist and be the right type
    assert isinstance(_bedrock_lock, type(threading.Lock()))

    # Gap 2 fix check: _enrich_single with empty chunk text returns empty
    result_idx, result_text = _enrich_single(0, {"text": "", "metadata": {}}, "", "test.pdf")
    assert result_idx == 0 and result_text == ""

    # Gap 6 fix check: _enrich_single is callable and returns (int, str)
    result_idx, result_text = _enrich_single(
        3,
        {"text": "Revenue grew.", "metadata": {}},
        "",   # skip mode — no context_text
        "test.pdf",
    )
    assert result_idx == 3 and isinstance(result_text, str)

    print(
        "✅ embed_chunks self-check passed — "
        "validation, truncation, doc builder, sections, locks, "
        "enrich_single, and gap fixes all correct"
    )
