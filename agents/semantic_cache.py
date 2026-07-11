"""
agents/semantic_cache.py — Semantic cache using a dedicated Qdrant collection.

How it works:
  - On every query: embed the question → search 'semantic_cache' collection
    filtered by tenant_id and created_at > (now - TTL). If similarity >= threshold,
    return the cached answer without running the full pipeline.
  - On cache miss: run the pipeline normally, then store the result here.

Design decisions:
  - Reuses Qdrant (already in the stack) instead of adding Redis vector search.
  - Single 'semantic_cache' collection with tenant_id payload filter — same pattern
    as documents. Scales to millions of tenants without separate collections.
  - TTL enforced by filtering created_at in the search query (no Qdrant TTL needed).
  - Threshold 0.92: high enough to avoid false positives on financial questions where
    "What is my balance?" and "What is my credit limit?" are semantically close but
    semantically different answers. Lower threshold = more hits but wrong answers.
  - SQL queries (balance, transactions) are NOT cached — live data must always be fresh.

Cache key: (question embedding, tenant_id, created_at > now - CACHE_TTL_SECONDS)
"""

import os
import time
import uuid
from typing import Optional

import boto3
import json
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct,
    Range, VectorParams,
)

from .logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_CACHE_COLLECTION   = "semantic_cache"
_CACHE_TTL_SECONDS  = int(os.getenv("CACHE_TTL_SECONDS", "3600"))   # 1 hour default
_CACHE_THRESHOLD    = float(os.getenv("CACHE_THRESHOLD", "0.92"))    # cosine similarity
_EMBED_MODEL        = "cohere.embed-english-v3"
_EMBED_DIMS         = 1024

# Singletons
import threading as _threading
_bedrock_client: Optional[object] = None
_qdrant_client:  Optional[QdrantClient] = None
_bedrock_lock = _threading.Lock()
_qdrant_lock  = _threading.Lock()


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


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(
                    host=os.getenv("QDRANT_HOST", "localhost"),
                    port=int(os.getenv("QDRANT_PORT", "6333")),
                    timeout=10.0,
                )
    return _qdrant_client


def _ensure_collection() -> None:
    """Create the semantic_cache collection if it doesn't exist."""
    qc = _get_qdrant()
    existing = {c.name for c in qc.get_collections().collections}
    if _CACHE_COLLECTION not in existing:
        qc.create_collection(
            collection_name=_CACHE_COLLECTION,
            vectors_config=VectorParams(size=_EMBED_DIMS, distance=Distance.COSINE),
        )
        logger.info(f"[cache] Created collection '{_CACHE_COLLECTION}'")


def _embed(text: str) -> list[float]:
    """Embed a single text via Cohere on Bedrock."""
    body = json.dumps({
        "texts": [text],
        "input_type": "search_query",
        "embedding_types": ["float"],
    })
    resp = _get_bedrock().invoke_model(
        modelId=f"cohere.embed-english-v3",
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    data = json.loads(resp["body"].read())
    return data["embeddings"]["float"][0]


def get_cached(question: str, tenant_id: str, data_source: str) -> Optional[dict]:
    """
    Return a cached result if a semantically similar question was answered recently.

    SQL queries are never cached — live data (balance, transactions) must always be fresh.
    Returns None on any error so a cache failure never blocks the pipeline.
    """
    if data_source in ("sql", "hybrid"):
        return None  # ponytail: never cache live financial data

    try:
        _ensure_collection()
        vector = _embed(question)
        cutoff = time.time() - _CACHE_TTL_SECONDS

        results = _get_qdrant().search(
            collection_name=_CACHE_COLLECTION,
            query_vector=vector,
            limit=1,
            score_threshold=_CACHE_THRESHOLD,
            query_filter=Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                    FieldCondition(key="created_at", range=Range(gte=cutoff)),
                ]
            ),
            with_payload=True,
        )

        if not results:
            return None

        hit = results[0]
        payload = hit.payload
        logger.info(
            f"[cache] HIT — tenant={tenant_id} score={hit.score:.3f} "
            f"question='{question[:60]}'"
        )
        return {
            "answer":       payload["answer"],
            "sources":      payload["sources"],
            "query_type":   payload["query_type"],
            "data_source":  payload["data_source"],
            "faithfulness": payload["faithfulness"],
            "relevance":    payload["relevance"],
            "cached":       True,
        }

    except Exception as e:
        logger.warning(f"[cache] get_cached failed (non-fatal): {type(e).__name__}: {e}")
        return None


def set_cache(question: str, tenant_id: str, result: dict) -> None:
    """
    Store a pipeline result in the semantic cache.

    Never called for SQL results. Errors are non-fatal — a cache write failure
    must never break the response to the caller.
    """
    if result.get("data_source") in ("sql", "hybrid"):
        return

    try:
        _ensure_collection()
        vector = _embed(question)

        _get_qdrant().upsert(
            collection_name=_CACHE_COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "tenant_id":   tenant_id,
                    "question":    question,
                    "answer":      result.get("answer", ""),
                    "sources":     result.get("sources", []),
                    "query_type":  result.get("query_type", ""),
                    "data_source": result.get("data_source", "rag"),
                    "faithfulness": result.get("faithfulness", 0.0),
                    "relevance":   result.get("relevance", 0.0),
                    "created_at":  time.time(),
                },
            )],
        )
        logger.info(f"[cache] SET — tenant={tenant_id} question='{question[:60]}'")

    except Exception as e:
        logger.warning(f"[cache] set_cache failed (non-fatal): {type(e).__name__}: {e}")
