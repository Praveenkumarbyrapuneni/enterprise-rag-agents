"""
api/main.py — FastAPI gateway for the Enterprise RAG system.

Endpoints:
  POST /query  — main query endpoint (authenticated, rate-limited)
  GET  /health — liveness check (no auth, for load balancer)
  GET  /ready  — readiness check (verifies Qdrant + PostgreSQL connectivity)

Auth:
  X-API-Key header. Valid keys are comma-separated in API_KEYS env var.
  Phase A: simple API key.
  Phase B: swap for Cognito JWT verification — same endpoint, different middleware.

Rate limiting:
  In-memory sliding window: 10 requests/minute per API key.
  Phase B: replace with Redis-backed rate limiting for multi-instance ECS deployment.

Tenant routing:
  tenant_id is supplied by the caller in the request body.
  Phase B: extract from JWT claims instead (Cognito User Pool groups → tenant_id).

Run locally:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
from collections import defaultdict, deque
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

load_dotenv()

# Import pipeline — done at module level so the LangGraph graph compiles once
from agents.graph import run_query
from agents.logger import get_logger

logger = get_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    description="Production-grade RAG system for financial institutions.",
    version="3.0.0",
    docs_url="/docs",       # Swagger UI (disable in production)
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Phase B: restrict to your frontend domain
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

_VALID_API_KEYS: set[str] = set(
    k.strip()
    for k in os.getenv("API_KEYS", "dev-key-001").split(",")
    if k.strip()
)


def _get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Validate API key. Phase B: replace with Cognito JWT verification."""
    if x_api_key not in _VALID_API_KEYS:
        logger.warning(f"[api] Rejected request with invalid API key: {x_api_key[:8]}...")
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return x_api_key


# ── Rate limiting (in-memory sliding window) ──────────────────────────────────

_RATE_WINDOW_S: int = 60
_RATE_LIMIT:    int = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))

# api_key → deque of request timestamps within the window
_rate_buckets: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(api_key: str) -> None:
    """Raise 429 if this key has exceeded _RATE_LIMIT requests in the last minute."""
    now    = time.time()
    bucket = _rate_buckets[api_key]

    # Evict timestamps outside the sliding window
    while bucket and bucket[0] < now - _RATE_WINDOW_S:
        bucket.popleft()

    if len(bucket) >= _RATE_LIMIT:
        logger.warning(f"[api] Rate limit exceeded for key {api_key[:8]}...")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {_RATE_LIMIT} requests per minute.",
            headers={"Retry-After": str(_RATE_WINDOW_S)},
        )
    bucket.append(now)


def _auth(api_key: str = Depends(_get_api_key)) -> str:
    """Combined auth + rate-limit dependency."""
    _check_rate_limit(api_key)
    return api_key


# ── Request / Response models ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question:    str
    tenant_id:   str
    customer_id: str = "default_user"

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty")
        if len(v) > 2000:
            raise ValueError("question exceeds 2000 character limit")
        return v

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("tenant_id is required")
        # Only allow safe characters — prevent injection at the application layer
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
        if not all(c in allowed for c in v):
            raise ValueError("tenant_id contains invalid characters (use a-z, 0-9, _, -)")
        return v

    @field_validator("customer_id")
    @classmethod
    def customer_id_safe(cls, v: str) -> str:
        v = v.strip() or "default_user"
        if len(v) > 100:
            raise ValueError("customer_id too long")
        return v


class QueryResponse(BaseModel):
    answer:       str
    sources:      list[str]
    query_type:   str
    data_source:  str
    faithfulness: float
    relevance:    float
    latency_ms:   float
    error:        Optional[str] = None


# ── Health / readiness ────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """
    Readiness probe — verifies Qdrant and PostgreSQL are reachable.
    Returns 503 if any dependency is down.
    """
    status: dict[str, str] = {}
    all_ok = True

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            timeout=5.0,
        )
        qc.get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {type(e).__name__}"
        all_ok = False

    # Check PostgreSQL
    try:
        import psycopg2
        url = os.getenv("DATABASE_URL")
        if url:
            conn = psycopg2.connect(url, connect_timeout=5)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            status["postgres"] = "ok"
        else:
            status["postgres"] = "error: DATABASE_URL not set"
            all_ok = False
    except Exception as e:
        status["postgres"] = f"error: {type(e).__name__}"
        all_ok = False

    if not all_ok:
        return JSONResponse(status_code=503, content={"status": "degraded", **status})
    return {"status": "ok", **status}


# ── Query endpoint ────────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest, api_key: str = Depends(_auth)):
    """
    Run a question through the RAG pipeline.

    The pipeline routes automatically:
      - "What is my balance?" → SQL lookup (no document search)
      - "Why was I charged X?" → Hybrid (SQL + document policy)
      - "What are Apple's risks?" → Document RAG only

    tenant_id scopes ALL searches — results are always isolated to the caller's tenant.
    """
    logger.info(
        f"[api] POST /query — tenant={req.tenant_id} customer={req.customer_id} "
        f"question='{req.question[:80]}'"
    )

    start = time.time()

    try:
        result     = run_query(req.question, tenant_id=req.tenant_id, customer_id=req.customer_id)
        latency_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            f"[api] /query done — {latency_ms}ms "
            f"data_source={result.get('data_source','rag')} "
            f"faith={result.get('faithfulness',0):.2f} "
            f"rel={result.get('relevance',0):.2f}"
        )

        return QueryResponse(
            answer       = result["answer"],
            sources      = result["sources"],
            query_type   = result["query_type"],
            data_source  = result.get("data_source", "rag"),
            faithfulness = result["faithfulness"],
            relevance    = result["relevance"],
            latency_ms   = latency_ms,
            error        = result.get("error"),
        )

    except HTTPException:
        raise   # re-raise FastAPI exceptions as-is

    except Exception as e:
        latency_ms = round((time.time() - start) * 1000, 2)
        msg        = f"{type(e).__name__}: {e}"
        logger.error(f"[api] Unhandled pipeline error after {latency_ms}ms: {msg}")
        raise HTTPException(status_code=500, detail="Internal pipeline error. Please try again.")


# ── Error handlers ────────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Catch-all: never expose internal stack traces to the caller."""
    logger.error(f"[api] Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please contact support."},
    )


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,    # auto-reload on code changes (dev only)
        log_level="info",
    )
