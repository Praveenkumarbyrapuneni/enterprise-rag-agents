"""
api/main.py — FastAPI gateway for the Enterprise RAG system.

Endpoints:
  POST /auth/register — create a user account under a tenant
  POST /auth/login    — returns a signed JWT
  POST /query         — main query endpoint (JWT-authenticated, rate-limited)
  GET  /health        — liveness check (no auth, for load balancer)
  GET  /ready         — readiness check (verifies Qdrant + PostgreSQL connectivity)

Auth:
  JWT Bearer token. Client logs in → receives token with tenant_id baked in.
  /query reads tenant_id from the token — callers never pass it manually.
  Phase B: swap _secret() to AWS Secrets Manager; add Cognito SSO option.

Rate limiting:
  In-memory sliding window: 10 requests/minute per user_id.
  Phase B: replace with Redis-backed rate limiting for multi-instance ECS deployment.

Run locally:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
from collections import defaultdict, deque
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, field_validator

load_dotenv()

# Import pipeline — done at module level so the LangGraph graph compiles once
from agents.graph import run_query
from agents.logger import get_logger
from api.auth import decode_token, router as auth_router

logger = get_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    description="Production-grade RAG system for financial institutions.",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Phase B: restrict to your frontend domain
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth_router)

# ── Auth — JWT Bearer ─────────────────────────────────────────────────────────

_bearer = HTTPBearer()


def _get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """
    Decode JWT from Authorization: Bearer <token> header.
    Returns the token payload: { sub, tenant_id, customer_id, exp }
    """
    return decode_token(creds.credentials)


# ── Rate limiting (in-memory sliding window) ─────────────────────────────────
# Keyed on user_id for /query, on username/IP for /auth/login (brute-force protection).
# Phase B: replace with Redis-backed window for multi-instance ECS deployment.

_RATE_WINDOW_S:    int = 60
_RATE_LIMIT:       int = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
_LOGIN_RATE_LIMIT: int = int(os.getenv("LOGIN_RATE_LIMIT_PER_MIN", "5"))
_rate_buckets: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(key: str, limit: int = _RATE_LIMIT) -> None:
    now    = time.time()
    bucket = _rate_buckets[key]
    while bucket and bucket[0] < now - _RATE_WINDOW_S:
        bucket.popleft()
    if len(bucket) >= limit:
        logger.warning(f"[api] Rate limit exceeded for key {key[:12]}...")
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {_RATE_WINDOW_S} seconds.",
            headers={"Retry-After": str(_RATE_WINDOW_S)},
        )
    bucket.append(now)


def _auth(user: dict = Depends(_get_current_user)) -> dict:
    """Combined JWT auth + rate-limit dependency for /query."""
    _check_rate_limit(user["sub"], _RATE_LIMIT)
    return user


# ── Request / Response models ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty")
        if len(v) > 2000:
            raise ValueError("question exceeds 2000 character limit")
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


# ── Compliance audit log ─────────────────────────────────────────────────────

def _write_audit_log(
    tenant_id: str, customer_id: str,
    question: str, result: dict, latency_ms: float,
) -> None:
    """
    Write every query + response to query_audit_log. Non-fatal — a log failure
    must never break the response to the caller.

    Required for SEC/FINRA compliance: AI responses to customers are business
    communications that must be retained for 7 years with full audit trail.
    """
    import json
    import psycopg2
    import psycopg2.extras
    url = os.getenv("DATABASE_URL")
    if not url:
        return
    try:
        conn = psycopg2.connect(url, connect_timeout=3)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_audit_log
                        (tenant_id, customer_id, question, query_type, data_source,
                         answer, sources, faithfulness, relevance, latency_ms, error)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        tenant_id, customer_id, question,
                        result.get("query_type"), result.get("data_source"),
                        result.get("answer"), psycopg2.extras.Json(result.get("sources", [])),
                        result.get("faithfulness"), result.get("relevance"),
                        latency_ms, result.get("error"),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"[api] audit log write failed (non-fatal): {type(e).__name__}: {e}")


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
def query(req: QueryRequest, user: dict = Depends(_auth)):
    """
    Run a question through the RAG pipeline.

    tenant_id and customer_id come from the JWT — not from the request body.
    The pipeline routes automatically:
      - "What is my balance?" → SQL lookup (no document search)
      - "Why was I charged X?" → Hybrid (SQL + document policy)
      - "What are Apple's risks?" → Document RAG only

    tenant_id scopes ALL searches — results are always isolated to the caller's tenant.
    """
    tenant_id   = user["tenant_id"]
    customer_id = user["customer_id"]

    logger.info(
        f"[api] POST /query — tenant={tenant_id} customer={customer_id} "
        f"question='{req.question[:80]}'"
    )

    start = time.time()

    try:
        result     = run_query(req.question, tenant_id=tenant_id, customer_id=customer_id)
        latency_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            f"[api] /query done — {latency_ms}ms "
            f"data_source={result.get('data_source','rag')} "
            f"faith={result.get('faithfulness',0):.2f} "
            f"rel={result.get('relevance',0):.2f}"
        )

        _write_audit_log(
            tenant_id=tenant_id, customer_id=customer_id,
            question=req.question, result=result, latency_ms=latency_ms,
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
