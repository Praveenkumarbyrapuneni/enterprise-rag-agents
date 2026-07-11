"""
api/main.py — FastAPI gateway for the Enterprise RAG system.

Endpoints:
  POST /auth/register — create a user account under a tenant
  POST /auth/login    — returns a signed JWT
  POST /query         — main query endpoint (JWT-authenticated, rate-limited)
  POST /feedback      — submit helpful/not-helpful feedback on a query response
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
import re
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
from api.feedback_router import router as feedback_router
from api.rate_limit import check_rate_limit, RATE_LIMIT

logger = get_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(
    title="Enterprise RAG API",
    description="Production-grade RAG system for financial institutions.",
    version="4.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    # Set ALLOWED_ORIGINS in .env for production (e.g. "https://app.yourbank.com").
    # Wildcard is only safe during local development — a real origin must be set
    # before this API serves financial data from a browser client.
    allow_origins=_ALLOWED_ORIGINS if _ALLOWED_ORIGINS else ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth_router)
app.include_router(feedback_router)


# ── Fork-safe startup hook ────────────────────────────────────────────────────

@app.on_event("startup")
async def _reset_singletons() -> None:
    """
    Reset all process-level boto3 and Qdrant singletons on worker startup.

    Problem this solves:
      uvicorn --workers N uses fork() to spawn N worker processes. If any
      boto3 or Qdrant client is created BEFORE the fork (e.g., during a
      health check warm-up), all workers inherit the SAME client object.
      boto3 clients are not fork-safe — concurrent AWS API calls from
      different OS processes on the same TCP connection produce ConnectionReset
      and BrokenPipe errors that look like random Bedrock failures.

    Fix: reset every singleton to None on startup. Each worker then creates
    its own client independently on first use. The double-checked locking
    in each _get_bedrock() function ensures only one client per worker.
    """
    import agents.retriever    as _r
    import agents.synthesizer  as _s
    import agents.evaluator    as _e
    import agents.query_analyzer as _q
    _r._bedrock_client = None
    _r._qdrant_client  = None
    _s._bedrock_client = None
    _e._bedrock_client = None
    _q._bedrock_client = None
    logger.info("[startup] singletons reset — each worker will create its own clients")

# ── Auth — JWT Bearer ─────────────────────────────────────────────────────────

_bearer = HTTPBearer()


def _get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """
    Decode JWT from Authorization: Bearer <token> header.
    decode_token validates all required claims — raises 401 on missing fields,
    never raises KeyError to the caller.
    """
    return decode_token(creds.credentials)


def _auth(user: dict = Depends(_get_current_user)) -> dict:
    """Combined JWT auth + rate-limit dependency for /query."""
    check_rate_limit(user["sub"], RATE_LIMIT)
    return user


# ── Request / Response models ─────────────────────────────────────────────────


# Prompt injection patterns — phrases that attempt to override system instructions.
# Not a complete defense (LLMs can be creative), but eliminates the obvious attacks.
# Real mitigation: strict system prompt (already in synthesizer.py) + output validation
# (evaluator.py) + tenant_id/customer_id never from LLM output (always from JWT).
_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions|"
    r"you\s+are\s+now\s+(a\s+)?|"
    r"disregard\s+(all\s+)?prior|"
    r"new\s+instructions?\s*:|"
    r"system\s*prompt\s*:|"
    r"<\s*system\s*>",
    re.IGNORECASE,
)


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
        # Strip null bytes and control characters before they reach any LLM prompt.
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        # Block obvious prompt injection attempts.
        if _INJECTION_PATTERNS.search(v):
            raise ValueError("question contains invalid content")
        return v


class QueryResponse(BaseModel):
    query_id:     str           # submit to POST /feedback to rate this answer
    answer:       str
    sources:      list[str]
    query_type:   str
    data_source:  str
    faithfulness: float
    relevance:    float
    latency_ms:   float
    cached:       bool          # True = served from semantic cache (⚡ tag this in UI)
    error:        Optional[str] = None


# ── Compliance audit log ─────────────────────────────────────────────────────

_BEDROCK_SONNET = os.getenv("BEDROCK_SONNET_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")


def _write_audit_log(
    tenant_id: str, customer_id: str,
    question: str, result: dict, latency_ms: float,
) -> None:
    """
    Write every query + response to query_audit_log. Non-fatal — a log failure
    must never break the response to the caller.

    Called from the finally block of /query so BOTH successful and failed queries
    are recorded — required for SEC/FINRA 7-year retention.
    """
    import psycopg2
    import psycopg2.extras
    url = os.getenv("DATABASE_URL")
    if not url:
        logger.warning("[api] audit log skipped — DATABASE_URL not set")
        return
    try:
        conn = psycopg2.connect(url, connect_timeout=3)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_audit_log
                        (tenant_id, customer_id, question, query_type, data_source,
                         answer, sources, faithfulness, relevance, latency_ms,
                         model_id, error)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        tenant_id, customer_id, question,
                        result.get("query_type"), result.get("data_source"),
                        result.get("answer"),
                        psycopg2.extras.Json(result.get("sources", [])),
                        result.get("faithfulness"), result.get("relevance"),
                        latency_ms,
                        _BEDROCK_SONNET,
                        result.get("error"),
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
    import time
    tenant_id   = user["tenant_id"]
    customer_id = user["customer_id"]

    logger.info(
        f"[api] POST /query — tenant={tenant_id} customer={customer_id} "
        f"question='{req.question[:80]}'"
    )

    start      = time.time()
    result     = {}
    latency_ms = 0.0

    try:
        result     = run_query(req.question, tenant_id=tenant_id, customer_id=customer_id)
        latency_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            f"[api] /query done — {latency_ms}ms "
            f"data_source={result.get('data_source','rag')} "
            f"faith={result.get('faithfulness',0):.2f} "
            f"rel={result.get('relevance',0):.2f}"
        )

        # Sanitize the error field — internal error strings (DB errors, Bedrock model IDs,
        # stack traces) must never reach the caller. Log them internally, return generic text.
        raw_error      = result.get("error")
        safe_error     = "Pipeline completed with reduced confidence." if raw_error else None

        return QueryResponse(
            query_id     = result["query_id"],
            answer       = result["answer"],
            sources      = result["sources"],
            query_type   = result["query_type"],
            data_source  = result.get("data_source", "rag"),
            faithfulness = result["faithfulness"],
            relevance    = result["relevance"],
            latency_ms   = latency_ms,
            cached       = result.get("cached", False),
            error        = safe_error,
        )

    except HTTPException:
        raise

    except Exception as e:
        latency_ms = round((time.time() - start) * 1000, 2)
        logger.error(f"[api] Unhandled pipeline error after {latency_ms}ms: {type(e).__name__}: {e}")
        result = {"error": f"{type(e).__name__}: {e}"}
        raise HTTPException(status_code=500, detail="Internal pipeline error. Please try again.")

    finally:
        # Always write audit log — success AND failure — for SEC/FINRA compliance.
        # result is {} on early exception but still records the attempt.
        _write_audit_log(
            tenant_id=tenant_id, customer_id=customer_id,
            question=req.question, result=result, latency_ms=latency_ms,
        )


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
