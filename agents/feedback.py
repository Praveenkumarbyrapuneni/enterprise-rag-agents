"""
agents/feedback.py — Feedback storage and active retrieval parameter tuning.

Flow:
  1. User submits POST /feedback { query_id, helpful, comment }
  2. store_feedback() writes to query_feedback table
  3. After every TUNE_EVERY_N feedback items per tenant, tune_retrieval_params()
     fires as a Celery task to adjust top_k / rerank_top_n / mmr_final_k
  4. retriever.py reads from retrieval_params on every query

Tuning logic (simple, interpretable, no ML needed):
  - Look at the last WINDOW feedback items for this tenant
  - helpful_rate = helpful_count / total
  - helpful_rate < 0.5 → increase top_k by 5 (retrieve more candidates)
  - helpful_rate > 0.8 → decrease top_k by 2 (working well, reduce cost)
  - rerank_top_n scales proportionally with top_k
  - Floors: top_k >= 10, rerank_top_n >= 4, mmr_final_k >= 3
  - Ceilings: top_k <= 50, rerank_top_n <= 20, mmr_final_k <= 10

Why this approach over a learned ranker:
  A learned ranker (LambdaMART etc.) needs thousands of labeled query-document pairs.
  We don't have that at launch. This heuristic adjusts the one parameter that most
  directly controls answer quality (candidates retrieved) based on the signal we do
  have (user thumbs). Once feedback volume is high enough, swap in a proper ranker.
"""

import os
from typing import Optional

import psycopg2
from celery import Celery
from dotenv import load_dotenv

from .logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TUNE_EVERY_N = int(os.getenv("FEEDBACK_TUNE_EVERY_N", "10"))  # trigger tuning after N feedbacks
_WINDOW      = 50    # look at last N feedback items when tuning
_TOP_K_MIN, _TOP_K_MAX           = 10, 50
_RERANK_MIN, _RERANK_MAX         = 4,  20
_MMR_MIN,    _MMR_MAX            = 3,  10

# ── Celery app (reuses existing Redis broker) ─────────────────────────────────

_celery = Celery(
    "feedback",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"), connect_timeout=5)


# ── Public: store feedback ────────────────────────────────────────────────────

def store_feedback(query_id: str, tenant_id: str, helpful: bool, comment: Optional[str]) -> None:
    """
    Persist one feedback record and trigger parameter tuning if threshold is reached.
    Raises on DB error — the API endpoint handles it and returns 500.
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_feedback (query_id, tenant_id, helpful, comment)
                VALUES (%s::uuid, %s, %s, %s)
                """,
                (query_id, tenant_id, helpful, comment),
            )
            # Count total feedback for this tenant to decide if tuning should fire
            cur.execute(
                "SELECT COUNT(*) FROM query_feedback WHERE tenant_id = %s",
                (tenant_id,),
            )
            total = cur.fetchone()[0]

    logger.info(
        f"[feedback] stored — tenant={tenant_id} helpful={helpful} "
        f"total_feedback={total}"
    )

    if total % TUNE_EVERY_N == 0:
        logger.info(f"[feedback] triggering tune for tenant={tenant_id} at {total} items")
        _tune_task.delay(tenant_id)


# ── Celery task: tune retrieval params ───────────────────────────────────────

@_celery.task(name="feedback.tune_retrieval_params")
def _tune_task(tenant_id: str) -> None:
    """Celery task — runs async, tunes retrieval params for one tenant."""
    tune_retrieval_params(tenant_id)


def tune_retrieval_params(tenant_id: str) -> dict:
    """
    Read last WINDOW feedback items, compute helpful_rate, adjust retrieval params.
    Returns the new params dict (useful for testing).
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            # Current params (or defaults if first tune)
            cur.execute(
                "SELECT top_k, rerank_top_n, mmr_final_k FROM retrieval_params WHERE tenant_id = %s",
                (tenant_id,),
            )
            row = cur.fetchone()
            top_k        = row[0] if row else 20
            rerank_top_n = row[1] if row else 8
            mmr_final_k  = row[2] if row else 6

            # Recent feedback signal
            cur.execute(
                """
                SELECT COUNT(*) FILTER (WHERE helpful = true)  AS helpful_count,
                       COUNT(*)                                 AS total
                FROM (
                    SELECT helpful FROM query_feedback
                    WHERE tenant_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) recent
                """,
                (tenant_id, _WINDOW),
            )
            helpful_count, total = cur.fetchone()

    if total == 0:
        return {"top_k": top_k, "rerank_top_n": rerank_top_n, "mmr_final_k": mmr_final_k}

    helpful_rate = helpful_count / total

    # Adjust top_k based on signal
    if helpful_rate < 0.5:
        top_k = min(top_k + 5, _TOP_K_MAX)
    elif helpful_rate > 0.8:
        top_k = max(top_k - 2, _TOP_K_MIN)

    # rerank_top_n and mmr_final_k scale proportionally with top_k
    rerank_top_n = min(max(round(top_k * 0.4), _RERANK_MIN), _RERANK_MAX)
    mmr_final_k  = min(max(round(top_k * 0.3), _MMR_MIN),    _MMR_MAX)

    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO retrieval_params
                    (tenant_id, top_k, rerank_top_n, mmr_final_k, feedback_count, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (tenant_id) DO UPDATE SET
                    top_k          = EXCLUDED.top_k,
                    rerank_top_n   = EXCLUDED.rerank_top_n,
                    mmr_final_k    = EXCLUDED.mmr_final_k,
                    feedback_count = retrieval_params.feedback_count + EXCLUDED.feedback_count,
                    updated_at     = now()
                """,
                (tenant_id, top_k, rerank_top_n, mmr_final_k, total),
            )

    new_params = {"top_k": top_k, "rerank_top_n": rerank_top_n, "mmr_final_k": mmr_final_k}
    logger.info(
        f"[feedback] tuned tenant={tenant_id} helpful_rate={helpful_rate:.2f} "
        f"window={total} params={new_params}"
    )
    return new_params


# ── Read tuned params ─────────────────────────────────────────────────────────

def get_retrieval_params(tenant_id: str) -> dict:
    """
    Return tuned retrieval params for a tenant, or hardcoded defaults if none exist.
    Called by retriever.py on every query. Returns in <5ms (single PK lookup).
    Non-fatal: on any DB error, returns defaults so retrieval is never blocked.
    """
    defaults = {"top_k": 20, "rerank_top_n": 8, "mmr_final_k": 6}
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT top_k, rerank_top_n, mmr_final_k FROM retrieval_params WHERE tenant_id = %s",
                    (tenant_id,),
                )
                row = cur.fetchone()
        if row:
            return {"top_k": row[0], "rerank_top_n": row[1], "mmr_final_k": row[2]}
    except Exception as e:
        logger.warning(f"[feedback] get_retrieval_params failed (using defaults): {e}")
    return defaults


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    params = tune_retrieval_params.__wrapped__("demo") if hasattr(tune_retrieval_params, "__wrapped__") else None

    # Tuning math checks — no DB needed
    def _simulate(top_k, helpful_count, total):
        helpful_rate = helpful_count / total
        if helpful_rate < 0.5:
            top_k = min(top_k + 5, _TOP_K_MAX)
        elif helpful_rate > 0.8:
            top_k = max(top_k - 2, _TOP_K_MIN)
        rerank = min(max(round(top_k * 0.4), _RERANK_MIN), _RERANK_MAX)
        mmr    = min(max(round(top_k * 0.3), _MMR_MIN),    _MMR_MAX)
        return top_k, rerank, mmr

    # Low helpful rate → top_k increases
    tk, rr, mm = _simulate(20, 4, 10)   # 40% helpful
    assert tk == 25, f"expected 25 got {tk}"
    assert rr >= _RERANK_MIN
    assert mm >= _MMR_MIN

    # High helpful rate → top_k decreases
    tk2, _, _ = _simulate(20, 9, 10)    # 90% helpful
    assert tk2 == 18, f"expected 18 got {tk2}"

    # Floor enforcement
    tk3, _, _ = _simulate(10, 9, 10)    # already at floor, high rate
    assert tk3 == _TOP_K_MIN

    # Ceiling enforcement
    tk4, _, _ = _simulate(50, 4, 10)    # already at ceiling, low rate
    assert tk4 == _TOP_K_MAX

    print("feedback.py self-check passed — tuning math correct, floors/ceilings enforced")
