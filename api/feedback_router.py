"""
api/feedback_router.py — POST /feedback endpoint.

User submits feedback on a query response using the query_id returned in /query.
Triggers async retrieval parameter tuning after every TUNE_EVERY_N submissions.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from agents.feedback import store_feedback
from agents.logger import get_logger
from api.auth import decode_token

logger = get_logger(__name__)
router = APIRouter()
_bearer = HTTPBearer()


def _get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    return decode_token(creds.credentials)


class FeedbackRequest(BaseModel):
    query_id: str           # UUID from the /query response
    helpful:  bool          # True = thumbs up, False = thumbs down
    comment:  Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


@router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def feedback(req: FeedbackRequest, user: dict = Depends(_get_current_user)):
    """
    Submit feedback for a previous query response.

    query_id comes from the /query response. The tenant_id is always taken from
    the JWT — users cannot submit feedback for another tenant's queries.
    """
    tenant_id = user["tenant_id"]

    try:
        store_feedback(
            query_id=req.query_id,
            tenant_id=tenant_id,
            helpful=req.helpful,
            comment=req.comment,
        )
        logger.info(
            f"[feedback] POST /feedback — tenant={tenant_id} "
            f"query_id={req.query_id} helpful={req.helpful}"
        )
        return FeedbackResponse(status="ok")

    except Exception as e:
        logger.error(f"[feedback] store failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Failed to store feedback.")
