"""
api/service_auth.py — Server-to-server token minting for embedded integrations.

POST /internal/mint-customer-token

Who calls this: a bank's OWN backend, never a customer's browser or app.
The customer already authenticated inside the bank's app (their login, not
ours). When that customer opens the embedded chat, the bank's backend calls
this endpoint with the customer_id it already trusts, and gets back a JWT
scoped to that customer — the same JWT shape /query already expects, so
nothing downstream (retriever, db_lookup, /query) changes.

Auth here is NOT username/password — there is no human filling a form.
The caller proves it IS the tenant's backend with a pre-shared secret
(X-Service-Key), provisioned once via `python -m db.schema --provision-key
<tenant_id>` and stored only as a SHA-256 hash (see db.schema.provision_service_key).

This is the highest-blast-radius endpoint in the system: get the secret
comparison wrong and any caller can mint a token for ANY customer_id and
read their balance. Constant-time comparison + a dummy hash for unknown
tenants prevent both secret-guessing and tenant-enumeration timing attacks.

Phase B: X-Service-Key stays the same shape, but require mTLS or an
API-Gateway-level allowlist per tenant on top of this — defense in depth,
not a replacement.
"""

import hashlib
import hmac
import os

import psycopg2
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, field_validator

from agents.logger import get_logger
from api.auth import TokenResponse, _make_token
from api.rate_limit import check_rate_limit, RATE_LIMIT

logger = get_logger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])

# Ephemeral, machine-minted, session-scoped — shorter-lived than a human's
# day-long login (JWT_EXPIRE_HOURS) to shrink the blast radius of a leak.
_MINT_TTL_HOURS = float(os.getenv("MINT_TOKEN_TTL_HOURS", "1"))

# Constant-time-compared against on every request where the tenant has no
# (or a not-yet-provisioned) service_key_hash — prevents timing-based
# tenant enumeration, same principle as auth.py's _DUMMY_HASH.
_DUMMY_KEY_HASH = hashlib.sha256(b"dummy").hexdigest()

_TENANT_ID_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


class MintTokenRequest(BaseModel):
    customer_id: str

    @field_validator("customer_id")
    @classmethod
    def _customer_id(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 100:
            raise ValueError("customer_id must be 1–100 characters")
        return v


def _pg():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(url)


def _valid_tenant_id(v: str) -> bool:
    return bool(v) and len(v) <= 50 and all(c in _TENANT_ID_ALLOWED for c in v)


def _verify_service_key(provided_key: str, stored_hash: str | None) -> bool:
    """
    Constant-time check of provided_key against stored_hash. Always hashes
    and always compares — even when stored_hash is None (tenant not
    provisioned) — so an attacker can't distinguish "wrong key" from
    "unknown/unprovisioned tenant" by response time.
    """
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    return hmac.compare_digest(provided_hash, stored_hash or _DUMMY_KEY_HASH)


@router.post("/mint-customer-token", response_model=TokenResponse)
def mint_customer_token(
    req: MintTokenRequest,
    request: Request,
    x_tenant_id: str = Header(..., alias="X-Tenant-Id"),
    x_service_key: str = Header(..., alias="X-Service-Key"),
):
    """
    Exchange a tenant's service key + a customer_id it already trusts for a
    short-lived JWT scoped to that customer. tenant_id and customer_id in the
    returned token flow into /query exactly like a password/SSO login would.
    """
    x_tenant_id = x_tenant_id.strip()
    if not _valid_tenant_id(x_tenant_id):
        raise HTTPException(status_code=401, detail="Invalid service credentials.")

    # Rate limited per tenant, not per IP — a bank's backend calls this from
    # a small set of server IPs but for potentially many customers/minute.
    check_rate_limit(f"mint:{x_tenant_id}", RATE_LIMIT)

    conn = _pg()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT service_key_hash FROM tenant_registry WHERE tenant_id = %s LIMIT 1",
                (x_tenant_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    stored_hash = row[0] if row else None
    if not _verify_service_key(x_service_key, stored_hash):
        # Same generic error whether the tenant doesn't exist, isn't
        # provisioned, or the key is simply wrong — don't leak which.
        raise HTTPException(status_code=401, detail="Invalid service credentials.")

    token = _make_token(
        user_id=f"ext:{x_tenant_id}:{req.customer_id}",
        tenant_id=x_tenant_id,
        customer_id=req.customer_id,
        ttl_hours=_MINT_TTL_HOURS,
    )
    logger.info(f"[service_auth] minted token tenant={x_tenant_id} customer={req.customer_id}")
    return TokenResponse(access_token=token, tenant_id=x_tenant_id, customer_id=req.customer_id)


# ── Self-check ────────────────────────────────────────────────────────────────
# ponytail: no pytest — this is the smallest thing that fails if the
# constant-time-compare logic breaks (the actual security-critical part).

def _selftest() -> None:
    real_hash = hashlib.sha256(b"correct-horse-battery-staple").hexdigest()
    assert _verify_service_key("correct-horse-battery-staple", real_hash) is True
    assert _verify_service_key("wrong-key", real_hash) is False
    assert _verify_service_key("anything", None) is False  # unprovisioned tenant
    assert _valid_tenant_id("goldman") is True
    assert _valid_tenant_id("Goldman!") is False
    assert _valid_tenant_id("") is False
    print("api/service_auth.py self-check passed")


if __name__ == "__main__":
    _selftest()
