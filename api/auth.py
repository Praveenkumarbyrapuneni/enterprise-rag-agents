"""
api/auth.py — User authentication endpoints.

POST /auth/register  — create a user under a tenant
POST /auth/login     — verify credentials, return a signed JWT

JWT payload: { sub: user_id, tenant_id, customer_id, exp }

Security invariants enforced here:
  - Email domain must match the tenant's registered domain (tenant_registry.email_domain)
  - customer_id must not already belong to another user in the same tenant
  - Rate limited: 5 login attempts/min per IP, 10 register attempts/min per IP
  - Constant-time password check regardless of whether username exists (timing-safe)
  - JWT fields validated on decode — missing fields raise 401, not 500

Phase B: swap _secret() to AWS Secrets Manager; replace bcrypt with Cognito User Pools.
"""

import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import psycopg2
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

from agents.logger import get_logger
from api.rate_limit import check_rate_limit, LOGIN_RATE_LIMIT, RATE_LIMIT

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_ALG = "HS256"
_TTL = int(os.getenv("JWT_EXPIRE_HOURS", "24"))


def _secret() -> str:
    s = os.getenv("JWT_SECRET_KEY", "")
    if not s:
        raise RuntimeError("JWT_SECRET_KEY not set in .env — generate with: openssl rand -hex 32")
    if len(s) < 32:
        raise RuntimeError("JWT_SECRET_KEY is too short — minimum 32 characters (use openssl rand -hex 32)")
    return s


def _pg():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(url)


# ── Models ────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username:    str
    email:       str
    password:    str
    tenant_id:   str
    customer_id: str

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        v = v.strip()
        if not 3 <= len(v) <= 50:
            raise ValueError("username must be 3–50 characters")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        if not all(c in allowed for c in v):
            raise ValueError("username: only letters, numbers, _ and - allowed")
        return v

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        if len(v) > 128:
            raise ValueError("password too long")
        return v

    @field_validator("tenant_id")
    @classmethod
    def _tenant(cls, v: str) -> str:
        v = v.strip()
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
        if not v or len(v) > 50 or not all(c in allowed for c in v):
            raise ValueError("tenant_id: only lowercase letters, numbers, _ and - allowed (max 50 chars)")
        return v

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or len(v) > 254:
            raise ValueError("invalid email address")
        return v

    @field_validator("customer_id")
    @classmethod
    def _customer_id(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 100:
            raise ValueError("customer_id must be 1–100 characters")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    tenant_id:    str
    customer_id:  str


# ── Password helpers ──────────────────────────────────────────────────────────

# Dummy hash — ensures bcrypt.checkpw always runs regardless of whether the
# username exists, preventing username enumeration via response-time differences.
_DUMMY_HASH = bcrypt.hashpw(b"dummy", bcrypt.gensalt()).decode()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Token helpers ─────────────────────────────────────────────────────────────

def _make_token(user_id: str, tenant_id: str, customer_id: str) -> str:
    payload = {
        "sub":         user_id,
        "tenant_id":   tenant_id,
        "customer_id": customer_id,
        "exp":         datetime.now(timezone.utc) + timedelta(hours=_TTL),
    }
    return jwt.encode(payload, _secret(), algorithm=_ALG)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT. Raises HTTPException(401) on any failure,
    including missing required claims — never raises KeyError to the caller.
    """
    try:
        payload = jwt.decode(token, _secret(), algorithms=[_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

    # Validate required claims explicitly — a token signed with the correct secret
    # but missing fields would otherwise cause KeyError inside query() → 500 not 401.
    for field in ("sub", "tenant_id", "customer_id"):
        if not payload.get(field):
            raise HTTPException(status_code=401, detail="Invalid token: missing required claims.")

    return payload


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register(req: RegisterRequest, request: Request):
    """
    Create a new user under a tenant.

    Security checks (in order):
      1. Rate limit — 10 registrations/min per IP (blocks tenant enumeration)
      2. Tenant exists in tenant_registry
      3. Email domain matches the tenant's registered domain (prevents cross-tenant registration)
      4. Username is unique globally
      5. customer_id is not already claimed by another user in the same tenant
         (prevents reading another customer's transactions by claiming their ID)
    """
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(f"register:{client_ip}", RATE_LIMIT)

    conn = _pg()
    try:
        with conn.cursor() as cur:
            # 1. Tenant must exist AND email domain must match
            email_domain = req.email.split("@")[1]
            cur.execute(
                "SELECT email_domain FROM tenant_registry WHERE tenant_id = %s LIMIT 1",
                (req.tenant_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Registration failed.")
            registered_domain = row[0]
            if email_domain != registered_domain:
                # Return generic error — don't reveal which domains are registered
                raise HTTPException(status_code=400, detail="Registration failed.")

            # 2. Username must be globally unique
            cur.execute("SELECT 1 FROM users WHERE username = %s", (req.username,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Username already taken.")

            # 3. customer_id must not already belong to another user in this tenant
            #    Prevents: attacker registers as "goldman" with customer_id="goldman_user_001"
            #    then reads that customer's transactions via /query.
            cur.execute(
                "SELECT 1 FROM users WHERE tenant_id = %s AND customer_id = %s",
                (req.tenant_id, req.customer_id),
            )
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Registration failed.")

            cur.execute(
                """
                INSERT INTO users (username, email, password_hash, tenant_id, customer_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING user_id::text
                """,
                (req.username, req.email, _hash_password(req.password),
                 req.tenant_id, req.customer_id),
            )
            user_id = cur.fetchone()[0]

        conn.commit()
        logger.info(f"[auth] registered username={req.username} tenant={req.tenant_id}")
        return {"user_id": user_id, "username": req.username, "tenant_id": req.tenant_id}

    except HTTPException:
        conn.rollback()
        raise
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=409, detail="Username or email already exists.")
    except Exception as e:
        conn.rollback()
        logger.error(f"[auth] register error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Registration failed.")
    finally:
        conn.close()


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request):
    """
    Authenticate and return a JWT.

    Rate-limited: 5 attempts/min per IP.
    Constant-time: always runs bcrypt whether or not the user exists.
    """
    # Read real IP from X-Forwarded-For when behind a proxy (ALB, nginx).
    # Falls back to direct connection IP, then "unknown" as last resort.
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = (
        forwarded_for.split(",")[0].strip()
        if forwarded_for
        else (request.client.host if request.client else "unknown")
    )
    check_rate_limit(f"login:{client_ip}", LOGIN_RATE_LIMIT)

    conn = _pg()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id::text, password_hash, tenant_id, customer_id "
                "FROM users WHERE username = %s",
                (req.username,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    # Always run bcrypt — prevents username enumeration via timing side-channel.
    stored_hash = row[1] if row else _DUMMY_HASH
    password_ok = _verify_password(req.password, stored_hash)
    if not row or not password_ok:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    user_id, _, tenant_id, customer_id = row
    token = _make_token(user_id, tenant_id, customer_id)
    logger.info(f"[auth] login username={req.username} tenant={tenant_id}")
    return TokenResponse(access_token=token, tenant_id=tenant_id, customer_id=customer_id)
