"""
api/auth.py — User authentication endpoints.

POST /auth/register  — create a user under a tenant
POST /auth/login     — verify credentials, return a signed JWT

JWT payload: { sub: user_id, tenant_id, customer_id, exp }

The /query endpoint reads tenant_id and customer_id from the JWT — callers
never pass them manually. Tenant isolation is enforced at login time, not
per-request.

Phase B: swap _jwt_secret() to AWS Secrets Manager; replace passlib with
Cognito user pools for enterprise SSO.
"""

import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import psycopg2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from agents.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_ALG  = "HS256"
_TTL  = int(os.getenv("JWT_EXPIRE_HOURS", "24"))


def _secret() -> str:
    s = os.getenv("JWT_SECRET_KEY", "")
    if not s:
        raise RuntimeError("JWT_SECRET_KEY not set in .env — generate one with: openssl rand -hex 32")
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
        if len(v) < 3:
            raise ValueError("username must be at least 3 characters")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        if not all(c in allowed for c in v):
            raise ValueError("username: only letters, numbers, _ and - allowed")
        return v

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v

    @field_validator("tenant_id")
    @classmethod
    def _tenant(cls, v: str) -> str:
        v = v.strip()
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
        if not v or not all(c in allowed for c in v):
            raise ValueError("tenant_id: only lowercase letters, numbers, _ and - allowed")
        return v

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v:
            raise ValueError("invalid email address")
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
    """Decode and validate a JWT. Raises HTTPException on any failure."""
    try:
        return jwt.decode(token, _secret(), algorithms=[_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register(req: RegisterRequest):
    """
    Create a new user.

    tenant_id must already exist in tenant_registry.
    customer_id links this user to their rows in the transactions table.
    """
    conn = _pg()
    try:
        with conn.cursor() as cur:
            # Tenant must exist
            cur.execute(
                "SELECT 1 FROM tenant_registry WHERE tenant_id = %s LIMIT 1",
                (req.tenant_id,),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=400, detail=f"Unknown tenant: {req.tenant_id}")

            # Username must be unique
            cur.execute("SELECT 1 FROM users WHERE username = %s", (req.username,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Username already taken.")

            cur.execute(
                """
                INSERT INTO users (username, email, password_hash, tenant_id, customer_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING user_id::text
                """,
                (req.username, req.email, _hash_password(req.password), req.tenant_id, req.customer_id),
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
        logger.error(f"[auth] register error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed.")
    finally:
        conn.close()


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """
    Authenticate and return a JWT.

    The token embeds tenant_id and customer_id — callers use it on /query
    without passing those fields manually.
    """
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

    # Same error for wrong username or wrong password — no user enumeration
    if not row or not _verify_password(req.password, row[1]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    user_id, _, tenant_id, customer_id = row
    token = _make_token(user_id, tenant_id, customer_id)
    logger.info(f"[auth] login username={req.username} tenant={tenant_id}")
    return TokenResponse(access_token=token, tenant_id=tenant_id, customer_id=customer_id)
