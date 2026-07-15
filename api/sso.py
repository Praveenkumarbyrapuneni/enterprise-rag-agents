"""
api/sso.py — Company login (SSO) via any standard identity provider.

Instead of creating a separate username/password for this app, employees log
in with the ONE company login they already have — the bank's own Okta, Azure
AD, Google Workspace, or any provider that speaks OIDC (the standard almost
every company identity system supports).

Flow:
  GET  /auth/sso/login    → redirect the browser to the company's login page
  GET  /auth/sso/callback → company's system redirects back here with proof
                              of who logged in; we find-or-create the user
                              row (auth_provider='sso', no password stored),
                              then issue the SAME JWT format as password login
                              — nothing downstream (retriever, synthesizer,
                              /query, /feedback) has to know the difference.

Config (.env) — swapping providers is a config change, not a code change:
  OIDC_ISSUER_URL     e.g. https://accounts.google.com
                            https://your-org.okta.com
                            https://login.microsoftonline.com/<tenant>/v2.0
  OIDC_CLIENT_ID      registered with that provider
  OIDC_CLIENT_SECRET  registered with that provider
  OIDC_REDIRECT_URI   must exactly match what's registered with the provider

Why the person is trusted without this app ever seeing a password:
  The identity provider checks the password on ITS OWN side, then hands back
  a signed proof ("this is definitely alice@bank.com") that only the real
  provider could have produced. We verify that signature, but never see or
  store a password for these users — auth_provider='sso' and password_hash
  stays NULL (see db/schema.py). If the company deactivates alice's account
  on their end, that signed proof can no longer be produced — she's locked
  out of everything connected to it, including this app, automatically.
"""

import os
import uuid

import psycopg2
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request

from agents.logger import get_logger
from api.auth import _make_token
from db.schema import get_tenant_id

logger = get_logger(__name__)

router = APIRouter(prefix="/auth/sso", tags=["auth"])

_ISSUER        = os.getenv("OIDC_ISSUER_URL", "").strip()
_CLIENT_ID     = os.getenv("OIDC_CLIENT_ID", "").strip()
_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET", "").strip()
_REDIRECT_URI  = os.getenv("OIDC_REDIRECT_URI", "").strip()

_oauth = OAuth()
if _ISSUER and _CLIENT_ID and _CLIENT_SECRET:
    _oauth.register(
        name="company_sso",
        server_metadata_url=f"{_ISSUER.rstrip('/')}/.well-known/openid-configuration",
        client_id=_CLIENT_ID,
        client_secret=_CLIENT_SECRET,
        client_kwargs={"scope": "openid email profile"},
    )


def _configured() -> bool:
    # SESSION_SECRET_KEY is required too — without it, SessionMiddleware isn't
    # installed (see api/main.py) and authlib's request.session access would
    # crash instead of returning this clean "not configured" response.
    return bool(_ISSUER and _CLIENT_ID and _CLIENT_SECRET and os.getenv("SESSION_SECRET_KEY", "").strip())


def _pg():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(url)


def _find_or_create_sso_user(email: str, tenant_id: str) -> tuple[str, str, str]:
    """
    Look up an existing SSO user by email, or create one on first login.
    Returns (user_id, tenant_id, customer_id).

    customer_id is just the email — SSO users never fill in a registration
    form, so there's no separate identifier to collect from them.
    """
    conn = _pg()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id::text, tenant_id, customer_id FROM users WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if row:
                return row[0], row[1], row[2]

            username = f"{email.split('@')[0]}_{uuid.uuid4().hex[:6]}"
            cur.execute(
                """
                INSERT INTO users (username, email, password_hash, auth_provider, tenant_id, customer_id)
                VALUES (%s, %s, NULL, 'sso', %s, %s)
                RETURNING user_id::text
                """,
                (username, email, tenant_id, email),
            )
            user_id = cur.fetchone()[0]
        conn.commit()
        logger.info(f"[sso] created new SSO user email={email} tenant={tenant_id}")
        return user_id, tenant_id, email
    finally:
        conn.close()


@router.get("/login")
async def sso_login(request: Request):
    """Redirect the browser to the company's own login page."""
    if not _configured():
        raise HTTPException(status_code=503, detail="SSO is not configured for this deployment.")
    return await _oauth.company_sso.authorize_redirect(request, _REDIRECT_URI)


@router.get("/callback")
async def sso_callback(request: Request):
    """
    The company's identity provider redirects here after the employee logs in
    on THEIR page. We never see a password — only a signed token proving who
    they are, verified against the provider's public keys before we trust it.
    """
    if not _configured():
        raise HTTPException(status_code=503, detail="SSO is not configured for this deployment.")

    token = await _oauth.company_sso.authorize_access_token(request)
    userinfo = token.get("userinfo") or {}
    email = (userinfo.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=401, detail="Identity provider did not return an email.")
    if not userinfo.get("email_verified", True):
        raise HTTPException(status_code=401, detail="Email not verified with identity provider.")

    tenant_id = get_tenant_id(email)
    if not tenant_id:
        raise HTTPException(status_code=403, detail="This email domain is not registered with any tenant.")

    user_id, tenant_id, customer_id = _find_or_create_sso_user(email, tenant_id)
    jwt_token = _make_token(user_id, tenant_id, customer_id)
    logger.info(f"[sso] login email={email} tenant={tenant_id}")
    return {
        "access_token": jwt_token,
        "token_type":   "bearer",
        "tenant_id":    tenant_id,
        "customer_id":  customer_id,
    }


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # No live identity provider needed for this check — only the logic that
    # doesn't require a network round trip: config detection.
    assert _configured() == bool(_ISSUER and _CLIENT_ID and _CLIENT_SECRET)
    print(
        "sso.py self-check passed — "
        f"configured={_configured()} "
        "(set OIDC_ISSUER_URL/OIDC_CLIENT_ID/OIDC_CLIENT_SECRET/OIDC_REDIRECT_URI "
        "in .env for a live provider)"
    )
