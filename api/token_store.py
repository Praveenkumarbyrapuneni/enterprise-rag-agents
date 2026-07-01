"""
api/token_store.py — In-memory JWT revocation store.

When a user logs out, their token's jti (JWT ID) is added here.
Every authenticated request checks this store before allowing access.

Why this matters:
  JWTs are self-contained — once issued they're valid until expiry (24h).
  Without revocation, a stolen token is usable for up to 24 hours even after
  the user logs out or an admin detects a breach.

Current implementation: in-memory set.
  Works correctly on a single process. On restart, the set is cleared —
  previously revoked tokens become valid again until they expire naturally.

Phase B: replace with Redis SET with TTL matching JWT_EXPIRE_HOURS.
  One line change: `_revoked.add(jti)` → `redis.setex(f"revoked:{jti}", TTL, 1)`
  This survives restarts and works across multiple ECS instances.
"""

_revoked: set[str] = set()  # ponytail: in-memory, Phase B → Redis


def revoke(jti: str) -> None:
    """Add a JWT ID to the revocation list on logout."""
    _revoked.add(jti)


def is_revoked(jti: str) -> bool:
    """Return True if this token has been explicitly revoked."""
    return jti in _revoked
