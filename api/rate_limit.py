"""
api/rate_limit.py — In-memory sliding window rate limiter.

Extracted into its own module to break the circular import between api/main.py
and api/auth.py (main imports auth for the router; auth needed main for this).

Phase B: replace _rate_buckets with Redis-backed sliding window for multi-instance
ECS deployment where in-process state is not shared across workers.
"""

import os
import time
from collections import defaultdict, deque

from fastapi import HTTPException

RATE_WINDOW_S:    int = 60
RATE_LIMIT:       int = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
LOGIN_RATE_LIMIT: int = int(os.getenv("LOGIN_RATE_LIMIT_PER_MIN", "5"))

# Bounded by eviction — keys are removed once their deque empties after the window.
_rate_buckets: dict[str, deque] = defaultdict(deque)


def check_rate_limit(key: str, limit: int = RATE_LIMIT) -> None:
    """
    Raise 429 if `key` has exceeded `limit` requests in the last RATE_WINDOW_S seconds.
    Evicts the key from the dict when its deque becomes empty — prevents unbounded growth.
    """
    now    = time.time()
    bucket = _rate_buckets[key]

    while bucket and bucket[0] < now - RATE_WINDOW_S:
        bucket.popleft()

    if len(bucket) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {RATE_WINDOW_S} seconds.",
            headers={"Retry-After": str(RATE_WINDOW_S)},
        )

    bucket.append(now)

    # Evict key once its window is empty — prevents unbounded dict growth
    # at Discover scale (millions of unique user IDs).
    if not bucket:
        del _rate_buckets[key]
