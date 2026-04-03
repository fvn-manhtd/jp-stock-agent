"""In-memory sliding-window rate limiter for jpstock-agent.

Design
------
Each API key gets a deque of call timestamps.  ``check()`` counts
timestamps within the current window and decides allow/deny.

The default window is **1 day** (86 400 s) so that limits map to
"calls per day".  For burst protection an optional secondary
per-minute window is also enforced.

Thread Safety
-------------
All public methods acquire a threading.Lock so the limiter is safe
for concurrent use from FastMCP's async/threaded handlers.

Production Note
---------------
For a multi-process or distributed deployment, swap this module
with a Redis-backed implementation (same interface).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RateLimitResult:
    """Outcome of a rate-limit check."""

    allowed: bool
    remaining: int = 0          # calls left in the window
    limit: int = 0              # total limit for the window
    reset_at: float = 0.0       # Unix ts when the oldest entry expires
    retry_after: float = 0.0    # seconds until next allowed call (0 if allowed)
    error: str = ""


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

_DAY = 86_400
_MINUTE = 60


class RateLimiter:
    """Sliding-window rate limiter.

    Parameters
    ----------
    default_daily_limit : int
        Fallback daily limit when no per-key override is set.
    burst_per_minute : int
        Maximum calls any single key can make per minute (burst cap).
        Set to 0 to disable burst checking.
    """

    def __init__(
        self,
        default_daily_limit: int = 50,
        burst_per_minute: int = 30,
    ):
        self._daily_limit = default_daily_limit
        self._burst = burst_per_minute
        self._lock = threading.Lock()
        # key_id → deque of timestamps
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        # key_id → custom daily limit (overrides default)
        self._limits: dict[str, int] = {}

    # -- configuration --

    def set_limit(self, key_id: str, daily_limit: int) -> None:
        """Override the daily limit for a specific key."""
        with self._lock:
            self._limits[key_id] = daily_limit

    def get_limit(self, key_id: str) -> int:
        """Return the effective daily limit for *key_id*."""
        return self._limits.get(key_id, self._daily_limit)

    # -- core --

    def check(self, key_id: str) -> RateLimitResult:
        """Record a call and check the rate limit.

        Returns an :class:`RateLimitResult` indicating whether the call
        is allowed. If not, ``retry_after`` gives the seconds to wait.
        """
        now = time.time()
        cutoff_day = now - _DAY
        cutoff_min = now - _MINUTE

        with self._lock:
            dq = self._windows[key_id]

            # Prune expired entries (older than 1 day)
            while dq and dq[0] < cutoff_day:
                dq.popleft()

            limit = self._limits.get(key_id, self._daily_limit)

            # -- daily check --
            if len(dq) >= limit:
                oldest = dq[0] if dq else now
                reset_at = oldest + _DAY
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=limit,
                    reset_at=reset_at,
                    retry_after=max(0.0, reset_at - now),
                    error=f"Daily limit exceeded ({limit} calls/day)",
                )

            # -- burst check (per-minute) --
            if self._burst > 0:
                recent = sum(1 for t in dq if t >= cutoff_min)
                if recent >= self._burst:
                    return RateLimitResult(
                        allowed=False,
                        remaining=max(0, limit - len(dq)),
                        limit=limit,
                        reset_at=now + _MINUTE,
                        retry_after=_MINUTE,
                        error=f"Burst limit exceeded ({self._burst} calls/min)",
                    )

            # -- allow --
            dq.append(now)
            remaining = limit - len(dq)
            reset_at = dq[0] + _DAY if dq else now + _DAY

            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                limit=limit,
                reset_at=reset_at,
            )

    def peek(self, key_id: str) -> RateLimitResult:
        """Check remaining quota without consuming a call."""
        now = time.time()
        cutoff_day = now - _DAY

        with self._lock:
            dq = self._windows[key_id]
            while dq and dq[0] < cutoff_day:
                dq.popleft()
            limit = self._limits.get(key_id, self._daily_limit)
            used = len(dq)
            remaining = max(0, limit - used)
            reset_at = dq[0] + _DAY if dq else now + _DAY

            return RateLimitResult(
                allowed=remaining > 0,
                remaining=remaining,
                limit=limit,
                reset_at=reset_at,
            )

    def reset(self, key_id: str) -> None:
        """Clear all recorded calls for a key."""
        with self._lock:
            self._windows.pop(key_id, None)

    def reset_all(self) -> None:
        """Clear all state."""
        with self._lock:
            self._windows.clear()
            self._limits.clear()

    def usage(self, key_id: str) -> dict:
        """Return usage stats for a key."""
        now = time.time()
        cutoff_day = now - _DAY
        cutoff_min = now - _MINUTE

        with self._lock:
            dq = self._windows[key_id]
            while dq and dq[0] < cutoff_day:
                dq.popleft()
            limit = self._limits.get(key_id, self._daily_limit)
            total = len(dq)
            recent = sum(1 for t in dq if t >= cutoff_min)

            return {
                "key_id": key_id[:12] + "..." if len(key_id) > 12 else key_id,
                "daily_used": total,
                "daily_limit": limit,
                "daily_remaining": max(0, limit - total),
                "minute_used": recent,
                "minute_limit": self._burst,
                "utilization_pct": round(total / limit * 100, 1) if limit else 0,
            }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_limiter: RateLimiter | None = None


def get_rate_limiter(
    default_daily_limit: int = 50,
    burst_per_minute: int = 30,
) -> RateLimiter:
    """Return the module-level RateLimiter singleton."""
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(default_daily_limit, burst_per_minute)
    return _limiter
