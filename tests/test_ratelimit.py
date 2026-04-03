"""Tests for ratelimit module – sliding window rate limiter."""

from __future__ import annotations

import threading
import time

import pytest

from jpstock_agent.ratelimit import RateLimiter, RateLimitResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def limiter():
    return RateLimiter(default_daily_limit=5, burst_per_minute=3)


# ---------------------------------------------------------------------------
# Basic checks
# ---------------------------------------------------------------------------


class TestRateLimitResult:
    def test_allowed_fields(self):
        r = RateLimitResult(allowed=True, remaining=4, limit=5)
        assert r.allowed is True
        assert r.remaining == 4
        assert r.limit == 5

    def test_denied_fields(self):
        r = RateLimitResult(allowed=False, error="limit exceeded")
        assert r.allowed is False
        assert r.error == "limit exceeded"


# ---------------------------------------------------------------------------
# Daily limiting
# ---------------------------------------------------------------------------


class TestDailyLimit:
    """Daily limit tests use burst_per_minute=0 to isolate daily logic."""

    @pytest.fixture
    def daily_limiter(self):
        return RateLimiter(default_daily_limit=5, burst_per_minute=0)

    def test_allows_under_limit(self, daily_limiter):
        for _ in range(5):
            r = daily_limiter.check("user1")
            assert r.allowed is True

    def test_denies_over_limit(self, daily_limiter):
        for _ in range(5):
            daily_limiter.check("user1")
        r = daily_limiter.check("user1")
        assert r.allowed is False
        assert "Daily limit" in r.error
        assert r.remaining == 0

    def test_remaining_decreases(self, daily_limiter):
        r1 = daily_limiter.check("user1")
        assert r1.remaining == 4
        r2 = daily_limiter.check("user1")
        assert r2.remaining == 3

    def test_different_keys_independent(self, daily_limiter):
        for _ in range(5):
            daily_limiter.check("user1")
        # user1 is exhausted
        assert daily_limiter.check("user1").allowed is False
        # user2 is fine
        assert daily_limiter.check("user2").allowed is True


# ---------------------------------------------------------------------------
# Burst limiting
# ---------------------------------------------------------------------------


class TestBurstLimit:
    def test_burst_denied(self, limiter):
        # Burst limit is 3/min, daily is 5
        for _ in range(3):
            r = limiter.check("burst_user")
            assert r.allowed is True
        # 4th call within same minute → burst denied
        r = limiter.check("burst_user")
        assert r.allowed is False
        assert "Burst" in r.error

    def test_no_burst_when_disabled(self):
        lim = RateLimiter(default_daily_limit=10, burst_per_minute=0)
        for _ in range(10):
            r = lim.check("user")
            assert r.allowed is True


# ---------------------------------------------------------------------------
# Custom per-key limits
# ---------------------------------------------------------------------------


class TestCustomLimits:
    def test_set_limit(self, limiter):
        limiter.set_limit("vip", 100)
        assert limiter.get_limit("vip") == 100

    def test_default_limit(self, limiter):
        assert limiter.get_limit("unknown_user") == 5

    def test_custom_limit_enforced(self):
        lim = RateLimiter(default_daily_limit=2, burst_per_minute=0)
        lim.set_limit("premium", 10)
        for _ in range(10):
            r = lim.check("premium")
            assert r.allowed is True
        r = lim.check("premium")
        assert r.allowed is False


# ---------------------------------------------------------------------------
# peek (read-only check)
# ---------------------------------------------------------------------------


class TestPeek:
    def test_peek_no_consume(self, limiter):
        r1 = limiter.peek("peeker")
        assert r1.allowed is True
        assert r1.remaining == 5  # default limit
        # peek again – should be same
        r2 = limiter.peek("peeker")
        assert r2.remaining == 5

    def test_peek_after_usage(self, limiter):
        limiter.check("peeker")
        limiter.check("peeker")
        r = limiter.peek("peeker")
        assert r.remaining == 3


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_single_key(self, limiter):
        for _ in range(5):
            limiter.check("user1")
        assert limiter.check("user1").allowed is False
        limiter.reset("user1")
        assert limiter.check("user1").allowed is True

    def test_reset_all(self, limiter):
        limiter.check("a")
        limiter.check("b")
        limiter.set_limit("a", 100)
        limiter.reset_all()
        # limits also cleared
        assert limiter.get_limit("a") == 5  # back to default


# ---------------------------------------------------------------------------
# usage stats
# ---------------------------------------------------------------------------


class TestUsage:
    def test_usage_stats(self, limiter):
        limiter.check("stats_user")
        limiter.check("stats_user")
        u = limiter.usage("stats_user")
        assert u["daily_used"] == 2
        assert u["daily_limit"] == 5
        assert u["daily_remaining"] == 3
        assert u["minute_limit"] == 3
        assert u["utilization_pct"] == 40.0

    def test_usage_empty(self, limiter):
        u = limiter.usage("new_user")
        assert u["daily_used"] == 0
        assert u["daily_remaining"] == 5


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_checks(self):
        lim = RateLimiter(default_daily_limit=100, burst_per_minute=0)
        results = []

        def worker():
            for _ in range(20):
                r = lim.check("concurrent")
                results.append(r.allowed)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 100 should be allowed
        assert sum(results) == 100
        assert len(results) == 100


# ---------------------------------------------------------------------------
# RateLimitResult fields
# ---------------------------------------------------------------------------


class TestRetryAfter:
    def test_retry_after_positive_on_deny(self, limiter):
        for _ in range(5):
            limiter.check("retry_user")
        r = limiter.check("retry_user")
        assert r.allowed is False
        assert r.retry_after > 0

    def test_reset_at_is_future(self, limiter):
        r = limiter.check("future_user")
        assert r.reset_at > time.time()
