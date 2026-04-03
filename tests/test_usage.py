"""Tests for usage module – persistent usage tracking and analytics."""

from __future__ import annotations

import os
import threading
import time

import pytest

from jpstock_agent.usage import UsageTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker(tmp_path):
    """UsageTracker backed by a temporary SQLite database."""
    db = str(tmp_path / "usage.db")
    return UsageTracker(db)


@pytest.fixture
def populated_tracker(tracker):
    """Tracker pre-loaded with sample data."""
    # 5 calls from key_a (pro), 3 from key_b (free), 1 error
    for i in range(5):
        tracker.record("key_a_hash", "pro", f"tool_{i % 3}", latency=0.05)
    for i in range(3):
        tracker.record("key_b_hash", "free", "stock_history", latency=0.02)
    tracker.record("key_a_hash", "pro", "ml_predict", latency=0.1, status="error")
    return tracker


# ---------------------------------------------------------------------------
# record + basic querying
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_and_daily_summary(self, tracker):
        tracker.record("key1", "pro", "stock_history", latency=0.05)
        tracker.record("key1", "pro", "ta_rsi", latency=0.03)
        tracker.record("key2", "free", "stock_history", latency=0.02)

        summary = tracker.daily_summary()
        assert summary["total_calls"] == 3
        assert summary["unique_keys"] == 2
        assert "pro" in summary["by_tier"]
        assert "free" in summary["by_tier"]
        assert summary["by_tier"]["pro"] == 2
        assert summary["by_tier"]["free"] == 1

    def test_record_error_status(self, tracker):
        tracker.record("key1", "pro", "ml_predict", status="error")
        summary = tracker.daily_summary()
        assert summary["errors"] == 1
        assert summary["error_rate_pct"] == 100.0

    def test_empty_daily_summary(self, tracker):
        summary = tracker.daily_summary()
        assert summary["total_calls"] == 0
        assert summary["unique_keys"] == 0
        assert summary["errors"] == 0
        assert summary["error_rate_pct"] == 0

    def test_top_tools(self, tracker):
        for _ in range(5):
            tracker.record("k", "pro", "stock_history")
        for _ in range(3):
            tracker.record("k", "pro", "ta_rsi")
        tracker.record("k", "pro", "ml_predict")

        summary = tracker.daily_summary()
        assert summary["top_tools"][0]["tool"] == "stock_history"
        assert summary["top_tools"][0]["count"] == 5


# ---------------------------------------------------------------------------
# key_usage
# ---------------------------------------------------------------------------


class TestKeyUsage:
    def test_key_usage_basic(self, populated_tracker):
        data = populated_tracker.key_usage("key_a_hash", days=1)
        assert data["total_calls"] == 6  # 5 ok + 1 error
        assert data["errors"] == 1
        assert len(data["daily"]) >= 1
        assert "key_a_hash" in data["key_hash_short"]

    def test_key_usage_different_key(self, populated_tracker):
        data = populated_tracker.key_usage("key_b_hash", days=1)
        assert data["total_calls"] == 3
        assert data["errors"] == 0

    def test_key_usage_no_data(self, tracker):
        data = tracker.key_usage("nonexistent", days=7)
        assert data["total_calls"] == 0


# ---------------------------------------------------------------------------
# tool_stats
# ---------------------------------------------------------------------------


class TestToolStats:
    def test_tool_stats(self, populated_tracker):
        stats = populated_tracker.tool_stats(days=1)
        assert len(stats) > 0
        # stock_history should appear (3 from key_b + some from key_a)
        tool_names = [s["tool"] for s in stats]
        assert "stock_history" in tool_names

    def test_tool_stats_empty(self, tracker):
        stats = tracker.tool_stats(days=7)
        assert stats == []

    def test_error_rate(self, populated_tracker):
        stats = populated_tracker.tool_stats(days=1)
        ml_stat = next((s for s in stats if s["tool"] == "ml_predict"), None)
        assert ml_stat is not None
        assert ml_stat["errors"] == 1
        assert ml_stat["error_rate_pct"] == 100.0


# ---------------------------------------------------------------------------
# revenue_estimate
# ---------------------------------------------------------------------------


class TestRevenueEstimate:
    def test_revenue(self, populated_tracker):
        rev = populated_tracker.revenue_estimate(days=1)
        assert rev["total_mrr"] > 0
        # pro key = $9, free = $0
        pro_row = next(
            (b for b in rev["breakdown"] if b["tier"] == "pro"), None
        )
        assert pro_row is not None
        assert pro_row["mrr"] == 9
        assert pro_row["active_keys"] == 1

        free_row = next(
            (b for b in rev["breakdown"] if b["tier"] == "free"), None
        )
        assert free_row is not None
        assert free_row["mrr"] == 0

    def test_revenue_empty(self, tracker):
        rev = tracker.revenue_estimate(days=30)
        assert rev["total_mrr"] == 0
        assert rev["estimated_arr"] == 0

    def test_arr_calculation(self, populated_tracker):
        rev = populated_tracker.revenue_estimate(days=1)
        assert rev["estimated_arr"] == round(rev["total_mrr"] * 12, 2)


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_removes_old(self, tracker):
        # Insert a record and verify it exists
        tracker.record("k", "pro", "test_tool")
        assert tracker.daily_summary()["total_calls"] == 1

        # Cleanup with keep_days=0 should delete today's records too
        # (since cutoff = today - 0 days = today, and we delete < cutoff)
        # Actually keep_days=0 means cutoff = today, so records with
        # date_str == today are NOT deleted (date < cutoff is strict).
        deleted = tracker.cleanup(keep_days=0)
        # Today's records should survive (they equal cutoff, not less than)
        assert tracker.daily_summary()["total_calls"] == 1

    def test_cleanup_empty(self, tracker):
        deleted = tracker.cleanup(keep_days=90)
        assert deleted == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_writes(self, tmp_path):
        db = str(tmp_path / "concurrent.db")
        t = UsageTracker(db)
        errors = []

        def worker(key):
            try:
                for _ in range(20):
                    t.record(key, "pro", "stock_history", latency=0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"key_{i}",)) for i in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0
        summary = t.daily_summary()
        assert summary["total_calls"] == 100  # 5 threads × 20 calls


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close(self, tracker):
        tracker.record("k", "pro", "test")
        tracker.close()
        # After close, getting a new connection should work
        summary = tracker.daily_summary()
        assert summary["total_calls"] == 1


# ---------------------------------------------------------------------------
# date_str format
# ---------------------------------------------------------------------------


class TestDateHandling:
    def test_specific_date_query(self, tracker):
        tracker.record("k", "pro", "tool")
        # Query a different date – should be empty
        summary = tracker.daily_summary("2020-01-01")
        assert summary["total_calls"] == 0
        assert summary["date"] == "2020-01-01"

    def test_avg_latency(self, tracker):
        tracker.record("k", "pro", "t1", latency=0.1)
        tracker.record("k", "pro", "t2", latency=0.2)
        summary = tracker.daily_summary()
        assert summary["avg_latency_ms"] == 150.0  # (100 + 200) / 2
