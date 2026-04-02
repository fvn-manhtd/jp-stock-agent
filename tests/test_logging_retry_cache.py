"""Tests for structured logging, retry logic, and caching.

Tests the new infrastructure modules added for reliability and observability.
"""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from jpstock_agent.core import (
    _RETRYABLE_EXCEPTIONS,
    _TTLCache,
    _data_cache,
    _safe_call_with_retry,
    cache_clear,
)
from jpstock_agent.logging import JSONFormatter, LogTimer, get_logger


# ---------------------------------------------------------------------------
# Structured Logging Tests
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    """Test the JSON log formatter."""

    def test_format_basic_message(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert '"message": "Hello world"' in output
        assert '"level": "INFO"' in output
        assert '"logger": "test"' in output

    def test_format_with_extra_fields(self):
        import json
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Fetch data", args=(), exc_info=None,
        )
        record.symbol = "7203"
        record.source = "yfinance"
        record.duration_ms = 150.5
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["symbol"] == "7203"
        assert parsed["source"] == "yfinance"
        assert parsed["duration_ms"] == 150.5

    def test_format_without_extra_fields(self):
        import json
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Simple", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "symbol" not in parsed
        assert "source" not in parsed


class TestGetLogger:
    """Test the get_logger factory."""

    def test_returns_logger(self):
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_module"

    def test_has_json_handler(self):
        log = get_logger("test_json_handler")
        assert any(isinstance(h.formatter, JSONFormatter) for h in log.handlers)

    def test_custom_level(self):
        log = get_logger("test_level", level=logging.DEBUG)
        assert log.level == logging.DEBUG


class TestLogTimer:
    """Test the LogTimer context manager."""

    def test_log_timer_success(self):
        log = get_logger("test_timer")
        with LogTimer(log, "test_operation", symbol="7203"):
            time.sleep(0.01)  # Small delay to measure

    def test_log_timer_failure(self):
        log = get_logger("test_timer_fail")
        with pytest.raises(ValueError):
            with LogTimer(log, "failing_op"):
                raise ValueError("Test error")


# ---------------------------------------------------------------------------
# Retry Logic Tests
# ---------------------------------------------------------------------------


class TestSafeCallWithRetry:
    """Test the _safe_call_with_retry function."""

    def test_success_on_first_try(self):
        func = MagicMock(return_value="success")
        result = _safe_call_with_retry(func, max_attempts=3)
        assert result == "success"
        assert func.call_count == 1

    def test_retry_on_connection_error(self):
        func = MagicMock(side_effect=[ConnectionError("fail"), "success"])
        result = _safe_call_with_retry(func, max_attempts=3)
        assert result == "success"
        assert func.call_count == 2

    def test_retry_on_timeout_error(self):
        func = MagicMock(side_effect=[TimeoutError("timeout"), "ok"])
        result = _safe_call_with_retry(func, max_attempts=3)
        assert result == "ok"
        assert func.call_count == 2

    def test_retry_on_os_error(self):
        func = MagicMock(side_effect=[OSError("network"), "ok"])
        result = _safe_call_with_retry(func, max_attempts=3)
        assert result == "ok"

    def test_no_retry_on_value_error(self):
        """Non-retryable exceptions should fail immediately."""
        func = MagicMock(side_effect=ValueError("bad input"))
        result = _safe_call_with_retry(func, max_attempts=3)
        assert isinstance(result, dict)
        assert "error" in result
        assert "ValueError" in result["error"]
        assert func.call_count == 1  # No retries

    def test_all_retries_exhausted(self):
        """All retries fail → returns error dict."""
        func = MagicMock(side_effect=ConnectionError("persistent failure"))
        result = _safe_call_with_retry(func, max_attempts=2)
        assert isinstance(result, dict)
        assert "error" in result
        assert "ConnectionError" in result["error"]
        assert func.call_count == 2

    def test_retry_with_args_and_kwargs(self):
        func = MagicMock(return_value=42)
        result = _safe_call_with_retry(func, "arg1", "arg2", max_attempts=1, key="val")
        assert result == 42
        func.assert_called_once_with("arg1", "arg2", key="val")

    @patch("jpstock_agent.core.time.sleep")
    def test_exponential_backoff_timing(self, mock_sleep):
        """Verify exponential backoff delays between retries."""
        func = MagicMock(side_effect=[
            ConnectionError("1"), ConnectionError("2"), "success"
        ])
        result = _safe_call_with_retry(func, max_attempts=3)
        assert result == "success"
        # Check backoff delays: 0.5s, then 1.0s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.5)
        mock_sleep.assert_any_call(1.0)


# ---------------------------------------------------------------------------
# TTL Cache Tests
# ---------------------------------------------------------------------------


class TestTTLCache:
    """Test the _TTLCache class."""

    def test_put_and_get(self):
        cache = _TTLCache(maxsize=10, ttl=60)
        cache.put("func", ("a",), {}, "value_a")
        hit, val = cache.get("func", ("a",), {})
        assert hit is True
        assert val == "value_a"

    def test_cache_miss(self):
        cache = _TTLCache(maxsize=10, ttl=60)
        hit, val = cache.get("func", ("nonexistent",), {})
        assert hit is False
        assert val is None

    def test_different_args_different_entries(self):
        cache = _TTLCache(maxsize=10, ttl=60)
        cache.put("func", ("a",), {}, "val_a")
        cache.put("func", ("b",), {}, "val_b")

        hit_a, val_a = cache.get("func", ("a",), {})
        hit_b, val_b = cache.get("func", ("b",), {})
        assert val_a == "val_a"
        assert val_b == "val_b"

    def test_ttl_expiry(self):
        cache = _TTLCache(maxsize=10, ttl=0)  # TTL = 0 → expires immediately
        cache.put("func", ("a",), {}, "value")
        time.sleep(0.01)
        hit, val = cache.get("func", ("a",), {})
        assert hit is False

    def test_maxsize_eviction(self):
        cache = _TTLCache(maxsize=2, ttl=60)
        cache.put("f", ("1",), {}, "v1")
        cache.put("f", ("2",), {}, "v2")
        cache.put("f", ("3",), {}, "v3")  # Should evict "1"

        assert cache.size == 2
        hit1, _ = cache.get("f", ("1",), {})
        assert hit1 is False  # Evicted
        hit3, val3 = cache.get("f", ("3",), {})
        assert hit3 is True
        assert val3 == "v3"

    def test_clear(self):
        cache = _TTLCache(maxsize=10, ttl=60)
        cache.put("f", ("a",), {}, "v")
        cache.put("f", ("b",), {}, "v")
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0

    def test_kwargs_in_key(self):
        cache = _TTLCache(maxsize=10, ttl=60)
        cache.put("func", (), {"k": "v1"}, "result1")
        cache.put("func", (), {"k": "v2"}, "result2")

        hit1, val1 = cache.get("func", (), {"k": "v1"})
        hit2, val2 = cache.get("func", (), {"k": "v2"})
        assert val1 == "result1"
        assert val2 == "result2"


class TestCacheClear:
    """Test the cache_clear() API."""

    def test_cache_clear_returns_message(self):
        # Ensure cache has something
        _data_cache.put("test", ("a",), {}, "val")
        result = cache_clear()
        assert "message" in result
        assert "cleared" in result["message"]
        assert _data_cache.size == 0


class TestStockHistoryWithCache:
    """Test that stock_history uses caching."""

    @patch("yfinance.Ticker")
    def test_second_call_uses_cache(self, mock_ticker_class):
        """Verify that calling stock_history twice uses cache on second call."""
        import pandas as pd
        from jpstock_agent.core import stock_history

        dates = pd.date_range("2026-01-01", periods=3)
        mock_df = pd.DataFrame({"close": [100, 101, 102]}, index=dates)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        # Clear cache first
        cache_clear()

        # First call - should hit yfinance
        result1 = stock_history("9999", start="2026-01-01", end="2026-01-03", source="yfinance")
        assert len(result1) == 3

        # Second call - should use cache, not call yfinance again
        result2 = stock_history("9999", start="2026-01-01", end="2026-01-03", source="yfinance")
        assert result1 == result2

        # yfinance Ticker should only be called once (cached on second call)
        assert mock_ticker.history.call_count == 1

        cache_clear()  # Cleanup
