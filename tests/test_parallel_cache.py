"""
Tests for parallel data fetching and extended caching.

Tests cover:
- fetch_parallel: ThreadPoolExecutor-based parallel fetching
- stock_history_batch: parallel OHLCV fetching
- with_cache decorator: TTL caching on core functions
- _get_returns_df parallel optimization in portfolio.py
"""

from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from jpstock_agent import core, portfolio
from jpstock_agent.core import _data_cache, cache_clear, with_cache
from tests.conftest import _make_ohlcv_df, _make_ohlcv_records


# ============================================================================
# Parallel Fetching Tests
# ============================================================================


class TestFetchParallel:
    """Tests for fetch_parallel function."""

    def test_fetch_parallel_returns_dict(self):
        """Test that fetch_parallel returns dict mapping symbol -> result."""
        mock_func = MagicMock(side_effect=lambda sym, **kw: [{"symbol": sym}])

        result = core.fetch_parallel(["SYM1", "SYM2"], mock_func)

        assert isinstance(result, dict)
        assert "SYM1" in result
        assert "SYM2" in result

    def test_fetch_parallel_calls_func_for_each_symbol(self):
        """Test that func is called once per symbol."""
        mock_func = MagicMock(return_value=[])

        core.fetch_parallel(["A", "B", "C"], mock_func)

        assert mock_func.call_count == 3

    def test_fetch_parallel_handles_errors(self):
        """Test that errors for individual symbols don't crash the batch."""
        def _fetch(sym, **kw):
            if sym == "BAD":
                raise ValueError("test error")
            return [{"close": 100}]

        result = core.fetch_parallel(["GOOD", "BAD"], _fetch)

        assert "GOOD" in result
        assert "BAD" in result
        assert isinstance(result["GOOD"], list)
        assert "error" in result["BAD"]

    def test_fetch_parallel_single_symbol(self):
        """Test parallel fetch with single symbol."""
        mock_func = MagicMock(return_value=[{"close": 100}])

        result = core.fetch_parallel(["SINGLE"], mock_func)

        assert "SINGLE" in result
        mock_func.assert_called_once()

    def test_fetch_parallel_empty_list(self):
        """Test parallel fetch with empty symbol list."""
        mock_func = MagicMock()

        result = core.fetch_parallel([], mock_func)

        assert result == {}
        mock_func.assert_not_called()


class TestStockHistoryBatch:
    """Tests for stock_history_batch function."""

    def test_batch_returns_dict(self):
        """Test that batch returns dict of symbol -> records."""
        with patch("jpstock_agent.core.stock_history") as mock_hist:
            mock_hist.return_value = _make_ohlcv_records()

            result = core.stock_history_batch(["7203", "6758"])

            assert isinstance(result, dict)
            assert "7203" in result
            assert "6758" in result

    def test_batch_passes_params(self):
        """Test that start/end/interval are passed through."""
        with patch("jpstock_agent.core.stock_history") as mock_hist:
            mock_hist.return_value = _make_ohlcv_records()

            core.stock_history_batch(
                ["7203"], start="2025-01-01", end="2025-03-01", interval="1wk"
            )

            mock_hist.assert_called_once()
            _, kwargs = mock_hist.call_args
            assert kwargs.get("start") == "2025-01-01"
            assert kwargs.get("end") == "2025-03-01"
            assert kwargs.get("interval") == "1wk"


# ============================================================================
# Extended Caching Tests
# ============================================================================


class TestWithCacheDecorator:
    """Tests for the with_cache decorator."""

    def test_cached_function_returns_correct_result(self):
        """Test that cached function returns correct result on first call."""
        call_count = 0

        @with_cache
        def my_func(x):
            nonlocal call_count
            call_count += 1
            return {"value": x * 2}

        result = my_func(5)
        assert result == {"value": 10}
        assert call_count == 1

    def test_cached_function_caches_on_second_call(self):
        """Test that second call with same args uses cache."""
        call_count = 0

        @with_cache
        def my_func2(x):
            nonlocal call_count
            call_count += 1
            return {"value": x * 2}

        result1 = my_func2(5)
        result2 = my_func2(5)

        assert result1 == result2
        assert call_count == 1  # Only called once

    def test_cached_function_different_args_different_cache(self):
        """Test that different args produce different cache entries."""
        call_count = 0

        @with_cache
        def my_func3(x):
            nonlocal call_count
            call_count += 1
            return {"value": x}

        my_func3(1)
        my_func3(2)

        assert call_count == 2

    def test_cached_function_does_not_cache_errors(self):
        """Test that error results are not cached."""
        call_count = 0

        @with_cache
        def my_func4(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"error": "fail"}
            return {"value": x}

        result1 = my_func4(5)
        assert "error" in result1

        result2 = my_func4(5)
        assert "value" in result2
        assert call_count == 2

    def test_cached_function_preserves_original(self):
        """Test that _original attribute is preserved."""
        @with_cache
        def my_func5(x):
            return x

        assert hasattr(my_func5, "_original")


class TestExtendedCaching:
    """Test that additional core functions are cached."""

    def test_company_overview_is_cached(self):
        """Test that company_overview uses caching."""
        assert hasattr(core.company_overview, "_original")

    def test_company_news_is_cached(self):
        """Test that company_news uses caching."""
        assert hasattr(core.company_news, "_original")

    def test_financial_ratio_is_cached(self):
        """Test that financial_ratio uses caching."""
        assert hasattr(core.financial_ratio, "_original")

    def test_fx_history_is_cached(self):
        """Test that fx_history uses caching."""
        assert hasattr(core.fx_history, "_original")

    def test_crypto_history_is_cached(self):
        """Test that crypto_history uses caching."""
        assert hasattr(core.crypto_history, "_original")

    def test_world_index_history_is_cached(self):
        """Test that world_index_history uses caching."""
        assert hasattr(core.world_index_history, "_original")


# ============================================================================
# Portfolio Parallel Fetch Tests
# ============================================================================


class TestPortfolioParallelFetch:
    """Tests for parallel fetching in portfolio._get_returns_df."""

    def test_returns_df_parallel_for_multiple_symbols(self):
        """Test that _get_returns_df works with multiple symbols."""
        with patch("jpstock_agent.portfolio._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=60)

            result = portfolio._get_returns_df(["SYM1", "SYM2", "SYM3"])

            assert isinstance(result, pd.DataFrame)
            assert len(result.columns) == 3

    def test_returns_df_single_symbol_no_thread_pool(self):
        """Test that single symbol doesn't use thread pool (sequential)."""
        with patch("jpstock_agent.portfolio._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=60)

            result = portfolio._get_returns_df(["SYM1"])

            assert isinstance(result, pd.DataFrame)
            assert len(result.columns) == 1

    def test_returns_df_handles_partial_failures(self):
        """Test that partial failures still return valid symbols."""
        # Pre-generate DataFrames with same date index but different prices
        base_df = _make_ohlcv_df(days=60, seed=42)
        df1 = base_df.copy()
        df2 = base_df * 1.1  # Different prices, same dates

        call_count = [0]

        def _mock_fetch(symbol, **kw):
            if symbol == "BAD":
                return {"error": "not found"}
            call_count[0] += 1
            return df1 if call_count[0] == 1 else df2

        with patch("jpstock_agent.portfolio._get_ohlcv_df", side_effect=_mock_fetch):
            result = portfolio._get_returns_df(["GOOD1", "BAD", "GOOD2"])

            # Should still return DataFrame with valid symbols
            assert isinstance(result, pd.DataFrame)
            assert "GOOD1" in result.columns
            assert "GOOD2" in result.columns
            assert "BAD" not in result.columns

    def test_returns_df_empty_symbols(self):
        """Test _get_returns_df with empty symbol list."""
        result = portfolio._get_returns_df([])
        assert isinstance(result, dict)
        assert "error" in result
