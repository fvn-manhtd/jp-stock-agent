"""Tests for the jpstock_agent.core module.

Tests data retrieval functions, helper utilities, error handling,
and integration with yfinance, J-Quants, and vnstock.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from jpstock_agent.core import (
    _default_dates,
    _df_to_records,
    _safe_call,
    company_overview,
    stock_history,
    stock_intraday,
    stock_price_depth,
)


class TestDefaultDates:
    """Test the _default_dates() helper function."""

    def test_default_dates_no_args(self):
        """Test that _default_dates() returns ~90 days of data with no args."""
        start, end = _default_dates(None, None)

        # Parse the returned date strings
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        start_dt = datetime.strptime(start, "%Y-%m-%d")

        # End should be today-ish (within 1 day due to execution time)
        today = datetime.now()
        assert (today - end_dt).days <= 1

        # Start should be ~90 days before end
        delta_days = (end_dt - start_dt).days
        assert 89 <= delta_days <= 91

    def test_default_dates_with_explicit_start(self):
        """Test that explicit start date is respected."""
        start, end = _default_dates("2026-01-01", None)
        assert start == "2026-01-01"
        # end should still be today-ish
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        today = datetime.now()
        assert (today - end_dt).days <= 1

    def test_default_dates_with_explicit_end(self):
        """Test that explicit end date is respected."""
        explicit_end = "2026-03-15"
        start, end = _default_dates(None, explicit_end)
        assert end == explicit_end
        # start should be ~90 days before *today*, not before the explicit end
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        today = datetime.now()
        delta_days = (today - start_dt).days
        assert 89 <= delta_days <= 91

    def test_default_dates_with_both_explicit(self):
        """Test that both explicit dates are respected."""
        start, end = _default_dates("2026-01-01", "2026-03-01")
        assert start == "2026-01-01"
        assert end == "2026-03-01"

    def test_default_dates_format_yyyy_mm_dd(self):
        """Test that returned dates are in YYYY-MM-DD format."""
        start, end = _default_dates(None, None)
        # Should parse without error
        datetime.strptime(start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")


class TestDfToRecords:
    """Test the _df_to_records() helper function."""

    def test_df_to_records_simple_dataframe(self):
        """Test conversion of a simple DataFrame to list[dict]."""
        df = pd.DataFrame({
            "ticker": ["7203", "6758"],
            "price": [2500.0, 100.0],
        })
        result = _df_to_records(df)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["ticker"] == "7203"
        assert result[0]["price"] == 2500.0
        assert result[1]["ticker"] == "6758"

    def test_df_to_records_with_index(self):
        """Test that non-RangeIndex is reset and included in records."""
        dates = pd.date_range("2026-01-01", periods=3)
        df = pd.DataFrame({
            "price": [100.0, 101.0, 102.0],
        }, index=dates)
        df.index.name = "date"

        result = _df_to_records(df)

        assert len(result) == 3
        # date should now be a column
        assert "date" in result[0]
        # Timestamp should be converted to ISO string
        assert isinstance(result[0]["date"], str)
        assert "2026-01-01" in result[0]["date"]

    def test_df_to_records_timestamp_conversion(self):
        """Test that pd.Timestamp values are converted to ISO strings."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "value": [1.0, 2.0],
        })
        result = _df_to_records(df)

        assert isinstance(result[0]["date"], str)
        assert result[0]["date"].startswith("2026-01-01")

    def test_df_to_records_nan_to_none(self):
        """Test that NaN values are converted to None."""
        df = pd.DataFrame({
            "value": [1.0, np.nan, 3.0],
        })
        result = _df_to_records(df)

        assert result[0]["value"] == 1.0
        assert result[1]["value"] is None
        assert result[2]["value"] == 3.0

    def test_df_to_records_inf_to_none(self):
        """Test that inf values are converted to None."""
        df = pd.DataFrame({
            "value": [1.0, np.inf, -np.inf],
        })
        result = _df_to_records(df)

        assert result[0]["value"] == 1.0
        assert result[1]["value"] is None
        assert result[2]["value"] is None

    def test_df_to_records_empty_dataframe(self):
        """Test that empty DataFrame returns empty list."""
        df = pd.DataFrame()
        result = _df_to_records(df)
        assert result == []

    def test_df_to_records_none_input(self):
        """Test that None input returns empty list."""
        result = _df_to_records(None)
        assert result == []

    def test_df_to_records_series_conversion(self):
        """Test that a Series is converted to list[dict] with one record."""
        series = pd.Series({"a": 1, "b": 2})
        result = _df_to_records(series)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["a"] == 1
        assert result[0]["b"] == 2

    def test_df_to_records_multiindex(self):
        """Test that MultiIndex is reset."""
        arrays = [["a", "a", "b", "b"], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        df = pd.DataFrame({"value": [10, 20, 30, 40]}, index=index)

        result = _df_to_records(df)

        assert len(result) == 4
        # Both index levels should now be columns
        assert "first" in result[0]
        assert "second" in result[0]

    def test_df_to_records_non_string_column_names(self):
        """Test that non-string column names are converted to strings."""
        df = pd.DataFrame({
            0: [1, 2],
            1: [3, 4],
        })
        result = _df_to_records(df)

        # Column names should be strings
        assert all(isinstance(k, str) for k in result[0].keys())
        assert "0" in result[0]
        assert "1" in result[0]


class TestSafeCall:
    """Test the _safe_call() helper function."""

    def test_safe_call_success(self):
        """Test that _safe_call returns the result on success."""
        def dummy_func(a, b):
            return a + b

        result = _safe_call(dummy_func, 2, 3)
        assert result == 5

    def test_safe_call_with_kwargs(self):
        """Test that _safe_call passes kwargs correctly."""
        def dummy_func(a, b=10):
            return a + b

        result = _safe_call(dummy_func, 5, b=20)
        assert result == 25

    def test_safe_call_exception_handling(self):
        """Test that _safe_call catches exceptions and returns error dict."""
        def failing_func():
            raise ValueError("Test error message")

        result = _safe_call(failing_func)

        assert isinstance(result, dict)
        assert "error" in result
        assert "ValueError" in result["error"]
        assert "Test error message" in result["error"]

    def test_safe_call_type_error(self):
        """Test that _safe_call handles TypeError correctly."""
        def dummy_func(a, b):
            return a + b

        result = _safe_call(dummy_func, "only_one_arg")

        assert isinstance(result, dict)
        assert "error" in result
        assert "TypeError" in result["error"]

    def test_safe_call_returns_none_gracefully(self):
        """Test that _safe_call handles functions returning None."""
        def dummy_func():
            return None

        result = _safe_call(dummy_func)
        assert result is None


class TestStockHistory:
    """Test the stock_history() function."""

    @patch("yfinance.Ticker")
    def test_stock_history_yfinance_success(self, mock_ticker_class):
        """Test stock_history with yfinance source returns list[dict]."""
        # Create a mock DataFrame
        dates = pd.date_range("2026-01-01", periods=5)
        mock_df = pd.DataFrame({
            "open": [2500.0, 2510.0, 2520.0, 2530.0, 2540.0],
            "high": [2510.0, 2520.0, 2530.0, 2540.0, 2550.0],
            "low": [2490.0, 2500.0, 2510.0, 2520.0, 2530.0],
            "close": [2505.0, 2515.0, 2525.0, 2535.0, 2545.0],
            "volume": [1000000.0, 1100000.0, 1200000.0, 1300000.0, 1400000.0],
        }, index=dates)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        result = stock_history("7203", source="yfinance")

        assert isinstance(result, list)
        assert len(result) == 5
        assert result[0]["close"] == 2505.0
        assert result[0]["volume"] == 1000000.0
        mock_ticker_class.assert_called_once_with("7203.T")

    @patch("yfinance.Ticker")
    def test_stock_history_yfinance_empty_result(self, mock_ticker_class):
        """Test stock_history with yfinance returns empty list when no data."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        result = stock_history("FAKE", source="yfinance")

        assert result == []

    @patch("yfinance.Ticker")
    def test_stock_history_yfinance_exception_handling(self, mock_ticker_class):
        """Test stock_history returns error dict on yfinance exception."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_ticker_class.return_value = mock_ticker

        result = stock_history("7203", source="yfinance")

        assert isinstance(result, dict)
        assert "error" in result
        assert "Exception" in result["error"]

    @patch("jpstock_agent.core.get_jquants_client")
    def test_stock_history_jquants_no_credentials(self, mock_get_client):
        """Test stock_history returns error when J-Quants credentials not configured."""
        mock_get_client.return_value = None

        result = stock_history("7203", source="jquants")

        assert isinstance(result, dict)
        assert "error" in result
        assert "J-Quants credentials not configured" in result["error"]

    @patch("yfinance.Ticker")
    def test_stock_history_respects_date_range(self, mock_ticker_class):
        """Test that stock_history respects explicit date range."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        stock_history("7203", start="2026-01-01", end="2026-03-01", source="yfinance")

        # Verify history was called with correct date range
        call_args = mock_ticker.history.call_args
        assert call_args[1]["start"] == "2026-01-01"
        assert call_args[1]["end"] == "2026-03-01"

    @patch("yfinance.Ticker")
    def test_stock_history_auto_detects_source_japanese(self, mock_ticker_class):
        """Test that stock_history auto-detects yfinance for Japanese tickers."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        stock_history("7203")  # No source specified

        mock_ticker_class.assert_called_once()

    def test_stock_history_auto_detects_source_vietnamese(self):
        """Test that stock_history auto-detects vnstocks for Vietnamese tickers."""
        with patch("jpstock_agent.core._vn_stock") as mock_vn_stock:
            mock_stock = MagicMock()
            mock_stock.quote.history.return_value = pd.DataFrame()
            mock_vn_stock.return_value = mock_stock

            stock_history("ACB")  # Vietnamese ticker, no source specified

            mock_vn_stock.assert_called_once()


class TestStockIntraday:
    """Test the stock_intraday() function."""

    @patch("yfinance.Ticker")
    def test_stock_intraday_yfinance(self, mock_ticker_class):
        """Test stock_intraday with yfinance returns intraday data."""
        # Create mock intraday data
        times = pd.date_range("2026-04-02 09:00", periods=5, freq="1min")
        mock_df = pd.DataFrame({
            "open": [2500.0, 2501.0, 2502.0, 2503.0, 2504.0],
            "high": [2501.0, 2502.0, 2503.0, 2504.0, 2505.0],
            "low": [2499.0, 2500.0, 2501.0, 2502.0, 2503.0],
            "close": [2500.5, 2501.5, 2502.5, 2503.5, 2504.5],
            "volume": [100000.0, 110000.0, 120000.0, 130000.0, 140000.0],
        }, index=times)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        result = stock_intraday("7203", source="yfinance")

        assert isinstance(result, list)
        assert len(result) == 5
        # Verify 1-minute interval was requested
        call_args = mock_ticker.history.call_args
        assert call_args[1]["interval"] == "1m"

    @patch("jpstock_agent.core.get_jquants_client")
    def test_stock_intraday_jquants_no_credentials(self, mock_get_client):
        """Test stock_intraday returns error when J-Quants credentials missing."""
        mock_get_client.return_value = None

        result = stock_intraday("7203", source="jquants")

        assert isinstance(result, dict)
        assert "error" in result
        assert "J-Quants credentials not configured" in result["error"]


class TestStockPriceDepth:
    """Test the stock_price_depth() function."""

    @patch("yfinance.Ticker")
    def test_stock_price_depth_yfinance(self, mock_ticker_class):
        """Test stock_price_depth extracts bid/ask data from yfinance."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "bid": 2500.0,
            "bidSize": 1000,
            "ask": 2510.0,
            "askSize": 2000,
            "lastPrice": 2505.0,
            "volume": 5000000,
            "regularMarketPrice": 2505.0,
            "other_field": "ignored",
        }
        mock_ticker_class.return_value = mock_ticker

        result = stock_price_depth("7203", source="yfinance")

        assert isinstance(result, dict)
        assert result["bid"] == 2500.0
        assert result["ask"] == 2510.0
        assert result["bidSize"] == 1000
        assert result["askSize"] == 2000
        # other_field should not be included
        assert "other_field" not in result

    def test_stock_price_depth_jquants_not_available(self):
        """Test that stock_price_depth returns error for jquants."""
        result = stock_price_depth("7203", source="jquants")

        assert isinstance(result, dict)
        assert "error" in result
        assert "not available" in result["error"]

    def test_stock_price_depth_vnstocks_not_available(self):
        """Test that stock_price_depth returns error for vnstocks."""
        result = stock_price_depth("ACB", source="vnstocks")

        assert isinstance(result, dict)
        assert "error" in result
        assert "not available" in result["error"]


class TestCompanyOverview:
    """Test the company_overview() function."""

    @patch("yfinance.Ticker")
    def test_company_overview_yfinance(self, mock_ticker_class):
        """Test company_overview with yfinance returns dict of overview info."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "shortName": "Toyota",
            "longName": "Toyota Motor Corporation",
            "symbol": "7203.T",
            "exchange": "JPX",
            "sector": "Automotive",
            "industry": "Auto Manufacturers",
            "longBusinessSummary": "Toyota is a leading automotive company...",
            "marketCap": 250000000000,
            "enterpriseValue": 260000000000,
            "trailingPE": 9.5,
            "dividendYield": 0.035,
            "beta": 1.2,
            "fiftyTwoWeekHigh": 2700.0,
            "fiftyTwoWeekLow": 2200.0,
            "currency": "JPY",
            "website": "https://www.toyota.co.jp",
            "fullTimeEmployees": 370000,
            "irrelevant_field": "ignored",
        }
        mock_ticker_class.return_value = mock_ticker

        result = company_overview("7203", source="yfinance")

        assert isinstance(result, dict)
        assert result["shortName"] == "Toyota"
        assert result["sector"] == "Automotive"
        assert result["marketCap"] == 250000000000
        assert result["trailingPE"] == 9.5
        # Irrelevant fields should be filtered out
        assert "irrelevant_field" not in result

    @patch("jpstock_agent.core.get_jquants_client")
    def test_company_overview_jquants_no_credentials(self, mock_get_client):
        """Test company_overview returns error when J-Quants credentials missing."""
        mock_get_client.return_value = None

        result = company_overview("7203", source="jquants")

        assert isinstance(result, dict)
        assert "error" in result
        assert "J-Quants credentials not configured" in result["error"]

    @patch("yfinance.Ticker")
    def test_company_overview_partial_info(self, mock_ticker_class):
        """Test company_overview handles partial info gracefully."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "shortName": "TestCorp",
            "sector": "Technology",
            # Many fields are missing
        }
        mock_ticker_class.return_value = mock_ticker

        result = company_overview("FAKE", source="yfinance")

        assert isinstance(result, dict)
        assert result["shortName"] == "TestCorp"
        assert result["sector"] == "Technology"
        # Missing fields should not raise errors
        assert len(result) == 2


class TestErrorHandling:
    """Test error handling across core functions."""

    @patch("yfinance.Ticker")
    def test_stock_history_network_error(self, mock_ticker_class):
        """Test that network errors are caught and returned as error dict."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = ConnectionError("Network failure")
        mock_ticker_class.return_value = mock_ticker

        result = stock_history("7203", source="yfinance")

        assert isinstance(result, dict)
        assert "error" in result
        assert "ConnectionError" in result["error"]

    @patch("yfinance.Ticker")
    def test_company_overview_invalid_ticker(self, mock_ticker_class):
        """Test that invalid ticker symbol errors are handled."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # Empty info for invalid ticker
        mock_ticker_class.return_value = mock_ticker

        result = company_overview("INVALID", source="yfinance")

        assert isinstance(result, dict)
        # Result may be empty dict, not an error dict
        # (yfinance doesn't always error on invalid tickers)


class TestIntegrationScenarios:
    """Integration-style tests combining multiple functions."""

    @patch("yfinance.Ticker")
    def test_stock_history_and_price_depth_integration(self, mock_ticker_class):
        """Test retrieving history and depth data for same symbol."""
        dates = pd.date_range("2026-01-01", periods=3)
        mock_df = pd.DataFrame({
            "close": [2500.0, 2510.0, 2520.0],
            "volume": [1000000.0, 1100000.0, 1200000.0],
        }, index=dates)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker.info = {
            "bid": 2520.0,
            "ask": 2530.0,
            "bidSize": 1000,
            "askSize": 2000,
        }
        mock_ticker_class.return_value = mock_ticker

        history = stock_history("7203", source="yfinance")
        depth = stock_price_depth("7203", source="yfinance")

        assert len(history) == 3
        assert history[0]["close"] == 2500.0
        assert depth["bid"] == 2520.0
        assert depth["ask"] == 2530.0
