"""Tests for server.py – MCP tool definitions.

These tests verify that server tools correctly call core/ta/backtest/etc.
functions and return JSON-serialized results.
"""

import json
from unittest.mock import patch

import pytest

from jpstock_agent import server


class TestServerApp:
    """Test the FastMCP app configuration."""

    def test_app_exists(self):
        assert server.app is not None

    def test_app_name(self):
        assert server.app.name == "JPStock Agent"


class TestQuoteTools:
    """Test quote-related server tools."""

    def test_stock_history_returns_json(self):
        mock_data = [{"date": "2026-01-01", "close": 2500.0}]
        with patch("jpstock_agent.server.core.stock_history", return_value=mock_data):
            result = server.stock_history("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)
            assert parsed[0]["close"] == 2500.0

    def test_stock_intraday_returns_json(self):
        mock_data = [{"time": "09:00", "price": 2500.0}]
        with patch("jpstock_agent.server.core.stock_intraday", return_value=mock_data):
            result = server.stock_intraday("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_stock_price_depth_returns_json(self):
        mock_data = {"bid": 2500, "ask": 2501}
        with patch("jpstock_agent.server.core.stock_price_depth", return_value=mock_data):
            result = server.stock_price_depth("7203")
            parsed = json.loads(result)
            assert "bid" in parsed

    def test_stock_history_error_returns_json(self):
        mock_data = {"error": "Symbol not found"}
        with patch("jpstock_agent.server.core.stock_history", return_value=mock_data):
            result = server.stock_history("XXXX")
            parsed = json.loads(result)
            assert "error" in parsed


class TestCompanyTools:
    """Test company info server tools."""

    def test_company_overview_returns_json(self):
        mock_data = {"name": "Toyota", "sector": "Auto"}
        with patch("jpstock_agent.server.core.company_overview", return_value=mock_data):
            result = server.company_overview("7203")
            parsed = json.loads(result)
            assert parsed["name"] == "Toyota"

    def test_company_news_returns_json(self):
        mock_data = [{"title": "News 1"}]
        with patch("jpstock_agent.server.core.company_news", return_value=mock_data):
            result = server.company_news("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_company_officers_returns_json(self):
        mock_data = [{"name": "CEO Name"}]
        with patch("jpstock_agent.server.core.company_officers", return_value=mock_data):
            result = server.company_officers("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_company_shareholders_returns_json(self):
        mock_data = [{"holder": "Holder1"}]
        with patch("jpstock_agent.server.core.company_shareholders", return_value=mock_data):
            result = server.company_shareholders("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_company_events_returns_json(self):
        mock_data = {"dividends": [], "splits": []}
        with patch("jpstock_agent.server.core.company_events", return_value=mock_data):
            result = server.company_events("7203")
            parsed = json.loads(result)
            assert "dividends" in parsed


class TestFinancialTools:
    """Test financial data server tools."""

    def test_financial_income_statement(self):
        mock_data = [{"revenue": 1000000}]
        with patch("jpstock_agent.server.core.financial_income_statement", return_value=mock_data):
            result = server.financial_income_statement("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_financial_balance_sheet(self):
        mock_data = [{"total_assets": 5000000}]
        with patch("jpstock_agent.server.core.financial_balance_sheet", return_value=mock_data):
            result = server.financial_balance_sheet("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_financial_cash_flow(self):
        mock_data = [{"operating_cash_flow": 300000}]
        with patch("jpstock_agent.server.core.financial_cash_flow", return_value=mock_data):
            result = server.financial_cash_flow("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_financial_ratio(self):
        mock_data = {"pe_ratio": 15.5, "pb_ratio": 1.2}
        with patch("jpstock_agent.server.core.financial_ratio", return_value=mock_data):
            result = server.financial_ratio("7203")
            parsed = json.loads(result)
            assert "pe_ratio" in parsed


class TestTATools:
    """Test technical analysis server tools."""

    def test_ta_sma_returns_json(self):
        mock_data = [{"date": "2026-01-01", "sma_20": 2500.0}]
        with patch("jpstock_agent.server.ta.ta_sma", return_value=mock_data):
            result = server.ta_sma("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_ta_rsi_returns_json(self):
        mock_data = [{"date": "2026-01-01", "rsi": 55.0}]
        with patch("jpstock_agent.server.ta.ta_rsi", return_value=mock_data):
            result = server.ta_rsi("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_ta_macd_returns_json(self):
        mock_data = [{"date": "2026-01-01", "macd": 10.0}]
        with patch("jpstock_agent.server.ta.ta_macd", return_value=mock_data):
            result = server.ta_macd("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_ta_bbands_returns_json(self):
        mock_data = [{"upper": 2600, "lower": 2400}]
        with patch("jpstock_agent.server.ta.ta_bbands", return_value=mock_data):
            result = server.ta_bbands("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_ta_multi_indicator_returns_json(self):
        mock_data = {"score": 42, "signal": "BUY"}
        with patch("jpstock_agent.server.ta.ta_multi_indicator", return_value=mock_data):
            result = server.ta_multi_indicator("7203")
            parsed = json.loads(result)
            assert parsed["signal"] == "BUY"

    def test_ta_screen_returns_json(self):
        mock_data = [{"symbol": "7203", "matched": True}]
        with patch("jpstock_agent.server.ta.ta_screen", return_value=mock_data):
            result = server.ta_screen("7203", "oversold")
            parsed = json.loads(result)
            assert isinstance(parsed, list)


class TestCandlestickTools:
    """Test candlestick server tools."""

    def test_ta_candlestick_scan_returns_json(self):
        mock_data = [{"pattern": "doji", "date": "2026-01-01"}]
        with patch("jpstock_agent.server.candlestick.ta_candlestick_scan", return_value=mock_data):
            result = server.ta_candlestick_scan("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_ta_candlestick_latest_returns_json(self):
        mock_data = {"date": "2026-01-01", "patterns": []}
        with patch("jpstock_agent.server.candlestick.ta_candlestick_latest", return_value=mock_data):
            result = server.ta_candlestick_latest("7203")
            parsed = json.loads(result)
            assert "patterns" in parsed

    def test_ta_candlestick_screen_returns_json(self):
        mock_data = [{"symbol": "7203", "patterns": []}]
        with patch("jpstock_agent.server.candlestick.ta_candlestick_screen", return_value=mock_data):
            result = server.ta_candlestick_screen("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)


class TestBacktestTools:
    """Test backtest server tools."""

    def test_backtest_strategy_returns_json(self):
        mock_data = {"total_return": 15.5, "sharpe_ratio": 1.2}
        with patch("jpstock_agent.server.backtest.backtest_strategy", return_value=mock_data):
            result = server.backtest_strategy("7203", "sma_crossover")
            parsed = json.loads(result)
            assert "total_return" in parsed

    def test_backtest_compare_returns_json(self):
        mock_data = [{"strategy": "sma_crossover", "return": 10}]
        with patch("jpstock_agent.server.backtest.backtest_compare", return_value=mock_data):
            result = server.backtest_compare("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_backtest_optimize_returns_json(self):
        mock_data = [{"params": {"period": 20}, "return": 12}]
        with patch("jpstock_agent.server.backtest.backtest_optimize", return_value=mock_data):
            result = server.backtest_optimize("7203", "sma_crossover")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_backtest_walk_forward_returns_json(self):
        mock_data = {"windows": [], "overall": {}}
        with patch("jpstock_agent.server.backtest.backtest_walk_forward", return_value=mock_data):
            result = server.backtest_walk_forward("7203", "sma_crossover")
            parsed = json.loads(result)
            assert "windows" in parsed

    def test_backtest_monte_carlo_returns_json(self):
        mock_data = {"mean_return": 10, "confidence_95": [5, 15]}
        with patch("jpstock_agent.server.backtest.backtest_monte_carlo", return_value=mock_data):
            result = server.backtest_monte_carlo("7203", "sma_crossover")
            parsed = json.loads(result)
            assert "mean_return" in parsed

    def test_backtest_advanced_metrics_returns_json(self):
        mock_data = {"sortino_ratio": 1.5, "calmar_ratio": 0.8}
        with patch("jpstock_agent.server.backtest.backtest_advanced_metrics", return_value=mock_data):
            result = server.backtest_advanced_metrics("7203", "sma_crossover")
            parsed = json.loads(result)
            assert "sortino_ratio" in parsed


class TestPortfolioTools:
    """Test portfolio server tools."""

    def test_portfolio_analyze_returns_json(self):
        mock_data = {"stocks": {}, "correlation": {}}
        with patch("jpstock_agent.server.portfolio.portfolio_analyze", return_value=mock_data):
            result = server.portfolio_analyze("7203,6758")
            parsed = json.loads(result)
            assert isinstance(parsed, dict)

    def test_portfolio_optimize_returns_json(self):
        mock_data = {"max_sharpe": {}, "min_volatility": {}}
        with patch("jpstock_agent.server.portfolio.portfolio_optimize", return_value=mock_data):
            result = server.portfolio_optimize("7203,6758")
            parsed = json.loads(result)
            assert isinstance(parsed, dict)

    def test_portfolio_risk_returns_json(self):
        mock_data = {"var_95": -0.02, "cvar": -0.03}
        with patch("jpstock_agent.server.portfolio.portfolio_risk", return_value=mock_data):
            result = server.portfolio_risk("7203,6758")
            parsed = json.loads(result)
            assert "var_95" in parsed

    def test_portfolio_correlation_returns_json(self):
        mock_data = {"correlation_matrix": {}, "most_correlated": []}
        with patch("jpstock_agent.server.portfolio.portfolio_correlation", return_value=mock_data):
            result = server.portfolio_correlation("7203,6758")
            parsed = json.loads(result)
            assert isinstance(parsed, dict)


class TestSentimentTools:
    """Test sentiment server tools."""

    def test_sentiment_news_returns_json(self):
        mock_data = {"score": 0.5, "label": "positive"}
        with patch("jpstock_agent.server.sentiment.sentiment_news", return_value=mock_data):
            result = server.sentiment_news("7203")
            parsed = json.loads(result)
            assert "score" in parsed

    def test_sentiment_market_returns_json(self):
        mock_data = [{"symbol": "7203", "score": 0.3}]
        with patch("jpstock_agent.server.sentiment.sentiment_market", return_value=mock_data):
            result = server.sentiment_market("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_sentiment_combined_returns_json(self):
        mock_data = {"ta_score": 50, "sentiment_score": 0.3, "combined": "BUY"}
        with patch("jpstock_agent.server.sentiment.sentiment_combined", return_value=mock_data):
            result = server.sentiment_combined("7203")
            parsed = json.loads(result)
            assert "combined" in parsed

    def test_sentiment_screen_returns_json(self):
        mock_data = [{"symbol": "7203", "score": 0.5}]
        with patch("jpstock_agent.server.sentiment.sentiment_screen", return_value=mock_data):
            result = server.sentiment_screen("7203")
            parsed = json.loads(result)
            assert isinstance(parsed, list)


class TestListingTools:
    """Test listing/market server tools."""

    def test_listing_all_symbols_returns_json(self):
        mock_data = [{"code": "7203", "name": "Toyota"}]
        with patch("jpstock_agent.server.core.listing_all_symbols", return_value=mock_data):
            result = server.listing_all_symbols()
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_listing_sectors_returns_json(self):
        mock_data = [{"sector": "Automobile"}]
        with patch("jpstock_agent.server.core.listing_sectors", return_value=mock_data):
            result = server.listing_sectors()
            parsed = json.loads(result)
            assert isinstance(parsed, list)


class TestMarketDataTools:
    """Test forex, crypto, world index tools."""

    def test_fx_history_returns_json(self):
        mock_data = [{"date": "2026-01-01", "close": 150.0}]
        with patch("jpstock_agent.server.core.fx_history", return_value=mock_data):
            result = server.fx_history("USDJPY=X")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_crypto_history_returns_json(self):
        mock_data = [{"date": "2026-01-01", "close": 50000.0}]
        with patch("jpstock_agent.server.core.crypto_history", return_value=mock_data):
            result = server.crypto_history("BTC-USD")
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_world_index_history_returns_json(self):
        mock_data = [{"date": "2026-01-01", "close": 30000.0}]
        with patch("jpstock_agent.server.core.world_index_history", return_value=mock_data):
            result = server.world_index_history("^N225")
            parsed = json.loads(result)
            assert isinstance(parsed, list)
