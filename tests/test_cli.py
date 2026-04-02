"""Tests for cli.py – Click CLI commands.

Tests the CLI output formatting and command invocations using Click's test runner.
"""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from jpstock_agent.cli import _format_output, cli


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# _format_output tests
# ---------------------------------------------------------------------------


class TestFormatOutput:
    """Test the _format_output helper function."""

    def test_json_format_dict(self):
        data = {"key": "value"}
        result = _format_output(data, "json")
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_json_format_list(self):
        data = [{"a": 1}, {"a": 2}]
        result = _format_output(data, "json")
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_table_format_error_dict(self):
        data = {"error": "Something went wrong"}
        result = _format_output(data, "table")
        assert "Error:" in result
        assert "Something went wrong" in result

    def test_table_format_message_dict(self):
        data = {"message": "Operation completed"}
        result = _format_output(data, "table")
        assert result == "Operation completed"

    def test_table_format_regular_dict(self):
        data = {"name": "Toyota", "sector": "Auto"}
        result = _format_output(data, "table")
        assert "name" in result or "Field" in result
        assert "Toyota" in result

    def test_table_format_list_of_dicts(self):
        data = [{"symbol": "7203", "price": 2500}]
        result = _format_output(data, "table")
        assert "7203" in result

    def test_table_format_empty_list(self):
        result = _format_output([], "table")
        assert result == "No data returned."

    def test_table_format_plain_list(self):
        data = ["a", "b", "c"]
        result = _format_output(data, "table")
        assert "a" in result

    def test_format_non_list_non_dict(self):
        result = _format_output("plain string", "table")
        assert result == "plain string"

    def test_format_integer(self):
        result = _format_output(42, "table")
        assert "42" in result


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestCLIGroup:
    """Test the main CLI group."""

    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "JPStock Agent" in result.output

    def test_cli_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestHistoryCommand:
    """Test the history CLI command."""

    def test_history_table_output(self, runner):
        mock_data = [{"date": "2026-01-01", "close": 2500.0, "volume": 1000000}]
        with patch("jpstock_agent.cli.core.stock_history", return_value=mock_data):
            result = runner.invoke(cli, ["history", "7203"])
            assert result.exit_code == 0
            assert "2500" in result.output

    def test_history_json_output(self, runner):
        mock_data = [{"date": "2026-01-01", "close": 2500.0}]
        with patch("jpstock_agent.cli.core.stock_history", return_value=mock_data):
            result = runner.invoke(cli, ["history", "7203", "-f", "json"])
            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert isinstance(parsed, list)

    def test_history_with_source(self, runner):
        mock_data = [{"date": "2026-01-01", "close": 2500.0}]
        with patch("jpstock_agent.cli.core.stock_history", return_value=mock_data):
            result = runner.invoke(cli, ["history", "7203", "-s", "yfinance"])
            assert result.exit_code == 0

    def test_history_error_output(self, runner):
        mock_data = {"error": "Symbol not found"}
        with patch("jpstock_agent.cli.core.stock_history", return_value=mock_data):
            result = runner.invoke(cli, ["history", "XXXX"])
            assert result.exit_code == 0
            assert "Error" in result.output


class TestOverviewCommand:
    """Test the overview CLI command."""

    def test_overview_output(self, runner):
        mock_data = {"name": "Toyota", "market_cap": 30000000000}
        with patch("jpstock_agent.cli.core.company_overview", return_value=mock_data):
            result = runner.invoke(cli, ["overview", "7203"])
            assert result.exit_code == 0
            assert "Toyota" in result.output


class TestTACommands:
    """Test TA CLI commands."""

    def test_sma_command(self, runner):
        mock_data = [{"date": "2026-01-01", "sma_20": 2500.0}]
        with patch("jpstock_agent.cli.ta.ta_sma", return_value=mock_data):
            result = runner.invoke(cli, ["ta-sma", "7203"])
            assert result.exit_code == 0

    def test_rsi_command(self, runner):
        mock_data = [{"date": "2026-01-01", "rsi": 55.0}]
        with patch("jpstock_agent.cli.ta.ta_rsi", return_value=mock_data):
            result = runner.invoke(cli, ["ta-rsi", "7203"])
            assert result.exit_code == 0

    def test_macd_command(self, runner):
        mock_data = [{"date": "2026-01-01", "macd": 10.0}]
        with patch("jpstock_agent.cli.ta.ta_macd", return_value=mock_data):
            result = runner.invoke(cli, ["ta-macd", "7203"])
            assert result.exit_code == 0


class TestBacktestCommands:
    """Test backtest CLI commands."""

    def test_backtest_command(self, runner):
        mock_data = {"total_return": 15.5, "sharpe_ratio": 1.2}
        with patch("jpstock_agent.cli.backtest.backtest_strategy", return_value=mock_data):
            result = runner.invoke(cli, ["backtest", "7203", "--strategy", "sma_crossover"])
            assert result.exit_code == 0
            assert "15.5" in result.output

    def test_backtest_compare_command(self, runner):
        mock_data = [{"strategy": "sma", "return": 10}]
        with patch("jpstock_agent.cli.backtest.backtest_compare", return_value=mock_data):
            result = runner.invoke(cli, ["backtest-compare", "7203"])
            assert result.exit_code == 0


class TestPortfolioCommands:
    """Test portfolio CLI commands."""

    def test_portfolio_command(self, runner):
        mock_data = {"stocks": {"7203": {"return": 10}}}
        with patch("jpstock_agent.cli.portfolio.portfolio_analyze", return_value=mock_data):
            result = runner.invoke(cli, ["portfolio", "7203,6758"])
            assert result.exit_code == 0

    def test_portfolio_optimize_command(self, runner):
        mock_data = {"max_sharpe": {"weights": {"7203": 0.6}}}
        with patch("jpstock_agent.cli.portfolio.portfolio_optimize", return_value=mock_data):
            result = runner.invoke(cli, ["portfolio-optimize", "7203,6758"])
            assert result.exit_code == 0


class TestSentimentCommands:
    """Test sentiment CLI commands."""

    def test_sentiment_command(self, runner):
        mock_data = {"score": 0.5, "label": "positive", "articles_analyzed": 10}
        with patch("jpstock_agent.cli.sentiment.sentiment_news", return_value=mock_data):
            result = runner.invoke(cli, ["sentiment", "7203"])
            assert result.exit_code == 0

    def test_sentiment_market_command(self, runner):
        mock_data = [{"symbol": "7203", "score": 0.3}]
        with patch("jpstock_agent.cli.sentiment.sentiment_market", return_value=mock_data):
            result = runner.invoke(cli, ["sentiment-market", "7203,6758"])
            assert result.exit_code == 0


class TestCandlestickCommands:
    """Test candlestick CLI commands."""

    def test_candle_scan_command(self, runner):
        mock_data = [{"pattern": "doji", "date": "2026-01-01"}]
        with patch("jpstock_agent.cli.candlestick.ta_candlestick_scan", return_value=mock_data):
            result = runner.invoke(cli, ["ta-candle-scan", "7203"])
            assert result.exit_code == 0

    def test_candle_latest_command(self, runner):
        mock_data = {"date": "2026-01-01", "patterns": []}
        with patch("jpstock_agent.cli.candlestick.ta_candlestick_latest", return_value=mock_data):
            result = runner.invoke(cli, ["ta-candle-latest", "7203"])
            assert result.exit_code == 0
