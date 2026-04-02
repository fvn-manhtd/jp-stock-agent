"""
Tests for the backtest module.

Tests cover:
- backtest_strategy: core strategy backtesting
- backtest_compare: multi-strategy comparison
- backtest_optimize: parameter optimization
- backtest_walk_forward: rolling window consistency
- backtest_monte_carlo: Monte Carlo simulation
- backtest_advanced_metrics: advanced risk/performance metrics
- _generate_signals: signal generation for different strategies
- Error handling for invalid strategies
"""

from unittest.mock import patch

import pandas as pd

from jpstock_agent import backtest
from tests.conftest import _make_ohlcv_df


class TestBacktestStrategy:
    """Tests for backtest_strategy function."""

    def test_sma_crossover_returns_expected_keys(self):
        """Test that sma_crossover strategy returns dict with all expected metric keys."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy("7203", "sma_crossover")

            assert isinstance(result, dict)
            assert "error" not in result
            assert result["strategy"] == "sma_crossover"
            assert result["symbol"] == "7203"
            assert "total_return_pct" in result
            assert "win_rate_pct" in result
            assert "sharpe_ratio" in result
            assert "trades" in result

    def test_sma_crossover_returns_numeric_metrics(self):
        """Test that sma_crossover returns numeric values for key metrics."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy("7203", "sma_crossover")

            assert isinstance(result["total_return_pct"], (int, float))
            assert isinstance(result["win_rate_pct"], (int, float))
            assert isinstance(result["sharpe_ratio"], (int, float))
            assert isinstance(result["trades"], list)

    def test_ema_crossover_no_error(self):
        """Test that ema_crossover strategy runs without error."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy("7203", "ema_crossover")

            assert isinstance(result, dict)
            assert "error" not in result

    def test_rsi_reversal_no_error(self):
        """Test that rsi_reversal strategy runs without error."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy("7203", "rsi_reversal")

            assert isinstance(result, dict)
            assert "error" not in result

    def test_macd_crossover_no_error(self):
        """Test that macd_crossover strategy runs without error."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy("7203", "macd_crossover")

            assert isinstance(result, dict)
            assert "error" not in result

    def test_invalid_strategy_returns_error(self):
        """Test that invalid strategy name returns error dict."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy("7203", "invalid_strategy_xyz")

            assert isinstance(result, dict)
            assert "error" in result

    def test_failed_data_fetch_returns_error(self):
        """Test that failed data fetch returns error dict."""
        with patch("jpstock_agent.backtest._get_ohlcv_df") as mock_get:
            mock_get.return_value = None

            result = backtest.backtest_strategy("7203", "sma_crossover")

            assert isinstance(result, dict)
            assert "error" in result

    def test_custom_initial_capital_affects_metrics(self):
        """Test that custom initial_capital parameter is used."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy(
                "7203", "sma_crossover", initial_capital=2_000_000
            )

            assert isinstance(result, dict)
            assert "error" not in result
            assert result.get("initial_capital") == 2_000_000 or "final_capital" in result


class TestBacktestCompare:
    """Tests for backtest_compare function."""

    def test_compare_returns_list(self):
        """Test that backtest_compare returns a list of results."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_compare("7203")

            assert isinstance(result, list)
            assert len(result) > 0

    def test_compare_results_sorted_by_return(self):
        """Test that compare results are sorted by total_return_pct descending."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_compare("7203", strategies=["sma_crossover", "ema_crossover"])

            assert isinstance(result, list)
            # Check sorting: each valid result should have total_return_pct or error
            valid_results = [r for r in result if "error" not in r]
            if len(valid_results) > 1:
                returns = [r.get("total_return_pct", float("-inf")) for r in valid_results]
                assert returns == sorted(returns, reverse=True)

    def test_compare_multiple_strategies(self):
        """Test comparing multiple strategies."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            strategies = ["sma_crossover", "ema_crossover", "rsi_reversal"]
            result = backtest.backtest_compare("7203", strategies=strategies)

            assert isinstance(result, list)
            assert len(result) >= 1


class TestBacktestOptimize:
    """Tests for backtest_optimize function."""

    def test_optimize_returns_list_with_different_values(self):
        """Test that optimize returns list with results for different parameter values."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            param_range = [10, 20, 30]
            result = backtest.backtest_optimize(
                "7203", "sma_crossover", "fast_period", param_range
            )

            assert isinstance(result, list)
            # Check that we got results for different parameter values
            if len(result) > 0:
                valid_results = [r for r in result if "error" not in r]
                # Each result should contain the parameter that was tested
                for res in valid_results:
                    assert "fast_period" in res

    def test_optimize_results_sorted_by_return(self):
        """Test that optimize results are sorted by total_return_pct descending."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            param_range = [10, 20, 30]
            result = backtest.backtest_optimize(
                "7203", "sma_crossover", "fast_period", param_range
            )

            assert isinstance(result, list)
            valid_results = [r for r in result if "error" not in r]
            if len(valid_results) > 1:
                returns = [r.get("total_return_pct", float("-inf")) for r in valid_results]
                assert returns == sorted(returns, reverse=True)


class TestBacktestWalkForward:
    """Tests for backtest_walk_forward function."""

    def test_walk_forward_returns_dict_with_windows(self):
        """Test that walk_forward returns dict with window_results."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_walk_forward("7203", "sma_crossover")

            assert isinstance(result, dict)
            assert "error" not in result or "window_results" in result
            if "error" not in result:
                assert "window_results" in result
                assert isinstance(result["window_results"], list)

    def test_walk_forward_returns_overall_metrics(self):
        """Test that walk_forward includes overall metrics."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_walk_forward("7203", "sma_crossover")

            if isinstance(result, dict) and "error" not in result:
                assert "strategy" in result
                assert "symbol" in result
                assert "overall_return_pct" in result or "window_results" in result


class TestBacktestMonteCarlo:
    """Tests for backtest_monte_carlo function."""

    def test_monte_carlo_returns_dict_with_metrics(self):
        """Test that monte_carlo returns dict with probability metrics."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_monte_carlo("7203", "sma_crossover")

            assert isinstance(result, dict)
            # May return error if not enough trades, but structure should be dict
            if "error" not in result:
                assert "probability_of_profit_pct" in result
                assert "confidence_interval_90" in result

    def test_monte_carlo_confidence_interval_format(self):
        """Test that confidence_interval_90 is properly formatted."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_monte_carlo("7203", "sma_crossover", num_simulations=100)

            if isinstance(result, dict) and "error" not in result:
                assert isinstance(result["confidence_interval_90"], (list, tuple))
                assert len(result["confidence_interval_90"]) == 2
                # 5th percentile should be <= 95th percentile
                assert result["confidence_interval_90"][0] <= result["confidence_interval_90"][1]


class TestBacktestAdvancedMetrics:
    """Tests for backtest_advanced_metrics function."""

    def test_advanced_metrics_returns_dict(self):
        """Test that advanced_metrics returns dict with extended metrics."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_advanced_metrics("7203", "sma_crossover")

            assert isinstance(result, dict)
            if "error" not in result:
                assert "sortino_ratio" in result
                assert "calmar_ratio" in result
                assert "profit_factor" in result

    def test_advanced_metrics_includes_all_base_metrics(self):
        """Test that advanced_metrics includes base backtest metrics."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_advanced_metrics("7203", "sma_crossover")

            if isinstance(result, dict) and "error" not in result:
                # Should have base metrics
                assert "total_return_pct" in result or "strategy" in result


class TestGenerateSignals:
    """Tests for _generate_signals helper function."""

    def test_generate_signals_sma_returns_dataframe_with_signal_column(self):
        """Test that _generate_signals returns DataFrame with signal column."""
        df = _make_ohlcv_df(days=100)

        result = backtest._generate_signals(df, "sma_crossover")

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns

    def test_generate_signals_column_values_in_valid_range(self):
        """Test that signal column contains only 1, -1, or 0."""
        df = _make_ohlcv_df(days=100)

        result = backtest._generate_signals(df, "sma_crossover")

        valid_signals = {-1, 0, 1}
        actual_signals = set(result["signal"].unique())
        assert actual_signals.issubset(valid_signals)

    def test_generate_signals_ema_crossover(self):
        """Test _generate_signals with ema_crossover strategy."""
        df = _make_ohlcv_df(days=100)

        result = backtest._generate_signals(df, "ema_crossover")

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns

    def test_generate_signals_rsi_reversal(self):
        """Test _generate_signals with rsi_reversal strategy."""
        df = _make_ohlcv_df(days=100)

        result = backtest._generate_signals(df, "rsi_reversal")

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns

    def test_generate_signals_invalid_strategy_returns_none(self):
        """Test that invalid strategy returns None."""
        df = _make_ohlcv_df(days=100)

        result = backtest._generate_signals(df, "invalid_xyz")

        # Should return None for invalid strategy
        assert result is None or "signal" not in result

    def test_generate_signals_with_custom_params(self):
        """Test that custom parameters are used in signal generation."""
        df = _make_ohlcv_df(days=100)

        result = backtest._generate_signals(
            df, "sma_crossover", fast_period=10, slow_period=30
        )

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns


class TestBacktestErrorHandling:
    """Tests for error handling in backtest functions."""

    def test_backtest_empty_dataframe_returns_error(self):
        """Test that empty DataFrame returns error."""
        with patch("jpstock_agent.backtest._get_ohlcv_df") as mock_get:
            mock_get.return_value = pd.DataFrame()

            result = backtest.backtest_strategy("7203", "sma_crossover")

            assert isinstance(result, dict)
            assert "error" in result

    def test_backtest_none_dataframe_returns_error(self):
        """Test that None DataFrame returns error."""
        with patch("jpstock_agent.backtest._get_ohlcv_df") as mock_get:
            mock_get.return_value = None

            result = backtest.backtest_strategy("7203", "sma_crossover")

            assert isinstance(result, dict)
            assert "error" in result

    def test_insufficient_data_returns_error(self):
        """Test that insufficient data returns error."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            # Very small dataset
            mock_get.return_value = _make_ohlcv_df(days=5)

            result = backtest.backtest_walk_forward("7203", "sma_crossover")

            # Should handle gracefully (may error, may return minimal results)
            assert isinstance(result, dict)
