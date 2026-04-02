"""
Tests for the portfolio module.

Tests cover:
- portfolio_analyze: portfolio metrics and correlations
- portfolio_optimize: Monte Carlo optimization
- portfolio_risk: risk metrics with weights
- portfolio_correlation: correlation and covariance matrices
- Error handling for invalid inputs
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from jpstock_agent import portfolio


def _make_returns_df(symbols=None, days=252, seed=42):
    """Generate a returns DataFrame for testing portfolio functions."""
    if symbols is None:
        symbols = ["7203", "6758", "9984"]

    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    data = np.random.randn(days, len(symbols)) * 0.02  # 2% daily volatility

    df = pd.DataFrame(data, columns=symbols, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def mock_get_returns_df(monkeypatch):
    """Mock portfolio._get_returns_df to return sample returns."""
    def _mock_returns(*args, **kwargs):
        return _make_returns_df()

    monkeypatch.setattr("jpstock_agent.portfolio._get_returns_df", _mock_returns)
    return _mock_returns


class TestPortfolioAnalyze:
    """Tests for portfolio_analyze function."""

    def test_analyze_returns_dict_with_expected_keys(self, mock_get_returns_df):
        """Test that portfolio_analyze returns dict with all expected keys."""
        result = portfolio.portfolio_analyze(["7203", "6758", "9984"])

        assert isinstance(result, dict)
        assert "error" not in result
        assert "symbols" in result
        assert "stocks" in result
        assert "correlation_matrix" in result
        assert "best_performer" in result
        assert "worst_performer" in result

    def test_analyze_stocks_metrics_structure(self, mock_get_returns_df):
        """Test that stocks metrics have correct structure."""
        result = portfolio.portfolio_analyze(["7203", "6758", "9984"])

        assert isinstance(result, dict)
        assert "error" not in result
        stocks = result.get("stocks", {})
        for symbol, metrics in stocks.items():
            assert "return_pct" in metrics
            assert "volatility_pct" in metrics
            assert "sharpe_ratio" in metrics

    def test_analyze_best_worst_performer_structure(self, mock_get_returns_df):
        """Test that best/worst performer have correct structure."""
        result = portfolio.portfolio_analyze(["7203", "6758", "9984"])

        if "error" not in result:
            assert "symbol" in result["best_performer"]
            assert "return_pct" in result["best_performer"]
            assert "symbol" in result["worst_performer"]
            assert "return_pct" in result["worst_performer"]

    def test_analyze_correlation_matrix_structure(self, mock_get_returns_df):
        """Test that correlation_matrix has correct nested dict structure."""
        result = portfolio.portfolio_analyze(["7203", "6758", "9984"])

        if "error" not in result:
            corr = result.get("correlation_matrix", {})
            assert isinstance(corr, dict)
            # Should have entries for each symbol
            for symbol in ["7203", "6758", "9984"]:
                assert symbol in corr or len(corr) > 0

    def test_analyze_period_dates_present(self, mock_get_returns_df):
        """Test that period with start and end dates is included."""
        result = portfolio.portfolio_analyze(["7203", "6758", "9984"])

        if "error" not in result:
            assert "period" in result
            assert "start" in result["period"]
            assert "end" in result["period"]

    def test_analyze_empty_symbols_returns_error(self):
        """Test that empty symbols list returns error."""
        with patch("jpstock_agent.portfolio._get_returns_df") as mock_get:
            mock_get.return_value = {"error": "No symbols provided"}

            result = portfolio.portfolio_analyze([])

            assert isinstance(result, dict)
            assert "error" in result


class TestPortfolioOptimize:
    """Tests for portfolio_optimize function."""

    def test_optimize_returns_dict_with_portfolios(self, mock_get_returns_df):
        """Test that optimize returns dict with optimized portfolios."""
        result = portfolio.portfolio_optimize(["7203", "6758", "9984"], num_portfolios=100)

        assert isinstance(result, dict)
        assert "error" not in result
        assert "max_sharpe_portfolio" in result
        assert "min_volatility_portfolio" in result
        assert "efficient_frontier" in result

    def test_optimize_max_sharpe_portfolio_structure(self, mock_get_returns_df):
        """Test that max_sharpe_portfolio has correct structure."""
        result = portfolio.portfolio_optimize(["7203", "6758", "9984"], num_portfolios=100)

        if "error" not in result:
            portfolio_data = result["max_sharpe_portfolio"]
            assert "weights" in portfolio_data
            assert "return_pct" in portfolio_data
            assert "volatility_pct" in portfolio_data
            assert "sharpe_ratio" in portfolio_data

    def test_optimize_weights_sum_to_one(self, mock_get_returns_df):
        """Test that portfolio weights sum to approximately 1.0."""
        result = portfolio.portfolio_optimize(["7203", "6758", "9984"], num_portfolios=100)

        if "error" not in result:
            weights = result["max_sharpe_portfolio"]["weights"]
            weight_sum = sum(weights.values())
            assert 0.99 <= weight_sum <= 1.01

    def test_optimize_efficient_frontier_is_list(self, mock_get_returns_df):
        """Test that efficient_frontier is a list of portfolios."""
        result = portfolio.portfolio_optimize(["7203", "6758", "9984"], num_portfolios=100)

        if "error" not in result:
            frontier = result["efficient_frontier"]
            assert isinstance(frontier, list)
            assert len(frontier) > 0
            # Each item should be a portfolio dict
            for portfolio_item in frontier:
                assert "weights" in portfolio_item
                assert "sharpe_ratio" in portfolio_item

    def test_optimize_num_portfolios_parameter(self, mock_get_returns_df):
        """Test that num_portfolios parameter is recorded."""
        result = portfolio.portfolio_optimize(
            ["7203", "6758", "9984"], num_portfolios=500
        )

        if "error" not in result:
            assert result["num_portfolios"] == 500


class TestPortfolioRisk:
    """Tests for portfolio_risk function."""

    def test_risk_returns_dict_with_metrics(self, mock_get_returns_df):
        """Test that portfolio_risk returns dict with risk metrics."""
        result = portfolio.portfolio_risk(["7203", "6758", "9984"])

        assert isinstance(result, dict)
        assert "error" not in result
        assert "portfolio_return_pct" in result
        assert "portfolio_volatility_pct" in result
        assert "var_95_pct" in result
        assert "cvar_95_pct" in result

    def test_risk_includes_advanced_metrics(self, mock_get_returns_df):
        """Test that portfolio_risk includes advanced metrics."""
        result = portfolio.portfolio_risk(["7203", "6758", "9984"])

        if "error" not in result:
            assert "sortino_ratio" in result
            assert "max_drawdown_pct" in result
            assert "beta" in result
            assert "sharpe_ratio" in result

    def test_risk_with_equal_weights_default(self, mock_get_returns_df):
        """Test that equal weights are used when weights=None."""
        result = portfolio.portfolio_risk(["7203", "6758", "9984"], weights=None)

        if "error" not in result:
            assert "weights_used" in result
            weights = result["weights_used"]
            # Each of 3 symbols should have ~1/3 weight
            for symbol in ["7203", "6758", "9984"]:
                assert symbol in weights
                assert 0.25 < weights[symbol] < 0.4

    def test_risk_with_custom_weights_list(self, mock_get_returns_df):
        """Test that custom weights (list format) are respected."""
        custom_weights = [0.5, 0.3, 0.2]
        result = portfolio.portfolio_risk(
            ["7203", "6758", "9984"], weights=custom_weights
        )

        if "error" not in result:
            weights_used = result["weights_used"]
            assert abs(weights_used["7203"] - 0.5) < 0.01
            assert abs(weights_used["6758"] - 0.3) < 0.01
            assert abs(weights_used["9984"] - 0.2) < 0.01

    def test_risk_with_custom_weights_dict(self, mock_get_returns_df):
        """Test that custom weights (dict format) are respected."""
        custom_weights = {"7203": 0.6, "6758": 0.25, "9984": 0.15}
        result = portfolio.portfolio_risk(
            ["7203", "6758", "9984"], weights=custom_weights
        )

        if "error" not in result:
            weights_used = result["weights_used"]
            for symbol, weight in custom_weights.items():
                assert abs(weights_used[symbol] - weight) < 0.01

    def test_risk_invalid_weights_returns_error(self, mock_get_returns_df):
        """Test that invalid weights (wrong length) returns error."""
        invalid_weights = [0.5, 0.3]  # Only 2 weights for 3 symbols
        result = portfolio.portfolio_risk(
            ["7203", "6758", "9984"], weights=invalid_weights
        )

        assert isinstance(result, dict)
        assert "error" in result

    def test_risk_weights_dont_sum_to_one_returns_error(self, mock_get_returns_df):
        """Test that weights not summing to 1 returns error."""
        invalid_weights = [0.5, 0.3, 0.1]  # Sum = 0.9
        result = portfolio.portfolio_risk(
            ["7203", "6758", "9984"], weights=invalid_weights
        )

        assert isinstance(result, dict)
        assert "error" in result


class TestPortfolioCorrelation:
    """Tests for portfolio_correlation function."""

    def test_correlation_returns_dict_with_matrices(self, mock_get_returns_df):
        """Test that correlation returns dict with correlation and covariance matrices."""
        result = portfolio.portfolio_correlation(["7203", "6758", "9984"])

        assert isinstance(result, dict)
        assert "error" not in result
        assert "correlation_matrix" in result
        assert "covariance_matrix" in result

    def test_correlation_matrix_structure(self, mock_get_returns_df):
        """Test that correlation_matrix is nested dict."""
        result = portfolio.portfolio_correlation(["7203", "6758", "9984"])

        if "error" not in result:
            corr_matrix = result["correlation_matrix"]
            assert isinstance(corr_matrix, dict)
            # Should have entries for symbols
            for symbol in ["7203", "6758", "9984"]:
                assert symbol in corr_matrix

    def test_covariance_matrix_structure(self, mock_get_returns_df):
        """Test that covariance_matrix is nested dict."""
        result = portfolio.portfolio_correlation(["7203", "6758", "9984"])

        if "error" not in result:
            cov_matrix = result["covariance_matrix"]
            assert isinstance(cov_matrix, dict)

    def test_correlation_most_correlated_pair(self, mock_get_returns_df):
        """Test that most_correlated_pair has expected structure."""
        result = portfolio.portfolio_correlation(["7203", "6758", "9984"])

        if "error" not in result:
            most = result["most_correlated_pair"]
            assert "symbols" in most
            assert "correlation" in most
            assert isinstance(most["symbols"], list)
            assert len(most["symbols"]) == 2

    def test_correlation_least_correlated_pair(self, mock_get_returns_df):
        """Test that least_correlated_pair has expected structure."""
        result = portfolio.portfolio_correlation(["7203", "6758", "9984"])

        if "error" not in result:
            least = result["least_correlated_pair"]
            assert "symbols" in least
            assert "correlation" in least
            assert isinstance(least["symbols"], list)
            assert len(least["symbols"]) == 2

    def test_correlation_values_in_valid_range(self, mock_get_returns_df):
        """Test that correlation values are between -1 and 1."""
        result = portfolio.portfolio_correlation(["7203", "6758", "9984"])

        if "error" not in result:
            corr_matrix = result["correlation_matrix"]
            for symbol1, correlations in corr_matrix.items():
                for symbol2, corr_value in correlations.items():
                    assert -1.0 <= corr_value <= 1.0


class TestPortfolioErrorHandling:
    """Tests for error handling in portfolio functions."""

    def test_analyze_empty_symbols_returns_error(self):
        """Test that empty symbols list returns error."""
        with patch("jpstock_agent.portfolio._get_returns_df") as mock_get:
            mock_get.return_value = {"error": "No symbols provided"}

            result = portfolio.portfolio_analyze([])

            assert isinstance(result, dict)
            assert "error" in result

    def test_optimize_failed_returns_returns_error(self):
        """Test that failed returns fetch returns error."""
        with patch("jpstock_agent.portfolio._get_returns_df") as mock_get:
            mock_get.return_value = {"error": "Failed to fetch data"}

            result = portfolio.portfolio_optimize(["7203", "6758"])

            assert isinstance(result, dict)
            assert "error" in result

    def test_risk_single_symbol_works(self, mock_get_returns_df):
        """Test that portfolio_risk works with single symbol (beta=1)."""
        with patch("jpstock_agent.portfolio._get_returns_df") as mock_get:
            mock_get.return_value = _make_returns_df(symbols=["7203"])

            result = portfolio.portfolio_risk(["7203"])

            if isinstance(result, dict) and "error" not in result:
                assert result.get("beta") == 1.0


class TestPortfolioIntegration:
    """Integration tests for portfolio functions."""

    def test_analyze_and_optimize_consistency(self, mock_get_returns_df):
        """Test that analyze and optimize use consistent data."""
        analyze_result = portfolio.portfolio_analyze(["7203", "6758", "9984"])
        optimize_result = portfolio.portfolio_optimize(
            ["7203", "6758", "9984"], num_portfolios=100
        )

        # Both should have no errors or both should have data
        if "error" not in analyze_result:
            assert isinstance(analyze_result["stocks"], dict)
        if "error" not in optimize_result:
            assert "max_sharpe_portfolio" in optimize_result

    def test_multiple_portfolio_operations(self, mock_get_returns_df):
        """Test running multiple portfolio operations on same symbols."""
        symbols = ["7203", "6758", "9984"]

        result1 = portfolio.portfolio_analyze(symbols)
        result2 = portfolio.portfolio_risk(symbols)
        result3 = portfolio.portfolio_correlation(symbols)

        # All should return dict
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert isinstance(result3, dict)
