"""
Tests for the options & derivatives module (options.py).

Tests cover:
- Black-Scholes Greeks calculations (_black_scholes_greeks)
- Math helpers (_norm_cdf, _norm_pdf)
- options_chain: options chain fetching
- options_greeks: Greeks calculation via yfinance
- options_iv_surface: IV surface construction
- options_unusual_activity: unusual activity detection
- options_put_call_ratio: P/C ratio calculation
- options_max_pain: max pain calculation
- Error handling for missing options data
"""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from jpstock_agent import options
from jpstock_agent.options import _black_scholes_greeks, _norm_cdf, _norm_pdf


# ============================================================================
# Math Helper Tests
# ============================================================================


class TestNormHelpers:
    """Tests for normal distribution helpers."""

    def test_norm_cdf_at_zero(self):
        """Test that CDF at 0 is 0.5."""
        assert abs(_norm_cdf(0) - 0.5) < 1e-10

    def test_norm_cdf_at_large_positive(self):
        """Test that CDF at large positive is ~1."""
        assert abs(_norm_cdf(5) - 1.0) < 1e-6

    def test_norm_cdf_at_large_negative(self):
        """Test that CDF at large negative is ~0."""
        assert abs(_norm_cdf(-5) - 0.0) < 1e-6

    def test_norm_cdf_symmetry(self):
        """Test CDF(x) + CDF(-x) = 1."""
        for x in [0.5, 1.0, 2.0]:
            assert abs(_norm_cdf(x) + _norm_cdf(-x) - 1.0) < 1e-10

    def test_norm_pdf_at_zero(self):
        """Test that PDF at 0 is ~0.3989."""
        assert abs(_norm_pdf(0) - 0.3989422804014327) < 1e-6

    def test_norm_pdf_positive(self):
        """Test that PDF is always positive."""
        for x in [-3, -1, 0, 1, 3]:
            assert _norm_pdf(x) > 0

    def test_norm_pdf_symmetry(self):
        """Test that PDF is symmetric."""
        for x in [0.5, 1.0, 2.0]:
            assert abs(_norm_pdf(x) - _norm_pdf(-x)) < 1e-10


# ============================================================================
# Black-Scholes Greeks Tests
# ============================================================================


class TestBlackScholesGreeks:
    """Tests for Black-Scholes Greeks calculation."""

    def test_call_greeks_returns_all_fields(self):
        """Test that call Greeks include all required fields."""
        greeks = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks
        assert "theoretical_price" in greeks

    def test_put_greeks_returns_all_fields(self):
        """Test that put Greeks include all required fields."""
        greeks = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="put")
        assert "delta" in greeks
        assert "theoretical_price" in greeks

    def test_call_delta_between_0_and_1(self):
        """Test that call delta is between 0 and 1."""
        greeks = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        assert 0 <= greeks["delta"] <= 1

    def test_put_delta_between_neg1_and_0(self):
        """Test that put delta is between -1 and 0."""
        greeks = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="put")
        assert -1 <= greeks["delta"] <= 0

    def test_atm_call_delta_near_05(self):
        """Test that ATM call delta is approximately 0.5."""
        greeks = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        assert abs(greeks["delta"] - 0.5) < 0.15

    def test_gamma_always_positive(self):
        """Test that gamma is always positive."""
        for opt_type in ["call", "put"]:
            greeks = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type=opt_type)
            assert greeks["gamma"] >= 0

    def test_call_gamma_equals_put_gamma(self):
        """Test that call and put gamma are equal (same strike/expiry)."""
        call = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        put = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="put")
        assert abs(call["gamma"] - put["gamma"]) < 1e-6

    def test_call_vega_equals_put_vega(self):
        """Test that call and put vega are equal."""
        call = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        put = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="put")
        assert abs(call["vega"] - put["vega"]) < 1e-6

    def test_theta_negative_for_long_options(self):
        """Test that theta is negative (time decay)."""
        greeks = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        assert greeks["theta"] < 0

    def test_deep_itm_call_delta_near_1(self):
        """Test that deep ITM call has delta near 1."""
        greeks = _black_scholes_greeks(S=150, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        assert greeks["delta"] > 0.9

    def test_deep_otm_call_delta_near_0(self):
        """Test that deep OTM call has delta near 0."""
        greeks = _black_scholes_greeks(S=50, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        assert greeks["delta"] < 0.1

    def test_zero_time_returns_zeros(self):
        """Test that T=0 returns zero Greeks."""
        greeks = _black_scholes_greeks(S=100, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert greeks["delta"] == 0.0

    def test_zero_vol_returns_zeros(self):
        """Test that sigma=0 returns zero Greeks."""
        greeks = _black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0, option_type="call")
        assert greeks["delta"] == 0.0

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)."""
        import math
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
        call = _black_scholes_greeks(S, K, T, r, sigma, "call")
        put = _black_scholes_greeks(S, K, T, r, sigma, "put")

        lhs = call["theoretical_price"] - put["theoretical_price"]
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 0.1


# ============================================================================
# Options Chain Tests (with mocked yfinance)
# ============================================================================

def _mock_option_chain():
    """Create a mock yfinance option chain."""
    calls = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0, 110.0],
        "lastPrice": [7.0, 3.5, 1.2, 0.3],
        "bid": [6.8, 3.3, 1.0, 0.2],
        "ask": [7.2, 3.7, 1.4, 0.4],
        "volume": [1000, 5000, 3000, 500],
        "openInterest": [5000, 10000, 8000, 2000],
        "impliedVolatility": [0.25, 0.22, 0.23, 0.28],
        "inTheMoney": [True, True, False, False],
    })
    puts = pd.DataFrame({
        "strike": [90.0, 95.0, 100.0, 105.0],
        "lastPrice": [0.2, 0.8, 3.0, 6.5],
        "bid": [0.1, 0.7, 2.8, 6.3],
        "ask": [0.3, 0.9, 3.2, 6.7],
        "volume": [200, 800, 4000, 2000],
        "openInterest": [1000, 3000, 9000, 4000],
        "impliedVolatility": [0.30, 0.26, 0.22, 0.20],
        "inTheMoney": [False, False, True, True],
    })
    mock_chain = MagicMock()
    mock_chain.calls = calls
    mock_chain.puts = puts
    return mock_chain


def _make_mock_ticker(price=100.0, expiries=("2026-05-15", "2026-06-19", "2026-07-17")):
    """Create a mock yfinance Ticker with options support."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"currentPrice": price}
    mock_ticker.options = list(expiries)
    mock_ticker.option_chain.return_value = _mock_option_chain()
    return mock_ticker


class TestOptionsChain:
    """Tests for options_chain function."""

    def test_returns_dict_with_calls_and_puts(self):
        """Test that options chain returns calls and puts."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_chain("AAPL")

            assert isinstance(result, dict)
            assert "error" not in result
            assert "calls" in result
            assert "puts" in result
            assert isinstance(result["calls"], list)
            assert isinstance(result["puts"], list)

    def test_summary_section(self):
        """Test that summary includes volume and OI totals."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_chain("AAPL")

            assert "summary" in result
            summary = result["summary"]
            assert "total_calls" in summary
            assert "total_puts" in summary
            assert "put_call_volume_ratio" in summary

    def test_call_option_fields(self):
        """Test that each call option has required fields."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_chain("AAPL")

            calls = result["calls"]
            assert len(calls) > 0
            first = calls[0]
            assert "strike" in first
            assert "volume" in first
            assert "open_interest" in first
            assert "implied_volatility" in first

    def test_expiry_selection(self):
        """Test that nearest expiry is used when none specified."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_chain("AAPL")

            assert result["expiry"] == "2026-05-15"

    def test_available_expiries(self):
        """Test that all available expiries are listed."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_chain("AAPL")

            assert "available_expiries" in result
            assert len(result["available_expiries"]) == 3


class TestOptionsGreeks:
    """Tests for options_greeks function."""

    def test_returns_options_with_greeks(self):
        """Test that Greeks are calculated for each option."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_greeks("AAPL")

            assert isinstance(result, dict)
            assert "error" not in result
            assert "options" in result
            assert len(result["options"]) > 0

            first = result["options"][0]
            assert "delta" in first
            assert "gamma" in first
            assert "theta" in first
            assert "vega" in first

    def test_atm_greeks_present(self):
        """Test that ATM Greeks are identified."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_greeks("AAPL")

            assert "atm_greeks" in result
            if result["atm_greeks"]:
                assert "strike" in result["atm_greeks"]
                assert "delta" in result["atm_greeks"]


class TestOptionsIvSurface:
    """Tests for options_iv_surface function."""

    def test_returns_surface_data(self):
        """Test that IV surface returns surface data points."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_iv_surface("AAPL", max_expiries=3)

            assert isinstance(result, dict)
            assert "error" not in result
            assert "surface" in result
            assert "term_structure" in result
            assert "skew_summary" in result

    def test_surface_point_fields(self):
        """Test that each surface point has required fields."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_iv_surface("AAPL", max_expiries=1)

            if result.get("surface"):
                point = result["surface"][0]
                assert "expiry" in point
                assert "strike" in point
                assert "iv_pct" in point
                assert "moneyness" in point


class TestOptionsUnusualActivity:
    """Tests for options_unusual_activity function."""

    def test_returns_unusual_options(self):
        """Test unusual activity detection."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_unusual_activity("AAPL", volume_threshold=0.1)

            assert isinstance(result, dict)
            assert "error" not in result
            assert "unusual_calls" in result
            assert "unusual_puts" in result
            assert "alert_level" in result

    def test_alert_level_values(self):
        """Test that alert level is valid."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_unusual_activity("AAPL")

            assert result["alert_level"] in ["HIGH", "MODERATE", "LOW"]


class TestOptionsPutCallRatio:
    """Tests for options_put_call_ratio function."""

    def test_returns_ratio_and_sentiment(self):
        """Test that P/C ratio returns sentiment."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_put_call_ratio("AAPL")

            assert isinstance(result, dict)
            assert "error" not in result
            assert "volume_put_call_ratio" in result
            assert "sentiment" in result
            assert result["sentiment"] in ["BEARISH", "NEUTRAL", "BULLISH"]

    def test_per_expiry_breakdown(self):
        """Test that per-expiry breakdown is included."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_put_call_ratio("AAPL")

            assert "per_expiry" in result
            assert isinstance(result["per_expiry"], list)


class TestOptionsMaxPain:
    """Tests for options_max_pain function."""

    def test_returns_max_pain_strike(self):
        """Test that max pain calculation returns a strike."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_max_pain("AAPL")

            assert isinstance(result, dict)
            assert "error" not in result
            assert "max_pain_strike" in result
            assert result["max_pain_strike"] > 0

    def test_pain_by_strike_present(self):
        """Test that pain by strike breakdown is included."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_max_pain("AAPL")

            assert "pain_by_strike" in result
            assert isinstance(result["pain_by_strike"], list)

    def test_distance_from_max_pain(self):
        """Test that distance percentage is calculated."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_gt.return_value = _make_mock_ticker()

            result = options.options_max_pain("AAPL")

            assert "distance_from_max_pain_pct" in result


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestOptionsErrorHandling:
    """Tests for error handling in options module."""

    def test_no_options_data_returns_error(self):
        """Test error when no options data available."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_ticker = MagicMock()
            mock_ticker.options = []
            mock_ticker.info = {"currentPrice": 100.0}
            mock_gt.return_value = mock_ticker

            result = options.options_chain("7203")

            assert isinstance(result, dict)
            assert "error" in result

    def test_ticker_error_handled(self):
        """Test that yfinance errors are handled gracefully."""
        with patch("jpstock_agent.options._get_ticker") as mock_gt:
            mock_ticker = MagicMock()
            type(mock_ticker).options = PropertyMock(side_effect=Exception("No data"))
            mock_ticker.info = {"currentPrice": 100.0}
            mock_gt.return_value = mock_ticker

            result = options.options_chain("INVALID")

            assert isinstance(result, dict)
            assert "error" in result
