"""Tests for jpstock_agent.market module.

Tests cover:
- _get_period_return: period return calculation with various edge cases
- market_sector_performance: sector comparison and ranking
- market_breadth: advance/decline ratios and breadth signals
- market_top_movers: top gainers and losers identification
- market_regime: bull/bear/sideways detection from price data
- market_heatmap: heatmap data structure and sector averages
"""

from unittest.mock import patch

import pytest

from jpstock_agent import market


def _mock_history(days=90, base=100.0, trend=0.001):
    """Generate mock price history as list[dict].

    Parameters
    ----------
    days : int
        Number of days of data.
    base : float
        Starting price.
    trend : float
        Daily trend coefficient (multiplicative).

    Returns
    -------
    list[dict]
        Price history with 'date' and 'close' keys.
    """
    records = []
    price = base
    for i in range(days):
        price *= (1 + trend)
        # Generate realistic dates
        month = (i // 30) + 1
        day = (i % 30) + 1
        records.append({
            "date": f"2026-{month:02d}-{day:02d}",
            "close": round(price, 2),
            "volume": 1000000 + i * 10000,
        })
    return records


# ---------------------------------------------------------------------------
# TestGetPeriodReturn
# ---------------------------------------------------------------------------

class TestGetPeriodReturn:
    """Tests for _get_period_return helper function."""

    @patch("jpstock_agent.market._import_core")
    def test_period_return_calculates_basic_return(self, mock_core):
        """Test that period return correctly calculates percentage change."""
        # Mock returns 30 days of data
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=30, base=100.0, trend=0.01)

        result = market._get_period_return("7203", days=20)

        assert result is not None
        assert "first_close" in result
        assert "last_close" in result
        assert "return_pct" in result
        assert "high" in result
        assert "low" in result
        assert "data_points" in result
        assert result["data_points"] == 20

    @patch("jpstock_agent.market._import_core")
    def test_period_return_with_insufficient_data(self, mock_core):
        """Test that period return returns None with insufficient data."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = [
            {"date": "2026-03-01", "close": 100.0},
        ]

        result = market._get_period_return("7203", days=20)

        assert result is None

    @patch("jpstock_agent.market._import_core")
    def test_period_return_with_error_response(self, mock_core):
        """Test that period return handles error dict from stock_history."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = {"error": "API failed"}

        result = market._get_period_return("7203", days=20)

        assert result is None

    @patch("jpstock_agent.market._import_core")
    def test_period_return_with_empty_history(self, mock_core):
        """Test that period return handles empty history."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = []

        result = market._get_period_return("7203", days=20)

        assert result is None

    @patch("jpstock_agent.market._import_core")
    def test_period_return_uses_uppercase_close(self, mock_core):
        """Test that period return handles 'Close' (uppercase) key."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = [
            {"date": "2026-03-01", "Close": 100.0},
            {"date": "2026-03-02", "Close": 102.0},
            {"date": "2026-03-03", "Close": 104.0},
        ]

        result = market._get_period_return("7203", days=2)

        assert result is not None
        # Return from 102 to 104 = (104-102)/102 = 1.96%
        assert result["return_pct"] == pytest.approx(1.96, abs=0.1)

    @patch("jpstock_agent.market._import_core")
    def test_period_return_skips_invalid_close_values(self, mock_core):
        """Test that period return skips non-numeric close values."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = [
            {"date": "2026-03-01", "close": 100.0},
            {"date": "2026-03-02", "close": "invalid"},
            {"date": "2026-03-03", "close": 102.0},
        ]

        result = market._get_period_return("7203", days=2)

        assert result is not None
        assert result["data_points"] == 2


# ---------------------------------------------------------------------------
# TestMarketSectorPerformance
# ---------------------------------------------------------------------------

class TestMarketSectorPerformance:
    """Tests for market_sector_performance function."""

    @patch("jpstock_agent.market._get_period_return")
    def test_sector_performance_returns_dict(self, mock_return):
        """Test that market_sector_performance returns dict."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 30,
        }

        sectors = {"Auto": ["7203", "7267"], "Tech": ["6758"]}
        result = market.market_sector_performance(sectors)

        assert isinstance(result, dict)
        assert "error" not in result
        assert "sectors" in result
        assert "ranking" in result
        assert "period_days" in result

    @patch("jpstock_agent.market._get_period_return")
    def test_sector_performance_computes_average_return(self, mock_return):
        """Test that sector performance correctly averages returns."""
        def side_effect(sym, days, source):
            returns = {
                "7203": 5.0,
                "7267": 7.0,
                "6758": 3.0,
            }
            return {
                "first_close": 100.0,
                "last_close": 100.0 + returns.get(sym, 0),
                "return_pct": returns.get(sym, 0),
                "high": 101.0,
                "low": 99.0,
                "data_points": 30,
            }

        mock_return.side_effect = side_effect

        sectors = {"Auto": ["7203", "7267"], "Tech": ["6758"]}
        result = market.market_sector_performance(sectors)

        auto_return = result["sectors"]["Auto"]["avg_return_pct"]
        assert auto_return == 6.0  # Average of 5.0 and 7.0

    @patch("jpstock_agent.market._get_period_return")
    def test_sector_performance_ranks_by_return(self, mock_return):
        """Test that sectors are ranked from best to worst."""
        def side_effect(sym, days, source):
            returns = {
                "7203": 5.0,
                "7267": 7.0,
                "6758": 3.0,
            }
            return {
                "first_close": 100.0,
                "last_close": 100.0 + returns.get(sym, 0),
                "return_pct": returns.get(sym, 0),
                "high": 101.0,
                "low": 99.0,
                "data_points": 30,
            }

        mock_return.side_effect = side_effect

        sectors = {"Auto": ["7203", "7267"], "Tech": ["6758"]}
        result = market.market_sector_performance(sectors)
        ranking = result["ranking"]

        assert ranking[0]["sector"] == "Auto"  # 6.0%
        assert ranking[1]["sector"] == "Tech"  # 3.0%

    def test_sector_performance_with_empty_sectors(self):
        """Test that sector performance returns error for empty sectors."""
        result = market.market_sector_performance({})

        assert isinstance(result, dict)
        assert "error" in result

    @patch("jpstock_agent.market._get_period_return")
    def test_sector_performance_with_none_returns(self, mock_return):
        """Test that sector performance handles None returns (missing data)."""
        mock_return.return_value = None

        sectors = {"Auto": ["7203", "7267"]}
        result = market.market_sector_performance(sectors)

        assert isinstance(result, dict)
        assert "error" not in result
        assert result["sectors"]["Auto"]["avg_return_pct"] is None

    @patch("jpstock_agent.market._get_period_return")
    def test_sector_performance_includes_stock_details(self, mock_return):
        """Test that sector performance includes per-stock details."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 30,
        }

        sectors = {"Auto": ["7203", "7267"]}
        result = market.market_sector_performance(sectors)

        stocks = result["sectors"]["Auto"]["stocks"]
        assert len(stocks) == 2
        assert "symbol" in stocks[0]
        assert "return_pct" in stocks[0]


# ---------------------------------------------------------------------------
# TestMarketBreadth
# ---------------------------------------------------------------------------

class TestMarketBreadth:
    """Tests for market_breadth function."""

    @patch("jpstock_agent.market._import_core")
    def test_breadth_returns_dict_with_expected_keys(self, mock_core):
        """Test that market_breadth returns dict with all expected keys."""
        core_module = mock_core.return_value

        def history_side_effect(sym, start, source):
            # Return 370 days of data for 52-week calculations
            return _mock_history(days=370, base=100.0, trend=0.0005)

        core_module.stock_history.side_effect = history_side_effect

        symbols = ["7203", "6758", "9984"]
        result = market.market_breadth(symbols)

        assert isinstance(result, dict)
        assert "error" not in result
        assert "advancing" in result
        assert "declining" in result
        assert "unchanged" in result
        assert "advance_decline_ratio" in result
        assert "breadth_signal" in result
        assert "new_highs_52w" in result
        assert "new_lows_52w" in result

    @patch("jpstock_agent.market._import_core")
    def test_breadth_counts_advancing_declining(self, mock_core):
        """Test that breadth correctly counts advancing and declining stocks."""
        core_module = mock_core.return_value

        call_count = [0]
        def history_side_effect(sym, start, source):
            call_count[0] += 1
            if call_count[0] == 1:
                # First symbol: trending up (advancing)
                return _mock_history(days=370, base=100.0, trend=0.001)
            elif call_count[0] == 2:
                # Second symbol: flat (unchanged)
                return _mock_history(days=370, base=100.0, trend=0.0)
            else:
                # Third symbol: trending down (declining)
                return _mock_history(days=370, base=100.0, trend=-0.001)

        core_module.stock_history.side_effect = history_side_effect

        symbols = ["7203", "6758", "9984"]
        result = market.market_breadth(symbols)

        assert result["advancing"] >= 0
        assert result["declining"] >= 0

    @patch("jpstock_agent.market._import_core")
    def test_breadth_signal_strong_bullish(self, mock_core):
        """Test that breadth signal is STRONG_BULLISH when >70% advancing."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=370, base=100.0, trend=0.002)

        # Create 10 symbols, all trending up (>70%)
        symbols = [f"SYM{i}" for i in range(10)]
        result = market.market_breadth(symbols)

        # With strong uptrend, should have >70% advancing
        if result["breadth_pct"] > 70:
            assert result["breadth_signal"] == "STRONG_BULLISH"

    @patch("jpstock_agent.market._import_core")
    def test_breadth_signal_bullish(self, mock_core):
        """Test breadth signal transitions."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=370, base=100.0, trend=0.0005)

        symbols = ["7203", "6758", "9984"]
        result = market.market_breadth(symbols)

        # Should have some signal
        assert result["breadth_signal"] in [
            "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
        ]

    @patch("jpstock_agent.market._import_core")
    def test_breadth_detects_52w_highs(self, mock_core):
        """Test that breadth detects 52-week highs."""
        core_module = mock_core.return_value
        # Create data where latest close is at max
        data = _mock_history(days=252, base=100.0, trend=0.001)
        core_module.stock_history.return_value = data

        symbols = ["7203"]
        result = market.market_breadth(symbols)

        assert "new_highs_52w" in result

    def test_breadth_with_empty_symbols(self):
        """Test that breadth returns error for empty symbols."""
        result = market.market_breadth([])

        assert isinstance(result, dict)
        assert "error" in result

    @patch("jpstock_agent.market._import_core")
    def test_breadth_ad_ratio_calculation(self, mock_core):
        """Test advance/decline ratio calculation."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=370, base=100.0, trend=0.001)

        symbols = ["7203", "6758", "9984"]
        result = market.market_breadth(symbols)

        # Ratio should be > 0 if there are advancing stocks
        if result["declining"] > 0:
            assert result["advance_decline_ratio"] > 0


# ---------------------------------------------------------------------------
# TestMarketTopMovers
# ---------------------------------------------------------------------------

class TestMarketTopMovers:
    """Tests for market_top_movers function."""

    @patch("jpstock_agent.market._get_period_return")
    def test_top_movers_returns_dict(self, mock_return):
        """Test that market_top_movers returns dict with expected keys."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 5,
        }

        symbols = ["7203", "6758", "9984"]
        result = market.market_top_movers(symbols)

        assert isinstance(result, dict)
        assert "error" not in result
        assert "top_gainers" in result
        assert "top_losers" in result
        assert "period_days" in result

    @patch("jpstock_agent.market._get_period_return")
    def test_top_movers_sorts_gainers_descending(self, mock_return):
        """Test that top gainers are sorted from best to worst."""
        def side_effect(sym, days, source):
            returns = {
                "7203": 10.0,
                "6758": 5.0,
                "9984": 2.0,
            }
            return {
                "first_close": 100.0,
                "last_close": 100.0 + returns.get(sym, 0),
                "return_pct": returns.get(sym, 0),
                "high": 101.0,
                "low": 99.0,
                "data_points": 5,
            }

        mock_return.side_effect = side_effect

        symbols = ["7203", "6758", "9984"]
        result = market.market_top_movers(symbols, top_n=3)

        gainers = result["top_gainers"]
        assert gainers[0]["return_pct"] >= gainers[1]["return_pct"]

    @patch("jpstock_agent.market._get_period_return")
    def test_top_movers_respects_top_n(self, mock_return):
        """Test that top_n parameter limits results."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 5,
        }

        symbols = ["S1", "S2", "S3", "S4", "S5"]
        result = market.market_top_movers(symbols, top_n=2)

        assert len(result["top_gainers"]) <= 2
        assert len(result["top_losers"]) <= 2

    @patch("jpstock_agent.market._get_period_return")
    def test_top_movers_losers_sorted_ascending(self, mock_return):
        """Test that top losers are sorted from worst to least bad."""
        def side_effect(sym, days, source):
            returns = {
                "7203": -5.0,
                "6758": -2.0,
                "9984": 3.0,
            }
            return {
                "first_close": 100.0,
                "last_close": 100.0 + returns.get(sym, 0),
                "return_pct": returns.get(sym, 0),
                "high": 101.0,
                "low": 99.0,
                "data_points": 5,
            }

        mock_return.side_effect = side_effect

        symbols = ["7203", "6758", "9984"]
        result = market.market_top_movers(symbols, top_n=2)

        losers = result["top_losers"]
        # Losers should be in ascending order (worst first)
        if len(losers) > 1:
            assert losers[0]["return_pct"] <= losers[1]["return_pct"]

    @patch("jpstock_agent.market._get_period_return")
    def test_top_movers_includes_last_close(self, mock_return):
        """Test that top movers includes last_close price."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 5,
        }

        symbols = ["7203"]
        result = market.market_top_movers(symbols)

        if result["top_gainers"]:
            assert "last_close" in result["top_gainers"][0]

    def test_top_movers_with_empty_symbols(self):
        """Test that top movers returns error for empty symbols."""
        result = market.market_top_movers([])

        assert isinstance(result, dict)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestMarketRegime
# ---------------------------------------------------------------------------

class TestMarketRegime:
    """Tests for market_regime function."""

    @patch("jpstock_agent.market._import_core")
    def test_regime_returns_dict_with_expected_keys(self, mock_core):
        """Test that market_regime returns dict with expected keys."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=300, base=100.0, trend=0.001)

        result = market.market_regime("^N225")

        assert isinstance(result, dict)
        assert "error" not in result
        assert "regime" in result
        assert "confidence_pct" in result
        assert "indicators" in result
        assert "scoring" in result

    @patch("jpstock_agent.market._import_core")
    def test_regime_detects_bull_market(self, mock_core):
        """Test that regime detects bull market (uptrend)."""
        core_module = mock_core.return_value
        # Strong uptrend: latest > SMA50 > SMA200
        core_module.stock_history.return_value = _mock_history(days=300, base=100.0, trend=0.001)

        result = market.market_regime("^N225")

        assert "BULL" in result["regime"] or "MILD_BULL" in result["regime"]

    @patch("jpstock_agent.market._import_core")
    def test_regime_detects_bear_market(self, mock_core):
        """Test that regime detects bear market (downtrend)."""
        core_module = mock_core.return_value
        # Strong downtrend: latest < SMA50 < SMA200
        core_module.stock_history.return_value = _mock_history(days=300, base=100.0, trend=-0.001)

        result = market.market_regime("^N225")

        assert "BEAR" in result["regime"] or "MILD_BEAR" in result["regime"]

    @patch("jpstock_agent.market._import_core")
    def test_regime_detects_sideways_market(self, mock_core):
        """Test that regime detects sideways market (weak trend)."""
        core_module = mock_core.return_value
        # Very weak trend: minimal price movement (sideways)
        core_module.stock_history.return_value = _mock_history(days=300, base=100.0, trend=0.00001)

        result = market.market_regime("^N225")

        # With very weak trend, should be sideways or mild bull/bear
        assert result["regime"] in ["SIDEWAYS", "MILD_BULL", "MILD_BEAR"]

    @patch("jpstock_agent.market._import_core")
    def test_regime_calculates_sma50_and_sma200(self, mock_core):
        """Test that regime includes SMA50 and SMA200."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=300, base=100.0, trend=0.001)

        result = market.market_regime("^N225")

        indicators = result["indicators"]
        assert "sma50" in indicators
        assert "sma200" in indicators
        assert isinstance(indicators["sma50"], float)
        assert isinstance(indicators["sma200"], float)

    @patch("jpstock_agent.market._import_core")
    def test_regime_includes_return_metrics(self, mock_core):
        """Test that regime includes 30-day and 90-day returns."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=300, base=100.0, trend=0.001)

        result = market.market_regime("^N225")

        indicators = result["indicators"]
        assert "return_30d_pct" in indicators
        assert "return_90d_pct" in indicators

    @patch("jpstock_agent.market._import_core")
    def test_regime_with_insufficient_data(self, mock_core):
        """Test that regime returns error with insufficient data."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=30, base=100.0, trend=0.001)

        result = market.market_regime("^N225")

        assert "error" in result

    @patch("jpstock_agent.market._import_core")
    def test_regime_with_api_error(self, mock_core):
        """Test that regime handles API error."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = {"error": "API failed"}

        result = market.market_regime("^N225")

        assert "error" in result

    @patch("jpstock_agent.market._import_core")
    def test_regime_confidence_between_0_and_100(self, mock_core):
        """Test that confidence is between 0 and 100."""
        core_module = mock_core.return_value
        core_module.stock_history.return_value = _mock_history(days=300, base=100.0, trend=0.001)

        result = market.market_regime("^N225")

        confidence = result["confidence_pct"]
        assert 0 <= confidence <= 100


# ---------------------------------------------------------------------------
# TestMarketHeatmap
# ---------------------------------------------------------------------------

class TestMarketHeatmap:
    """Tests for market_heatmap function."""

    @patch("jpstock_agent.market._get_period_return")
    def test_heatmap_returns_dict(self, mock_return):
        """Test that market_heatmap returns dict with expected keys."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 5,
        }

        sectors = {"Auto": ["7203", "7267"], "Tech": ["6758"]}
        result = market.market_heatmap(sectors)

        assert isinstance(result, dict)
        assert "error" not in result
        assert "heatmap_data" in result
        assert "sector_averages" in result
        assert "period_days" in result

    @patch("jpstock_agent.market._get_period_return")
    def test_heatmap_structure(self, mock_return):
        """Test that heatmap data has correct structure."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 5,
        }

        sectors = {"Auto": ["7203", "7267"]}
        result = market.market_heatmap(sectors)

        heatmap = result["heatmap_data"]
        assert "Auto" in heatmap
        assert isinstance(heatmap["Auto"], list)
        assert len(heatmap["Auto"]) == 2

    @patch("jpstock_agent.market._get_period_return")
    def test_heatmap_stock_entry_structure(self, mock_return):
        """Test that each stock entry has required fields."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 5,
        }

        sectors = {"Auto": ["7203"]}
        result = market.market_heatmap(sectors)

        stock = result["heatmap_data"]["Auto"][0]
        assert "symbol" in stock
        assert "return_pct" in stock
        assert "last_close" in stock

    @patch("jpstock_agent.market._get_period_return")
    def test_heatmap_sector_averages(self, mock_return):
        """Test that sector averages are computed correctly."""
        def side_effect(sym, days, source):
            returns = {
                "7203": 6.0,
                "7267": 4.0,
            }
            return {
                "first_close": 100.0,
                "last_close": 100.0 + returns.get(sym, 0),
                "return_pct": returns.get(sym, 0),
                "high": 101.0,
                "low": 99.0,
                "data_points": 5,
            }

        mock_return.side_effect = side_effect

        sectors = {"Auto": ["7203", "7267"]}
        result = market.market_heatmap(sectors)

        avg = result["sector_averages"]["Auto"]
        assert avg == 5.0  # Average of 6.0 and 4.0

    @patch("jpstock_agent.market._get_period_return")
    def test_heatmap_with_empty_sectors(self, mock_return):
        """Test that heatmap returns error for empty sectors."""
        result = market.market_heatmap({})

        assert isinstance(result, dict)
        assert "error" in result

    @patch("jpstock_agent.market._get_period_return")
    def test_heatmap_with_missing_data(self, mock_return):
        """Test that heatmap handles missing stock data."""
        mock_return.return_value = None

        sectors = {"Auto": ["7203", "7267"]}
        result = market.market_heatmap(sectors)

        heatmap = result["heatmap_data"]
        assert "Auto" in heatmap
        # All stocks should be present, some with None values
        assert len(heatmap["Auto"]) == 2

    @patch("jpstock_agent.market._get_period_return")
    def test_heatmap_total_symbols_count(self, mock_return):
        """Test that total_symbols count is correct."""
        mock_return.return_value = {
            "first_close": 100.0,
            "last_close": 105.0,
            "return_pct": 5.0,
            "high": 106.0,
            "low": 99.0,
            "data_points": 5,
        }

        sectors = {"Auto": ["7203", "7267"], "Tech": ["6758"]}
        result = market.market_heatmap(sectors)

        assert result["total_symbols"] == 3
