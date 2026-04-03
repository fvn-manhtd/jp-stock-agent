"""Tests for jpstock_agent.alert module.

Tests alert conditions, watchlist checking, and fundamental alerts.
All external API calls are mocked to keep tests fast.
"""

from unittest.mock import MagicMock, patch

from jpstock_agent import alert
from tests.conftest import _make_ohlcv_records

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_history_with_volume(days=25, base_price=100.0, volumes=None):
    """Create stock history with custom volumes."""
    records = _make_ohlcv_records(days=days, base_price=base_price)
    if volumes:
        for i, vol in enumerate(volumes):
            if i < len(records):
                records[i]["volume"] = vol
    return records


# ---------------------------------------------------------------------------
# Test: alert_list_conditions
# ---------------------------------------------------------------------------

class TestAlertListConditions:
    """Test alert_list_conditions returns all 16 conditions."""

    def test_list_conditions_returns_dict_with_conditions_key(self):
        """Test that list_conditions returns dict with 'conditions' key."""
        result = alert.alert_list_conditions()
        assert isinstance(result, dict)
        assert "conditions" in result
        assert "total" in result

    def test_list_conditions_has_16_total(self):
        """Test that 16 conditions are defined."""
        result = alert.alert_list_conditions()
        assert result["total"] == 16

    def test_list_conditions_includes_rsi_oversold(self):
        """Test that rsi_oversold is in conditions."""
        result = alert.alert_list_conditions()
        assert "rsi_oversold" in result["conditions"]
        assert "description" in result["conditions"]["rsi_oversold"]
        assert "default_params" in result["conditions"]["rsi_oversold"]

    def test_list_conditions_includes_macd_bullish_cross(self):
        """Test that macd_bullish_cross is in conditions."""
        result = alert.alert_list_conditions()
        assert "macd_bullish_cross" in result["conditions"]

    def test_list_conditions_includes_volume_spike(self):
        """Test that volume_spike is in conditions."""
        result = alert.alert_list_conditions()
        assert "volume_spike" in result["conditions"]

    def test_list_conditions_includes_all_16_names(self):
        """Test all expected condition names are present."""
        expected = {
            "rsi_oversold", "rsi_overbought",
            "macd_bullish_cross", "macd_bearish_cross",
            "bb_squeeze", "bb_breakout_upper", "bb_breakout_lower",
            "golden_cross", "death_cross", "volume_spike",
            "price_above_sma", "price_below_sma",
            "supertrend_bullish", "supertrend_bearish",
            "new_high_52w", "new_low_52w",
        }
        result = alert.alert_list_conditions()
        actual = set(result["conditions"].keys())
        assert actual == expected


# ---------------------------------------------------------------------------
# Test: alert_price
# ---------------------------------------------------------------------------

class TestAlertPrice:
    """Test price-level alerts (above/below thresholds)."""

    @patch("jpstock_agent.alert._import_core")
    def test_alert_price_above_threshold_triggered(self, mock_import_core):
        """Test price above threshold is triggered."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.stock_history.return_value = [{"close": 150.0}]

        result = alert.alert_price("7203", above=100.0)

        assert result["symbol"] == "7203"
        assert result["current_price"] == 150.0
        assert result["any_triggered"] is True
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["triggered"] is True
        assert result["alerts"][0]["type"] == "price_above"

    @patch("jpstock_agent.alert._import_core")
    def test_alert_price_above_threshold_not_triggered(self, mock_import_core):
        """Test price above threshold not triggered when price lower."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.stock_history.return_value = [{"close": 50.0}]

        result = alert.alert_price("7203", above=100.0)

        assert result["current_price"] == 50.0
        assert result["any_triggered"] is False
        assert len(result["alerts"]) == 0

    @patch("jpstock_agent.alert._import_core")
    def test_alert_price_below_threshold_triggered(self, mock_import_core):
        """Test price below threshold is triggered."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.stock_history.return_value = [{"close": 50.0}]

        result = alert.alert_price("7203", below=100.0)

        assert result["any_triggered"] is True
        assert result["alerts"][0]["triggered"] is True
        assert result["alerts"][0]["type"] == "price_below"

    @patch("jpstock_agent.alert._import_core")
    def test_alert_price_both_thresholds(self, mock_import_core):
        """Test both above and below thresholds."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.stock_history.return_value = [{"close": 100.0}]

        result = alert.alert_price("7203", above=50.0, below=150.0)

        assert result["any_triggered"] is True
        assert len(result["alerts"]) == 2  # Both triggered (100 > 50 AND 100 < 150)

    @patch("jpstock_agent.alert._import_core")
    def test_alert_price_missing_params_returns_error(self, mock_import_core):
        """Test that missing both above and below returns error."""
        result = alert.alert_price("7203")
        assert "error" in result

    @patch("jpstock_agent.alert._import_core")
    def test_alert_price_no_history_returns_error(self, mock_import_core):
        """Test that no history data returns error."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.stock_history.return_value = []

        result = alert.alert_price("7203", above=100.0)
        assert "error" in result

    @patch("jpstock_agent.alert._import_core")
    def test_alert_price_handles_Close_key(self, mock_import_core):
        """Test that Close key (uppercase) is handled."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.stock_history.return_value = [{"Close": 150.0}]

        result = alert.alert_price("7203", above=100.0)

        assert result["any_triggered"] is True


# ---------------------------------------------------------------------------
# Test: _eval_rsi_oversold
# ---------------------------------------------------------------------------

class TestEvalRsiOversold:
    """Test RSI oversold condition evaluation."""

    @patch("jpstock_agent.ta.ta_rsi")
    def test_rsi_oversold_triggered(self, mock_ta_rsi):
        """Test RSI oversold is triggered when RSI < 30."""
        mock_ta_rsi.return_value = [{"rsi": 25}]
        ta_mod = MagicMock()
        ta_mod.ta_rsi = mock_ta_rsi

        result = alert._eval_rsi_oversold(ta_mod, "7203", None, {})

        assert result["triggered"] is True
        assert result["value"] == 25
        assert result["threshold"] == 30

    @patch("jpstock_agent.ta.ta_rsi")
    def test_rsi_oversold_not_triggered(self, mock_ta_rsi):
        """Test RSI oversold not triggered when RSI >= 30."""
        mock_ta_rsi.return_value = [{"rsi": 35}]
        ta_mod = MagicMock()
        ta_mod.ta_rsi = mock_ta_rsi

        result = alert._eval_rsi_oversold(ta_mod, "7203", None, {})

        assert result["triggered"] is False

    @patch("jpstock_agent.ta.ta_rsi")
    def test_rsi_oversold_custom_threshold(self, mock_ta_rsi):
        """Test RSI oversold with custom threshold."""
        mock_ta_rsi.return_value = [{"rsi": 25}]
        ta_mod = MagicMock()
        ta_mod.ta_rsi = mock_ta_rsi

        result = alert._eval_rsi_oversold(ta_mod, "7203", None, {"threshold": 20})

        assert result["triggered"] is False  # 25 is not < 20


# ---------------------------------------------------------------------------
# Test: _eval_rsi_overbought
# ---------------------------------------------------------------------------

class TestEvalRsiOverbought:
    """Test RSI overbought condition evaluation."""

    @patch("jpstock_agent.ta.ta_rsi")
    def test_rsi_overbought_triggered(self, mock_ta_rsi):
        """Test RSI overbought is triggered when RSI > 70."""
        mock_ta_rsi.return_value = [{"rsi": 75}]
        ta_mod = MagicMock()
        ta_mod.ta_rsi = mock_ta_rsi

        result = alert._eval_rsi_overbought(ta_mod, "7203", None, {})

        assert result["triggered"] is True
        assert result["value"] == 75
        assert result["threshold"] == 70


# ---------------------------------------------------------------------------
# Test: _eval_macd_cross
# ---------------------------------------------------------------------------

class TestEvalMacdCross:
    """Test MACD crossover detection."""

    @patch("jpstock_agent.ta.ta_macd")
    def test_macd_bullish_cross_triggered(self, mock_ta_macd):
        """Test bullish MACD cross is detected."""
        # Previous: MACD below signal, Current: MACD above signal
        mock_ta_macd.return_value = [
            {"macd": 1.0, "macd_signal": 2.0},
            {"macd": 2.5, "macd_signal": 2.0},
        ]
        ta_mod = MagicMock()
        ta_mod.ta_macd = mock_ta_macd

        result = alert._eval_macd_cross(ta_mod, "7203", None, direction="bullish")

        assert result["triggered"] is True
        assert "bullish" in result["message"]

    @patch("jpstock_agent.ta.ta_macd")
    def test_macd_bullish_cross_not_triggered(self, mock_ta_macd):
        """Test bullish cross not triggered when MACD stays below signal."""
        mock_ta_macd.return_value = [
            {"macd": 1.0, "macd_signal": 2.0},
            {"macd": 1.5, "macd_signal": 2.0},
        ]
        ta_mod = MagicMock()
        ta_mod.ta_macd = mock_ta_macd

        result = alert._eval_macd_cross(ta_mod, "7203", None, direction="bullish")

        assert result["triggered"] is False

    @patch("jpstock_agent.ta.ta_macd")
    def test_macd_bearish_cross_triggered(self, mock_ta_macd):
        """Test bearish MACD cross is detected."""
        # Previous: MACD above signal, Current: MACD below signal
        mock_ta_macd.return_value = [
            {"macd": 2.5, "macd_signal": 2.0},
            {"macd": 1.0, "macd_signal": 2.0},
        ]
        ta_mod = MagicMock()
        ta_mod.ta_macd = mock_ta_macd

        result = alert._eval_macd_cross(ta_mod, "7203", None, direction="bearish")

        assert result["triggered"] is True
        assert "bearish" in result["message"]


# ---------------------------------------------------------------------------
# Test: _eval_volume_spike_condition
# ---------------------------------------------------------------------------

class TestEvalVolumeSpikeCondition:
    """Test volume spike detection."""

    @patch("jpstock_agent.alert._import_core")
    def test_volume_spike_triggered(self, mock_import_core):
        """Test volume spike is triggered."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core

        # Create history: 20 days of 1M volume, then spike to 3M
        volumes = [1_000_000.0] * 20 + [3_000_000.0]
        mock_core.stock_history.return_value = _make_history_with_volume(
            days=21, volumes=volumes
        )

        ta_mod = MagicMock()
        result = alert._eval_volume_spike(ta_mod, "7203", None, {"multiplier": 2.0})

        assert result["triggered"] is True
        assert result["ratio"] > 2.0

    @patch("jpstock_agent.alert._import_core")
    def test_volume_spike_not_triggered(self, mock_import_core):
        """Test volume spike not triggered when below multiplier."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core

        # All volumes around 1M
        volumes = [1_000_000.0] * 21
        mock_core.stock_history.return_value = _make_history_with_volume(
            days=21, volumes=volumes
        )

        ta_mod = MagicMock()
        result = alert._eval_volume_spike(ta_mod, "7203", None, {"multiplier": 2.0})

        assert result["triggered"] is False


# ---------------------------------------------------------------------------
# Test: alert_ta
# ---------------------------------------------------------------------------

class TestAlertTa:
    """Test TA-based alerts."""

    @patch("jpstock_agent.alert.alert_check")
    def test_alert_ta_default_conditions(self, mock_check):
        """Test alert_ta uses default conditions."""
        mock_check.return_value = {"triggered": [], "not_triggered_count": 5}

        alert.alert_ta("7203")

        # Verify alert_check was called
        assert mock_check.called
        call_args = mock_check.call_args
        conditions = call_args[0][1]
        assert len(conditions) == 8  # Default is 8 conditions

    @patch("jpstock_agent.alert.alert_check")
    def test_alert_ta_custom_conditions(self, mock_check):
        """Test alert_ta with custom condition list."""
        mock_check.return_value = {"triggered": []}

        alert.alert_ta("7203", conditions=["rsi_oversold"])

        call_args = mock_check.call_args
        conditions = call_args[0][1]
        assert len(conditions) == 1
        assert conditions[0]["condition"] == "rsi_oversold"


# ---------------------------------------------------------------------------
# Test: alert_check
# ---------------------------------------------------------------------------

class TestAlertCheck:
    """Test alert_check evaluation of multiple conditions."""

    @patch("jpstock_agent.alert._eval_rsi_oversold")
    def test_alert_check_single_condition_triggered(self, mock_eval):
        """Test alert_check with one condition triggered."""
        mock_eval.return_value = {"triggered": True, "value": 25}

        conditions = [{"condition": "rsi_oversold"}]
        result = alert.alert_check("7203", conditions)

        assert result["symbol"] == "7203"
        assert result["triggered_count"] == 1
        assert len(result["triggered"]) == 1

    @patch("jpstock_agent.alert._eval_rsi_oversold")
    @patch("jpstock_agent.alert._eval_rsi_overbought")
    def test_alert_check_multiple_conditions(self, mock_eval_ob, mock_eval_os):
        """Test alert_check with multiple conditions."""
        mock_eval_os.return_value = {"triggered": True, "value": 25}
        mock_eval_ob.return_value = {"triggered": False}

        conditions = [
            {"condition": "rsi_oversold"},
            {"condition": "rsi_overbought"},
        ]
        result = alert.alert_check("7203", conditions)

        assert result["total_conditions"] == 2
        assert result["triggered_count"] == 1
        assert result["not_triggered_count"] == 1

    def test_alert_check_no_conditions_returns_error(self):
        """Test alert_check with empty conditions list."""
        result = alert.alert_check("7203", [])
        assert "error" in result

    @patch("jpstock_agent.alert._eval_rsi_oversold")
    def test_alert_check_unknown_condition(self, mock_eval):
        """Test alert_check with unknown condition name."""
        conditions = [{"condition": "nonexistent_condition"}]
        result = alert.alert_check("7203", conditions)

        assert result["error_count"] == 1

    @patch("jpstock_agent.alert._eval_rsi_oversold")
    def test_alert_check_condition_with_params(self, mock_eval):
        """Test alert_check passes params to evaluator."""
        mock_eval.return_value = {"triggered": False}

        conditions = [{"condition": "rsi_oversold", "params": {"threshold": 25}}]
        alert.alert_check("7203", conditions)

        # Verify the evaluator was called
        assert mock_eval.called


# ---------------------------------------------------------------------------
# Test: alert_fundamental
# ---------------------------------------------------------------------------

class TestAlertFundamental:
    """Test fundamental-based alerts."""

    @patch("jpstock_agent.alert._import_financial")
    @patch("jpstock_agent.alert._import_core")
    def test_alert_fundamental_pe_below_triggered(self, mock_import_core, mock_import_fin):
        """Test P/E below threshold alert."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.financial_ratio.return_value = {"trailingPE": 10.0}

        mock_fin = MagicMock()
        mock_import_fin.return_value = mock_fin

        result = alert.alert_fundamental("7203", pe_below=15.0)

        assert result["triggered_count"] == 1
        assert result["any_triggered"] is True
        assert result["alerts"][0]["condition"] == "pe_below"

    @patch("jpstock_agent.alert._import_financial")
    @patch("jpstock_agent.alert._import_core")
    def test_alert_fundamental_pe_above_triggered(self, mock_import_core, mock_import_fin):
        """Test P/E above threshold alert."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.financial_ratio.return_value = {"trailingPE": 25.0}

        mock_fin = MagicMock()
        mock_import_fin.return_value = mock_fin

        result = alert.alert_fundamental("7203", pe_above=20.0)

        assert result["triggered_count"] == 1
        assert result["alerts"][0]["condition"] == "pe_above"

    @patch("jpstock_agent.alert._import_financial")
    @patch("jpstock_agent.alert._import_core")
    def test_alert_fundamental_yield_triggered(self, mock_import_core, mock_import_fin):
        """Test dividend yield alert."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.financial_ratio.return_value = {"dividendYield": 0.05}

        mock_fin = MagicMock()
        mock_import_fin.return_value = mock_fin

        result = alert.alert_fundamental("7203", yield_above=0.03)

        assert result["triggered_count"] == 1
        assert result["alerts"][0]["condition"] == "yield_above"

    @patch("jpstock_agent.alert._import_financial")
    @patch("jpstock_agent.alert._import_core")
    def test_alert_fundamental_roe_triggered(self, mock_import_core, mock_import_fin):
        """Test ROE alert."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.financial_ratio.return_value = {"returnOnEquity": 0.20}

        mock_fin = MagicMock()
        mock_import_fin.return_value = mock_fin

        result = alert.alert_fundamental("7203", roe_above=0.15)

        assert result["triggered_count"] == 1
        assert result["alerts"][0]["condition"] == "roe_above"

    @patch("jpstock_agent.alert._import_financial")
    @patch("jpstock_agent.alert._import_core")
    def test_alert_fundamental_debt_to_equity_triggered(self, mock_import_core, mock_import_fin):
        """Test debt-to-equity ratio alert."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.financial_ratio.return_value = {"debtToEquity": 0.5}

        mock_fin = MagicMock()
        mock_import_fin.return_value = mock_fin

        result = alert.alert_fundamental("7203", debt_to_equity_below=1.0)

        assert result["triggered_count"] == 1
        assert result["alerts"][0]["condition"] == "debt_to_equity_below"

    @patch("jpstock_agent.alert._import_financial")
    @patch("jpstock_agent.alert._import_core")
    def test_alert_fundamental_f_score_triggered(self, mock_import_core, mock_import_fin):
        """Test Piotroski F-score alert."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.financial_ratio.return_value = {}

        mock_fin = MagicMock()
        mock_import_fin.return_value = mock_fin
        mock_fin.financial_health.return_value = {"piotroski_f": {"score": 8}}

        result = alert.alert_fundamental("7203", f_score_above=6)

        assert result["triggered_count"] == 1
        assert result["alerts"][0]["condition"] == "f_score_above"

    @patch("jpstock_agent.alert._import_financial")
    @patch("jpstock_agent.alert._import_core")
    def test_alert_fundamental_no_alerts(self, mock_import_core, mock_import_fin):
        """Test when no conditions are triggered."""
        mock_core = MagicMock()
        mock_import_core.return_value = mock_core
        mock_core.financial_ratio.return_value = {"trailingPE": 20.0}

        mock_fin = MagicMock()
        mock_import_fin.return_value = mock_fin

        result = alert.alert_fundamental("7203", pe_below=10.0)

        assert result["triggered_count"] == 0
        assert result["any_triggered"] is False


# ---------------------------------------------------------------------------
# Test: alert_watchlist
# ---------------------------------------------------------------------------

class TestAlertWatchlist:
    """Test multi-symbol watchlist checking."""

    @patch("jpstock_agent.alert.alert_ta")
    def test_alert_watchlist_checks_multiple_symbols(self, mock_alert_ta):
        """Test watchlist checks all symbols."""
        mock_alert_ta.side_effect = [
            {"symbol": "7203", "triggered_count": 1, "triggered": [{"condition": "rsi_oversold"}]},
            {"symbol": "6758", "triggered_count": 0, "triggered": []},
        ]

        result = alert.alert_watchlist(["7203", "6758"])

        assert result["symbol_count"] == 2
        assert len(result["results"]) == 2
        assert mock_alert_ta.call_count == 2

    @patch("jpstock_agent.alert.alert_ta")
    def test_alert_watchlist_identifies_triggered_symbols(self, mock_alert_ta):
        """Test watchlist identifies symbols with triggered alerts."""
        mock_alert_ta.side_effect = [
            {"symbol": "7203", "triggered_count": 1},
            {"symbol": "6758", "triggered_count": 0},
        ]

        result = alert.alert_watchlist(["7203", "6758"])

        assert result["triggered_symbol_count"] == 1
        assert "7203" in result["triggered_symbols"]
        assert "6758" not in result["triggered_symbols"]

    def test_alert_watchlist_no_symbols_returns_error(self):
        """Test watchlist with empty symbols list."""
        result = alert.alert_watchlist([])
        assert "error" in result

    @patch("jpstock_agent.alert.alert_ta")
    def test_alert_watchlist_preserves_symbol_order(self, mock_alert_ta):
        """Test watchlist preserves input symbol order."""
        mock_alert_ta.side_effect = [
            {"symbol": "9984", "triggered_count": 0},
            {"symbol": "7203", "triggered_count": 1},
            {"symbol": "6758", "triggered_count": 0},
        ]

        result = alert.alert_watchlist(["9984", "7203", "6758"])

        # Results should be in input order
        symbols = [r["symbol"] for r in result["results"]]
        assert symbols == ["9984", "7203", "6758"]


# ---------------------------------------------------------------------------
# Test: Bollinger Bands alerts
# ---------------------------------------------------------------------------

class TestEvalBbSqueezeCondition:
    """Test Bollinger Band squeeze detection."""

    @patch("jpstock_agent.ta.ta_bbands")
    def test_bb_squeeze_triggered(self, mock_ta_bbands):
        """Test BB squeeze is triggered when width < threshold."""
        mock_ta_bbands.return_value = [
            {
                "close": 100.0,
                "bb_upper": 102.0,
                "bb_lower": 98.0,
                "bb_middle": 100.0,
            }
        ]
        ta_mod = MagicMock()
        ta_mod.ta_bbands = mock_ta_bbands

        result = alert._eval_bb_squeeze(ta_mod, "7203", None, {"threshold": 0.1})

        assert result["triggered"] is True
        # width = (102-98)/100 = 0.04 < 0.1

    @patch("jpstock_agent.ta.ta_bbands")
    def test_bb_breakout_upper_triggered(self, mock_ta_bbands):
        """Test BB upper breakout is triggered."""
        mock_ta_bbands.return_value = [
            {
                "close": 105.0,
                "bb_upper": 102.0,
                "bb_lower": 98.0,
                "bb_middle": 100.0,
            }
        ]
        ta_mod = MagicMock()
        ta_mod.ta_bbands = mock_ta_bbands

        result = alert._eval_bb_breakout(ta_mod, "7203", None, direction="upper")

        assert result["triggered"] is True


# ---------------------------------------------------------------------------
# Test: Price vs SMA
# ---------------------------------------------------------------------------

class TestEvalPriceVsSma:
    """Test price vs SMA comparisons."""

    @patch("jpstock_agent.ta.ta_sma")
    def test_price_above_sma_triggered(self, mock_ta_sma):
        """Test price above SMA is triggered."""
        mock_ta_sma.return_value = [
            {"close": 105.0, "sma": 100.0}
        ]
        ta_mod = MagicMock()
        ta_mod.ta_sma = mock_ta_sma

        result = alert._eval_price_vs_sma(ta_mod, "7203", None, {}, direction="above")

        assert result["triggered"] is True

    @patch("jpstock_agent.ta.ta_sma")
    def test_price_below_sma_triggered(self, mock_ta_sma):
        """Test price below SMA is triggered."""
        mock_ta_sma.return_value = [
            {"close": 95.0, "sma": 100.0}
        ]
        ta_mod = MagicMock()
        ta_mod.ta_sma = mock_ta_sma

        result = alert._eval_price_vs_sma(ta_mod, "7203", None, {}, direction="below")

        assert result["triggered"] is True


# ---------------------------------------------------------------------------
# Test: 52-week extremes
# ---------------------------------------------------------------------------

class TestEval52WeekExtreme:
    """Test 52-week high/low detection."""

    @patch("jpstock_agent.alert._import_core")
    def test_52w_high_triggered(self, mock_import_core):
        """Test 52-week high is triggered."""
        mock_core = MagicMock()
        # Create 25 records; latest is highest
        history = [{"close": 100.0 + i} for i in range(24)] + [{"close": 150.0}]
        mock_core.stock_history.return_value = history

        result = alert._eval_52w_extreme(mock_core, "7203", None, direction="high")

        assert result["triggered"] is True

    @patch("jpstock_agent.alert._import_core")
    def test_52w_low_triggered(self, mock_import_core):
        """Test 52-week low is triggered."""
        mock_core = MagicMock()
        # Create 25 records; latest is lowest
        history = [{"close": 100.0 + i} for i in range(24)] + [{"close": 50.0}]
        mock_core.stock_history.return_value = history

        result = alert._eval_52w_extreme(mock_core, "7203", None, direction="low")

        assert result["triggered"] is True
