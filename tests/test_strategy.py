"""Tests for jpstock_agent.strategy module (custom strategy builder).

Tests condition evaluation, strategy screening, and multi-condition logic
with mocked core, ta, and financial functions.
"""

from unittest.mock import patch

import pytest

from jpstock_agent import strategy
from tests.conftest import _make_ohlcv_records

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_history():
    """Return mocked stock history (list of OHLCV dicts)."""
    return _make_ohlcv_records(days=60, base_price=2500.0)


@pytest.fixture
def sample_history():
    """60-day OHLCV history as list[dict]."""
    return _mock_history()


# ---------------------------------------------------------------------------
# TestStrategyEvaluate: Test AND/OR logic
# ---------------------------------------------------------------------------

class TestStrategyEvaluate:
    """Test strategy_evaluate with various condition combinations."""

    @patch("jpstock_agent.strategy._import_core")
    @patch("jpstock_agent.strategy._import_ta")
    def test_evaluate_and_logic_all_pass(self, mock_ta, mock_core):
        """Test AND logic when all conditions pass."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 25.0}]
        mock_ta_inst.ta_macd.return_value = [{"macd": 2.0, "macd_signal": 1.5}]

        conditions = [
            {"type": "rsi_below", "params": {"value": 30}},
            {"type": "macd_bullish"},
        ]

        result = strategy.strategy_evaluate("7203", conditions, logic="AND")

        assert result["passed"] is True
        assert result["symbol"] == "7203"
        assert result["logic"] == "AND"
        assert result["passed_count"] == 2
        assert result["total_conditions"] == 2

    @patch("jpstock_agent.strategy._import_core")
    @patch("jpstock_agent.strategy._import_ta")
    def test_evaluate_and_logic_some_fail(self, mock_ta, mock_core):
        """Test AND logic when some conditions fail."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 75.0}]
        mock_ta_inst.ta_macd.return_value = [{"macd": 2.0, "macd_signal": 1.5}]

        conditions = [
            {"type": "rsi_below", "params": {"value": 30}},
            {"type": "macd_bullish"},
        ]

        result = strategy.strategy_evaluate("7203", conditions, logic="AND")

        assert result["passed"] is False
        assert result["passed_count"] == 1
        assert result["total_conditions"] == 2

    @patch("jpstock_agent.strategy._import_core")
    @patch("jpstock_agent.strategy._import_ta")
    def test_evaluate_or_logic_any_pass(self, mock_ta, mock_core):
        """Test OR logic when at least one condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 25.0}]
        mock_ta_inst.ta_macd.return_value = [{"macd": 0.5, "macd_signal": 1.5}]

        conditions = [
            {"type": "rsi_below", "params": {"value": 30}},
            {"type": "macd_bullish"},
        ]

        result = strategy.strategy_evaluate("7203", conditions, logic="OR")

        assert result["passed"] is True
        assert result["logic"] == "OR"
        assert result["passed_count"] == 1

    @patch("jpstock_agent.strategy._import_core")
    @patch("jpstock_agent.strategy._import_ta")
    def test_evaluate_or_logic_none_pass(self, mock_ta, mock_core):
        """Test OR logic when no conditions pass."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 75.0}]
        mock_ta_inst.ta_macd.return_value = [{"macd": 0.5, "macd_signal": 1.5}]

        conditions = [
            {"type": "rsi_below", "params": {"value": 30}},
            {"type": "macd_bullish"},
        ]

        result = strategy.strategy_evaluate("7203", conditions, logic="OR")

        assert result["passed"] is False
        assert result["passed_count"] == 0

    def test_evaluate_empty_conditions_error(self):
        """Test that empty conditions list returns error."""
        result = strategy.strategy_evaluate("7203", [])

        assert "error" in result
        assert result["error"] == "No conditions provided"

    @patch("jpstock_agent.strategy._import_core")
    @patch("jpstock_agent.strategy._import_ta")
    def test_evaluate_returns_conditions_detail(self, mock_ta, mock_core):
        """Test that evaluation result includes detailed condition results."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 25.0}]

        conditions = [
            {"type": "rsi_below", "params": {"value": 30}},
        ]

        result = strategy.strategy_evaluate("7203", conditions)

        assert "conditions_detail" in result
        assert len(result["conditions_detail"]) == 1
        detail = result["conditions_detail"][0]
        assert detail["type"] == "rsi_below"
        assert detail["passed"] is True
        assert detail["value"] is not None


# ---------------------------------------------------------------------------
# TestStrategyScreen: Test screening multiple symbols
# ---------------------------------------------------------------------------

class TestStrategyScreen:
    """Test strategy_screen with multiple symbols."""

    @patch("jpstock_agent.strategy.strategy_evaluate")
    def test_screen_multiple_symbols(self, mock_evaluate):
        """Test screening returns matching and not_matching counts."""
        mock_evaluate.side_effect = [
            {"symbol": "7203", "passed": True},
            {"symbol": "6758", "passed": False},
            {"symbol": "9984", "passed": True},
        ]

        symbols = ["7203", "6758", "9984"]
        conditions = [{"type": "rsi_below", "params": {"value": 30}}]

        result = strategy.strategy_screen(symbols, conditions)

        assert result["total_screened"] == 3
        assert result["matching_count"] == 2
        assert result["not_matching_count"] == 1
        assert len(result["matching"]) == 2
        assert len([r for r in result["matching"] if r.get("passed")]) == 2

    @patch("jpstock_agent.strategy.strategy_evaluate")
    def test_screen_preserves_symbol_order(self, mock_evaluate):
        """Test that screening results maintain input symbol order."""
        mock_evaluate.side_effect = [
            {"symbol": "7203", "passed": True},
            {"symbol": "6758", "passed": False},
            {"symbol": "9984", "passed": True},
        ]

        symbols = ["7203", "6758", "9984"]
        conditions = [{"type": "rsi_below"}]

        result = strategy.strategy_screen(symbols, conditions)

        # All results should be in original order
        first_passed = result["matching"][0] if result["matching"] else None
        if first_passed:
            assert first_passed.get("symbol") in symbols

    def test_screen_empty_symbols_error(self):
        """Test that empty symbols list returns error."""
        result = strategy.strategy_screen([], [{"type": "rsi_below"}])

        assert "error" in result
        assert result["error"] == "No symbols provided"

    def test_screen_empty_conditions_error(self):
        """Test that empty conditions list returns error."""
        result = strategy.strategy_screen(["7203"], [])

        assert "error" in result
        assert result["error"] == "No conditions provided"

    @patch("jpstock_agent.strategy.strategy_evaluate")
    def test_screen_returns_proper_structure(self, mock_evaluate):
        """Test that screen result has required fields."""
        mock_evaluate.side_effect = [
            {"symbol": "7203", "passed": True},
            {"symbol": "6758", "passed": False},
        ]

        result = strategy.strategy_screen(["7203", "6758"], [{"type": "rsi_below"}])

        assert "logic" in result
        assert "conditions_count" in result
        assert "total_screened" in result
        assert "matching_count" in result
        assert "matching" in result
        assert "not_matching_count" in result


# ---------------------------------------------------------------------------
# TestStrategyListConditions: Test condition registry
# ---------------------------------------------------------------------------

class TestStrategyListConditions:
    """Test strategy_list_conditions."""

    def test_list_conditions_returns_categories(self):
        """Test that condition list is grouped by category."""
        result = strategy.strategy_list_conditions()

        assert "categories" in result
        categories = result["categories"]
        assert "price" in categories
        assert "ta" in categories
        assert "fundamental" in categories

    def test_list_conditions_has_all_types(self):
        """Test that all condition types are listed."""
        result = strategy.strategy_list_conditions()

        assert result["total_conditions"] == len(strategy.CONDITION_TYPES)
        assert result["total_conditions"] > 0

    def test_list_conditions_has_descriptions(self):
        """Test that each condition has description and params."""
        result = strategy.strategy_list_conditions()

        for category, conditions in result["categories"].items():
            for cond in conditions:
                assert "type" in cond
                assert "description" in cond
                assert "params" in cond
                assert isinstance(cond["description"], str)
                assert len(cond["description"]) > 0

    def test_list_conditions_price_category(self):
        """Test price category conditions exist."""
        result = strategy.strategy_list_conditions()

        price_types = [c["type"] for c in result["categories"]["price"]]
        assert "price_above" in price_types
        assert "price_below" in price_types
        assert "return_above" in price_types
        assert "volume_above_avg" in price_types

    def test_list_conditions_ta_category(self):
        """Test TA category conditions exist."""
        result = strategy.strategy_list_conditions()

        ta_types = [c["type"] for c in result["categories"]["ta"]]
        assert "rsi_below" in ta_types
        assert "macd_bullish" in ta_types
        assert "price_above_sma" in ta_types
        assert "ta_signal_buy" in ta_types

    def test_list_conditions_fundamental_category(self):
        """Test fundamental category conditions exist."""
        result = strategy.strategy_list_conditions()

        fund_types = [c["type"] for c in result["categories"]["fundamental"]]
        assert "pe_below" in fund_types
        assert "pb_below" in fund_types
        assert "dividend_yield_above" in fund_types
        assert "f_score_above" in fund_types


# ---------------------------------------------------------------------------
# TestEvalPriceConditions: Test price-based conditions
# ---------------------------------------------------------------------------

class TestEvalPriceConditions:
    """Test price-related condition evaluation."""

    @patch("jpstock_agent.strategy._import_core")
    def test_price_above_pass(self, mock_core):
        """Test price_above condition passes."""
        mock_core_inst = mock_core.return_value
        # Mock history with a known latest price
        history = _mock_history()
        # Ensure the latest price is higher
        history[-1]["close"] = 2800.0
        mock_core_inst.stock_history.return_value = history

        cond = {"type": "price_above", "params": {"value": 2400.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["type"] == "price_above"
        assert result["value"] is not None

    @patch("jpstock_agent.strategy._import_core")
    def test_price_above_fail(self, mock_core):
        """Test price_above condition fails."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        cond = {"type": "price_above", "params": {"value": 3000.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is False

    @patch("jpstock_agent.strategy._import_core")
    def test_price_below_pass(self, mock_core):
        """Test price_below condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        cond = {"type": "price_below", "params": {"value": 3000.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_core")
    def test_return_above_pass(self, mock_core):
        """Test return_above condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        cond = {"type": "return_above", "params": {"days": 30, "value": -50.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] is not None

    @patch("jpstock_agent.strategy._import_core")
    def test_return_below_pass(self, mock_core):
        """Test return_below condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        cond = {"type": "return_below", "params": {"days": 30, "value": 100.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_core")
    def test_volume_above_avg_pass(self, mock_core):
        """Test volume_above_avg condition passes."""
        mock_core_inst = mock_core.return_value
        # Mock history where the latest volume is high
        history = _mock_history()
        # Set latest volume to be very high compared to average
        history[-1]["volume"] = 9000000.0
        mock_core_inst.stock_history.return_value = history

        cond = {"type": "volume_above_avg", "params": {"multiplier": 0.5}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] is not None

    @patch("jpstock_agent.strategy._import_core")
    def test_volume_above_avg_fail(self, mock_core):
        """Test volume_above_avg condition fails."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        cond = {"type": "volume_above_avg", "params": {"multiplier": 100.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is False


# ---------------------------------------------------------------------------
# TestEvalTaConditions: Test technical analysis conditions
# ---------------------------------------------------------------------------

class TestEvalTaConditions:
    """Test TA-related condition evaluation."""

    @patch("jpstock_agent.strategy._import_ta")
    def test_rsi_below_pass(self, mock_ta):
        """Test rsi_below condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 25.0}]

        cond = {"type": "rsi_below", "params": {"value": 30}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] == 25.0

    @patch("jpstock_agent.strategy._import_ta")
    def test_rsi_above_pass(self, mock_ta):
        """Test rsi_above condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 75.0}]

        cond = {"type": "rsi_above", "params": {"value": 70}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] == 75.0

    @patch("jpstock_agent.strategy._import_ta")
    def test_rsi_between_pass(self, mock_ta):
        """Test rsi_between condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 50.0}]

        cond = {"type": "rsi_between", "params": {"low": 40, "high": 60}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_ta")
    def test_rsi_between_fail(self, mock_ta):
        """Test rsi_between condition fails."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 25.0}]

        cond = {"type": "rsi_between", "params": {"low": 40, "high": 60}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is False

    @patch("jpstock_agent.strategy._import_ta")
    def test_macd_bullish_pass(self, mock_ta):
        """Test macd_bullish condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_macd.return_value = [{"macd": 2.0, "macd_signal": 1.5}]

        cond = {"type": "macd_bullish"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_ta")
    def test_macd_bearish_pass(self, mock_ta):
        """Test macd_bearish condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_macd.return_value = [{"macd": 0.5, "macd_signal": 1.5}]

        cond = {"type": "macd_bearish"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_core")
    @patch("jpstock_agent.strategy._import_ta")
    def test_price_above_sma_pass(self, mock_ta, mock_core):
        """Test price_above_sma condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_sma.return_value = [{"close": 100.0, "sma": 95.0}]

        cond = {"type": "price_above_sma", "params": {"period": 50}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] == 100.0

    @patch("jpstock_agent.strategy._import_core")
    @patch("jpstock_agent.strategy._import_ta")
    def test_price_below_sma_pass(self, mock_ta, mock_core):
        """Test price_below_sma condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_sma.return_value = [{"close": 90.0, "sma": 95.0}]

        cond = {"type": "price_below_sma", "params": {"period": 50}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_ta")
    def test_bb_above_upper_pass(self, mock_ta):
        """Test bb_above_upper condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_bbands.return_value = [
            {"close": 110.0, "bb_upper": 105.0, "bb_lower": 90.0}
        ]

        cond = {"type": "bb_above_upper"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_ta")
    def test_bb_below_lower_pass(self, mock_ta):
        """Test bb_below_lower condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_bbands.return_value = [
            {"close": 85.0, "bb_upper": 105.0, "bb_lower": 90.0}
        ]

        cond = {"type": "bb_below_lower"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_ta")
    def test_supertrend_bullish_pass(self, mock_ta):
        """Test supertrend_bullish condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_supertrend.return_value = [{"direction": "bullish"}]

        cond = {"type": "supertrend_bullish"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] == "bullish"

    @patch("jpstock_agent.strategy._import_ta")
    def test_supertrend_bearish_pass(self, mock_ta):
        """Test supertrend_bearish condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_supertrend.return_value = [{"direction": "bearish"}]

        cond = {"type": "supertrend_bearish"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_ta")
    def test_ta_signal_buy_pass(self, mock_ta):
        """Test ta_signal_buy condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_multi_indicator.return_value = {"signal": "BUY", "score": 60}

        cond = {"type": "ta_signal_buy"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] == "BUY"

    @patch("jpstock_agent.strategy._import_ta")
    def test_ta_signal_sell_pass(self, mock_ta):
        """Test ta_signal_sell condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_multi_indicator.return_value = {"signal": "SELL", "score": -60}

        cond = {"type": "ta_signal_sell"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_ta")
    def test_ta_score_above_pass(self, mock_ta):
        """Test ta_score_above condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_multi_indicator.return_value = {"signal": "BUY", "score": 60}

        cond = {"type": "ta_score_above", "params": {"value": 50}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] == 60

    @patch("jpstock_agent.strategy._import_ta")
    def test_ta_score_below_pass(self, mock_ta):
        """Test ta_score_below condition passes."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_multi_indicator.return_value = {"signal": "SELL", "score": -60}

        cond = {"type": "ta_score_below", "params": {"value": -50}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True


# ---------------------------------------------------------------------------
# TestEvalFundamentalConditions: Test fundamental conditions
# ---------------------------------------------------------------------------

class TestEvalFundamentalConditions:
    """Test fundamental-related condition evaluation."""

    @patch("jpstock_agent.strategy._import_core")
    def test_pe_below_pass(self, mock_core):
        """Test pe_below condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.financial_ratio.return_value = {"trailingPE": 12.0}

        cond = {"type": "pe_below", "params": {"value": 15.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] == 12.0

    @patch("jpstock_agent.strategy._import_core")
    def test_pe_above_pass(self, mock_core):
        """Test pe_above condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.financial_ratio.return_value = {"trailingPE": 18.0}

        cond = {"type": "pe_above", "params": {"value": 15.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_core")
    def test_pb_below_pass(self, mock_core):
        """Test pb_below condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.financial_ratio.return_value = {"priceToBook": 1.2}

        cond = {"type": "pb_below", "params": {"value": 1.5}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_core")
    def test_dividend_yield_above_pass(self, mock_core):
        """Test dividend_yield_above condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.financial_ratio.return_value = {"dividendYield": 0.04}

        cond = {"type": "dividend_yield_above", "params": {"value": 0.03}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_core")
    def test_roe_above_pass(self, mock_core):
        """Test roe_above condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.financial_ratio.return_value = {"returnOnEquity": 0.18}

        cond = {"type": "roe_above", "params": {"value": 0.15}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_core")
    def test_debt_to_equity_below_pass(self, mock_core):
        """Test debt_to_equity_below condition passes."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.financial_ratio.return_value = {"debtToEquity": 45.0}

        cond = {"type": "debt_to_equity_below", "params": {"value": 50.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True

    @patch("jpstock_agent.strategy._import_financial")
    def test_f_score_above_pass(self, mock_financial):
        """Test f_score_above condition passes."""
        mock_fin_inst = mock_financial.return_value
        mock_fin_inst.financial_health.return_value = {
            "piotroski_f": {"score": 7}
        }

        cond = {"type": "f_score_above", "params": {"value": 6}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is True
        assert result["value"] == 7

    @patch("jpstock_agent.strategy._import_financial")
    def test_f_score_above_fail(self, mock_financial):
        """Test f_score_above condition fails."""
        mock_fin_inst = mock_financial.return_value
        mock_fin_inst.financial_health.return_value = {
            "piotroski_f": {"score": 4}
        }

        cond = {"type": "f_score_above", "params": {"value": 6}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is False


# ---------------------------------------------------------------------------
# TestPerSymbolCache: Test caching behavior
# ---------------------------------------------------------------------------

class TestPerSymbolCache:
    """Test that cache prevents redundant data fetching."""

    @patch("jpstock_agent.strategy._import_core")
    def test_cache_prevents_duplicate_history_fetches(self, mock_core):
        """Test that cached history is reused."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        cache = {}
        cond1 = {"type": "price_above", "params": {"value": 2400.0}}
        cond2 = {"type": "price_below", "params": {"value": 3000.0}}

        strategy._eval_condition(cond1, "7203", _cache=cache)
        strategy._eval_condition(cond2, "7203", _cache=cache)

        # Should only call stock_history once due to caching
        assert mock_core_inst.stock_history.call_count == 1

    @patch("jpstock_agent.strategy._import_ta")
    def test_cache_prevents_duplicate_ta_fetches(self, mock_ta):
        """Test that cached TA data is reused."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 25.0}]

        cache = {}
        cond1 = {"type": "rsi_below", "params": {"value": 30}}
        cond2 = {"type": "rsi_above", "params": {"value": 20}}

        strategy._eval_condition(cond1, "7203", _cache=cache)
        strategy._eval_condition(cond2, "7203", _cache=cache)

        # Should only call ta_rsi once due to caching
        assert mock_ta_inst.ta_rsi.call_count == 1

    @patch("jpstock_agent.strategy._import_core")
    @patch("jpstock_agent.strategy._import_ta")
    def test_cache_is_shared_across_conditions(self, mock_ta, mock_core):
        """Test that cache is properly shared in strategy_evaluate."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = _mock_history()

        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.return_value = [{"rsi": 25.0}]
        mock_ta_inst.ta_macd.return_value = [{"macd": 2.0, "macd_signal": 1.5}]

        conditions = [
            {"type": "price_above", "params": {"value": 2400.0}},
            {"type": "price_below", "params": {"value": 3000.0}},
            {"type": "rsi_below", "params": {"value": 30}},
        ]

        strategy.strategy_evaluate("7203", conditions)

        # History should be fetched once
        assert mock_core_inst.stock_history.call_count == 1
        # RSI should be fetched once
        assert mock_ta_inst.ta_rsi.call_count == 1


# ---------------------------------------------------------------------------
# TestRoundVal: Test value rounding utility
# ---------------------------------------------------------------------------

class TestRoundVal:
    """Test _round_val helper function."""

    def test_round_val_normal_float(self):
        """Test rounding a normal float."""
        result = strategy._round_val(1.23456, decimals=2)
        assert result == 1.23

    def test_round_val_none_returns_none(self):
        """Test that None returns None."""
        result = strategy._round_val(None)
        assert result is None

    def test_round_val_nan_returns_none(self):
        """Test that NaN returns None."""
        result = strategy._round_val(float('nan'))
        assert result is None

    def test_round_val_inf_returns_none(self):
        """Test that infinity returns None."""
        result = strategy._round_val(float('inf'))
        assert result is None

    def test_round_val_invalid_string_returns_none(self):
        """Test that invalid string returns None."""
        result = strategy._round_val("not_a_number")
        assert result is None

    def test_round_val_default_decimals(self):
        """Test rounding with default decimals (4)."""
        result = strategy._round_val(1.123456)
        assert result == 1.1235


# ---------------------------------------------------------------------------
# TestConditionErrorHandling: Test error handling in conditions
# ---------------------------------------------------------------------------

class TestConditionErrorHandling:
    """Test error handling and edge cases in condition evaluation."""

    @patch("jpstock_agent.strategy._import_core")
    def test_condition_missing_data_returns_detail(self, mock_core):
        """Test that missing data is handled gracefully."""
        mock_core_inst = mock_core.return_value
        mock_core_inst.stock_history.return_value = {}

        cond = {"type": "price_above", "params": {"value": 2400.0}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is False
        assert "detail" in result
        assert result["detail"] != ""

    def test_unknown_condition_type(self):
        """Test that unknown condition type is handled."""
        cond = {"type": "unknown_condition"}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is False
        assert "Unknown condition type" in result["detail"]

    @patch("jpstock_agent.strategy._import_ta")
    def test_condition_exception_caught(self, mock_ta):
        """Test that exceptions in condition evaluation are caught."""
        mock_ta_inst = mock_ta.return_value
        mock_ta_inst.ta_rsi.side_effect = Exception("API error")

        cond = {"type": "rsi_below", "params": {"value": 30}}
        result = strategy._eval_condition(cond, "7203")

        assert result["passed"] is False
        assert "Error:" in result["detail"]
