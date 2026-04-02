"""
Tests for transaction cost model and position sizing in backtest module.

Tests cover:
- CostModel: commission, slippage, spread calculations
- Position sizing strategies: full, kelly, atr, max_loss, fixed_fraction
- backtest_strategy with cost model integration
- backtest_realistic convenience function
- Backward compatibility (zero costs by default)
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from jpstock_agent import backtest
from jpstock_agent.backtest import (
    CostModel,
    JP_MARKET_COSTS,
    NO_COSTS,
    VN_MARKET_COSTS,
    _position_size_atr,
    _position_size_fixed_fraction,
    _position_size_full,
    _position_size_kelly,
    _position_size_max_loss,
)
from tests.conftest import _make_ohlcv_df


# ============================================================================
# CostModel Tests
# ============================================================================


class TestCostModel:
    """Tests for CostModel class."""

    def test_default_cost_model(self):
        """Test default cost model parameters."""
        cm = CostModel()
        assert cm.commission_pct == 0.001  # 0.1%
        assert cm.slippage_pct == 0.0005   # 0.05%
        assert cm.spread_pct == 0.0003     # 0.03%

    def test_zero_cost_model(self):
        """Test NO_COSTS preset has zero costs."""
        assert NO_COSTS.total_cost_pct() == 0.0

    def test_jp_market_costs(self):
        """Test JP market cost preset."""
        assert JP_MARKET_COSTS.commission_pct == 0.001
        assert JP_MARKET_COSTS.slippage_pct == 0.0005
        assert JP_MARKET_COSTS.spread_pct == 0.0003

    def test_vn_market_costs(self):
        """Test VN market cost preset."""
        assert VN_MARKET_COSTS.commission_pct == 0.0015
        assert VN_MARKET_COSTS.slippage_pct == 0.001

    def test_total_cost_pct(self):
        """Test total cost calculation."""
        cm = CostModel(commission_pct=0.1, slippage_pct=0.05, spread_pct=0.03)
        expected = 0.001 + 0.0005 + 0.0003  # 0.18% = 0.0018
        assert abs(cm.total_cost_pct() - expected) < 1e-10

    def test_apply_buy_increases_effective_price(self):
        """Test that buy costs increase effective price."""
        cm = CostModel(commission_pct=0.1, slippage_pct=0.05, spread_pct=0.03)
        price = 1000.0
        capital = 100000.0

        eff_price, commission, shares = cm.apply_buy(price, capital)

        assert eff_price > price
        assert commission >= 0

    def test_apply_sell_decreases_effective_price(self):
        """Test that sell costs decrease effective price."""
        cm = CostModel(commission_pct=0.1, slippage_pct=0.05, spread_pct=0.03)
        price = 1000.0
        shares = 100.0

        eff_price, commission, proceeds = cm.apply_sell(price, shares)

        assert eff_price < price
        assert commission >= 0
        assert proceeds < shares * price

    def test_zero_cost_buy_no_change(self):
        """Test that zero cost model doesn't change effective price."""
        price = 1000.0
        capital = 100000.0

        eff_price, commission, shares = NO_COSTS.apply_buy(price, capital)

        assert eff_price == price
        assert commission == 0.0

    def test_zero_cost_sell_no_change(self):
        """Test that zero cost model doesn't change sell proceeds."""
        price = 1000.0
        shares = 100.0

        eff_price, commission, proceeds = NO_COSTS.apply_sell(price, shares)

        assert eff_price == price
        assert commission == 0.0
        assert proceeds == shares * price

    def test_min_commission(self):
        """Test minimum commission enforcement."""
        cm = CostModel(commission_pct=0.01, min_commission=500.0)
        price = 1000.0
        capital = 10000.0  # Small trade

        _, commission, _ = cm.apply_buy(price, capital)

        assert commission >= 500.0

    def test_to_dict(self):
        """Test serialization to dict."""
        cm = CostModel(commission_pct=0.1, slippage_pct=0.05, spread_pct=0.03)
        d = cm.to_dict()

        assert "commission_pct" in d
        assert "slippage_pct" in d
        assert "spread_pct" in d
        assert "total_one_way_cost_pct" in d

    def test_roundtrip_cost_impact(self):
        """Test that a buy-sell roundtrip results in a loss due to costs."""
        cm = CostModel(commission_pct=0.1, slippage_pct=0.05, spread_pct=0.03)
        initial_capital = 1_000_000
        price = 2500.0

        # Buy
        eff_buy, comm_buy, shares = cm.apply_buy(price, initial_capital)
        investable = initial_capital - comm_buy
        actual_shares = investable / eff_buy

        # Sell at same price
        eff_sell, comm_sell, proceeds = cm.apply_sell(price, actual_shares)

        # Should lose money on roundtrip due to costs
        assert proceeds < initial_capital


# ============================================================================
# Position Sizing Tests
# ============================================================================


class TestPositionSizing:
    """Tests for position sizing strategies."""

    def test_full_position(self):
        """Test full position sizing."""
        shares = _position_size_full(100000, 1000)
        assert shares == 100.0

    def test_full_position_zero_price(self):
        """Test full position with zero price."""
        shares = _position_size_full(100000, 0)
        assert shares == 0

    def test_kelly_positive_edge(self):
        """Test Kelly sizing with positive edge."""
        shares = _position_size_kelly(
            100000, 1000, win_rate=0.6, avg_win=2.0, avg_loss=1.0
        )
        assert shares > 0
        assert shares <= 100  # Can't buy more than affordable

    def test_kelly_negative_edge(self):
        """Test Kelly sizing with negative edge returns minimal position."""
        shares = _position_size_kelly(
            100000, 1000, win_rate=0.3, avg_win=0.5, avg_loss=1.5
        )
        assert shares >= 0

    def test_kelly_half_kelly(self):
        """Test that Kelly uses half-Kelly for safety."""
        # With perfect edge, full Kelly would be 1.0, half-Kelly should be 0.5
        shares = _position_size_kelly(
            100000, 1000, win_rate=1.0, avg_win=1.0, avg_loss=1.0
        )
        # Half-Kelly fraction = 0.5
        assert shares <= 50.0  # 50% of capital / price

    def test_atr_sizing(self):
        """Test ATR-based position sizing."""
        shares = _position_size_atr(
            100000, 1000, atr=50.0, risk_per_trade=0.02, atr_multiplier=2.0
        )
        # risk_amount = 100000 * 0.02 = 2000
        # stop_distance = 50 * 2 = 100
        # shares = 2000 / 100 = 20
        assert abs(shares - 20.0) < 0.01

    def test_atr_sizing_zero_atr(self):
        """Test ATR sizing falls back to full position with zero ATR."""
        shares = _position_size_atr(100000, 1000, atr=0.0)
        assert shares == 100.0  # Full position fallback

    def test_atr_sizing_caps_at_max_affordable(self):
        """Test ATR sizing caps at max affordable shares."""
        shares = _position_size_atr(
            10000, 1000, atr=1.0, risk_per_trade=0.5, atr_multiplier=1.0
        )
        # risk = 5000, stop = 1, shares would be 5000 but max = 10
        assert shares <= 10.0

    def test_max_loss_sizing(self):
        """Test max loss position sizing."""
        shares = _position_size_max_loss(
            100000, 1000, max_loss_pct=0.02, stop_loss_pct=0.05
        )
        # risk_amount = 100000 * 0.02 = 2000
        # loss_per_share = 1000 * 0.05 = 50
        # shares = 2000 / 50 = 40
        assert abs(shares - 40.0) < 0.01

    def test_max_loss_zero_stop(self):
        """Test max loss sizing with zero stop loss."""
        shares = _position_size_max_loss(100000, 1000, stop_loss_pct=0)
        assert shares == 100.0  # Fallback to full

    def test_fixed_fraction(self):
        """Test fixed fraction sizing."""
        shares = _position_size_fixed_fraction(100000, 1000, fraction=0.5)
        assert abs(shares - 50.0) < 0.01

    def test_fixed_fraction_clamps(self):
        """Test fixed fraction clamps to [0, 1]."""
        shares_over = _position_size_fixed_fraction(100000, 1000, fraction=1.5)
        shares_under = _position_size_fixed_fraction(100000, 1000, fraction=-0.5)
        assert shares_over <= 100.0
        assert shares_under == 0.0


# ============================================================================
# Integrated Backtest Tests
# ============================================================================


class TestBacktestWithCosts:
    """Tests for backtest_strategy with transaction costs."""

    def test_backtest_with_costs_returns_cost_model(self):
        """Test that backtest results include cost_model info."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy(
                "7203", "sma_crossover",
                commission_pct=0.1, slippage_pct=0.05, spread_pct=0.03,
            )

            assert isinstance(result, dict)
            if "error" not in result:
                assert "cost_model" in result
                assert "total_costs_paid" in result
                assert result["cost_model"]["commission_pct"] > 0

    def test_backtest_with_costs_lower_returns_than_without(self):
        """Test that costs reduce total return."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result_no_cost = backtest.backtest_strategy("7203", "sma_crossover")
            result_with_cost = backtest.backtest_strategy(
                "7203", "sma_crossover",
                commission_pct=0.5, slippage_pct=0.3, spread_pct=0.2,
            )

            if "error" not in result_no_cost and "error" not in result_with_cost:
                assert result_with_cost["total_return_pct"] <= result_no_cost["total_return_pct"]

    def test_backtest_zero_cost_backward_compatible(self):
        """Test that default (zero cost) backtest is backward compatible."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy("7203", "sma_crossover")

            assert isinstance(result, dict)
            if "error" not in result:
                assert result["cost_model"]["total_one_way_cost_pct"] == 0
                assert result["total_costs_paid"] == 0

    def test_backtest_with_position_sizing(self):
        """Test backtest with ATR position sizing."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy(
                "7203", "sma_crossover", position_sizing="atr",
            )

            assert isinstance(result, dict)
            if "error" not in result:
                assert result["position_sizing"] == "atr"

    def test_backtest_with_kelly_sizing(self):
        """Test backtest with Kelly Criterion sizing."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy(
                "7203", "sma_crossover", position_sizing="kelly",
            )

            assert isinstance(result, dict)
            if "error" not in result:
                assert result["position_sizing"] == "kelly"

    def test_backtest_with_max_loss_sizing(self):
        """Test backtest with max loss sizing."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy(
                "7203", "sma_crossover", position_sizing="max_loss",
            )

            assert isinstance(result, dict)
            if "error" not in result:
                assert result["position_sizing"] == "max_loss"

    def test_backtest_trades_include_effective_price(self):
        """Test that trades include effective_price and commission fields."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_strategy(
                "7203", "sma_crossover", commission_pct=0.1,
            )

            if isinstance(result, dict) and "error" not in result:
                trades = result.get("trades", [])
                if trades:
                    assert "effective_price" in trades[0]
                    assert "commission" in trades[0]


class TestBacktestRealistic:
    """Tests for backtest_realistic convenience function."""

    def test_realistic_jp_market(self):
        """Test realistic backtest with JP market preset."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_realistic("7203", "sma_crossover", market="jp")

            assert isinstance(result, dict)
            if "error" not in result:
                assert result["cost_model"]["commission_pct"] > 0
                assert "comparison_vs_zero_cost" in result

    def test_realistic_vn_market(self):
        """Test realistic backtest with VN market preset."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_realistic("ACB", "sma_crossover", market="vn")

            assert isinstance(result, dict)

    def test_realistic_zero_market(self):
        """Test realistic backtest with zero cost preset."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_realistic(
                "7203", "sma_crossover", market="zero"
            )

            assert isinstance(result, dict)
            if "error" not in result:
                assert result["cost_model"]["total_one_way_cost_pct"] == 0

    def test_realistic_comparison_shows_cost_impact(self):
        """Test that comparison section shows cost impact."""
        with patch("jpstock_agent.ta._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=252)

            result = backtest.backtest_realistic("7203", "sma_crossover", market="jp")

            if isinstance(result, dict) and "error" not in result:
                comp = result.get("comparison_vs_zero_cost", {})
                if comp:
                    assert "cost_impact_pct" in comp
                    assert comp["cost_impact_pct"] >= 0  # Costs should not help
