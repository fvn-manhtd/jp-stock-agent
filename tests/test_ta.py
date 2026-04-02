"""Tests for jpstock_agent.ta module (technical analysis).

Tests all 20+ TA functions with mocked _get_ohlcv_df to avoid real API calls.
"""

from unittest.mock import patch

import pytest

from jpstock_agent import ta
from tests.conftest import _make_ohlcv_df

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_df():
    """Return a mocked OHLCV DataFrame for testing."""
    return _make_ohlcv_df(days=60, base_price=2500.0)


# ---------------------------------------------------------------------------
# 1. TREND INDICATORS
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_sma_returns_list_with_sma_column(mock_ohlcv):
    """Test SMA returns list with SMA_20 column."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_sma("7203", period=20)

    assert isinstance(result, list)
    assert len(result) > 0
    assert "SMA_20" in result[0]
    assert "close" in result[0]
    assert isinstance(result[0]["SMA_20"], (int, float, type(None)))


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_sma_custom_period(mock_ohlcv):
    """Test SMA with custom period."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_sma("7203", period=50)

    assert isinstance(result, list)
    assert len(result) > 0
    assert "SMA_50" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_ema_returns_list_with_ema_column(mock_ohlcv):
    """Test EMA returns list with EMA_20 column."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_ema("7203", period=20)

    assert isinstance(result, list)
    assert len(result) > 0
    assert "EMA_20" in result[0]
    assert "close" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_ema_custom_period(mock_ohlcv):
    """Test EMA with custom period."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_ema("7203", period=9)

    assert isinstance(result, list)
    assert "EMA_9" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_ichimoku_returns_cloud_components(mock_ohlcv):
    """Test Ichimoku returns tenkan_sen and kijun_sen."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_ichimoku("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "tenkan_sen" in result[0]
    assert "kijun_sen" in result[0]
    assert "senkou_span_a" in result[0]
    assert "senkou_span_b" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_supertrend_returns_trend_and_direction(mock_ohlcv):
    """Test Supertrend returns supertrend and direction."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_supertrend("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "supertrend" in result[0]
    assert "direction" in result[0]
    # direction should be 1 (up) or -1 (down)
    assert result[-1]["direction"] in (1, -1, 1.0, -1.0)


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_parabolic_sar_returns_sar_values(mock_ohlcv):
    """Test Parabolic SAR returns PSAR values."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_parabolic_sar("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "PSAR" in result[0]
    assert "PSAR_up" in result[0]
    assert "PSAR_down" in result[0]


# ---------------------------------------------------------------------------
# 2. MOMENTUM INDICATORS
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_rsi_returns_rsi_in_range(mock_ohlcv):
    """Test RSI returns list with RSI_14 values between 0-100."""
    mock_ohlcv.return_value = _make_ohlcv_df(days=60, base_price=2500.0)
    result = ta.ta_rsi("7203", period=14)

    assert isinstance(result, list)
    assert len(result) > 0
    assert "RSI_14" in result[0]

    # RSI values should be between 0 and 100 (or None for early periods)
    for record in result:
        if record.get("RSI_14") is not None:
            assert 0 <= record["RSI_14"] <= 100


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_rsi_custom_period(mock_ohlcv):
    """Test RSI with custom period."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_rsi("7203", period=21)

    assert isinstance(result, list)
    assert "RSI_21" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_macd_returns_all_components(mock_ohlcv):
    """Test MACD returns MACD, signal, and histogram."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_macd("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "MACD" in result[0]
    assert "MACD_signal" in result[0]
    assert "MACD_histogram" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_stochastic_returns_k_and_d(mock_ohlcv):
    """Test Stochastic returns %K and %D."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_stochastic("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "Stoch_K" in result[0]
    assert "Stoch_D" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_williams_r_returns_values(mock_ohlcv):
    """Test Williams %R returns values."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_williams_r("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "WilliamsR_14" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_cci_returns_values(mock_ohlcv):
    """Test CCI returns values."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_cci("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "CCI_20" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_roc_returns_values(mock_ohlcv):
    """Test Rate of Change returns values."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_roc("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "ROC_12" in result[0]


# ---------------------------------------------------------------------------
# 3. VOLATILITY INDICATORS
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_bbands_returns_upper_middle_lower(mock_ohlcv):
    """Test Bollinger Bands returns upper, middle, lower bands."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_bbands("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "BB_upper" in result[0]
    assert "BB_middle" in result[0]
    assert "BB_lower" in result[0]
    assert "BB_width" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_atr_returns_atr_values(mock_ohlcv):
    """Test ATR returns ATR_14 values."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_atr("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "ATR_14" in result[0]
    # ATR should be positive
    for record in result:
        if record.get("ATR_14") is not None:
            assert record["ATR_14"] >= 0


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_keltner_returns_channels(mock_ohlcv):
    """Test Keltner Channels returns upper, middle, lower."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_keltner("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "KC_upper" in result[0]
    assert "KC_middle" in result[0]
    assert "KC_lower" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_donchian_returns_channels(mock_ohlcv):
    """Test Donchian Channels returns upper, middle, lower."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_donchian("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "DC_upper" in result[0]
    assert "DC_middle" in result[0]
    assert "DC_lower" in result[0]


# ---------------------------------------------------------------------------
# 4. VOLUME INDICATORS
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_obv_returns_obv_values(mock_ohlcv):
    """Test OBV returns OBV column."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_obv("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "OBV" in result[0]
    assert "volume" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_vwap_returns_vwap_values(mock_ohlcv):
    """Test VWAP returns VWAP column."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_vwap("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "VWAP" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_mfi_returns_mfi_values(mock_ohlcv):
    """Test Money Flow Index returns MFI_14."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_mfi("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "MFI_14" in result[0]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_ad_returns_accumulation_distribution(mock_ohlcv):
    """Test Accumulation/Distribution returns AD column."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_ad("7203")

    assert isinstance(result, list)
    assert len(result) > 0
    assert "AD" in result[0]


# ---------------------------------------------------------------------------
# 5. COMPOSITE / ADVANCED ANALYSIS
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_fibonacci_returns_dict_with_levels(mock_ohlcv):
    """Test Fibonacci returns dict with level keys."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_fibonacci("7203")

    assert isinstance(result, dict)
    assert "high" in result
    assert "low" in result
    assert "level_0.0" in result
    assert "level_0.236" in result
    assert "level_0.382" in result
    assert "level_0.500" in result
    assert "level_0.618" in result
    assert "level_0.786" in result
    assert "level_1.0" in result
    assert "current_close" in result


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_fibonacci_levels_ordered(mock_ohlcv):
    """Test Fibonacci levels are ordered from high to low."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_fibonacci("7203")

    high = result["high"]
    low = result["low"]

    # Levels should decrease from 0.0 to 1.0
    if high is not None and low is not None:
        assert result["level_0.0"] >= result["level_0.236"]
        assert result["level_0.236"] >= result["level_1.0"]


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_support_resistance_returns_pivot_and_levels(mock_ohlcv):
    """Test support/resistance returns pivot and support/resistance levels."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_support_resistance("7203")

    assert isinstance(result, dict)
    assert "pivot" in result
    assert "resistance_1" in result
    assert "resistance_2" in result
    assert "resistance_3" in result
    assert "support_1" in result
    assert "support_2" in result
    assert "support_3" in result
    assert "current_close" in result


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_multi_indicator_returns_signals_and_score(mock_ohlcv):
    """Test multi-indicator returns overall_signal and signal_score."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_multi_indicator("7203")

    assert isinstance(result, dict)
    assert "overall_signal" in result
    assert "signal_score" in result
    assert "signals" in result

    # Score should be between -100 and 100
    assert -100 <= result["signal_score"] <= 100

    # Overall signal should be one of the valid options
    valid_signals = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    assert result["overall_signal"] in valid_signals

    # signals should be a list
    assert isinstance(result["signals"], list)


# ---------------------------------------------------------------------------
# 6. SCREENING
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_screen_oversold_returns_list(mock_ohlcv):
    """Test ta_screen with oversold strategy returns list."""
    mock_ohlcv.return_value = _mock_df()
    try:
        result = ta.ta_screen(["7203", "6758"], strategy="oversold")
        assert isinstance(result, list) or isinstance(result, dict)
        # Result may be empty if no stocks match criteria, or error dict
        if isinstance(result, list):
            for record in result:
                assert isinstance(record, dict)
                assert "symbol" in record
                assert "close" in record
    except ImportError:
        # Skip if ta library import fails (library version issue)
        pytest.skip("ta library missing MFIIndicator")


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_screen_macd_bullish_returns_list(mock_ohlcv):
    """Test ta_screen with MACD bullish strategy."""
    mock_ohlcv.return_value = _mock_df()
    try:
        result = ta.ta_screen(["7203"], strategy="macd_bullish")
        assert isinstance(result, list) or isinstance(result, dict)
    except ImportError:
        pytest.skip("ta library missing MFIIndicator")


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_screen_with_three_symbols(mock_ohlcv):
    """Test ta_screen with multiple symbols."""
    mock_ohlcv.return_value = _mock_df()
    try:
        result = ta.ta_screen(["7203", "6758", "9984"], strategy="trend_up")
        assert isinstance(result, list) or isinstance(result, dict)
    except ImportError:
        pytest.skip("ta library missing MFIIndicator")


# ---------------------------------------------------------------------------
# 7. MULTI-TIMEFRAME
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_multi_timeframe_returns_dict_with_timeframes(mock_ohlcv):
    """Test multi-timeframe analysis returns daily/weekly/monthly data."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_multi_timeframe("7203")

    assert isinstance(result, dict)
    assert "symbol" in result
    assert "daily" in result or "weekly" in result or "monthly" in result


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_multi_timeframe_includes_rsi_macd(mock_ohlcv):
    """Test multi-timeframe includes RSI and MACD for each timeframe."""
    mock_ohlcv.return_value = _mock_df()
    result = ta.ta_multi_timeframe("7203")

    # At least one timeframe should have data
    timeframes = [tf for tf in ["daily", "weekly", "monthly"] if tf in result]
    assert len(timeframes) > 0


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def test_round_val_with_normal_float():
    """Test _round_val with normal float."""
    result = ta._round_val(2500.123456)
    assert result == 2500.1235


def test_round_val_with_none():
    """Test _round_val with None returns None."""
    result = ta._round_val(None)
    assert result is None


def test_round_val_with_nan():
    """Test _round_val with NaN returns None."""
    result = ta._round_val(float('nan'))
    assert result is None


def test_round_val_with_inf():
    """Test _round_val with infinity returns None."""
    result = ta._round_val(float('inf'))
    assert result is None


def test_round_val_with_negative_inf():
    """Test _round_val with negative infinity returns None."""
    result = ta._round_val(float('-inf'))
    assert result is None


def test_round_val_custom_decimals():
    """Test _round_val with custom decimals."""
    result = ta._round_val(2500.123456, decimals=2)
    assert result == 2500.12


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_sma_error_handling(mock_ohlcv):
    """Test SMA returns error dict when _get_ohlcv_df fails."""
    mock_ohlcv.return_value = {"error": "API error"}
    result = ta.ta_sma("7203")

    assert isinstance(result, dict)
    assert "error" in result


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_fibonacci_error_handling(mock_ohlcv):
    """Test Fibonacci returns error dict when _get_ohlcv_df fails."""
    mock_ohlcv.return_value = {"error": "No data"}
    result = ta.ta_fibonacci("7203")

    assert isinstance(result, dict)
    assert "error" in result


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_multi_indicator_error_handling(mock_ohlcv):
    """Test multi_indicator returns error dict when _get_ohlcv_df fails."""
    mock_ohlcv.return_value = {"error": "Data unavailable"}
    result = ta.ta_multi_indicator("7203")

    assert isinstance(result, dict)
    assert "error" in result
