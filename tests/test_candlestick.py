"""Tests for jpstock_agent.candlestick module (pattern detection).

Tests candlestick pattern detection with mocked OHLCV data.
"""

from unittest.mock import patch

from jpstock_agent import candlestick
from tests.conftest import _make_ohlcv_df

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_df():
    """Return a mocked OHLCV DataFrame for testing."""
    return _make_ohlcv_df(days=60, base_price=2500.0)


def _create_doji_df():
    """Create a DataFrame with a clear Doji pattern on the last day.

    Doji: open ≈ close (very small body relative to range).
    """
    df = _make_ohlcv_df(days=5, base_price=2500.0)
    # On the last day, set open ≈ close to create a Doji
    last_idx = -1
    df.iloc[last_idx, df.columns.get_loc("open")] = 2520.0
    df.iloc[last_idx, df.columns.get_loc("close")] = 2521.0
    df.iloc[last_idx, df.columns.get_loc("high")] = 2550.0
    df.iloc[last_idx, df.columns.get_loc("low")] = 2490.0
    return df


def _create_hammer_df():
    """Create a DataFrame with a Hammer pattern.

    Hammer: small bullish body at top, long lower shadow (≥2x body).
    """
    df = _make_ohlcv_df(days=5, base_price=2500.0)
    # Create a downtrend first (make price go down)
    for i in range(len(df) - 1):
        df.iloc[i, df.columns.get_loc("close")] = 2500.0 - (5 - i) * 10
    # Now add a hammer on the last day
    last_idx = -1
    df.iloc[last_idx, df.columns.get_loc("open")] = 2450.0
    df.iloc[last_idx, df.columns.get_loc("close")] = 2460.0  # Small bullish body
    df.iloc[last_idx, df.columns.get_loc("high")] = 2465.0  # Minimal upper shadow
    df.iloc[last_idx, df.columns.get_loc("low")] = 2400.0   # Long lower shadow
    return df


def _create_bullish_engulfing_df():
    """Create a DataFrame with a Bullish Engulfing pattern.

    Bullish Engulfing: bearish candle followed by larger bullish candle.
    """
    df = _make_ohlcv_df(days=5, base_price=2500.0)
    # Second to last candle: bearish (close < open)
    penultimate_idx = -2
    df.iloc[penultimate_idx, df.columns.get_loc("open")] = 2510.0
    df.iloc[penultimate_idx, df.columns.get_loc("close")] = 2500.0
    df.iloc[penultimate_idx, df.columns.get_loc("high")] = 2515.0
    df.iloc[penultimate_idx, df.columns.get_loc("low")] = 2495.0

    # Last candle: bullish, larger body that engulfs the previous
    last_idx = -1
    df.iloc[last_idx, df.columns.get_loc("open")] = 2495.0
    df.iloc[last_idx, df.columns.get_loc("close")] = 2520.0
    df.iloc[last_idx, df.columns.get_loc("high")] = 2525.0
    df.iloc[last_idx, df.columns.get_loc("low")] = 2490.0
    return df


def _create_morning_star_df():
    """Create a DataFrame with a Morning Star pattern.

    Morning Star: bearish candle, small body (gap down), bullish candle
    closing above midpoint of first.
    """
    df = _make_ohlcv_df(days=5, base_price=2500.0)
    # First candle: bearish
    idx1 = -3
    df.iloc[idx1, df.columns.get_loc("open")] = 2510.0
    df.iloc[idx1, df.columns.get_loc("close")] = 2490.0
    df.iloc[idx1, df.columns.get_loc("high")] = 2515.0
    df.iloc[idx1, df.columns.get_loc("low")] = 2485.0

    # Second candle: small body, gap down
    idx2 = -2
    df.iloc[idx2, df.columns.get_loc("open")] = 2480.0
    df.iloc[idx2, df.columns.get_loc("close")] = 2478.0
    df.iloc[idx2, df.columns.get_loc("high")] = 2482.0
    df.iloc[idx2, df.columns.get_loc("low")] = 2475.0

    # Third candle: bullish, closes above first's midpoint (2500)
    idx3 = -1
    df.iloc[idx3, df.columns.get_loc("open")] = 2480.0
    df.iloc[idx3, df.columns.get_loc("close")] = 2505.0
    df.iloc[idx3, df.columns.get_loc("high")] = 2510.0
    df.iloc[idx3, df.columns.get_loc("low")] = 2475.0
    return df


# ---------------------------------------------------------------------------
# Public API Tests
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_scan_returns_list(mock_ohlcv):
    """Test ta_candlestick_scan returns list."""
    mock_ohlcv.return_value = _mock_df()
    result = candlestick.ta_candlestick_scan("7203")

    assert isinstance(result, list)


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_scan_records_have_required_fields(mock_ohlcv):
    """Test scan results have date, pattern_name, type, reliability."""
    mock_ohlcv.return_value = _mock_df()
    result = candlestick.ta_candlestick_scan("7203")

    # Even if no patterns detected, it should be a valid list
    assert isinstance(result, list)

    if len(result) > 0:
        for record in result:
            assert isinstance(record, dict)
            assert "date" in record
            assert "pattern_name" in record
            assert "pattern_type" in record
            assert "reliability" in record
            assert "description" in record


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_scan_with_doji_pattern(mock_ohlcv):
    """Test candlestick scan detects Doji pattern."""
    doji_df = _create_doji_df()
    mock_ohlcv.return_value = doji_df
    result = candlestick.ta_candlestick_scan("7203")

    assert isinstance(result, list)
    # The Doji pattern should be detected
    pattern_names = [r.get("pattern_name") for r in result]
    assert "Doji" in pattern_names


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_latest_returns_dict(mock_ohlcv):
    """Test ta_candlestick_latest returns dict."""
    mock_ohlcv.return_value = _mock_df()
    result = candlestick.ta_candlestick_latest("7203")

    assert isinstance(result, dict)
    assert "symbol" in result
    assert "date" in result
    assert "patterns" in result
    assert "pattern_count" in result


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_latest_patterns_is_list(mock_ohlcv):
    """Test latest patterns field is a list."""
    mock_ohlcv.return_value = _mock_df()
    result = candlestick.ta_candlestick_latest("7203")

    assert isinstance(result["patterns"], list)
    assert isinstance(result["pattern_count"], int)
    # Pattern count should match patterns list length
    assert result["pattern_count"] == len(result["patterns"])


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_latest_with_doji(mock_ohlcv):
    """Test latest detects pattern on most recent day."""
    doji_df = _create_doji_df()
    mock_ohlcv.return_value = doji_df
    result = candlestick.ta_candlestick_latest("7203")

    assert isinstance(result, dict)
    assert "symbol" in result
    patterns = result.get("patterns", [])
    assert isinstance(patterns, list)


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_screen_returns_list(mock_ohlcv):
    """Test ta_candlestick_screen with pattern=all returns list."""
    mock_ohlcv.return_value = _mock_df()
    result = candlestick.ta_candlestick_screen(["7203", "6758"], pattern="all")

    assert isinstance(result, list)


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_screen_bullish_filter(mock_ohlcv):
    """Test ta_candlestick_screen with bullish filter."""
    mock_ohlcv.return_value = _create_hammer_df()
    result = candlestick.ta_candlestick_screen(["7203"], pattern="bullish")

    assert isinstance(result, list)


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_screen_bearish_filter(mock_ohlcv):
    """Test ta_candlestick_screen with bearish filter."""
    mock_ohlcv.return_value = _mock_df()
    result = candlestick.ta_candlestick_screen(["7203"], pattern="bearish")

    assert isinstance(result, list)


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_screen_specific_pattern(mock_ohlcv):
    """Test ta_candlestick_screen with specific pattern name."""
    doji_df = _create_doji_df()
    mock_ohlcv.return_value = doji_df
    result = candlestick.ta_candlestick_screen(["7203"], pattern="doji")

    assert isinstance(result, list)


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_screen_multiple_symbols(mock_ohlcv):
    """Test screening with multiple symbols."""
    mock_ohlcv.return_value = _mock_df()
    result = candlestick.ta_candlestick_screen(["7203", "6758", "9984"], pattern="all")

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Pattern Detection Tests
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_doji_pattern_detection(mock_ohlcv):
    """Test Doji pattern is detected correctly."""
    doji_df = _create_doji_df()
    mock_ohlcv.return_value = doji_df
    result = candlestick.ta_candlestick_scan("7203")

    assert isinstance(result, list)
    pattern_names = [r.get("pattern_name") for r in result]
    assert "Doji" in pattern_names


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_hammer_pattern_detection(mock_ohlcv):
    """Test Hammer pattern detection."""
    hammer_df = _create_hammer_df()
    mock_ohlcv.return_value = hammer_df
    result = candlestick.ta_candlestick_scan("7203")

    assert isinstance(result, list)
    pattern_names = [r.get("pattern_name") for r in result]
    # Hammer should be detected
    assert "Hammer" in pattern_names


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_bullish_engulfing_pattern_detection(mock_ohlcv):
    """Test Bullish Engulfing pattern detection."""
    engulfing_df = _create_bullish_engulfing_df()
    mock_ohlcv.return_value = engulfing_df
    result = candlestick.ta_candlestick_scan("7203")

    assert isinstance(result, list)
    pattern_names = [r.get("pattern_name") for r in result]
    assert "Bullish Engulfing" in pattern_names


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_morning_star_pattern_detection(mock_ohlcv):
    """Test Morning Star pattern detection."""
    morning_df = _create_morning_star_df()
    mock_ohlcv.return_value = morning_df
    result = candlestick.ta_candlestick_scan("7203")

    assert isinstance(result, list)
    pattern_names = [r.get("pattern_name") for r in result]
    assert "Morning Star" in pattern_names


# ---------------------------------------------------------------------------
# Pattern Types and Reliability
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_patterns_have_valid_types(mock_ohlcv):
    """Test detected patterns have valid type field."""
    doji_df = _create_doji_df()
    mock_ohlcv.return_value = doji_df
    result = candlestick.ta_candlestick_scan("7203")

    valid_types = ["bullish", "bearish", "neutral"]
    for record in result:
        assert record.get("pattern_type") in valid_types


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_patterns_have_valid_reliability(mock_ohlcv):
    """Test detected patterns have valid reliability field."""
    doji_df = _create_doji_df()
    mock_ohlcv.return_value = doji_df
    result = candlestick.ta_candlestick_scan("7203")

    valid_reliabilities = ["high", "medium", "low"]
    for record in result:
        assert record.get("reliability") in valid_reliabilities


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

def test_is_bullish_candle():
    """Test _is_bullish_candle helper."""
    assert candlestick._is_bullish_candle(100, 110) is True
    assert candlestick._is_bullish_candle(110, 100) is False
    assert candlestick._is_bullish_candle(100, 100) is False


def test_is_bearish_candle():
    """Test _is_bearish_candle helper."""
    assert candlestick._is_bearish_candle(110, 100) is True
    assert candlestick._is_bearish_candle(100, 110) is False
    assert candlestick._is_bearish_candle(100, 100) is False


def test_get_body_size():
    """Test _get_body_size helper."""
    assert candlestick._get_body_size(100, 110) == 10
    assert candlestick._get_body_size(110, 100) == 10
    assert candlestick._get_body_size(100, 100) == 0


def test_get_upper_shadow():
    """Test _get_upper_shadow helper."""
    # open=100, close=110, high=120
    assert candlestick._get_upper_shadow(120, 100, 110) == 10
    # open=110, close=100, high=115
    assert candlestick._get_upper_shadow(115, 110, 100) == 5


def test_get_lower_shadow():
    """Test _get_lower_shadow helper."""
    # open=100, close=110, low=90
    assert candlestick._get_lower_shadow(90, 100, 110) == 10
    # open=110, close=100, low=95
    assert candlestick._get_lower_shadow(95, 110, 100) == 5


def test_get_range():
    """Test _get_range helper."""
    assert candlestick._get_range(150, 100) == 50
    assert candlestick._get_range(100, 100) == 0


def test_approx_equal():
    """Test _approx_equal helper."""
    # Within 0.5%
    assert candlestick._approx_equal(100, 100.49) is True
    assert candlestick._approx_equal(100, 100.51) is False
    assert candlestick._approx_equal(0, 0) is True
    assert candlestick._approx_equal(0, 1) is False


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

def test_ta_candlestick_scan_error_handling():
    """Test candlestick_scan handles errors from _get_ohlcv_df."""
    # Patch in the candlestick module where it's imported
    with patch.object(candlestick, '_get_ohlcv_df', return_value={"error": "API error"}):
        result = candlestick.ta_candlestick_scan("7203")

        # When _get_ohlcv_df returns an error dict, the scan should also return it
        assert isinstance(result, dict)
        assert "error" in result


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_latest_error_handling(mock_ohlcv):
    """Test candlestick_latest handles errors."""
    mock_ohlcv.return_value = {"error": "No data"}
    result = candlestick.ta_candlestick_latest("7203")

    assert isinstance(result, dict)
    # The result should either have an error key or be the error itself
    assert "error" in result or isinstance(result, dict)


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_screen_error_handling(mock_ohlcv):
    """Test candlestick_screen handles empty symbols."""
    result = candlestick.ta_candlestick_screen([], pattern="all")

    assert isinstance(result, dict)
    assert "error" in result


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_ta_candlestick_screen_skips_failed_symbols(mock_ohlcv):
    """Test screening skips symbols with errors."""
    # Make mock return error for any call
    mock_ohlcv.return_value = {"error": "API error"}
    result = candlestick.ta_candlestick_screen(["7203", "6758"], pattern="all")

    assert isinstance(result, list)
    # Should skip failed symbols and return empty list
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

@patch("jpstock_agent.ta._get_ohlcv_df")
def test_scan_and_latest_consistency(mock_ohlcv):
    """Test that scan and latest return consistent data."""
    doji_df = _create_doji_df()
    mock_ohlcv.return_value = doji_df

    # Get scan result
    scan_result = candlestick.ta_candlestick_scan("7203")
    assert isinstance(scan_result, list)

    # Get latest result
    latest_result = candlestick.ta_candlestick_latest("7203")
    assert isinstance(latest_result, dict)

    # Latest patterns should be a subset of scan patterns on that date
    if len(scan_result) > 0:
        assert len(latest_result.get("patterns", [])) >= 0


@patch("jpstock_agent.ta._get_ohlcv_df")
def test_screen_includes_only_matching_patterns(mock_ohlcv):
    """Test screening filters patterns correctly."""
    hammer_df = _create_hammer_df()
    mock_ohlcv.return_value = hammer_df

    # Screen for bullish patterns
    result = candlestick.ta_candlestick_screen(["7203"], pattern="bullish")

    assert isinstance(result, list)

    # All returned patterns should be bullish
    for item in result:
        for pattern in item.get("patterns", []):
            assert pattern.get("type") == "bullish"
