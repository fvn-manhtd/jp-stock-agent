"""Candlestick pattern detection module for jpstock-agent.

Detects 20 Japanese candlestick patterns from OHLCV data.

Patterns include:
  - Single Candle (Bullish): Hammer, Inverted Hammer, Dragonfly Doji, Bullish Marubozu
  - Single Candle (Bearish): Hanging Man, Shooting Star, Gravestone Doji, Bearish Marubozu
  - Single Candle (Neutral): Doji, Spinning Top, High Wave
  - Two Candles: Bullish Engulfing, Bearish Engulfing, Tweezer Top, Tweezer Bottom, Piercing Line
  - Three Candles: Morning Star, Evening Star, Three White Soldiers, Three Black Crows

Inspired by TradingView's candlestick patterns (15 patterns) but extends to 20.

Every public function returns ``list[dict]`` or ``dict`` on success,
or ``{"error": str}`` on failure.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .core import _df_to_records, _safe_call
from .ta import _get_ohlcv_df, _round_val

# ---------------------------------------------------------------------------
# Candlestick pattern detection thresholds
# ---------------------------------------------------------------------------

# Body size thresholds (as fraction of candle range = high - low)
DOJI_THRESHOLD = 0.1  # body_size < 10% of range
SMALL_BODY_THRESHOLD = 0.3  # body_size < 30% of range
LONG_SHADOW_RATIO = 2.0  # shadow >= 2.0 * body_size
MARUBOZU_SHADOW_RATIO = 0.05  # both shadows < 5% of range
APPROX_EQUAL_THRESHOLD = 0.005  # within 0.5% of price

# Pattern types
PATTERN_TYPE_BULLISH = "bullish"
PATTERN_TYPE_BEARISH = "bearish"
PATTERN_TYPE_NEUTRAL = "neutral"

# Reliability levels
RELIABILITY_HIGH = "high"
RELIABILITY_MEDIUM = "medium"
RELIABILITY_LOW = "low"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_bullish_candle(open_p: float, close_p: float) -> bool:
    """Return True if candle is bullish (close > open)."""
    return close_p > open_p


def _is_bearish_candle(open_p: float, close_p: float) -> bool:
    """Return True if candle is bearish (close < open)."""
    return close_p < open_p


def _get_body_size(open_p: float, close_p: float) -> float:
    """Return absolute body size (distance between open and close)."""
    return abs(close_p - open_p)


def _get_upper_shadow(high_p: float, open_p: float, close_p: float) -> float:
    """Return upper shadow size (distance from max(open, close) to high)."""
    return max(0.0, high_p - max(open_p, close_p))


def _get_lower_shadow(low_p: float, open_p: float, close_p: float) -> float:
    """Return lower shadow size (distance from low to min(open, close))."""
    return max(0.0, min(open_p, close_p) - low_p)


def _get_range(high_p: float, low_p: float) -> float:
    """Return candle range (high - low)."""
    return high_p - low_p


def _approx_equal(a: float, b: float, threshold: float = APPROX_EQUAL_THRESHOLD) -> bool:
    """Return True if two values are approximately equal (within threshold %)."""
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) <= threshold


def _is_in_uptrend(df: pd.DataFrame, idx: int, period: int = 5) -> bool:
    """Return True if price is in uptrend (close > SMA5 for last 3 periods)."""
    if idx < period:
        return False
    sma = df["close"].iloc[idx - period + 1 : idx + 1].mean()
    return df["close"].iloc[idx] > sma


def _is_in_downtrend(df: pd.DataFrame, idx: int, period: int = 5) -> bool:
    """Return True if price is in downtrend (close < SMA5 for last 3 periods)."""
    if idx < period:
        return False
    sma = df["close"].iloc[idx - period + 1 : idx + 1].mean()
    return df["close"].iloc[idx] < sma


def _get_ohlcv_row(df: pd.DataFrame, idx: int) -> tuple[float, float, float, float, float]:
    """Return (open, high, low, close, volume) for row at idx."""
    return (
        float(df["open"].iloc[idx]),
        float(df["high"].iloc[idx]),
        float(df["low"].iloc[idx]),
        float(df["close"].iloc[idx]),
        float(df["volume"].iloc[idx]) if "volume" in df.columns else 0.0,
    )


# ---------------------------------------------------------------------------
# Single candle pattern detection
# ---------------------------------------------------------------------------


def _detect_doji(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Doji pattern: open ≈ close (very small body relative to range)."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    if body_size < DOJI_THRESHOLD * candle_range and _approx_equal(open_p, close_p):
        return {
            "name": "Doji",
            "type": PATTERN_TYPE_NEUTRAL,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Open and close are nearly equal with small body",
        }
    return None


def _detect_spinning_top(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Spinning Top: small body, similar upper and lower shadows."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    if (body_size < SMALL_BODY_THRESHOLD * candle_range and
            upper_shadow > 0 and lower_shadow > 0 and
            _approx_equal(upper_shadow, lower_shadow, 0.2)):
        return {
            "name": "Spinning Top",
            "type": PATTERN_TYPE_NEUTRAL,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Small body with roughly equal upper and lower shadows",
        }
    return None


def _detect_high_wave(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect High Wave: very small body, very long shadows on both sides."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    if (body_size < DOJI_THRESHOLD * candle_range and
            upper_shadow >= LONG_SHADOW_RATIO * body_size and
            lower_shadow >= LONG_SHADOW_RATIO * body_size):
        return {
            "name": "High Wave",
            "type": PATTERN_TYPE_NEUTRAL,
            "reliability": RELIABILITY_LOW,
            "description": "Very small body with very long shadows on both sides",
        }
    return None


def _detect_hammer(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Hammer: small body at top, long lower shadow (≥2x body), little/no upper shadow.
    Must be in downtrend or at support level."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0 or idx < 1:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    is_bullish = _is_bullish_candle(open_p, close_p)
    body_at_top = close_p > open_p if is_bullish else open_p > close_p

    if (is_bullish and body_size < SMALL_BODY_THRESHOLD * candle_range and
            lower_shadow >= LONG_SHADOW_RATIO * body_size and
            upper_shadow < 0.5 * body_size and
            body_at_top):
        # Check if previous candles show downtrend
        in_downtrend = _is_in_downtrend(df, idx - 1) if idx > 0 else True
        return {
            "name": "Hammer",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_HIGH if in_downtrend else RELIABILITY_MEDIUM,
            "description": "Small bullish body at top with long lower shadow, reversal signal",
        }
    return None


def _detect_inverted_hammer(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Inverted Hammer: small body at bottom, long upper shadow, little/no lower shadow."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0 or idx < 1:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    is_bearish = _is_bearish_candle(open_p, close_p)
    body_at_bottom = close_p < open_p if is_bearish else open_p < close_p

    if (is_bearish and body_size < SMALL_BODY_THRESHOLD * candle_range and
            upper_shadow >= LONG_SHADOW_RATIO * body_size and
            lower_shadow < 0.5 * body_size and
            body_at_bottom):
        in_downtrend = _is_in_downtrend(df, idx - 1) if idx > 0 else True
        return {
            "name": "Inverted Hammer",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Small bearish body at bottom with long upper shadow",
        }
    return None


def _detect_dragonfly_doji(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Dragonfly Doji: open ≈ close ≈ high, long lower shadow."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    if (_approx_equal(open_p, close_p) and _approx_equal(close_p, high_p) and
            body_size < DOJI_THRESHOLD * candle_range and
            lower_shadow >= LONG_SHADOW_RATIO * body_size):
        return {
            "name": "Dragonfly Doji",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Doji with long lower shadow, potential reversal",
        }
    return None


def _detect_gravestone_doji(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Gravestone Doji: open ≈ close ≈ low, long upper shadow."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)

    if (_approx_equal(open_p, close_p) and _approx_equal(close_p, low_p) and
            body_size < DOJI_THRESHOLD * candle_range and
            upper_shadow >= LONG_SHADOW_RATIO * body_size):
        return {
            "name": "Gravestone Doji",
            "type": PATTERN_TYPE_BEARISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Doji with long upper shadow, bearish reversal",
        }
    return None


def _detect_bullish_marubozu(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Bullish Marubozu: large bullish candle, no shadows (open≈low, close≈high)."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    if (_is_bullish_candle(open_p, close_p) and
            body_size > 0.7 * candle_range and
            upper_shadow < MARUBOZU_SHADOW_RATIO * candle_range and
            lower_shadow < MARUBOZU_SHADOW_RATIO * candle_range and
            _approx_equal(open_p, low_p) and _approx_equal(close_p, high_p)):
        return {
            "name": "Bullish Marubozu",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_HIGH,
            "description": "Large bullish candle with no shadows, strong uptrend",
        }
    return None


def _detect_bearish_marubozu(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Bearish Marubozu: large bearish candle, no shadows (open≈high, close≈low)."""
    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    if (_is_bearish_candle(open_p, close_p) and
            body_size > 0.7 * candle_range and
            upper_shadow < MARUBOZU_SHADOW_RATIO * candle_range and
            lower_shadow < MARUBOZU_SHADOW_RATIO * candle_range and
            _approx_equal(open_p, high_p) and _approx_equal(close_p, low_p)):
        return {
            "name": "Bearish Marubozu",
            "type": PATTERN_TYPE_BEARISH,
            "reliability": RELIABILITY_HIGH,
            "description": "Large bearish candle with no shadows, strong downtrend",
        }
    return None


def _detect_hanging_man(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Hanging Man: same shape as hammer but appears in uptrend."""
    if idx < 1:
        return None

    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    # Shape is like hammer (bullish), but in uptrend
    is_bullish = _is_bullish_candle(open_p, close_p)
    in_uptrend = _is_in_uptrend(df, idx - 1)

    if (is_bullish and body_size < SMALL_BODY_THRESHOLD * candle_range and
            lower_shadow >= LONG_SHADOW_RATIO * body_size and
            upper_shadow < 0.5 * body_size and
            in_uptrend):
        return {
            "name": "Hanging Man",
            "type": PATTERN_TYPE_BEARISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Hammer shape in uptrend, potential bearish reversal",
        }
    return None


def _detect_shooting_star(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Shooting Star: same as inverted hammer but in uptrend."""
    if idx < 1:
        return None

    open_p, high_p, low_p, close_p, _ = _get_ohlcv_row(df, idx)
    candle_range = _get_range(high_p, low_p)
    if candle_range == 0:
        return None

    body_size = _get_body_size(open_p, close_p)
    upper_shadow = _get_upper_shadow(high_p, open_p, close_p)
    lower_shadow = _get_lower_shadow(low_p, open_p, close_p)

    is_bearish = _is_bearish_candle(open_p, close_p)
    in_uptrend = _is_in_uptrend(df, idx - 1)

    if (is_bearish and body_size < SMALL_BODY_THRESHOLD * candle_range and
            upper_shadow >= LONG_SHADOW_RATIO * body_size and
            lower_shadow < 0.5 * body_size and
            in_uptrend):
        return {
            "name": "Shooting Star",
            "type": PATTERN_TYPE_BEARISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Inverted hammer in uptrend, bearish reversal signal",
        }
    return None


# ---------------------------------------------------------------------------
# Two candle pattern detection
# ---------------------------------------------------------------------------


def _detect_bullish_engulfing(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Bullish Engulfing: bearish candle followed by larger bullish candle."""
    if idx < 1:
        return None

    o1, h1, l1, c1, _ = _get_ohlcv_row(df, idx - 1)
    o2, h2, l2, c2, _ = _get_ohlcv_row(df, idx)

    # First candle must be bearish, second must be bullish
    if not (_is_bearish_candle(o1, c1) and _is_bullish_candle(o2, c2)):
        return None

    # Second candle engulfs first: open < first close AND close > first open
    if o2 <= c1 and c2 >= o1:
        return {
            "name": "Bullish Engulfing",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_HIGH,
            "description": "Bearish candle engulfed by larger bullish candle, strong reversal",
        }
    return None


def _detect_bearish_engulfing(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Bearish Engulfing: bullish candle followed by larger bearish candle."""
    if idx < 1:
        return None

    o1, h1, l1, c1, _ = _get_ohlcv_row(df, idx - 1)
    o2, h2, l2, c2, _ = _get_ohlcv_row(df, idx)

    # First candle must be bullish, second must be bearish
    if not (_is_bullish_candle(o1, c1) and _is_bearish_candle(o2, c2)):
        return None

    # Second candle engulfs first: open > first close AND close < first open
    if o2 >= c1 and c2 <= o1:
        return {
            "name": "Bearish Engulfing",
            "type": PATTERN_TYPE_BEARISH,
            "reliability": RELIABILITY_HIGH,
            "description": "Bullish candle engulfed by larger bearish candle, strong reversal",
        }
    return None


def _detect_tweezer_top(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Tweezer Top: two candles with same/similar highs at top of uptrend."""
    if idx < 1:
        return None

    _, h1, _, _, _ = _get_ohlcv_row(df, idx - 1)
    _, h2, _, _, _ = _get_ohlcv_row(df, idx)

    in_uptrend = _is_in_uptrend(df, idx)
    if in_uptrend and _approx_equal(h1, h2, 0.01):
        return {
            "name": "Tweezer Top",
            "type": PATTERN_TYPE_BEARISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Two candles with equal highs at uptrend peak, reversal signal",
        }
    return None


def _detect_tweezer_bottom(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Tweezer Bottom: two candles with same/similar lows at bottom of downtrend."""
    if idx < 1:
        return None

    _, _, l1, _, _ = _get_ohlcv_row(df, idx - 1)
    _, _, l2, _, _ = _get_ohlcv_row(df, idx)

    in_downtrend = _is_in_downtrend(df, idx)
    if in_downtrend and _approx_equal(l1, l2, 0.01):
        return {
            "name": "Tweezer Bottom",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Two candles with equal lows at downtrend bottom, reversal signal",
        }
    return None


def _detect_piercing_line(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Piercing Line: bearish candle, then bullish candle opening below prev low,
    closing above 50% of prev body."""
    if idx < 1:
        return None

    o1, h1, l1, c1, _ = _get_ohlcv_row(df, idx - 1)
    o2, h2, l2, c2, _ = _get_ohlcv_row(df, idx)

    if not (_is_bearish_candle(o1, c1) and _is_bullish_candle(o2, c2)):
        return None

    midpoint = (o1 + c1) / 2.0
    if o2 < l1 and c2 > midpoint and c2 < o1:
        return {
            "name": "Piercing Line",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Bullish candle pierces 50% of previous bearish candle body",
        }
    return None


# ---------------------------------------------------------------------------
# Three candle pattern detection
# ---------------------------------------------------------------------------


def _detect_morning_star(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Morning Star: bearish candle, small body (gap down), bullish candle
    closing above midpoint of first."""
    if idx < 2:
        return None

    o1, h1, l1, c1, _ = _get_ohlcv_row(df, idx - 2)
    o2, h2, l2, c2, _ = _get_ohlcv_row(df, idx - 1)
    o3, h3, l3, c3, _ = _get_ohlcv_row(df, idx)

    if not _is_bearish_candle(o1, c1):
        return None

    body_size2 = _get_body_size(o2, c2)
    range2 = _get_range(h2, l2)

    # Second candle is small (gap down and small)
    if not (l2 < l1 and body_size2 < SMALL_BODY_THRESHOLD * max(1.0, range2)):
        return None

    # Third is bullish, closing above first's midpoint
    if _is_bullish_candle(o3, c3) and c3 > (o1 + c1) / 2.0:
        return {
            "name": "Morning Star",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Three-candle reversal pattern at bottom of downtrend",
        }
    return None


def _detect_evening_star(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Evening Star: bullish candle, small body (gap up), bearish candle
    closing below midpoint of first."""
    if idx < 2:
        return None

    o1, h1, l1, c1, _ = _get_ohlcv_row(df, idx - 2)
    o2, h2, l2, c2, _ = _get_ohlcv_row(df, idx - 1)
    o3, h3, l3, c3, _ = _get_ohlcv_row(df, idx)

    if not _is_bullish_candle(o1, c1):
        return None

    body_size2 = _get_body_size(o2, c2)
    range2 = _get_range(h2, l2)

    # Second candle is small (gap up and small)
    if not (h2 > h1 and body_size2 < SMALL_BODY_THRESHOLD * max(1.0, range2)):
        return None

    # Third is bearish, closing below first's midpoint
    if _is_bearish_candle(o3, c3) and c3 < (o1 + c1) / 2.0:
        return {
            "name": "Evening Star",
            "type": PATTERN_TYPE_BEARISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Three-candle reversal pattern at top of uptrend",
        }
    return None


def _detect_three_white_soldiers(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Three White Soldiers: three consecutive bullish candles, each opening within
    prev body, closing progressively higher."""
    if idx < 2:
        return None

    o1, h1, l1, c1, _ = _get_ohlcv_row(df, idx - 2)
    o2, h2, l2, c2, _ = _get_ohlcv_row(df, idx - 1)
    o3, h3, l3, c3, _ = _get_ohlcv_row(df, idx)

    # All three must be bullish
    if not (_is_bullish_candle(o1, c1) and _is_bullish_candle(o2, c2) and _is_bullish_candle(o3, c3)):
        return None

    # Each opens within previous body, closes higher
    if (o2 >= o1 and o2 <= c1 and c2 > c1 and
            o3 >= o2 and o3 <= c2 and c3 > c2):
        return {
            "name": "Three White Soldiers",
            "type": PATTERN_TYPE_BULLISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Three consecutive bullish candles with progressive closes higher",
        }
    return None


def _detect_three_black_crows(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    """Detect Three Black Crows: three consecutive bearish candles, each opening within
    prev body, closing progressively lower."""
    if idx < 2:
        return None

    o1, h1, l1, c1, _ = _get_ohlcv_row(df, idx - 2)
    o2, h2, l2, c2, _ = _get_ohlcv_row(df, idx - 1)
    o3, h3, l3, c3, _ = _get_ohlcv_row(df, idx)

    # All three must be bearish
    if not (_is_bearish_candle(o1, c1) and _is_bearish_candle(o2, c2) and _is_bearish_candle(o3, c3)):
        return None

    # Each opens within previous body, closes lower
    if (o2 <= o1 and o2 >= c1 and c2 < c1 and
            o3 <= o2 and o3 >= c2 and c3 < c2):
        return {
            "name": "Three Black Crows",
            "type": PATTERN_TYPE_BEARISH,
            "reliability": RELIABILITY_MEDIUM,
            "description": "Three consecutive bearish candles with progressive closes lower",
        }
    return None


# ---------------------------------------------------------------------------
# Pattern detection aggregator
# ---------------------------------------------------------------------------


def _detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'patterns' column (list of pattern dicts per row)."""
    if df.empty or len(df) < 3:
        df["patterns"] = [[] for _ in range(len(df))]
        return df

    patterns_list = []

    for idx in range(len(df)):
        patterns = []

        # Single candle patterns
        if doji := _detect_doji(df, idx):
            patterns.append(doji)
        if spinning_top := _detect_spinning_top(df, idx):
            patterns.append(spinning_top)
        if high_wave := _detect_high_wave(df, idx):
            patterns.append(high_wave)
        if hammer := _detect_hammer(df, idx):
            patterns.append(hammer)
        if inv_hammer := _detect_inverted_hammer(df, idx):
            patterns.append(inv_hammer)
        if dragonfly := _detect_dragonfly_doji(df, idx):
            patterns.append(dragonfly)
        if gravestone := _detect_gravestone_doji(df, idx):
            patterns.append(gravestone)
        if bull_marubozu := _detect_bullish_marubozu(df, idx):
            patterns.append(bull_marubozu)
        if bear_marubozu := _detect_bearish_marubozu(df, idx):
            patterns.append(bear_marubozu)
        if hanging_man := _detect_hanging_man(df, idx):
            patterns.append(hanging_man)
        if shooting_star := _detect_shooting_star(df, idx):
            patterns.append(shooting_star)

        # Two candle patterns
        if idx >= 1:
            if bull_eng := _detect_bullish_engulfing(df, idx):
                patterns.append(bull_eng)
            if bear_eng := _detect_bearish_engulfing(df, idx):
                patterns.append(bear_eng)
            if tweezer_top := _detect_tweezer_top(df, idx):
                patterns.append(tweezer_top)
            if tweezer_bottom := _detect_tweezer_bottom(df, idx):
                patterns.append(tweezer_bottom)
            if piercing := _detect_piercing_line(df, idx):
                patterns.append(piercing)

        # Three candle patterns
        if idx >= 2:
            if morning := _detect_morning_star(df, idx):
                patterns.append(morning)
            if evening := _detect_evening_star(df, idx):
                patterns.append(evening)
            if three_white := _detect_three_white_soldiers(df, idx):
                patterns.append(three_white)
            if three_black := _detect_three_black_crows(df, idx):
                patterns.append(three_black)

        patterns_list.append(patterns)

    df["patterns"] = patterns_list
    return df


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def ta_candlestick_scan(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> list[dict] | dict:
    """Scan OHLCV data and detect ALL candlestick patterns present.

    Args:
        symbol: Ticker code (e.g., "7203", "ACB").
        start: Start date (YYYY-MM-DD). Defaults to 90 days ago.
        end: End date (YYYY-MM-DD). Defaults to today.
        source: Data source ("yfinance", "jquants", "vnstocks").

    Returns:
        List of dicts with: date, pattern_name, pattern_type, reliability, description.
        Or error dict on failure.
    """
    def _scan():
        df = _get_ohlcv_df(symbol, start, end, source=source)
        if isinstance(df, dict):
            return df

        df = _detect_patterns(df)

        results = []
        for idx, row in df.iterrows():
            for pattern in row.get("patterns", []):
                results.append({
                    "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "pattern_name": pattern["name"],
                    "pattern_type": pattern["type"],
                    "reliability": pattern["reliability"],
                    "description": pattern["description"],
                })

        return results

    return _safe_call(_scan)


def ta_candlestick_latest(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> dict:
    """Return patterns detected on the most recent trading day.

    Args:
        symbol: Ticker code (e.g., "7203", "ACB").
        start: Start date (YYYY-MM-DD). Defaults to 90 days ago.
        end: End date (YYYY-MM-DD). Defaults to today.
        source: Data source ("yfinance", "jquants", "vnstocks").

    Returns:
        Dict with: symbol, date, patterns (list of pattern dicts), pattern_count.
        Or error dict on failure.
    """
    def _latest():
        df = _get_ohlcv_df(symbol, start, end, source=source)
        if isinstance(df, dict):
            return df

        df = _detect_patterns(df)

        # Get last row
        if df.empty:
            return {"error": "No data available"}

        last_row = df.iloc[-1]
        patterns = last_row.get("patterns", [])

        return {
            "symbol": symbol,
            "date": last_row.name.isoformat() if hasattr(last_row.name, "isoformat") else str(last_row.name),
            "patterns": patterns,
            "pattern_count": len(patterns),
        }

    return _safe_call(_latest)


def ta_candlestick_screen(
    symbols: list[str],
    pattern: str = "all",
    source: str | None = None,
) -> list[dict] | dict:
    """Screen multiple stocks for specific candlestick patterns.

    Args:
        symbols: List of ticker codes (e.g., ["7203", "6758", "ACB"]).
        pattern: Filter patterns by type:
            - "all": all patterns
            - "bullish": bullish patterns only
            - "bearish": bearish patterns only
            - or specific pattern name like "hammer", "doji", "morning_star"
        source: Data source ("yfinance", "jquants", "vnstocks").

    Returns:
        List of dicts with: symbol, date, patterns (filtered), pattern_count.
        Or error dict on failure.
    """
    def _screen():
        if not symbols:
            return {"error": "symbols list is empty"}

        results = []

        for symbol in symbols:
            latest = ta_candlestick_latest(symbol, source=source)
            if isinstance(latest, dict) and "error" in latest:
                continue

            patterns = latest.get("patterns", [])

            # Filter patterns
            if pattern != "all":
                if pattern in (PATTERN_TYPE_BULLISH, PATTERN_TYPE_BEARISH, PATTERN_TYPE_NEUTRAL):
                    patterns = [p for p in patterns if p.get("type") == pattern]
                else:
                    # Specific pattern name (case-insensitive)
                    pattern_lower = pattern.lower()
                    patterns = [p for p in patterns if p.get("name", "").lower() == pattern_lower]

            if patterns:
                results.append({
                    "symbol": symbol,
                    "date": latest.get("date"),
                    "patterns": patterns,
                    "pattern_count": len(patterns),
                })

        return results

    return _safe_call(_screen)
