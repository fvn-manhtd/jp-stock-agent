"""Technical Analysis module for jpstock-agent.

Provides 20+ technical indicators, signal generation, stock screening,
and multi-timeframe analysis for Japanese and Vietnamese markets.

Uses the ``ta`` library (by bukosabino) for indicator calculations.

Inspired by:
  - MaverickMCP: screening strategies, multi-indicator analysis
  - TradingView MCP: multi-timeframe, backtesting signals
  - Crypto Indicators MCP: comprehensive indicator coverage
  - fintools-mcp: Fibonacci retracement, support/resistance, VWAP
  - dexter-kabu-jp: Japan-specific analysis (Ichimoku focus)

Every public function returns ``list[dict]`` or ``dict`` on success,
or ``{"error": str}`` on failure.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .core import _df_to_records, stock_history

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_ohlcv_df(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    source: str | None = None,
) -> pd.DataFrame | dict:
    """Fetch OHLCV data and return as a pandas DataFrame."""
    records = stock_history(symbol, start, end, interval, source)
    if isinstance(records, dict) and "error" in records:
        return records
    if not records:
        return {"error": f"No price data available for {symbol}"}

    df = pd.DataFrame(records)

    # Normalize column names for ta library compatibility
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "high", "low", "close", "volume", "date"):
            col_map[c] = cl
    df = df.rename(columns=col_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.set_index("date").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _round_val(v, decimals=4):
    """Round a numeric value, returning None for NaN/inf."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return round(v, decimals)


def _series_to_records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Extract specific columns from a DataFrame and return as records."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return []
    return _df_to_records(df[available].dropna(how="all"))


# ---------------------------------------------------------------------------
# 1. TREND INDICATORS
# ---------------------------------------------------------------------------


def ta_sma(
    symbol: str, period: int = 20,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Simple Moving Average (SMA)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    df[f"SMA_{period}"] = df["close"].rolling(window=period).mean()
    return _series_to_records(df, ["close", f"SMA_{period}"])


def ta_ema(
    symbol: str, period: int = 20,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Exponential Moving Average (EMA)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return _series_to_records(df, ["close", f"EMA_{period}"])


def ta_ichimoku(
    symbol: str, tenkan: int = 9, kijun: int = 26, senkou: int = 52,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Ichimoku Cloud (Ichimoku Kinko Hyo).

    Japan's most popular technical indicator.
    """
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.trend import IchimokuIndicator

    ich = IchimokuIndicator(df["high"], df["low"], window1=tenkan, window2=kijun, window3=senkou)
    df["tenkan_sen"] = ich.ichimoku_conversion_line()
    df["kijun_sen"] = ich.ichimoku_base_line()
    df["senkou_span_a"] = ich.ichimoku_a()
    df["senkou_span_b"] = ich.ichimoku_b()
    return _series_to_records(df, ["close", "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b"])


def ta_supertrend(
    symbol: str, period: int = 10, multiplier: float = 3.0,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Supertrend indicator (trend direction + levels)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volatility import AverageTrueRange

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period)
    atr_vals = atr.average_true_range()
    hl2 = (df["high"] + df["low"]) / 2
    df["upper_band"] = hl2 + multiplier * atr_vals
    df["lower_band"] = hl2 - multiplier * atr_vals

    # Simple supertrend logic
    df["supertrend"] = 0.0
    df["direction"] = 1  # 1=up, -1=down
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["upper_band"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("direction")] = 1
        elif df["close"].iloc[i] < df["lower_band"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("direction")] = -1
        else:
            df.iloc[i, df.columns.get_loc("direction")] = df["direction"].iloc[i - 1]

        if df["direction"].iloc[i] == 1:
            df.iloc[i, df.columns.get_loc("supertrend")] = df["lower_band"].iloc[i]
        else:
            df.iloc[i, df.columns.get_loc("supertrend")] = df["upper_band"].iloc[i]

    return _series_to_records(df, ["close", "supertrend", "direction"])


def ta_parabolic_sar(
    symbol: str, step: float = 0.02, max_step: float = 0.2,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Parabolic SAR (Stop and Reverse)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.trend import PSARIndicator

    psar = PSARIndicator(df["high"], df["low"], df["close"], step=step, max_step=max_step)
    df["PSAR"] = psar.psar()
    df["PSAR_up"] = psar.psar_up()
    df["PSAR_down"] = psar.psar_down()
    return _series_to_records(df, ["close", "PSAR", "PSAR_up", "PSAR_down"])


# ---------------------------------------------------------------------------
# 2. MOMENTUM INDICATORS
# ---------------------------------------------------------------------------


def ta_rsi(
    symbol: str, period: int = 14,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Relative Strength Index (RSI). <30=oversold, >70=overbought."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.momentum import RSIIndicator

    rsi = RSIIndicator(df["close"], window=period)
    df[f"RSI_{period}"] = rsi.rsi()
    return _series_to_records(df, ["close", f"RSI_{period}"])


def ta_macd(
    symbol: str, fast: int = 12, slow: int = 26, signal: int = 9,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate MACD (line, signal, histogram)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.trend import MACD

    macd = MACD(df["close"], window_slow=slow, window_fast=fast, window_sign=signal)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_histogram"] = macd.macd_diff()
    return _series_to_records(df, ["close", "MACD", "MACD_signal", "MACD_histogram"])


def ta_stochastic(
    symbol: str, k_period: int = 14, d_period: int = 3,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Stochastic Oscillator (%K, %D). <20=oversold, >80=overbought."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.momentum import StochasticOscillator

    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=k_period, smooth_window=d_period)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    return _series_to_records(df, ["close", "Stoch_K", "Stoch_D"])


def ta_williams_r(
    symbol: str, period: int = 14,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Williams %R indicator."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.momentum import WilliamsRIndicator

    wr = WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=period)
    df[f"WilliamsR_{period}"] = wr.williams_r()
    return _series_to_records(df, ["close", f"WilliamsR_{period}"])


def ta_cci(
    symbol: str, period: int = 20,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Commodity Channel Index (CCI)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.trend import CCIIndicator

    cci = CCIIndicator(df["high"], df["low"], df["close"], window=period)
    df[f"CCI_{period}"] = cci.cci()
    return _series_to_records(df, ["close", f"CCI_{period}"])


def ta_roc(
    symbol: str, period: int = 12,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Rate of Change (ROC)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.momentum import ROCIndicator

    roc = ROCIndicator(df["close"], window=period)
    df[f"ROC_{period}"] = roc.roc()
    return _series_to_records(df, ["close", f"ROC_{period}"])


# ---------------------------------------------------------------------------
# 3. VOLATILITY INDICATORS
# ---------------------------------------------------------------------------


def ta_bbands(
    symbol: str, period: int = 20, std_dev: float = 2.0,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Bollinger Bands (upper, middle, lower)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volatility import BollingerBands

    bb = BollingerBands(df["close"], window=period, window_dev=std_dev)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()
    return _series_to_records(df, ["close", "BB_upper", "BB_middle", "BB_lower", "BB_width"])


def ta_atr(
    symbol: str, period: int = 14,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Average True Range (ATR) - volatility measure."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volatility import AverageTrueRange

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period)
    df[f"ATR_{period}"] = atr.average_true_range()
    return _series_to_records(df, ["close", f"ATR_{period}"])


def ta_keltner(
    symbol: str, period: int = 20, multiplier: float = 2.0,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Keltner Channels."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volatility import KeltnerChannel

    kc = KeltnerChannel(df["high"], df["low"], df["close"], window=period, window_atr=period,
                        multiplier=multiplier)
    df["KC_upper"] = kc.keltner_channel_hband()
    df["KC_middle"] = kc.keltner_channel_mband()
    df["KC_lower"] = kc.keltner_channel_lband()
    return _series_to_records(df, ["close", "KC_upper", "KC_middle", "KC_lower"])


def ta_donchian(
    symbol: str, period: int = 20,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Donchian Channels (highest high / lowest low)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volatility import DonchianChannel

    dc = DonchianChannel(df["high"], df["low"], df["close"], window=period)
    df["DC_upper"] = dc.donchian_channel_hband()
    df["DC_middle"] = dc.donchian_channel_mband()
    df["DC_lower"] = dc.donchian_channel_lband()
    return _series_to_records(df, ["close", "DC_upper", "DC_middle", "DC_lower"])


# ---------------------------------------------------------------------------
# 4. VOLUME INDICATORS
# ---------------------------------------------------------------------------


def ta_obv(
    symbol: str,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate On-Balance Volume (OBV)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volume import OnBalanceVolumeIndicator

    obv = OnBalanceVolumeIndicator(df["close"], df["volume"])
    df["OBV"] = obv.on_balance_volume()
    return _series_to_records(df, ["close", "volume", "OBV"])


def ta_vwap(
    symbol: str,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Volume Weighted Average Price (VWAP)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volume import VolumeWeightedAveragePrice

    vwap = VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"])
    df["VWAP"] = vwap.volume_weighted_average_price()
    return _series_to_records(df, ["close", "volume", "VWAP"])


def ta_mfi(
    symbol: str, period: int = 14,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Money Flow Index (MFI). <20=oversold, >80=overbought."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volume import MFIIndicator

    mfi = MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=period)
    df[f"MFI_{period}"] = mfi.money_flow_index()
    return _series_to_records(df, ["close", "volume", f"MFI_{period}"])


def ta_ad(
    symbol: str,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> list[dict] | dict:
    """Calculate Accumulation/Distribution Line."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df
    from ta.volume import AccDistIndexIndicator

    ad = AccDistIndexIndicator(df["high"], df["low"], df["close"], df["volume"])
    df["AD"] = ad.acc_dist_index()
    return _series_to_records(df, ["close", "volume", "AD"])


# ---------------------------------------------------------------------------
# 5. COMPOSITE / ADVANCED ANALYSIS
# ---------------------------------------------------------------------------


def ta_fibonacci(
    symbol: str,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> dict:
    """Calculate Fibonacci retracement levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%)."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df

    high = float(df["high"].max())
    low = float(df["low"].min())
    diff = high - low

    return {
        "high": _round_val(high),
        "low": _round_val(low),
        "level_0.0": _round_val(high),
        "level_0.236": _round_val(high - 0.236 * diff),
        "level_0.382": _round_val(high - 0.382 * diff),
        "level_0.500": _round_val(high - 0.500 * diff),
        "level_0.618": _round_val(high - 0.618 * diff),
        "level_0.786": _round_val(high - 0.786 * diff),
        "level_1.0": _round_val(low),
        "current_close": _round_val(float(df["close"].iloc[-1])),
    }


def ta_support_resistance(
    symbol: str, window: int = 20,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> dict:
    """Detect support and resistance using pivot points."""
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df

    last = df.iloc[-1]
    pivot = (last["high"] + last["low"] + last["close"]) / 3
    r1 = 2 * pivot - last["low"]
    s1 = 2 * pivot - last["high"]
    r2 = pivot + (last["high"] - last["low"])
    s2 = pivot - (last["high"] - last["low"])
    r3 = last["high"] + 2 * (pivot - last["low"])
    s3 = last["low"] - 2 * (last["high"] - pivot)

    rolling_high = df["high"].rolling(window=window).max()
    rolling_low = df["low"].rolling(window=window).min()

    return {
        "current_close": _round_val(float(last["close"])),
        "pivot": _round_val(float(pivot)),
        "resistance_1": _round_val(float(r1)),
        "resistance_2": _round_val(float(r2)),
        "resistance_3": _round_val(float(r3)),
        "support_1": _round_val(float(s1)),
        "support_2": _round_val(float(s2)),
        "support_3": _round_val(float(s3)),
        f"rolling_{window}d_high": _round_val(float(rolling_high.iloc[-1])),
        f"rolling_{window}d_low": _round_val(float(rolling_low.iloc[-1])),
    }


def ta_multi_indicator(
    symbol: str,
    start: str | None = None, end: str | None = None, source: str | None = None,
) -> dict:
    """Run comprehensive multi-indicator analysis with BUY/SELL/HOLD signal.

    Combines RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV,
    moving averages, and generates an overall signal score (-100 to +100).
    """
    df = _get_ohlcv_df(symbol, start, end, source=source)
    if isinstance(df, dict):
        return df

    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD
    from ta.volatility import AverageTrueRange, BollingerBands

    close = df["close"]
    high = df["high"]
    low = df["low"]
    last_close = float(close.iloc[-1])

    # Calculate all indicators
    rsi = RSIIndicator(close, window=14).rsi()
    macd_obj = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd()
    macd_sig = macd_obj.macd_signal()
    macd_hist = macd_obj.macd_diff()
    bb = BollingerBands(close, window=20, window_dev=2)
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()

    def _last(s):
        if s is not None and len(s.dropna()) > 0:
            return _round_val(float(s.iloc[-1]))
        return None

    latest = {
        "symbol": symbol,
        "close": _round_val(last_close),
        "RSI_14": _last(rsi),
        "MACD": _last(macd_line),
        "MACD_signal": _last(macd_sig),
        "MACD_histogram": _last(macd_hist),
        "BB_upper": _last(bb.bollinger_hband()),
        "BB_middle": _last(bb.bollinger_mavg()),
        "BB_lower": _last(bb.bollinger_lband()),
        "Stoch_K": _last(stoch.stoch()),
        "Stoch_D": _last(stoch.stoch_signal()),
        "ATR_14": _last(atr),
        "SMA_20": _last(sma_20),
        "SMA_50": _last(sma_50),
        "SMA_200": _last(sma_200),
        "EMA_9": _last(ema_9),
        "EMA_21": _last(ema_21),
    }

    # Generate signals
    signals = []
    score = 0

    rsi_val = latest["RSI_14"]
    if rsi_val is not None:
        if rsi_val < 30:
            signals.append("RSI: OVERSOLD (<30)")
            score += 20
        elif rsi_val > 70:
            signals.append("RSI: OVERBOUGHT (>70)")
            score -= 20
        else:
            signals.append(f"RSI: NEUTRAL ({rsi_val})")

    if latest["MACD"] is not None and latest["MACD_signal"] is not None:
        if latest["MACD"] > latest["MACD_signal"]:
            signals.append("MACD: BULLISH (above signal)")
            score += 15
        else:
            signals.append("MACD: BEARISH (below signal)")
            score -= 15

    if latest["BB_upper"] is not None and latest["BB_lower"] is not None:
        if last_close > latest["BB_upper"]:
            signals.append("Bollinger: ABOVE upper band")
            score -= 10
        elif last_close < latest["BB_lower"]:
            signals.append("Bollinger: BELOW lower band")
            score += 10

    if latest["Stoch_K"] is not None:
        if latest["Stoch_K"] < 20:
            signals.append("Stochastic: OVERSOLD (<20)")
            score += 15
        elif latest["Stoch_K"] > 80:
            signals.append("Stochastic: OVERBOUGHT (>80)")
            score -= 15

    s20, s50, s200 = latest["SMA_20"], latest["SMA_50"], latest["SMA_200"]
    if s20 is not None and s50 is not None:
        if s20 > s50:
            signals.append("SMA20 > SMA50: short-term bullish")
            score += 10
        else:
            signals.append("SMA20 < SMA50: short-term bearish")
            score -= 10
    if s50 is not None and s200 is not None:
        if s50 > s200:
            signals.append("Golden Cross: SMA50 > SMA200")
            score += 15
        else:
            signals.append("Death Cross: SMA50 < SMA200")
            score -= 15
    if s200 is not None:
        if last_close > s200:
            signals.append("Price above SMA200")
            score += 5
        else:
            signals.append("Price below SMA200")
            score -= 5

    score = max(-100, min(100, score))
    if score >= 40:
        overall = "STRONG BUY"
    elif score >= 15:
        overall = "BUY"
    elif score <= -40:
        overall = "STRONG SELL"
    elif score <= -15:
        overall = "SELL"
    else:
        overall = "HOLD"

    latest["signals"] = signals
    latest["signal_score"] = score
    latest["overall_signal"] = overall
    return latest


# ---------------------------------------------------------------------------
# 6. SCREENING
# ---------------------------------------------------------------------------


def ta_screen(
    symbols: list[str],
    strategy: str = "oversold",
    source: str | None = None,
) -> list[dict] | dict:
    """Screen multiple stocks for technical signals.

    29 strategies available:
    - Momentum: oversold, overbought, rsi_divergence_bull, rsi_divergence_bear,
      mfi_oversold, mfi_overbought
    - MACD: macd_bullish, macd_bearish
    - Bollinger Bands: bb_squeeze, bb_breakout_up, bb_breakout_down
    - Gaps: gap_up, gap_down
    - Bar patterns: inside_bar, outside_bar
    - Moving Averages: golden_cross, death_cross, trend_up, trend_down,
      ema_bullish_cross, ema_bearish_cross
    - 52-week: new_high_52w, new_low_52w
    - Breakouts: breakout_up, breakout_down
    - Volume: volume_spike, high_volume_gain
    - Supertrend: supertrend_bullish, supertrend_bearish
    """
    from ta.momentum import MFIIndicator, RSIIndicator
    from ta.trend import MACD
    from ta.volatility import AverageTrueRange, BollingerBands

    results = []
    for sym in symbols:
        try:
            df = _get_ohlcv_df(sym, source=source)
            if isinstance(df, dict) or df is None or len(df) < 30:
                continue

            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            last_close = float(close.iloc[-1])
            info: dict[str, Any] = {"symbol": sym, "close": _round_val(last_close)}
            match = False

            if strategy in ("oversold", "overbought"):
                rsi = RSIIndicator(close, window=14).rsi()
                rsi_val = float(rsi.iloc[-1]) if len(rsi.dropna()) > 0 else None
                if rsi_val is not None:
                    info["RSI_14"] = _round_val(rsi_val)
                    match = (strategy == "oversold" and rsi_val < 30) or (strategy == "overbought" and rsi_val > 70)

            elif strategy in ("rsi_divergence_bull", "rsi_divergence_bear"):
                if len(df) >= 20:
                    rsi = RSIIndicator(close, window=14).rsi()
                    rsi_vals = rsi.dropna()
                    if len(rsi_vals) >= 20 and len(close) >= 20:
                        close_vals = close.iloc[-20:].values
                        rsi_recent = rsi_vals.iloc[-20:].values
                        if strategy == "rsi_divergence_bull":
                            local_close_mins = []
                            local_rsi_mins = []
                            for i in range(1, len(close_vals) - 1):
                                if close_vals[i] < close_vals[i-1] and close_vals[i] < close_vals[i+1]:
                                    local_close_mins.append((i, close_vals[i]))
                                if rsi_recent[i] < rsi_recent[i-1] and rsi_recent[i] < rsi_recent[i+1]:
                                    local_rsi_mins.append((i, rsi_recent[i]))
                            if len(local_close_mins) >= 2 and len(local_rsi_mins) >= 2:
                                match = (local_close_mins[-1][1] < local_close_mins[-2][1] and
                                        local_rsi_mins[-1][1] > local_rsi_mins[-2][1])
                        else:
                            local_close_maxs = []
                            local_rsi_maxs = []
                            for i in range(1, len(close_vals) - 1):
                                if close_vals[i] > close_vals[i-1] and close_vals[i] > close_vals[i+1]:
                                    local_close_maxs.append((i, close_vals[i]))
                                if rsi_recent[i] > rsi_recent[i-1] and rsi_recent[i] > rsi_recent[i+1]:
                                    local_rsi_maxs.append((i, rsi_recent[i]))
                            if len(local_close_maxs) >= 2 and len(local_rsi_maxs) >= 2:
                                match = (local_close_maxs[-1][1] > local_close_maxs[-2][1] and
                                        local_rsi_maxs[-1][1] < local_rsi_maxs[-2][1])

            elif strategy in ("macd_bullish", "macd_bearish"):
                macd_obj = MACD(close, window_slow=26, window_fast=12, window_sign=9)
                m = macd_obj.macd()
                s = macd_obj.macd_signal()
                if len(m.dropna()) >= 2 and len(s.dropna()) >= 2:
                    cm, cs = float(m.iloc[-1]), float(s.iloc[-1])
                    pm, ps = float(m.iloc[-2]), float(s.iloc[-2])
                    info["MACD"] = _round_val(cm)
                    info["MACD_signal"] = _round_val(cs)
                    if strategy == "macd_bullish":
                        match = pm <= ps and cm > cs
                    else:
                        match = pm >= ps and cm < cs

            elif strategy == "bb_squeeze":
                bb = BollingerBands(close, window=20, window_dev=2)
                lower = float(bb.bollinger_lband().iloc[-1])
                info["BB_lower"] = _round_val(lower)
                match = last_close <= lower * 1.02

            elif strategy in ("bb_breakout_up", "bb_breakout_down"):
                bb = BollingerBands(close, window=20, window_dev=2)
                upper = float(bb.bollinger_hband().iloc[-1])
                lower = float(bb.bollinger_lband().iloc[-1])
                info["BB_upper"] = _round_val(upper)
                info["BB_lower"] = _round_val(lower)
                if strategy == "bb_breakout_up":
                    match = last_close > upper
                else:
                    match = last_close < lower

            elif strategy in ("gap_up", "gap_down"):
                if len(df) >= 2:
                    prev_high = float(high.iloc[-2])
                    prev_low = float(low.iloc[-2])
                    curr_low = float(low.iloc[-1])
                    curr_high = float(high.iloc[-1])
                    info["prev_high"] = _round_val(prev_high)
                    info["prev_low"] = _round_val(prev_low)
                    info["curr_low"] = _round_val(curr_low)
                    info["curr_high"] = _round_val(curr_high)
                    if strategy == "gap_up":
                        match = curr_low > prev_high
                    else:
                        match = curr_high < prev_low

            elif strategy in ("inside_bar", "outside_bar"):
                if len(df) >= 2:
                    prev_high = float(high.iloc[-2])
                    prev_low = float(low.iloc[-2])
                    curr_high = float(high.iloc[-1])
                    curr_low = float(low.iloc[-1])
                    if strategy == "inside_bar":
                        match = curr_high < prev_high and curr_low > prev_low
                    else:
                        match = curr_high > prev_high and curr_low < prev_low

            elif strategy in ("new_high_52w", "new_low_52w"):
                if len(df) >= 252:
                    high_252 = float(high.iloc[-252:].max())
                    low_252 = float(low.iloc[-252:].min())
                    info["52w_high"] = _round_val(high_252)
                    info["52w_low"] = _round_val(low_252)
                    if strategy == "new_high_52w":
                        match = last_close >= high_252 * 0.99
                    else:
                        match = last_close <= low_252 * 1.01

            elif strategy in ("breakout_up", "breakout_down"):
                if len(df) >= 21:
                    high_20 = float(high.iloc[-20:].max())
                    low_20 = float(low.iloc[-20:].min())
                    avg_vol_20 = float(volume.rolling(20).mean().iloc[-1])
                    curr_vol = float(volume.iloc[-1])
                    info["20d_high"] = _round_val(high_20)
                    info["20d_low"] = _round_val(low_20)
                    info["volume_ratio"] = _round_val(curr_vol / avg_vol_20) if avg_vol_20 > 0 else None
                    if strategy == "breakout_up":
                        match = last_close > high_20 and curr_vol > 1.5 * avg_vol_20
                    else:
                        match = last_close < low_20 and curr_vol > 1.5 * avg_vol_20

            elif strategy in ("golden_cross", "death_cross"):
                sma50 = close.rolling(50).mean().dropna()
                sma200 = close.rolling(200).mean().dropna()
                if len(sma50) > 0 and len(sma200) > 0:
                    s50, s200 = float(sma50.iloc[-1]), float(sma200.iloc[-1])
                    info["SMA_50"] = _round_val(s50)
                    info["SMA_200"] = _round_val(s200)
                    match = (strategy == "golden_cross" and s50 > s200) or (strategy == "death_cross" and s50 < s200)

            elif strategy in ("ema_bullish_cross", "ema_bearish_cross"):
                if len(df) >= 22:
                    ema9 = close.ewm(span=9, adjust=False).mean()
                    ema21 = close.ewm(span=21, adjust=False).mean()
                    if len(ema9.dropna()) >= 2 and len(ema21.dropna()) >= 2:
                        ema9_curr, ema9_prev = float(ema9.iloc[-1]), float(ema9.iloc[-2])
                        ema21_curr, ema21_prev = float(ema21.iloc[-1]), float(ema21.iloc[-2])
                        info["EMA_9"] = _round_val(ema9_curr)
                        info["EMA_21"] = _round_val(ema21_curr)
                        if strategy == "ema_bullish_cross":
                            match = ema9_prev <= ema21_prev and ema9_curr > ema21_curr
                        else:
                            match = ema9_prev >= ema21_prev and ema9_curr < ema21_curr

            elif strategy in ("mfi_oversold", "mfi_overbought"):
                if len(df) >= 15:
                    mfi = MFIIndicator(high, low, close, volume, window=14).money_flow_index()
                    mfi_val = float(mfi.iloc[-1]) if len(mfi.dropna()) > 0 else None
                    if mfi_val is not None:
                        info["MFI_14"] = _round_val(mfi_val)
                        match = (strategy == "mfi_oversold" and mfi_val < 20) or \
                            (strategy == "mfi_overbought" and mfi_val > 80)

            elif strategy in ("supertrend_bullish", "supertrend_bearish"):
                if len(df) >= 11:
                    atr = AverageTrueRange(high, low, close, window=10).average_true_range()
                    atr_vals = atr.dropna()
                    if len(atr_vals) >= 2:
                        hl2 = (high + low) / 2
                        upper = hl2 + 3.0 * atr_vals
                        lower = hl2 - 3.0 * atr_vals
                        direction = [1]
                        for i in range(1, len(close)):
                            if close.iloc[i] > upper.iloc[i - 1]:
                                direction.append(1)
                            elif close.iloc[i] < lower.iloc[i - 1]:
                                direction.append(-1)
                            else:
                                direction.append(direction[-1])
                        if len(direction) >= 2:
                            prev_dir, curr_dir = direction[-2], direction[-1]
                            if strategy == "supertrend_bullish":
                                match = prev_dir == -1 and curr_dir == 1
                            else:
                                match = prev_dir == 1 and curr_dir == -1

            elif strategy == "volume_spike":
                vol = df["volume"]
                avg = float(vol.rolling(20).mean().iloc[-1])
                curr = float(vol.iloc[-1])
                info["volume"] = curr
                info["avg_volume_20d"] = _round_val(avg)
                info["volume_ratio"] = _round_val(curr / avg) if avg > 0 else None
                match = avg > 0 and curr > 2 * avg

            elif strategy == "high_volume_gain":
                if len(df) >= 21:
                    prev_close = float(close.iloc[-2])
                    pct_gain = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else 0
                    avg_vol_20 = float(volume.rolling(20).mean().iloc[-1])
                    curr_vol = float(volume.iloc[-1])
                    info["pct_change"] = _round_val(pct_gain)
                    info["volume_ratio"] = _round_val(curr_vol / avg_vol_20) if avg_vol_20 > 0 else None
                    match = pct_gain > 3 and curr_vol > 2 * avg_vol_20

            elif strategy in ("trend_up", "trend_down"):
                sma20 = close.rolling(20).mean().dropna()
                sma50 = close.rolling(50).mean().dropna()
                if len(sma20) > 0 and len(sma50) > 0:
                    s20, s50 = float(sma20.iloc[-1]), float(sma50.iloc[-1])
                    info["SMA_20"] = _round_val(s20)
                    info["SMA_50"] = _round_val(s50)
                    match = (strategy == "trend_up" and last_close > s20 > s50) or \
                            (strategy == "trend_down" and last_close < s20 < s50)

            if match:
                info["strategy"] = strategy
                results.append(info)
        except Exception:
            continue

    return results if results else {"message": f"No stocks matched strategy '{strategy}'."}


# ---------------------------------------------------------------------------
# 7. MULTI-TIMEFRAME ANALYSIS
# ---------------------------------------------------------------------------


def ta_multi_timeframe(
    symbol: str, source: str | None = None,
) -> dict:
    """Analyze across daily, weekly, monthly timeframes."""
    from ta.momentum import RSIIndicator
    from ta.trend import MACD

    timeframes = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
    result = {"symbol": symbol}

    for tf_name, tf_interval in timeframes.items():
        df = _get_ohlcv_df(symbol, interval=tf_interval, source=source)
        if isinstance(df, dict) or df is None or len(df) < 30:
            result[tf_name] = {"status": "insufficient data"}
            continue

        close = df["close"]
        tf_data: dict[str, Any] = {"close": _round_val(float(close.iloc[-1])), "data_points": len(df)}

        rsi = RSIIndicator(close, window=14).rsi()
        if len(rsi.dropna()) > 0:
            tf_data["RSI_14"] = _round_val(float(rsi.iloc[-1]))

        macd_obj = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        m, s = macd_obj.macd(), macd_obj.macd_signal()
        if len(m.dropna()) > 0 and len(s.dropna()) > 0:
            mv, sv = float(m.iloc[-1]), float(s.iloc[-1])
            tf_data["MACD"] = _round_val(mv)
            tf_data["MACD_signal"] = _round_val(sv)
            tf_data["MACD_direction"] = "bullish" if mv > sv else "bearish"

        sma20 = close.rolling(20).mean()
        if len(sma20.dropna()) > 0:
            s20 = float(sma20.iloc[-1])
            tf_data["SMA_20"] = _round_val(s20)
            tf_data["trend"] = "bullish" if float(close.iloc[-1]) > s20 else "bearish"

        result[tf_name] = tf_data
    return result
