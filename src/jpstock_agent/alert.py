"""Alert & Watchlist Module.

Provides condition-based stock monitoring — check if any user-defined
alert conditions are currently triggered for given symbols.

This module is **stateless** — it evaluates conditions at call time
rather than running a persistent process. Designed to be called by
scheduled tasks or AI agents on a regular interval.

Public Functions:
- alert_check: Evaluate a list of alert conditions against current data
- alert_price: Quick price-level alert (above/below threshold)
- alert_ta: TA-based alerts (RSI extremes, MACD cross, BB squeeze, etc.)
- alert_fundamental: Fundamental alerts (P/E threshold, yield threshold, etc.)
- alert_watchlist: Check multiple symbols against a set of common conditions

All functions return dict on success or {"error": str} on failure.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Alert condition definitions
# ---------------------------------------------------------------------------

# Built-in TA alert conditions
TA_CONDITIONS = {
    "rsi_oversold": {
        "description": "RSI below oversold threshold",
        "default_params": {"threshold": 30},
    },
    "rsi_overbought": {
        "description": "RSI above overbought threshold",
        "default_params": {"threshold": 70},
    },
    "macd_bullish_cross": {
        "description": "MACD line crosses above signal line",
        "default_params": {},
    },
    "macd_bearish_cross": {
        "description": "MACD line crosses below signal line",
        "default_params": {},
    },
    "bb_squeeze": {
        "description": "Bollinger Band width below threshold (low volatility)",
        "default_params": {"threshold": 0.05},
    },
    "bb_breakout_upper": {
        "description": "Price breaks above upper Bollinger Band",
        "default_params": {},
    },
    "bb_breakout_lower": {
        "description": "Price breaks below lower Bollinger Band",
        "default_params": {},
    },
    "golden_cross": {
        "description": "50-day SMA crosses above 200-day SMA",
        "default_params": {},
    },
    "death_cross": {
        "description": "50-day SMA crosses below 200-day SMA",
        "default_params": {},
    },
    "volume_spike": {
        "description": "Volume exceeds N times the 20-day average",
        "default_params": {"multiplier": 2.0},
    },
    "price_above_sma": {
        "description": "Price above N-day SMA",
        "default_params": {"period": 200},
    },
    "price_below_sma": {
        "description": "Price below N-day SMA",
        "default_params": {"period": 200},
    },
    "supertrend_bullish": {
        "description": "Supertrend flips to bullish",
        "default_params": {},
    },
    "supertrend_bearish": {
        "description": "Supertrend flips to bearish",
        "default_params": {},
    },
    "new_high_52w": {
        "description": "Price at 52-week high",
        "default_params": {},
    },
    "new_low_52w": {
        "description": "Price at 52-week low",
        "default_params": {},
    },
}


def _import_ta():
    """Lazy import of TA module."""
    from . import ta
    return ta


def _import_core():
    """Lazy import of core module."""
    from . import core
    return core


def _import_financial():
    """Lazy import of financial module."""
    from . import financial
    return financial


# ---------------------------------------------------------------------------
# TA condition evaluators
# ---------------------------------------------------------------------------


def _eval_rsi_oversold(ta_mod, symbol, source, params):
    threshold = params.get("threshold", 30)
    result = ta_mod.ta_rsi(symbol, source=source)
    if isinstance(result, dict) and "error" in result:
        return None
    if isinstance(result, list) and result:
        latest_rsi = result[-1].get("rsi")
        if latest_rsi is not None and latest_rsi < threshold:
            return {"triggered": True, "value": round(latest_rsi, 2), "threshold": threshold,
                    "message": f"RSI={round(latest_rsi, 2)} < {threshold}"}
    return {"triggered": False}


def _eval_rsi_overbought(ta_mod, symbol, source, params):
    threshold = params.get("threshold", 70)
    result = ta_mod.ta_rsi(symbol, source=source)
    if isinstance(result, dict) and "error" in result:
        return None
    if isinstance(result, list) and result:
        latest_rsi = result[-1].get("rsi")
        if latest_rsi is not None and latest_rsi > threshold:
            return {"triggered": True, "value": round(latest_rsi, 2), "threshold": threshold,
                    "message": f"RSI={round(latest_rsi, 2)} > {threshold}"}
    return {"triggered": False}


def _eval_macd_cross(ta_mod, symbol, source, direction="bullish"):
    result = ta_mod.ta_macd(symbol, source=source)
    if isinstance(result, dict) and "error" in result:
        return None
    if isinstance(result, list) and len(result) >= 2:
        curr = result[-1]
        prev = result[-2]
        macd_curr = curr.get("macd")
        signal_curr = curr.get("macd_signal")
        macd_prev = prev.get("macd")
        signal_prev = prev.get("macd_signal")
        if all(v is not None for v in [macd_curr, signal_curr, macd_prev, signal_prev]):
            if direction == "bullish":
                if macd_prev <= signal_prev and macd_curr > signal_curr:
                    return {"triggered": True, "message": "MACD bullish crossover detected"}
            else:
                if macd_prev >= signal_prev and macd_curr < signal_curr:
                    return {"triggered": True, "message": "MACD bearish crossover detected"}
    return {"triggered": False}


def _eval_bb_squeeze(ta_mod, symbol, source, params):
    threshold = params.get("threshold", 0.05)
    result = ta_mod.ta_bbands(symbol, source=source)
    if isinstance(result, dict) and "error" in result:
        return None
    if isinstance(result, list) and result:
        latest = result[-1]
        upper = latest.get("bb_upper")
        lower = latest.get("bb_lower")
        middle = latest.get("bb_middle")
        if all(v is not None for v in [upper, lower, middle]) and middle != 0:
            width = (upper - lower) / middle
            if width < threshold:
                return {"triggered": True, "value": round(width, 4), "threshold": threshold,
                        "message": f"BB width={round(width, 4)} < {threshold} (squeeze)"}
    return {"triggered": False}


def _eval_bb_breakout(ta_mod, symbol, source, direction="upper"):
    result = ta_mod.ta_bbands(symbol, source=source)
    if isinstance(result, dict) and "error" in result:
        return None
    if isinstance(result, list) and result:
        latest = result[-1]
        close = latest.get("close")
        upper = latest.get("bb_upper")
        lower = latest.get("bb_lower")
        if close is not None:
            if direction == "upper" and upper is not None and close > upper:
                return {"triggered": True, "message": f"Price {close} above upper BB {round(upper, 2)}"}
            if direction == "lower" and lower is not None and close < lower:
                return {"triggered": True, "message": f"Price {close} below lower BB {round(lower, 2)}"}
    return {"triggered": False}


def _eval_sma_cross(ta_mod, symbol, source, cross_type="golden"):
    """Evaluate golden cross (50 > 200) or death cross (50 < 200)."""
    result = ta_mod.ta_sma(symbol, period=50, source=source)
    result200 = ta_mod.ta_sma(symbol, period=200, source=source)
    if isinstance(result, dict) or isinstance(result200, dict):
        return None
    if isinstance(result, list) and isinstance(result200, list) and len(result) >= 2 and len(result200) >= 2:
        # Align by taking last 2 from each
        sma50_curr = result[-1].get("sma")
        sma200_curr = result200[-1].get("sma")
        sma50_prev = result[-2].get("sma")
        sma200_prev = result200[-2].get("sma")
        if all(v is not None for v in [sma50_curr, sma200_curr, sma50_prev, sma200_prev]):
            if cross_type == "golden":
                if sma50_prev <= sma200_prev and sma50_curr > sma200_curr:
                    return {"triggered": True, "message": "Golden cross: SMA50 crossed above SMA200"}
            else:
                if sma50_prev >= sma200_prev and sma50_curr < sma200_curr:
                    return {"triggered": True, "message": "Death cross: SMA50 crossed below SMA200"}
    return {"triggered": False}


def _eval_volume_spike(ta_mod, symbol, source, params):
    multiplier = params.get("multiplier", 2.0)
    core = _import_core()
    history = core.stock_history(symbol, source=source)
    if isinstance(history, dict) or not history or len(history) < 21:
        return None
    volumes = []
    for rec in history:
        v = rec.get("volume") or rec.get("Volume")
        if v is not None:
            try:
                volumes.append(float(v))
            except (TypeError, ValueError):
                pass
    if len(volumes) < 21:
        return None
    avg_20 = sum(volumes[-21:-1]) / 20
    latest_vol = volumes[-1]
    if avg_20 > 0 and latest_vol > avg_20 * multiplier:
        return {"triggered": True, "value": latest_vol, "avg_20": round(avg_20, 0),
                "ratio": round(latest_vol / avg_20, 2),
                "message": f"Volume {latest_vol:.0f} is {latest_vol / avg_20:.1f}x the 20-day average"}
    return {"triggered": False}


def _eval_price_vs_sma(ta_mod, symbol, source, params, direction="above"):
    period = params.get("period", 200)
    result = ta_mod.ta_sma(symbol, period=period, source=source)
    if isinstance(result, dict) or not result:
        return None
    latest = result[-1]
    close = latest.get("close")
    sma = latest.get("sma")
    if close is not None and sma is not None:
        if direction == "above" and close > sma:
            return {"triggered": True,
                    "message": f"Price {close:.2f} above SMA{period} {sma:.2f}"}
        if direction == "below" and close < sma:
            return {"triggered": True,
                    "message": f"Price {close:.2f} below SMA{period} {sma:.2f}"}
    return {"triggered": False}


def _eval_supertrend(ta_mod, symbol, source, direction="bullish"):
    result = ta_mod.ta_supertrend(symbol, source=source)
    if isinstance(result, dict) or not result:
        return None
    if len(result) >= 2:
        curr = result[-1]
        prev = result[-2]
        curr_dir = curr.get("direction")
        prev_dir = prev.get("direction")
        if direction == "bullish" and prev_dir == "bearish" and curr_dir == "bullish":
            return {"triggered": True, "message": "Supertrend flipped to BULLISH"}
        if direction == "bearish" and prev_dir == "bullish" and curr_dir == "bearish":
            return {"triggered": True, "message": "Supertrend flipped to BEARISH"}
    return {"triggered": False}


def _eval_52w_extreme(core, symbol, source, direction="high"):
    """Check if price is at 52-week high or low."""
    from datetime import datetime, timedelta
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    history = core.stock_history(symbol, start=start, source=source)
    if isinstance(history, dict) or not history or len(history) < 20:
        return None
    closes = []
    for rec in history:
        c = rec.get("close") or rec.get("Close")
        if c is not None:
            try:
                closes.append(float(c))
            except (TypeError, ValueError):
                pass
    if not closes:
        return None
    latest = closes[-1]
    if direction == "high" and latest >= max(closes):
        return {"triggered": True, "value": latest, "message": f"At 52-week high: {latest:.2f}"}
    if direction == "low" and latest <= min(closes):
        return {"triggered": True, "value": latest, "message": f"At 52-week low: {latest:.2f}"}
    return {"triggered": False}


# Map condition names to evaluator functions
def _get_evaluator(condition_name):
    """Return (evaluator_function, needs_ta_mod, needs_core) for a condition."""
    evaluators = {
        "rsi_oversold": lambda ta, core, sym, src, p: _eval_rsi_oversold(ta, sym, src, p),
        "rsi_overbought": lambda ta, core, sym, src, p: _eval_rsi_overbought(ta, sym, src, p),
        "macd_bullish_cross": lambda ta, core, sym, src, p: _eval_macd_cross(ta, sym, src, "bullish"),
        "macd_bearish_cross": lambda ta, core, sym, src, p: _eval_macd_cross(ta, sym, src, "bearish"),
        "bb_squeeze": lambda ta, core, sym, src, p: _eval_bb_squeeze(ta, sym, src, p),
        "bb_breakout_upper": lambda ta, core, sym, src, p: _eval_bb_breakout(ta, sym, src, "upper"),
        "bb_breakout_lower": lambda ta, core, sym, src, p: _eval_bb_breakout(ta, sym, src, "lower"),
        "golden_cross": lambda ta, core, sym, src, p: _eval_sma_cross(ta, sym, src, "golden"),
        "death_cross": lambda ta, core, sym, src, p: _eval_sma_cross(ta, sym, src, "death"),
        "volume_spike": lambda ta, core, sym, src, p: _eval_volume_spike(ta, sym, src, p),
        "price_above_sma": lambda ta, core, sym, src, p: _eval_price_vs_sma(ta, sym, src, p, "above"),
        "price_below_sma": lambda ta, core, sym, src, p: _eval_price_vs_sma(ta, sym, src, p, "below"),
        "supertrend_bullish": lambda ta, core, sym, src, p: _eval_supertrend(ta, sym, src, "bullish"),
        "supertrend_bearish": lambda ta, core, sym, src, p: _eval_supertrend(ta, sym, src, "bearish"),
        "new_high_52w": lambda ta, core, sym, src, p: _eval_52w_extreme(core, sym, src, "high"),
        "new_low_52w": lambda ta, core, sym, src, p: _eval_52w_extreme(core, sym, src, "low"),
    }
    return evaluators.get(condition_name)


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def alert_check(
    symbol: str,
    conditions: list[dict],
    source: str | None = None,
) -> dict:
    """Evaluate a list of alert conditions for a symbol.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    conditions : list[dict]
        Each dict has: {"condition": str, "params": dict (optional)}.
        Example: [{"condition": "rsi_oversold", "params": {"threshold": 25}},
                  {"condition": "volume_spike"}]
    source : str | None
        Data source override.

    Returns
    -------
    dict with: symbol, triggered (list of triggered alerts), not_triggered (count),
    total_conditions, alert_summary.
    """
    if not conditions:
        return {"error": "No conditions provided"}

    ta_mod = _import_ta()
    core = _import_core()

    triggered = []
    not_triggered = 0
    errors = 0

    for cond_spec in conditions:
        cond_name = cond_spec.get("condition", "")
        params = cond_spec.get("params", {})

        evaluator = _get_evaluator(cond_name)
        if evaluator is None:
            errors += 1
            continue

        try:
            result = evaluator(ta_mod, core, symbol, source, params)
            if result is None:
                errors += 1
            elif result.get("triggered"):
                result["condition"] = cond_name
                triggered.append(result)
            else:
                not_triggered += 1
        except Exception as e:
            logger.warning(f"Alert condition '{cond_name}' failed for {symbol}: {e}")
            errors += 1

    return {
        "symbol": symbol,
        "triggered": triggered,
        "triggered_count": len(triggered),
        "not_triggered_count": not_triggered,
        "error_count": errors,
        "total_conditions": len(conditions),
    }


def alert_price(
    symbol: str,
    above: float | None = None,
    below: float | None = None,
    source: str | None = None,
) -> dict:
    """Quick price-level alert.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    above : float | None
        Alert if price is above this level.
    below : float | None
        Alert if price is below this level.

    Returns
    -------
    dict with: symbol, current_price, alerts.
    """
    if above is None and below is None:
        return {"error": "Specify at least one of 'above' or 'below'"}

    core = _import_core()
    history = core.stock_history(symbol, source=source)
    if isinstance(history, dict) and "error" in history:
        return history
    if not history:
        return {"error": f"No price data for {symbol}"}

    latest = history[-1]
    price = latest.get("close") or latest.get("Close")
    if price is None:
        return {"error": "Could not determine current price"}

    try:
        price = float(price)
    except (TypeError, ValueError):
        return {"error": f"Invalid price value: {price}"}

    alerts = []
    if above is not None and price > above:
        alerts.append({
            "type": "price_above",
            "triggered": True,
            "current_price": round(price, 2),
            "threshold": above,
            "message": f"Price {price:.2f} is ABOVE {above}",
        })
    if below is not None and price < below:
        alerts.append({
            "type": "price_below",
            "triggered": True,
            "current_price": round(price, 2),
            "threshold": below,
            "message": f"Price {price:.2f} is BELOW {below}",
        })

    return {
        "symbol": symbol,
        "current_price": round(price, 2),
        "alerts": alerts,
        "any_triggered": len(alerts) > 0,
    }


def alert_ta(
    symbol: str,
    conditions: list[str] | None = None,
    source: str | None = None,
) -> dict:
    """TA-based alerts using predefined conditions.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    conditions : list[str] | None
        List of condition names from TA_CONDITIONS.
        Default: checks all common conditions.

    Returns
    -------
    dict with triggered alerts.
    """
    if conditions is None:
        conditions = [
            "rsi_oversold", "rsi_overbought",
            "macd_bullish_cross", "macd_bearish_cross",
            "bb_squeeze", "volume_spike",
            "golden_cross", "death_cross",
        ]

    cond_list = [
        {"condition": c, "params": TA_CONDITIONS.get(c, {}).get("default_params", {})}
        for c in conditions
        if c in TA_CONDITIONS
    ]

    return alert_check(symbol, cond_list, source)


def alert_fundamental(
    symbol: str,
    pe_below: float | None = None,
    pe_above: float | None = None,
    yield_above: float | None = None,
    roe_above: float | None = None,
    debt_to_equity_below: float | None = None,
    f_score_above: int | None = None,
) -> dict:
    """Fundamental-based alerts.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    pe_below, pe_above : float | None
        P/E ratio thresholds.
    yield_above : float | None
        Dividend yield threshold (e.g. 0.03 for 3%).
    roe_above : float | None
        Return on equity threshold.
    debt_to_equity_below : float | None
        Debt/equity ratio threshold.
    f_score_above : int | None
        Piotroski F-score threshold (0-9).

    Returns
    -------
    dict with triggered alerts.
    """
    core = _import_core()
    financial_mod = _import_financial()

    alerts = []

    # Fetch ratios
    ratios = core.financial_ratio(symbol)
    if isinstance(ratios, dict) and "error" not in ratios:
        pe = ratios.get("trailingPE")
        if pe is not None:
            if pe_below is not None and pe < pe_below:
                alerts.append({"condition": "pe_below", "triggered": True,
                               "value": pe, "threshold": pe_below,
                               "message": f"P/E {pe:.1f} < {pe_below}"})
            if pe_above is not None and pe > pe_above:
                alerts.append({"condition": "pe_above", "triggered": True,
                               "value": pe, "threshold": pe_above,
                               "message": f"P/E {pe:.1f} > {pe_above}"})

        dy = ratios.get("dividendYield")
        if dy is not None and yield_above is not None and dy > yield_above:
            alerts.append({"condition": "yield_above", "triggered": True,
                           "value": round(dy, 4), "threshold": yield_above,
                           "message": f"Dividend yield {dy:.2%} > {yield_above:.2%}"})

        roe = ratios.get("returnOnEquity")
        if roe is not None and roe_above is not None and roe > roe_above:
            alerts.append({"condition": "roe_above", "triggered": True,
                           "value": round(roe, 4), "threshold": roe_above,
                           "message": f"ROE {roe:.2%} > {roe_above:.2%}"})

        dte = ratios.get("debtToEquity")
        if dte is not None and debt_to_equity_below is not None and dte < debt_to_equity_below:
            alerts.append({"condition": "debt_to_equity_below", "triggered": True,
                           "value": round(dte, 2), "threshold": debt_to_equity_below,
                           "message": f"D/E {dte:.1f} < {debt_to_equity_below}"})

    # F-score check
    if f_score_above is not None:
        health = financial_mod.financial_health(symbol)
        if isinstance(health, dict) and "error" not in health:
            f = health.get("piotroski_f", {}).get("score")
            if f is not None and f > f_score_above:
                alerts.append({"condition": "f_score_above", "triggered": True,
                               "value": f, "threshold": f_score_above,
                               "message": f"Piotroski F-score {f}/9 > {f_score_above}"})

    return {
        "symbol": symbol,
        "alerts": alerts,
        "triggered_count": len(alerts),
        "any_triggered": len(alerts) > 0,
    }


def alert_watchlist(
    symbols: list[str],
    conditions: list[str] | None = None,
    source: str | None = None,
) -> dict:
    """Check multiple symbols against common alert conditions.

    Parameters
    ----------
    symbols : list[str]
        List of stock symbols.
    conditions : list[str] | None
        TA condition names. Default: common set.

    Returns
    -------
    dict with: results (per-symbol), triggered_symbols (symbols with any alert).
    """
    if not symbols:
        return {"error": "No symbols provided"}

    def _check_one(sym):
        return alert_ta(sym, conditions, source)

    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as executor:
        futures = {executor.submit(_check_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"symbol": futures[future], "error": str(e)})

    # Sort by input order
    sym_order = {s: i for i, s in enumerate(symbols)}
    results.sort(key=lambda r: sym_order.get(r.get("symbol", ""), 999))

    triggered_symbols = [
        r["symbol"] for r in results
        if r.get("triggered_count", 0) > 0
    ]

    return {
        "results": results,
        "symbol_count": len(symbols),
        "triggered_symbols": triggered_symbols,
        "triggered_symbol_count": len(triggered_symbols),
    }


def alert_list_conditions() -> dict:
    """List all available alert conditions with descriptions.

    Returns
    -------
    dict with condition names, descriptions, and default parameters.
    """
    return {
        "conditions": {
            name: {
                "description": info["description"],
                "default_params": info["default_params"],
            }
            for name, info in TA_CONDITIONS.items()
        },
        "total": len(TA_CONDITIONS),
    }
