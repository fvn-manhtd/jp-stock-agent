"""Custom Strategy Builder Module.

Allows composing multiple TA, fundamental, and price conditions into
custom screening strategies. Supports AND/OR logic for combining conditions.

Public Functions:
- strategy_screen: Screen symbols with custom composed conditions (AND/OR)
- strategy_evaluate: Evaluate a custom strategy against a single symbol
- strategy_list_conditions: List all available condition types

All functions return dict on success or {"error": str} on failure.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging import get_logger

logger = get_logger(__name__)


def _import_ta():
    from . import ta
    return ta


def _import_core():
    from . import core
    return core


def _import_financial():
    from . import financial
    return financial


def _round_val(v, decimals=4):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, decimals)


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

CONDITION_TYPES = {
    # ---- Price conditions ----
    "price_above": {
        "category": "price",
        "description": "Current price above threshold",
        "params": {"value": "float — price threshold"},
    },
    "price_below": {
        "category": "price",
        "description": "Current price below threshold",
        "params": {"value": "float — price threshold"},
    },
    "return_above": {
        "category": "price",
        "description": "N-day return above threshold",
        "params": {"days": "int (default 30)", "value": "float — return % threshold"},
    },
    "return_below": {
        "category": "price",
        "description": "N-day return below threshold",
        "params": {"days": "int (default 30)", "value": "float — return % threshold"},
    },
    "volume_above_avg": {
        "category": "price",
        "description": "Volume above N times 20-day average",
        "params": {"multiplier": "float (default 2.0)"},
    },

    # ---- TA conditions ----
    "rsi_below": {
        "category": "ta",
        "description": "RSI below threshold (oversold)",
        "params": {"value": "float (default 30)"},
    },
    "rsi_above": {
        "category": "ta",
        "description": "RSI above threshold (overbought)",
        "params": {"value": "float (default 70)"},
    },
    "rsi_between": {
        "category": "ta",
        "description": "RSI between two values",
        "params": {"low": "float", "high": "float"},
    },
    "macd_bullish": {
        "category": "ta",
        "description": "MACD above signal line",
        "params": {},
    },
    "macd_bearish": {
        "category": "ta",
        "description": "MACD below signal line",
        "params": {},
    },
    "price_above_sma": {
        "category": "ta",
        "description": "Price above N-period SMA",
        "params": {"period": "int (default 50)"},
    },
    "price_below_sma": {
        "category": "ta",
        "description": "Price below N-period SMA",
        "params": {"period": "int (default 50)"},
    },
    "bb_above_upper": {
        "category": "ta",
        "description": "Price above upper Bollinger Band",
        "params": {},
    },
    "bb_below_lower": {
        "category": "ta",
        "description": "Price below lower Bollinger Band",
        "params": {},
    },
    "supertrend_bullish": {
        "category": "ta",
        "description": "Supertrend direction is bullish",
        "params": {},
    },
    "supertrend_bearish": {
        "category": "ta",
        "description": "Supertrend direction is bearish",
        "params": {},
    },
    "ta_signal_buy": {
        "category": "ta",
        "description": "Multi-indicator composite signal is BUY",
        "params": {},
    },
    "ta_signal_sell": {
        "category": "ta",
        "description": "Multi-indicator composite signal is SELL",
        "params": {},
    },
    "ta_score_above": {
        "category": "ta",
        "description": "Multi-indicator score above threshold",
        "params": {"value": "float (-100 to 100)"},
    },
    "ta_score_below": {
        "category": "ta",
        "description": "Multi-indicator score below threshold",
        "params": {"value": "float (-100 to 100)"},
    },

    # ---- Fundamental conditions ----
    "pe_below": {
        "category": "fundamental",
        "description": "Trailing P/E below threshold",
        "params": {"value": "float"},
    },
    "pe_above": {
        "category": "fundamental",
        "description": "Trailing P/E above threshold",
        "params": {"value": "float"},
    },
    "pb_below": {
        "category": "fundamental",
        "description": "Price-to-Book below threshold",
        "params": {"value": "float"},
    },
    "dividend_yield_above": {
        "category": "fundamental",
        "description": "Dividend yield above threshold",
        "params": {"value": "float (e.g. 0.03 for 3%)"},
    },
    "roe_above": {
        "category": "fundamental",
        "description": "Return on equity above threshold",
        "params": {"value": "float (e.g. 0.15 for 15%)"},
    },
    "debt_to_equity_below": {
        "category": "fundamental",
        "description": "Debt/equity ratio below threshold",
        "params": {"value": "float"},
    },
    "f_score_above": {
        "category": "fundamental",
        "description": "Piotroski F-score above threshold (0-9)",
        "params": {"value": "int"},
    },
}


# ---------------------------------------------------------------------------
# Condition evaluators
# ---------------------------------------------------------------------------


def _eval_condition(condition: dict, symbol: str, source=None, _cache=None) -> dict:
    """Evaluate a single condition against a symbol.

    Parameters
    ----------
    condition : dict
        {"type": str, "params": dict (optional)}
    symbol : str
    source : str | None
    _cache : dict | None
        Shared cache for reusing fetched data within a strategy.

    Returns
    -------
    dict with: type, passed (bool), value, detail.
    """
    ctype = condition.get("type", "")
    params = condition.get("params", {})
    cache = _cache if _cache is not None else {}

    result = {"type": ctype, "passed": False, "value": None, "detail": ""}

    try:
        if ctype in ("price_above", "price_below"):
            val = _get_latest_close(symbol, source, cache)
            threshold = params.get("value")
            if val is None or threshold is None:
                result["detail"] = "No price data or missing threshold"
                return result
            result["value"] = val
            if ctype == "price_above":
                result["passed"] = val > threshold
            else:
                result["passed"] = val < threshold
            result["detail"] = f"Price {val} vs threshold {threshold}"

        elif ctype in ("return_above", "return_below"):
            days = params.get("days", 30)
            threshold = params.get("value", 0)
            ret = _get_return(symbol, days, source, cache)
            if ret is None:
                result["detail"] = "Insufficient data for return"
                return result
            result["value"] = ret
            if ctype == "return_above":
                result["passed"] = ret > threshold
            else:
                result["passed"] = ret < threshold
            result["detail"] = f"{days}d return {ret}% vs {threshold}%"

        elif ctype == "volume_above_avg":
            mult = params.get("multiplier", 2.0)
            vol_data = _get_volume_ratio(symbol, source, cache)
            if vol_data is None:
                result["detail"] = "Insufficient volume data"
                return result
            result["value"] = vol_data["ratio"]
            result["passed"] = vol_data["ratio"] > mult
            result["detail"] = f"Volume ratio {vol_data['ratio']}x vs {mult}x threshold"

        elif ctype in ("rsi_below", "rsi_above", "rsi_between"):
            rsi = _get_rsi(symbol, source, cache)
            if rsi is None:
                result["detail"] = "No RSI data"
                return result
            result["value"] = rsi
            if ctype == "rsi_below":
                threshold = params.get("value", 30)
                result["passed"] = rsi < threshold
                result["detail"] = f"RSI {rsi} vs {threshold}"
            elif ctype == "rsi_above":
                threshold = params.get("value", 70)
                result["passed"] = rsi > threshold
                result["detail"] = f"RSI {rsi} vs {threshold}"
            else:
                low = params.get("low", 40)
                high = params.get("high", 60)
                result["passed"] = low <= rsi <= high
                result["detail"] = f"RSI {rsi} in [{low}, {high}]"

        elif ctype in ("macd_bullish", "macd_bearish"):
            macd_data = _get_macd(symbol, source, cache)
            if macd_data is None:
                result["detail"] = "No MACD data"
                return result
            macd_val, signal_val = macd_data
            result["value"] = round(macd_val - signal_val, 4)
            if ctype == "macd_bullish":
                result["passed"] = macd_val > signal_val
            else:
                result["passed"] = macd_val < signal_val
            result["detail"] = f"MACD {round(macd_val, 2)} vs Signal {round(signal_val, 2)}"

        elif ctype in ("price_above_sma", "price_below_sma"):
            period = params.get("period", 50)
            sma_data = _get_sma(symbol, period, source, cache)
            if sma_data is None:
                result["detail"] = f"No SMA{period} data"
                return result
            price, sma = sma_data
            result["value"] = round(price, 2)
            if ctype == "price_above_sma":
                result["passed"] = price > sma
            else:
                result["passed"] = price < sma
            result["detail"] = f"Price {round(price, 2)} vs SMA{period} {round(sma, 2)}"

        elif ctype in ("bb_above_upper", "bb_below_lower"):
            bb = _get_bbands(symbol, source, cache)
            if bb is None:
                result["detail"] = "No Bollinger Band data"
                return result
            close, upper, lower = bb
            result["value"] = round(close, 2)
            if ctype == "bb_above_upper":
                result["passed"] = close > upper
                result["detail"] = f"Price {round(close, 2)} vs Upper BB {round(upper, 2)}"
            else:
                result["passed"] = close < lower
                result["detail"] = f"Price {round(close, 2)} vs Lower BB {round(lower, 2)}"

        elif ctype in ("supertrend_bullish", "supertrend_bearish"):
            direction = _get_supertrend(symbol, source, cache)
            if direction is None:
                result["detail"] = "No Supertrend data"
                return result
            result["value"] = direction
            if ctype == "supertrend_bullish":
                result["passed"] = direction == "bullish"
            else:
                result["passed"] = direction == "bearish"
            result["detail"] = f"Supertrend: {direction}"

        elif ctype in ("ta_signal_buy", "ta_signal_sell", "ta_score_above", "ta_score_below"):
            ta_data = _get_multi_indicator(symbol, source, cache)
            if ta_data is None:
                result["detail"] = "No TA data"
                return result
            signal, score = ta_data
            if ctype == "ta_signal_buy":
                result["passed"] = signal == "BUY"
                result["value"] = signal
            elif ctype == "ta_signal_sell":
                result["passed"] = signal == "SELL"
                result["value"] = signal
            elif ctype == "ta_score_above":
                threshold = params.get("value", 0)
                result["passed"] = score > threshold
                result["value"] = score
            else:
                threshold = params.get("value", 0)
                result["passed"] = score < threshold
                result["value"] = score
            result["detail"] = f"Signal: {signal}, Score: {score}"

        elif ctype in ("pe_below", "pe_above", "pb_below", "dividend_yield_above",
                        "roe_above", "debt_to_equity_below"):
            ratios = _get_ratios(symbol, source, cache)
            if ratios is None:
                result["detail"] = "No ratio data"
                return result
            threshold = params.get("value")
            if threshold is None:
                result["detail"] = "Missing threshold value"
                return result

            mapping = {
                "pe_below": ("trailingPE", "lt"),
                "pe_above": ("trailingPE", "gt"),
                "pb_below": ("priceToBook", "lt"),
                "dividend_yield_above": ("dividendYield", "gt"),
                "roe_above": ("returnOnEquity", "gt"),
                "debt_to_equity_below": ("debtToEquity", "lt"),
            }
            key, op = mapping[ctype]
            val = ratios.get(key)
            if val is None:
                result["detail"] = f"{key} not available"
                return result
            result["value"] = _round_val(val)
            result["passed"] = val < threshold if op == "lt" else val > threshold
            result["detail"] = f"{key}={_round_val(val)} {'<' if op == 'lt' else '>'} {threshold}"

        elif ctype == "f_score_above":
            f = _get_f_score(symbol, cache)
            if f is None:
                result["detail"] = "No F-score data"
                return result
            threshold = params.get("value", 5)
            result["value"] = f
            result["passed"] = f > threshold
            result["detail"] = f"F-score {f}/9 vs {threshold}"

        else:
            result["detail"] = f"Unknown condition type: {ctype}"

    except Exception as e:
        result["detail"] = f"Error: {e}"

    return result


# ---------------------------------------------------------------------------
# Data fetchers (with per-strategy caching)
# ---------------------------------------------------------------------------


def _get_latest_close(symbol, source, cache):
    key = f"close:{symbol}"
    if key in cache:
        return cache[key]
    core = _import_core()
    history = core.stock_history(symbol, source=source)
    if isinstance(history, dict) or not history:
        return None
    latest = history[-1]
    val = latest.get("close") or latest.get("Close")
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    cache[key] = val
    cache[f"history:{symbol}"] = history
    return val


def _get_return(symbol, days, source, cache):
    key = f"history:{symbol}"
    core = _import_core()
    if key in cache:
        history = cache[key]
    else:
        history = core.stock_history(symbol, source=source)
        if isinstance(history, dict) or not history:
            return None
        cache[key] = history

    closes = []
    for rec in history:
        c = rec.get("close") or rec.get("Close")
        if c is not None:
            try:
                closes.append(float(c))
            except (TypeError, ValueError):
                pass
    if len(closes) < days + 1:
        return None
    return round((closes[-1] - closes[-(days + 1)]) / closes[-(days + 1)] * 100, 2)


def _get_volume_ratio(symbol, source, cache):
    key = f"history:{symbol}"
    core = _import_core()
    if key in cache:
        history = cache[key]
    else:
        history = core.stock_history(symbol, source=source)
        if isinstance(history, dict) or not history:
            return None
        cache[key] = history

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
    avg = sum(volumes[-21:-1]) / 20
    return {"ratio": round(volumes[-1] / avg, 2) if avg > 0 else 0, "latest": volumes[-1], "avg_20": avg}


def _get_rsi(symbol, source, cache):
    key = f"rsi:{symbol}"
    if key in cache:
        return cache[key]
    ta_mod = _import_ta()
    result = ta_mod.ta_rsi(symbol, source=source)
    if isinstance(result, dict) or not result:
        return None
    rsi = result[-1].get("rsi")
    if rsi is not None:
        rsi = round(rsi, 2)
    cache[key] = rsi
    return rsi


def _get_macd(symbol, source, cache):
    key = f"macd:{symbol}"
    if key in cache:
        return cache[key]
    ta_mod = _import_ta()
    result = ta_mod.ta_macd(symbol, source=source)
    if isinstance(result, dict) or not result:
        return None
    latest = result[-1]
    macd = latest.get("macd")
    signal = latest.get("macd_signal")
    if macd is None or signal is None:
        return None
    val = (macd, signal)
    cache[key] = val
    return val


def _get_sma(symbol, period, source, cache):
    key = f"sma{period}:{symbol}"
    if key in cache:
        return cache[key]
    ta_mod = _import_ta()
    result = ta_mod.ta_sma(symbol, period=period, source=source)
    if isinstance(result, dict) or not result:
        return None
    latest = result[-1]
    close = latest.get("close")
    sma = latest.get("sma")
    if close is None or sma is None:
        return None
    val = (close, sma)
    cache[key] = val
    return val


def _get_bbands(symbol, source, cache):
    key = f"bbands:{symbol}"
    if key in cache:
        return cache[key]
    ta_mod = _import_ta()
    result = ta_mod.ta_bbands(symbol, source=source)
    if isinstance(result, dict) or not result:
        return None
    latest = result[-1]
    close = latest.get("close")
    upper = latest.get("bb_upper")
    lower = latest.get("bb_lower")
    if any(v is None for v in [close, upper, lower]):
        return None
    val = (close, upper, lower)
    cache[key] = val
    return val


def _get_supertrend(symbol, source, cache):
    key = f"supertrend:{symbol}"
    if key in cache:
        return cache[key]
    ta_mod = _import_ta()
    result = ta_mod.ta_supertrend(symbol, source=source)
    if isinstance(result, dict) or not result:
        return None
    direction = result[-1].get("direction")
    cache[key] = direction
    return direction


def _get_multi_indicator(symbol, source, cache):
    key = f"multi:{symbol}"
    if key in cache:
        return cache[key]
    ta_mod = _import_ta()
    result = ta_mod.ta_multi_indicator(symbol, source=source)
    if isinstance(result, dict) and "error" not in result:
        signal = result.get("signal")
        score = result.get("score")
        val = (signal, score)
        cache[key] = val
        return val
    return None


def _get_ratios(symbol, source, cache):
    key = f"ratios:{symbol}"
    if key in cache:
        return cache[key]
    core = _import_core()
    result = core.financial_ratio(symbol, source=source)
    if isinstance(result, dict) and "error" not in result:
        cache[key] = result
        return result
    return None


def _get_f_score(symbol, cache):
    key = f"fscore:{symbol}"
    if key in cache:
        return cache[key]
    fin = _import_financial()
    result = fin.financial_health(symbol)
    if isinstance(result, dict) and "error" not in result:
        f = result.get("piotroski_f", {}).get("score")
        cache[key] = f
        return f
    return None


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def strategy_evaluate(
    symbol: str,
    conditions: list[dict],
    logic: str = "AND",
    source: str | None = None,
) -> dict:
    """Evaluate a custom strategy against a single symbol.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    conditions : list[dict]
        Each dict: {"type": str, "params": dict (optional)}.
        Example: [{"type": "rsi_below", "params": {"value": 30}},
                  {"type": "macd_bullish"},
                  {"type": "f_score_above", "params": {"value": 6}}]
    logic : str
        "AND" (all must pass) or "OR" (any must pass). Default "AND".

    Returns
    -------
    dict with: symbol, passed (bool), logic, conditions_detail, summary.
    """
    if not conditions:
        return {"error": "No conditions provided"}

    cache = {}
    results = []
    for cond in conditions:
        r = _eval_condition(cond, symbol, source, cache)
        results.append(r)

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    if logic.upper() == "OR":
        overall = passed_count > 0
    else:
        overall = passed_count == total

    return {
        "symbol": symbol,
        "passed": overall,
        "logic": logic.upper(),
        "passed_count": passed_count,
        "total_conditions": total,
        "conditions_detail": results,
    }


def strategy_screen(
    symbols: list[str],
    conditions: list[dict],
    logic: str = "AND",
    source: str | None = None,
) -> dict:
    """Screen multiple symbols with custom composed conditions.

    Parameters
    ----------
    symbols : list[str]
        List of stock symbols to screen.
    conditions : list[dict]
        Condition definitions (same format as strategy_evaluate).
    logic : str
        "AND" or "OR". Default "AND".

    Returns
    -------
    dict with: matching (list of passing symbols), not_matching, total_screened.
    """
    if not symbols:
        return {"error": "No symbols provided"}
    if not conditions:
        return {"error": "No conditions provided"}

    def _screen_one(sym):
        return strategy_evaluate(sym, conditions, logic, source)

    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as executor:
        futures = {executor.submit(_screen_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"symbol": futures[future], "passed": False, "error": str(e)})

    # Sort by input order
    sym_order = {s: i for i, s in enumerate(symbols)}
    results.sort(key=lambda r: sym_order.get(r.get("symbol", ""), 999))

    matching = [r for r in results if r.get("passed")]
    not_matching = [r for r in results if not r.get("passed")]

    return {
        "logic": logic.upper(),
        "conditions_count": len(conditions),
        "total_screened": len(symbols),
        "matching_count": len(matching),
        "matching": matching,
        "not_matching_count": len(not_matching),
    }


def strategy_list_conditions() -> dict:
    """List all available condition types grouped by category.

    Returns
    -------
    dict with conditions grouped by category, with descriptions and params.
    """
    by_category = {}
    for name, info in CONDITION_TYPES.items():
        cat = info["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "type": name,
            "description": info["description"],
            "params": info["params"],
        })

    return {
        "categories": by_category,
        "total_conditions": len(CONDITION_TYPES),
    }
