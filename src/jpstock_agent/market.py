"""Sector & Market Analysis Module.

Provides market-level analysis beyond individual stock analysis:
sector performance, market breadth, top movers, sector rotation,
and market regime detection.

Public Functions:
- market_sector_performance: Performance comparison across sectors
- market_breadth: Advance/decline ratio, new highs/lows for a list of stocks
- market_top_movers: Top gainers and losers from a list of symbols
- market_regime: Detect bull/bear/sideways market regime from an index
- market_heatmap: Sector-grouped performance data for visualization

All functions return dict on success or {"error": str} on failure.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from .logging import get_logger

logger = get_logger(__name__)


def _import_core():
    from . import core
    return core


def _import_ta():
    from . import ta
    return ta


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


def _get_period_return(symbol, days=30, source=None):
    """Get period return for a symbol."""
    core = _import_core()
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    history = core.stock_history(symbol, start=start, source=source)
    if isinstance(history, dict) or not history:
        return None

    closes = []
    for rec in history:
        c = rec.get("close") or rec.get("Close")
        if c is not None:
            try:
                closes.append(float(c))
            except (TypeError, ValueError):
                pass

    if len(closes) < 2:
        return None

    # Take last N trading days
    if len(closes) > days:
        closes = closes[-days:]

    return {
        "first_close": closes[0],
        "last_close": closes[-1],
        "return_pct": round((closes[-1] - closes[0]) / closes[0] * 100, 2),
        "high": max(closes),
        "low": min(closes),
        "data_points": len(closes),
    }


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def market_sector_performance(
    sectors: dict[str, list[str]],
    days: int = 30,
    source: str | None = None,
) -> dict:
    """Compare performance across sectors.

    Parameters
    ----------
    sectors : dict
        Mapping of sector_name → list of representative symbols.
        Example: {"Auto": ["7203", "7267"], "Tech": ["6758", "9984"]}
    days : int
        Lookback period in days (default 30).

    Returns
    -------
    dict with: sectors (per-sector avg return), ranking (best to worst).
    """
    if not sectors:
        return {"error": "No sectors provided"}

    all_symbols = []
    sym_to_sector = {}
    for sector, syms in sectors.items():
        for s in syms:
            all_symbols.append(s)
            sym_to_sector[s] = sector

    # Fetch returns in parallel
    returns = {}
    with ThreadPoolExecutor(max_workers=min(8, len(all_symbols))) as executor:
        futures = {
            executor.submit(_get_period_return, sym, days, source): sym
            for sym in all_symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                returns[sym] = future.result()
            except Exception:
                returns[sym] = None

    # Aggregate by sector
    sector_results = {}
    for sector, syms in sectors.items():
        sector_returns = []
        stock_details = []
        for s in syms:
            r = returns.get(s)
            if r and r.get("return_pct") is not None:
                sector_returns.append(r["return_pct"])
                stock_details.append({"symbol": s, "return_pct": r["return_pct"]})
            else:
                stock_details.append({"symbol": s, "return_pct": None})

        avg_return = round(sum(sector_returns) / len(sector_returns), 2) if sector_returns else None
        sector_results[sector] = {
            "avg_return_pct": avg_return,
            "stock_count": len(syms),
            "stocks": stock_details,
        }

    # Ranking
    ranked = sorted(
        [(name, data["avg_return_pct"]) for name, data in sector_results.items() if data["avg_return_pct"] is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    ranking = [{"sector": name, "avg_return_pct": ret} for name, ret in ranked]

    return {
        "period_days": days,
        "sectors": sector_results,
        "ranking": ranking,
    }


def market_breadth(
    symbols: list[str],
    days: int = 1,
    source: str | None = None,
) -> dict:
    """Calculate market breadth indicators for a list of stocks.

    Parameters
    ----------
    symbols : list[str]
        Universe of symbols to analyze.
    days : int
        Lookback period (default 1 = today vs yesterday).

    Returns
    -------
    dict with: advancing, declining, unchanged, advance_decline_ratio,
    new_highs_52w, new_lows_52w, breadth_signal.
    """
    if not symbols:
        return {"error": "No symbols provided"}

    core = _import_core()

    def _analyze_one(sym):
        start = (datetime.now() - timedelta(days=max(days + 10, 370))).strftime("%Y-%m-%d")
        history = core.stock_history(sym, start=start, source=source)
        if isinstance(history, dict) or not history:
            return None

        closes = []
        for rec in history:
            c = rec.get("close") or rec.get("Close")
            if c is not None:
                try:
                    closes.append(float(c))
                except (TypeError, ValueError):
                    pass

        if len(closes) < 2:
            return None

        latest = closes[-1]
        ref = closes[-(days + 1)] if len(closes) > days else closes[0]
        change_pct = (latest - ref) / ref * 100

        # 52-week check
        year_closes = closes[-252:] if len(closes) >= 252 else closes
        is_52w_high = latest >= max(year_closes)
        is_52w_low = latest <= min(year_closes)

        return {
            "symbol": sym,
            "change_pct": round(change_pct, 2),
            "is_52w_high": is_52w_high,
            "is_52w_low": is_52w_low,
        }

    # Parallel analysis
    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as executor:
        futures = {executor.submit(_analyze_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception:
                pass

    advancing = sum(1 for r in results if r["change_pct"] > 0)
    declining = sum(1 for r in results if r["change_pct"] < 0)
    unchanged = sum(1 for r in results if r["change_pct"] == 0)
    new_highs = sum(1 for r in results if r["is_52w_high"])
    new_lows = sum(1 for r in results if r["is_52w_low"])

    ad_ratio = round(advancing / declining, 2) if declining > 0 else (
        float("inf") if advancing > 0 else 0
    )

    # Breadth signal
    if advancing + declining > 0:
        breadth_pct = advancing / (advancing + declining) * 100
    else:
        breadth_pct = 50

    if breadth_pct > 70:
        signal = "STRONG_BULLISH"
    elif breadth_pct > 55:
        signal = "BULLISH"
    elif breadth_pct > 45:
        signal = "NEUTRAL"
    elif breadth_pct > 30:
        signal = "BEARISH"
    else:
        signal = "STRONG_BEARISH"

    return {
        "period_days": days,
        "total_symbols": len(results),
        "advancing": advancing,
        "declining": declining,
        "unchanged": unchanged,
        "advance_decline_ratio": _round_val(ad_ratio, 2) if ad_ratio != float("inf") else "inf",
        "breadth_pct": round(breadth_pct, 1),
        "new_highs_52w": new_highs,
        "new_lows_52w": new_lows,
        "breadth_signal": signal,
    }


def market_top_movers(
    symbols: list[str],
    days: int = 1,
    top_n: int = 5,
    source: str | None = None,
) -> dict:
    """Find top gainers and losers from a list of symbols.

    Parameters
    ----------
    symbols : list[str]
        Universe of symbols.
    days : int
        Lookback period (default 1).
    top_n : int
        Number of top gainers/losers to return (default 5).

    Returns
    -------
    dict with: top_gainers, top_losers.
    """
    if not symbols:
        return {"error": "No symbols provided"}

    # Fetch returns in parallel
    returns = {}
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as executor:
        futures = {
            executor.submit(_get_period_return, sym, max(days, 5), source): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                returns[sym] = future.result()
            except Exception:
                returns[sym] = None

    # Build ranked list
    ranked = []
    for sym in symbols:
        r = returns.get(sym)
        if r and r.get("return_pct") is not None:
            ranked.append({"symbol": sym, "return_pct": r["return_pct"],
                           "last_close": r["last_close"]})

    ranked.sort(key=lambda x: x["return_pct"], reverse=True)

    return {
        "period_days": days,
        "total_analyzed": len(ranked),
        "top_gainers": ranked[:top_n],
        "top_losers": ranked[-top_n:][::-1] if len(ranked) >= top_n else list(reversed(ranked)),
    }


def market_regime(
    symbol: str = "^N225",
    source: str | None = None,
) -> dict:
    """Detect market regime (bull/bear/sideways) from an index.

    Uses SMA trend, volatility, and momentum to classify regime.

    Parameters
    ----------
    symbol : str
        Index symbol. Default "^N225" (Nikkei 225).
        Other options: "^GSPC" (S&P 500), "^VNI" (VN-Index).

    Returns
    -------
    dict with: regime, confidence, indicators used.
    """
    core = _import_core()

    # Get 1 year of data
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    history = core.stock_history(symbol, start=start, source=source)
    if isinstance(history, dict) and "error" in history:
        return history
    if not history or len(history) < 50:
        return {"error": f"Insufficient data for {symbol}"}

    closes = []
    for rec in history:
        c = rec.get("close") or rec.get("Close")
        if c is not None:
            try:
                closes.append(float(c))
            except (TypeError, ValueError):
                pass

    if len(closes) < 50:
        return {"error": "Insufficient price data"}

    latest = closes[-1]

    # SMA indicators
    sma50 = sum(closes[-50:]) / 50
    sma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else sum(closes) / len(closes)

    # Trend direction
    price_vs_sma50 = "above" if latest > sma50 else "below"
    price_vs_sma200 = "above" if latest > sma200 else "below"
    sma50_vs_200 = "above" if sma50 > sma200 else "below"

    # Returns
    ret_30d = (closes[-1] - closes[-min(22, len(closes))]) / closes[-min(22, len(closes))] * 100
    ret_90d = (closes[-1] - closes[-min(63, len(closes))]) / closes[-min(63, len(closes))] * 100

    # Volatility (20-day)
    if len(closes) >= 21:
        daily_returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(-20, 0)]
        mean_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        volatility_20d = math.sqrt(variance) * math.sqrt(252) * 100  # Annualized %
    else:
        volatility_20d = None

    # Regime scoring
    bull_score = 0
    bear_score = 0

    if price_vs_sma50 == "above":
        bull_score += 1
    else:
        bear_score += 1

    if price_vs_sma200 == "above":
        bull_score += 1
    else:
        bear_score += 1

    if sma50_vs_200 == "above":
        bull_score += 1
    else:
        bear_score += 1

    if ret_30d > 2:
        bull_score += 1
    elif ret_30d < -2:
        bear_score += 1

    if ret_90d > 5:
        bull_score += 1
    elif ret_90d < -5:
        bear_score += 1

    # Determine regime
    total = bull_score + bear_score
    if total == 0:
        regime = "SIDEWAYS"
        confidence = 50
    elif bull_score >= 4:
        regime = "BULL"
        confidence = min(round(bull_score / total * 100), 95)
    elif bear_score >= 4:
        regime = "BEAR"
        confidence = min(round(bear_score / total * 100), 95)
    elif bull_score > bear_score:
        regime = "MILD_BULL"
        confidence = round(bull_score / total * 100)
    elif bear_score > bull_score:
        regime = "MILD_BEAR"
        confidence = round(bear_score / total * 100)
    else:
        regime = "SIDEWAYS"
        confidence = 50

    return {
        "symbol": symbol,
        "regime": regime,
        "confidence_pct": confidence,
        "indicators": {
            "latest_close": round(latest, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "price_vs_sma50": price_vs_sma50,
            "price_vs_sma200": price_vs_sma200,
            "sma50_vs_sma200": sma50_vs_200,
            "return_30d_pct": round(ret_30d, 2),
            "return_90d_pct": round(ret_90d, 2),
            "volatility_20d_annualized_pct": _round_val(volatility_20d, 2),
        },
        "scoring": {
            "bull_points": bull_score,
            "bear_points": bear_score,
        },
    }


def market_heatmap(
    sectors: dict[str, list[str]],
    days: int = 1,
    source: str | None = None,
) -> dict:
    """Generate sector-grouped performance data suitable for heatmap visualization.

    Parameters
    ----------
    sectors : dict
        Mapping of sector_name → list of symbols.
    days : int
        Lookback period (default 1).

    Returns
    -------
    dict with: heatmap_data (nested sector → stock → return), sector_averages.
    """
    if not sectors:
        return {"error": "No sectors provided"}

    all_symbols = []
    sym_to_sector = {}
    for sector, syms in sectors.items():
        for s in syms:
            all_symbols.append(s)
            sym_to_sector[s] = sector

    # Fetch all returns
    returns = {}
    with ThreadPoolExecutor(max_workers=min(8, len(all_symbols))) as executor:
        futures = {
            executor.submit(_get_period_return, sym, max(days, 5), source): sym
            for sym in all_symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                returns[sym] = future.result()
            except Exception:
                returns[sym] = None

    # Build heatmap structure
    heatmap = {}
    sector_avgs = {}
    for sector, syms in sectors.items():
        sector_data = []
        sector_rets = []
        for s in syms:
            r = returns.get(s)
            ret_pct = r.get("return_pct") if r else None
            sector_data.append({
                "symbol": s,
                "return_pct": ret_pct,
                "last_close": r.get("last_close") if r else None,
            })
            if ret_pct is not None:
                sector_rets.append(ret_pct)

        heatmap[sector] = sector_data
        sector_avgs[sector] = round(sum(sector_rets) / len(sector_rets), 2) if sector_rets else None

    return {
        "period_days": days,
        "heatmap_data": heatmap,
        "sector_averages": sector_avgs,
        "total_symbols": len(all_symbols),
    }
