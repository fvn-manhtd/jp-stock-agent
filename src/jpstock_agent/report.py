"""Stock Report Generator Module.

Aggregates data from all analysis modules into comprehensive, unified reports.
Eliminates the need to call 8-10 separate tools for a full stock analysis.

Public Functions:
- stock_report: Full analysis report for a single stock (TA + financial + sentiment + ML + candlestick)
- stock_report_quick: Lightweight summary report (key metrics only, faster)
- stock_report_compare: Side-by-side comparison report for multiple stocks

All functions return dict on success or {"error": str} on failure.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy module imports — avoid circular imports and keep startup fast
# ---------------------------------------------------------------------------


def _import_modules():
    """Import analysis modules lazily."""
    from . import candlestick, financial, portfolio, sentiment, ta

    return {
        "ta": ta,
        "candlestick": candlestick,
        "financial": financial,
        "sentiment": sentiment,
        "portfolio": portfolio,
    }


def _import_ml():
    """Import ML module separately (optional dependency)."""
    try:
        from . import ml
        return ml
    except Exception:
        return None


def _import_core():
    """Import core module."""
    from . import core
    return core


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_section(func, section_name: str) -> dict:
    """Run a section function safely, returning error dict on failure."""
    try:
        result = func()
        if isinstance(result, dict) and "error" in result:
            return {"_error": result["error"]}
        return result
    except Exception as e:
        logger.warning(f"Report section '{section_name}' failed: {e}")
        return {"_error": str(e)}


def _parallel_sections(sections: dict) -> dict:
    """Run multiple report sections in parallel.

    Parameters
    ----------
    sections : dict
        Mapping of section_name → callable.

    Returns
    -------
    dict mapping section_name → result.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=min(8, len(sections))) as executor:
        futures = {
            executor.submit(_safe_section, func, name): name
            for name, func in sections.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"_error": str(e)}
    return results


def _clean_result(data):
    """Remove internal _error keys and empty sections."""
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not (isinstance(v, dict) and "_error" in v)}
    return data


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def stock_report(
    symbol: str,
    include_ml: bool = True,
    include_options: bool = False,
    period: str = "annual",
    source: str | None = None,
) -> dict:
    """Generate a comprehensive stock analysis report.

    Aggregates: company overview, price data, TA signals, candlestick patterns,
    financial health, growth trends, valuation, sentiment, and optionally ML prediction.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g. "7203", "ACB", "AAPL").
    include_ml : bool
        Include ML prediction section (requires scikit-learn). Default True.
    include_options : bool
        Include options analysis (only for US/JP stocks with options data). Default False.
    period : str
        Financial statement period: "annual" or "quarterly".
    source : str | None
        Data source override.

    Returns
    -------
    dict with sections: overview, price_summary, technical, candlestick,
    financial_health, growth, valuation, sentiment, ml_prediction, options.
    """
    start_time = time.time()
    mods = _import_modules()
    core = _import_core()

    # Define all sections as callables for parallel execution
    sections = {
        "overview": lambda: core.company_overview(symbol, source=source),
        "technical": lambda: mods["ta"].ta_multi_indicator(symbol, source=source),
        "candlestick": lambda: mods["candlestick"].ta_candlestick_latest(symbol, source=source),
        "financial_health": lambda: mods["financial"].financial_health(symbol, period),
        "growth": lambda: mods["financial"].financial_growth(symbol, period),
        "valuation": lambda: mods["financial"].financial_valuation(symbol),
        "dividend": lambda: mods["financial"].financial_dividend(symbol),
        "sentiment": lambda: mods["sentiment"].sentiment_news(symbol, source=source),
    }

    if include_ml:
        ml_mod = _import_ml()
        if ml_mod:
            sections["ml_prediction"] = lambda: ml_mod.ml_signal(symbol, source=source)

    if include_options:
        try:
            from . import options
            sections["options_summary"] = lambda: options.options_put_call_ratio(symbol)
        except Exception:
            pass

    # Run all sections in parallel
    raw = _parallel_sections(sections)

    # Build price summary from recent history
    price_summary = {}
    history = core.stock_history(symbol, source=source)
    if isinstance(history, list) and history:
        latest = history[-1]
        price_summary["latest_close"] = latest.get("close") or latest.get("Close")
        price_summary["latest_date"] = latest.get("date") or latest.get("Date")
        price_summary["data_points"] = len(history)

        closes = []
        for rec in history:
            c = rec.get("close") or rec.get("Close")
            if c is not None:
                try:
                    closes.append(float(c))
                except (TypeError, ValueError):
                    pass

        if len(closes) >= 2:
            price_summary["period_high"] = max(closes)
            price_summary["period_low"] = min(closes)
            price_summary["period_return_pct"] = round(
                (closes[-1] - closes[0]) / closes[0] * 100, 2,
            )

    # Assemble report
    report = {"symbol": symbol, "report_type": "comprehensive"}
    report["price_summary"] = price_summary

    # Add each section, cleaning errors
    for key in ["overview", "technical", "candlestick", "financial_health",
                 "growth", "valuation", "dividend", "sentiment",
                 "ml_prediction", "options_summary"]:
        if key in raw:
            section = raw[key]
            if not (isinstance(section, dict) and "_error" in section):
                report[key] = section

    # Generate executive summary
    report["executive_summary"] = _build_executive_summary(report)

    elapsed = round((time.time() - start_time) * 1000, 0)
    report["generation_time_ms"] = elapsed

    return report


def stock_report_quick(
    symbol: str,
    source: str | None = None,
) -> dict:
    """Generate a lightweight quick report with key metrics only.

    Much faster than full report — only fetches price, basic TA, and ratios.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    source : str | None
        Data source override.

    Returns
    -------
    dict with: symbol, price_summary, signal, key_ratios, quick_summary.
    """
    start_time = time.time()
    mods = _import_modules()
    core = _import_core()

    sections = {
        "technical": lambda: mods["ta"].ta_multi_indicator(symbol, source=source),
        "ratios": lambda: core.financial_ratio(symbol, source=source),
    }

    raw = _parallel_sections(sections)

    # Price summary
    price_summary = {}
    history = core.stock_history(symbol, source=source)
    if isinstance(history, list) and history:
        latest = history[-1]
        price_summary["latest_close"] = latest.get("close") or latest.get("Close")
        price_summary["latest_date"] = latest.get("date") or latest.get("Date")

        closes = []
        for rec in history:
            c = rec.get("close") or rec.get("Close")
            if c is not None:
                try:
                    closes.append(float(c))
                except (TypeError, ValueError):
                    pass
        if len(closes) >= 2:
            price_summary["period_return_pct"] = round(
                (closes[-1] - closes[0]) / closes[0] * 100, 2,
            )

    report = {"symbol": symbol, "report_type": "quick"}
    report["price_summary"] = price_summary

    # TA signal
    ta_data = raw.get("technical", {})
    if isinstance(ta_data, dict) and "_error" not in ta_data:
        report["signal"] = ta_data.get("signal")
        report["signal_score"] = ta_data.get("score")
        report["rsi"] = ta_data.get("rsi")
        report["macd_signal"] = ta_data.get("macd_signal")

    # Key ratios
    ratios = raw.get("ratios", {})
    if isinstance(ratios, dict) and "_error" not in ratios:
        report["key_ratios"] = {
            k: ratios[k] for k in ["trailingPE", "forwardPE", "priceToBook",
                                    "dividendYield", "returnOnEquity", "debtToEquity"]
            if k in ratios
        }

    # Quick summary
    signals = []
    if report.get("signal"):
        signals.append(f"TA Signal: {report['signal']} (score: {report.get('signal_score')})")
    pe = (report.get("key_ratios") or {}).get("trailingPE")
    if pe is not None:
        signals.append(f"P/E: {pe}x")
    if price_summary.get("period_return_pct") is not None:
        signals.append(f"90-day return: {price_summary['period_return_pct']}%")
    report["quick_summary"] = signals

    elapsed = round((time.time() - start_time) * 1000, 0)
    report["generation_time_ms"] = elapsed

    return report


def stock_report_compare(
    symbols: list[str],
    source: str | None = None,
) -> dict:
    """Generate a side-by-side comparison report for multiple stocks.

    For each stock: price return, TA signal, key ratios, financial health score.
    Plus cross-comparison ranking.

    Parameters
    ----------
    symbols : list[str]
        List of stock symbols (2-10 recommended).
    source : str | None
        Data source override.

    Returns
    -------
    dict with: stocks (list of per-stock summaries), rankings, comparison_summary.
    """
    if not symbols:
        return {"error": "No symbols provided"}
    if len(symbols) < 2:
        return {"error": "Need at least 2 symbols for comparison"}

    start_time = time.time()

    def _analyze_one(sym: str) -> dict:
        """Generate quick analysis for one symbol."""
        result = stock_report_quick(sym, source=source)
        # Add financial health score
        try:
            mods = _import_modules()
            health = mods["financial"].financial_health(sym)
            if isinstance(health, dict) and "error" not in health:
                z = health.get("altman_z", {})
                f = health.get("piotroski_f", {})
                result["z_score"] = z.get("z_score")
                result["z_zone"] = z.get("zone")
                result["f_score"] = f.get("score")
        except Exception:
            pass
        return result

    # Parallel per-symbol analysis
    stocks = []
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as executor:
        futures = {executor.submit(_analyze_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                stocks.append(future.result())
            except Exception as e:
                stocks.append({"symbol": futures[future], "error": str(e)})

    # Sort by input order
    sym_order = {s: i for i, s in enumerate(symbols)}
    stocks.sort(key=lambda s: sym_order.get(s.get("symbol", ""), 999))

    # Rankings
    rankings = {}

    # Rank by period return
    return_data = [
        (s["symbol"], s["price_summary"]["period_return_pct"])
        for s in stocks
        if s.get("price_summary", {}).get("period_return_pct") is not None
    ]
    if return_data:
        return_data.sort(key=lambda x: x[1], reverse=True)
        rankings["by_return"] = [{"symbol": s, "return_pct": v} for s, v in return_data]

    # Rank by TA score
    score_data = [
        (s["symbol"], s["signal_score"])
        for s in stocks
        if s.get("signal_score") is not None
    ]
    if score_data:
        score_data.sort(key=lambda x: x[1], reverse=True)
        rankings["by_ta_score"] = [{"symbol": s, "score": v} for s, v in score_data]

    # Rank by F-score
    fscore_data = [
        (s["symbol"], s["f_score"])
        for s in stocks
        if s.get("f_score") is not None
    ]
    if fscore_data:
        fscore_data.sort(key=lambda x: x[1], reverse=True)
        rankings["by_f_score"] = [{"symbol": s, "f_score": v} for s, v in fscore_data]

    elapsed = round((time.time() - start_time) * 1000, 0)

    return {
        "report_type": "comparison",
        "symbol_count": len(symbols),
        "stocks": stocks,
        "rankings": rankings,
        "generation_time_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# Executive Summary Builder
# ---------------------------------------------------------------------------


def _build_executive_summary(report: dict) -> list[str]:
    """Build a human-readable executive summary from report sections."""
    summary = []
    symbol = report.get("symbol", "")

    # Price action
    ps = report.get("price_summary", {})
    ret = ps.get("period_return_pct")
    if ret is not None:
        direction = "up" if ret > 0 else "down"
        summary.append(f"{symbol} is {direction} {abs(ret)}% over the period")

    # TA signal
    ta = report.get("technical", {})
    if isinstance(ta, dict):
        signal = ta.get("signal")
        score = ta.get("score")
        if signal:
            summary.append(f"Technical signal: {signal} (score {score}/100)")

    # Financial health
    fh = report.get("financial_health", {})
    if isinstance(fh, dict):
        z = fh.get("altman_z", {})
        f = fh.get("piotroski_f", {})
        if z.get("zone"):
            summary.append(f"Altman Z: {z['zone']} zone (Z={z.get('z_score')})")
        if f.get("score") is not None:
            summary.append(f"Piotroski F-score: {f['score']}/9 ({f.get('interpretation', '')})")

    # Valuation
    val = report.get("valuation", {})
    if isinstance(val, dict):
        dcf = val.get("dcf", {})
        if dcf.get("upside_pct") is not None:
            up = dcf["upside_pct"]
            if up > 20:
                summary.append(f"DCF suggests undervalued ({up}% upside)")
            elif up < -20:
                summary.append(f"DCF suggests overvalued ({up}% downside)")
            else:
                summary.append(f"DCF suggests fair value (upside {up}%)")

    # Sentiment
    sent = report.get("sentiment", {})
    if isinstance(sent, dict):
        avg = sent.get("average_score")
        if avg is not None:
            if avg > 0.2:
                summary.append(f"News sentiment: Positive ({avg:.2f})")
            elif avg < -0.2:
                summary.append(f"News sentiment: Negative ({avg:.2f})")
            else:
                summary.append(f"News sentiment: Neutral ({avg:.2f})")

    # ML prediction
    ml = report.get("ml_prediction", {})
    if isinstance(ml, dict) and ml.get("combined_signal"):
        summary.append(f"ML+TA combined: {ml['combined_signal']} (score {ml.get('combined_score')})")

    # Growth
    growth = report.get("growth", {})
    if isinstance(growth, dict):
        growth_summary = growth.get("summary", [])
        for item in growth_summary[:2]:  # Take top 2 growth insights
            summary.append(item)

    return summary
