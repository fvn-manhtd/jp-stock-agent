"""FastMCP server exposing Japanese stock market data as MCP tools.

Supports three transport modes:
  - stdio  (default) – for Claude Desktop, Cursor, etc.
  - sse    – Server-Sent Events over HTTP
  - http   – Standard HTTP

Authentication & Rate Limiting
------------------------------
When ``JPSTOCK_AUTH_ENABLED=true``, every tool call requires a valid
API key passed via the ``api_key`` parameter.  Rate limits are enforced
per key according to the key's tier (free/pro/enterprise).
"""

from __future__ import annotations

import json

from fastmcp import FastMCP

from . import (
    alert,
    backtest,
    candlestick,
    core,
    financial,
    market,
    ml,
    options,
    portfolio,
    report,
    sentiment,
    strategy,
    ta,
)
from .auth import get_key_store
from .config import get_settings
from .ratelimit import get_rate_limiter

app = FastMCP(
    "JPStock Agent",
    instructions=(
        "Stock market data agent for Japanese and Vietnamese markets. "
        "Provides access to stock prices, company information, financial statements, "
        "market listings, forex, crypto, and world index data. "
        "Supports three data sources: 'yfinance' (default, Yahoo Finance), "
        "'jquants' (JPX official data), and 'vnstocks' (Vietnamese markets: HOSE/HNX/UPCOM). "
        "For Japanese tickers, use 4-digit codes like '7203' (Toyota) or '7203.T'. "
        "For Vietnamese tickers, use 3-letter codes like 'ACB', 'VNM', 'VIC'."
    ),
)


# ---------------------------------------------------------------------------
# Auth management tools (always accessible, no gate)
# ---------------------------------------------------------------------------


@app.tool()
def auth_usage(api_key: str = "") -> str:
    """Check your API key usage and remaining quota.

    Args:
        api_key: Your API key.
    """
    settings = get_settings()
    if not settings.jpstock_auth_enabled:
        return json.dumps({"message": "Auth is disabled on this server."})
    if not api_key:
        return json.dumps({"error": "Provide your api_key."})

    store = get_key_store(settings.jpstock_auth_key_file or None)
    result = store.validate(api_key)
    if not result.authenticated:
        return json.dumps({"error": result.error})

    limiter = get_rate_limiter()
    usage = limiter.usage(result.key_hash)
    usage["tier"] = result.tier
    usage["owner"] = result.owner
    return json.dumps(usage, default=str)


@app.tool()
def auth_tiers() -> str:
    """List available subscription tiers and their limits."""
    from .auth import TIERS
    return json.dumps(TIERS, default=str, ensure_ascii=False)


@app.tool()
def usage_daily(date: str | None = None) -> str:
    """Get daily usage summary: total calls, unique keys, top tools.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today (UTC).
    """
    from .usage import get_usage_tracker
    tracker = get_usage_tracker()
    return json.dumps(tracker.daily_summary(date), default=str)


@app.tool()
def usage_key(api_key: str = "", days: int = 7) -> str:
    """Get usage breakdown for a specific API key.

    Args:
        api_key: Your API key.
        days: Lookback period in days (default 7).
    """
    if not api_key:
        return json.dumps({"error": "Provide api_key."})
    settings = get_settings()
    store = get_key_store(settings.jpstock_auth_key_file or None)
    result = store.validate(api_key)
    if not result.authenticated:
        return json.dumps({"error": result.error})
    from .usage import get_usage_tracker
    tracker = get_usage_tracker()
    return json.dumps(tracker.key_usage(result.key_hash, days), default=str)


@app.tool()
def usage_tools(days: int = 7) -> str:
    """Get per-tool usage stats: call counts, latency, error rates.

    Args:
        days: Lookback period in days (default 7).
    """
    from .usage import get_usage_tracker
    tracker = get_usage_tracker()
    return json.dumps(tracker.tool_stats(days), default=str)


@app.tool()
def usage_revenue(days: int = 30) -> str:
    """Estimate monthly recurring revenue from active subscriptions.

    Args:
        days: Period to look back for active keys (default 30).
    """
    from .usage import get_usage_tracker
    tracker = get_usage_tracker()
    return json.dumps(tracker.revenue_estimate(days), default=str)


# ---------------------------------------------------------------------------
# Quote Tools
# ---------------------------------------------------------------------------


@app.tool()
def stock_history(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    source: str | None = None,
) -> str:
    """Fetch OHLCV historical price data for a Japanese stock.

    Args:
        symbol: Ticker code, e.g. "7203" (Toyota), "6758" (Sony).
        start: Start date (YYYY-MM-DD). Defaults to 90 days ago.
        end: End date (YYYY-MM-DD). Defaults to today.
        interval: "1d", "1wk", "1mo" for yfinance. "daily" for jquants.
        source: "yfinance" or "jquants". Uses default if not set.
    """
    result = core.stock_history(symbol, start, end, interval, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def stock_intraday(
    symbol: str,
    source: str | None = None,
) -> str:
    """Fetch intraday (1-minute) price data for today's trading session.

    Args:
        symbol: Ticker code, e.g. "7203".
        source: "yfinance" or "jquants".
    """
    result = core.stock_intraday(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def stock_price_depth(
    symbol: str,
    source: str | None = None,
) -> str:
    """Fetch bid/ask price depth (order book snapshot).

    Args:
        symbol: Ticker code, e.g. "7203".
        source: "yfinance" or "jquants".
    """
    result = core.stock_price_depth(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Company Tools
# ---------------------------------------------------------------------------


@app.tool()
def company_overview(
    symbol: str,
    source: str | None = None,
) -> str:
    """Fetch company overview: sector, industry, market cap, description.

    Args:
        symbol: Ticker code, e.g. "7203" (Toyota).
        source: "yfinance" or "jquants".
    """
    result = core.company_overview(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def company_shareholders(
    symbol: str,
    source: str | None = None,
) -> str:
    """Fetch major and institutional shareholders.

    Args:
        symbol: Ticker code.
        source: "yfinance" (jquants not supported for this).
    """
    result = core.company_shareholders(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def company_officers(
    symbol: str,
    source: str | None = None,
) -> str:
    """Fetch company officers and key executives.

    Args:
        symbol: Ticker code.
        source: "yfinance" (jquants not supported for this).
    """
    result = core.company_officers(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def company_news(
    symbol: str,
    source: str | None = None,
) -> str:
    """Fetch recent news articles for a company.

    Args:
        symbol: Ticker code.
        source: "yfinance" (jquants not supported for this).
    """
    result = core.company_news(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def company_events(
    symbol: str,
    source: str | None = None,
) -> str:
    """Fetch upcoming events: earnings dates, dividends, stock splits.

    Args:
        symbol: Ticker code.
        source: "yfinance" (jquants not supported for this).
    """
    result = core.company_events(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Financial Statement Tools
# ---------------------------------------------------------------------------


@app.tool()
def financial_balance_sheet(
    symbol: str,
    period: str = "annual",
    source: str | None = None,
) -> str:
    """Fetch balance sheet data (annual or quarterly).

    Args:
        symbol: Ticker code.
        period: "annual" or "quarterly".
        source: "yfinance" or "jquants".
    """
    result = core.financial_balance_sheet(symbol, period, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def financial_income_statement(
    symbol: str,
    period: str = "annual",
    source: str | None = None,
) -> str:
    """Fetch income statement data (annual or quarterly).

    Args:
        symbol: Ticker code.
        period: "annual" or "quarterly".
        source: "yfinance" or "jquants".
    """
    result = core.financial_income_statement(symbol, period, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def financial_cash_flow(
    symbol: str,
    period: str = "annual",
    source: str | None = None,
) -> str:
    """Fetch cash flow statement data (annual or quarterly).

    Args:
        symbol: Ticker code.
        period: "annual" or "quarterly".
        source: "yfinance" or "jquants".
    """
    result = core.financial_cash_flow(symbol, period, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def financial_ratio(
    symbol: str,
    source: str | None = None,
) -> str:
    """Fetch key financial ratios: PE, PB, ROE, margins, etc.

    Args:
        symbol: Ticker code.
        source: "yfinance" (jquants not supported for ratios).
    """
    result = core.financial_ratio(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Listing Tools
# ---------------------------------------------------------------------------


@app.tool()
def listing_all_symbols(
    source: str | None = None,
) -> str:
    """List all securities on the Tokyo Stock Exchange.

    Args:
        source: "jquants" recommended for full listing. "yfinance" returns common indices only.
    """
    result = core.listing_all_symbols(source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def listing_symbols_by_sector(
    sector: str = "",
    source: str | None = None,
) -> str:
    """List symbols filtered by sector/industry.

    Args:
        sector: Sector name to filter, e.g. "Electric Appliances", "Banks".
        source: "jquants" required.
    """
    result = core.listing_symbols_by_sector(sector, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def listing_symbols_by_market(
    market: str = "Prime",
    source: str | None = None,
) -> str:
    """List symbols by TSE market segment.

    Args:
        market: "Prime", "Standard", or "Growth".
        source: "jquants" required.
    """
    result = core.listing_symbols_by_market(market, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def listing_sectors(
    source: str | None = None,
) -> str:
    """List all TSE sector/industry classifications.

    Args:
        source: "jquants" for dynamic data, "yfinance" for static reference list.
    """
    result = core.listing_sectors(source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Trading Tools
# ---------------------------------------------------------------------------


@app.tool()
def trading_price_board(
    symbols: str,
    source: str | None = None,
) -> str:
    """Fetch snapshot prices for multiple symbols at once.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        source: "yfinance" or "jquants".
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = core.trading_price_board(sym_list, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Global Market Tools
# ---------------------------------------------------------------------------


@app.tool()
def fx_history(
    pair: str = "USDJPY=X",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> str:
    """Fetch forex exchange rate history.

    Args:
        pair: Currency pair, e.g. "USDJPY=X", "EURJPY=X", "GBPJPY=X".
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        interval: "1d", "1wk", "1mo".
    """
    result = core.fx_history(pair, start, end, interval)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def crypto_history(
    symbol: str = "BTC-JPY",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> str:
    """Fetch cryptocurrency price history in JPY.

    Args:
        symbol: Crypto pair, e.g. "BTC-JPY", "ETH-JPY".
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        interval: "1d", "1wk", "1mo".
    """
    result = core.crypto_history(symbol, start, end, interval)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def world_index_history(
    symbol: str = "^N225",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> str:
    """Fetch world index historical data.

    Args:
        symbol: Index symbol – "^N225" (Nikkei), "^TOPX" (TOPIX), "^GSPC" (S&P 500).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        interval: "1d", "1wk", "1mo".
    """
    result = core.world_index_history(symbol, start, end, interval)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# vnstock Specific Tools
# ---------------------------------------------------------------------------


@app.tool()
def vnstocks_listing(
    exchange: str = "HOSE",
) -> str:
    """List all securities on a Vietnamese stock exchange (HOSE, HNX, UPCOM).

    Args:
        exchange: "HOSE" (Ho Chi Minh), "HNX" (Hanoi), "UPCOM", or "all".
    """
    result = core.vnstocks_listing(exchange)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def vnstocks_price_board(
    symbols: str,
) -> str:
    """Fetch current price snapshot for multiple Vietnamese stocks.

    Args:
        symbols: Comma-separated Vietnamese ticker codes, e.g. "ACB,VNM,VIC".
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = core.vnstocks_price_board(sym_list)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# J-Quants Specific Tools
# ---------------------------------------------------------------------------


@app.tool()
def jquants_financial_statements(
    symbol: str,
) -> str:
    """Fetch financial statements from J-Quants API (requires J-Quants credentials).

    Args:
        symbol: Ticker code, e.g. "7203".
    """
    result = core.jquants_financial_statements(symbol)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def jquants_trading_calendar(
    from_date: str | None = None,
    to_date: str | None = None,
) -> str:
    """Fetch TSE trading calendar (market open/close days).

    Args:
        from_date: Start date (YYYY-MM-DD).
        to_date: End date (YYYY-MM-DD).
    """
    result = core.jquants_trading_calendar(from_date, to_date)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Technical Analysis Tools
# ---------------------------------------------------------------------------


@app.tool()
def ta_sma(
    symbol: str,
    period: int = 20,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Simple Moving Average (SMA).

    Args:
        symbol: Ticker code, e.g. "7203" or "ACB".
        period: Lookback period (default 20).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    result = ta.ta_sma(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_ema(
    symbol: str,
    period: int = 20,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Exponential Moving Average (EMA).

    Args:
        symbol: Ticker code.
        period: Lookback period (default 20).
    """
    result = ta.ta_ema(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_rsi(
    symbol: str,
    period: int = 14,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Relative Strength Index (RSI). RSI<30=oversold, RSI>70=overbought.

    Args:
        symbol: Ticker code.
        period: Lookback period (default 14).
    """
    result = ta.ta_rsi(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_macd(
    symbol: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        symbol: Ticker code.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line period (default 9).
    """
    result = ta.ta_macd(symbol, fast, slow, signal, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_bbands(
    symbol: str,
    period: int = 20,
    std_dev: float = 2.0,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Bollinger Bands (upper, middle, lower).

    Args:
        symbol: Ticker code.
        period: Lookback period (default 20).
        std_dev: Standard deviation multiplier (default 2.0).
    """
    result = ta.ta_bbands(symbol, period, std_dev, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_ichimoku(
    symbol: str,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Ichimoku Cloud (Japan's most popular indicator).

    Args:
        symbol: Ticker code.
        tenkan: Conversion line period (default 9).
        kijun: Base line period (default 26).
        senkou: Leading span B period (default 52).
    """
    result = ta.ta_ichimoku(symbol, tenkan, kijun, senkou, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_stochastic(
    symbol: str,
    k_period: int = 14,
    d_period: int = 3,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Stochastic Oscillator (%K, %D). %K<20=oversold, %K>80=overbought.

    Args:
        symbol: Ticker code.
        k_period: %K period (default 14).
        d_period: %D period (default 3).
    """
    result = ta.ta_stochastic(symbol, k_period, d_period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_atr(
    symbol: str,
    period: int = 14,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Average True Range (ATR) - volatility measure.

    Args:
        symbol: Ticker code.
        period: Lookback period (default 14).
    """
    result = ta.ta_atr(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_supertrend(
    symbol: str,
    period: int = 10,
    multiplier: float = 3.0,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Supertrend indicator (trend direction + support/resistance).

    Args:
        symbol: Ticker code.
        period: ATR period (default 10).
        multiplier: ATR multiplier (default 3.0).
    """
    result = ta.ta_supertrend(symbol, period, multiplier, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_obv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate On-Balance Volume (OBV) - volume trend indicator.

    Args:
        symbol: Ticker code.
    """
    result = ta.ta_obv(symbol, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_vwap(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Volume Weighted Average Price (VWAP).

    Args:
        symbol: Ticker code.
    """
    result = ta.ta_vwap(symbol, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_mfi(
    symbol: str,
    period: int = 14,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Money Flow Index (MFI) - volume-weighted RSI. MFI<20=oversold, MFI>80=overbought.

    Args:
        symbol: Ticker code.
        period: Lookback period (default 14).
    """
    result = ta.ta_mfi(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_williams_r(
    symbol: str,
    period: int = 14,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Williams %R indicator.

    Args:
        symbol: Ticker code.
        period: Lookback period (default 14).
    """
    result = ta.ta_williams_r(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_cci(
    symbol: str,
    period: int = 20,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Commodity Channel Index (CCI).

    Args:
        symbol: Ticker code.
        period: Lookback period (default 20).
    """
    result = ta.ta_cci(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_roc(
    symbol: str,
    period: int = 12,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Rate of Change (ROC).

    Args:
        symbol: Ticker code.
        period: Lookback period (default 12).
    """
    result = ta.ta_roc(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_keltner(
    symbol: str,
    period: int = 20,
    multiplier: float = 2.0,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Keltner Channels.

    Args:
        symbol: Ticker code.
        period: Lookback period (default 20).
        multiplier: ATR multiplier (default 2.0).
    """
    result = ta.ta_keltner(symbol, period, multiplier, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_donchian(
    symbol: str,
    period: int = 20,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Donchian Channels (highest high / lowest low).

    Args:
        symbol: Ticker code.
        period: Lookback period (default 20).
    """
    result = ta.ta_donchian(symbol, period, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_parabolic_sar(
    symbol: str,
    af: float = 0.02,
    max_af: float = 0.2,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Parabolic SAR (Stop and Reverse).

    Args:
        symbol: Ticker code.
        af: Acceleration factor (default 0.02).
        max_af: Maximum acceleration factor (default 0.2).
    """
    result = ta.ta_parabolic_sar(symbol, af, max_af, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_ad(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Accumulation/Distribution Line.

    Args:
        symbol: Ticker code.
    """
    result = ta.ta_ad(symbol, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_fibonacci(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Calculate Fibonacci retracement levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%).

    Args:
        symbol: Ticker code.
    """
    result = ta.ta_fibonacci(symbol, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_support_resistance(
    symbol: str,
    window: int = 20,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Detect support and resistance levels using pivot points.

    Args:
        symbol: Ticker code.
        window: Rolling window for local highs/lows (default 20).
    """
    result = ta.ta_support_resistance(symbol, window, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_multi_indicator(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Run comprehensive multi-indicator analysis with BUY/SELL/HOLD signal.

    Combines RSI, MACD, Bollinger Bands, Stochastic, moving averages,
    and generates an overall signal score from -100 (strong sell) to +100 (strong buy).

    Args:
        symbol: Ticker code.
    """
    result = ta.ta_multi_indicator(symbol, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_screen(
    symbols: str,
    strategy: str = "oversold",
    source: str | None = None,
) -> str:
    """Screen multiple stocks for technical signals.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984" or "ACB,VNM,VIC".
        strategy: One of: "oversold", "overbought", "macd_bullish", "macd_bearish",
                  "bb_squeeze", "golden_cross", "death_cross", "volume_spike",
                  "trend_up", "trend_down".
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = ta.ta_screen(sym_list, strategy, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_multi_timeframe(
    symbol: str,
    source: str | None = None,
) -> str:
    """Analyze a stock across daily, weekly, and monthly timeframes.

    Returns RSI, MACD, trend direction for each timeframe.

    Args:
        symbol: Ticker code.
    """
    result = ta.ta_multi_timeframe(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Candlestick Pattern Tools
# ---------------------------------------------------------------------------


@app.tool()
def ta_candlestick_scan(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Scan for all Japanese candlestick patterns in recent price data.

    Detects 20 patterns: Hammer, Inverted Hammer, Hanging Man, Shooting Star,
    Doji, Spinning Top, High Wave, Marubozu, Engulfing, Tweezer, Piercing Line,
    Morning Star, Evening Star, Three White Soldiers, Three Black Crows, etc.

    Args:
        symbol: Ticker code, e.g. "7203" or "ACB".
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    result = candlestick.ta_candlestick_scan(symbol, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_candlestick_latest(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Get candlestick patterns detected on the most recent trading day.

    Args:
        symbol: Ticker code.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    result = candlestick.ta_candlestick_latest(symbol, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ta_candlestick_screen(
    symbols: str,
    pattern: str = "all",
    source: str | None = None,
) -> str:
    """Screen multiple stocks for candlestick patterns.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        pattern: Filter - "all", "bullish", "bearish", or specific pattern name
                 (e.g. "hammer", "doji", "engulfing").
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = candlestick.ta_candlestick_screen(sym_list, pattern, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Backtesting Tools
# ---------------------------------------------------------------------------


@app.tool()
def backtest_strategy(
    symbol: str,
    strategy: str = "sma_crossover",
    start: str | None = None,
    end: str | None = None,
    initial_capital: float = 1000000,
    source: str | None = None,
) -> str:
    """Backtest a trading strategy on historical data.

    Runs the strategy and returns performance metrics: total return, win rate,
    max drawdown, Sharpe ratio, alpha vs buy-and-hold.

    Args:
        symbol: Ticker code, e.g. "7203".
        strategy: One of: "sma_crossover", "ema_crossover", "rsi_reversal",
                  "macd_crossover", "bollinger_bounce", "supertrend",
                  "ichimoku_cloud", "golden_cross", "mean_reversion",
                  "momentum", "breakout", "vwap_strategy".
        start: Start date (YYYY-MM-DD). Defaults to 1 year ago.
        end: End date (YYYY-MM-DD).
        initial_capital: Starting capital in JPY (default 1,000,000).
        source: Data source override.
    """
    result = backtest.backtest_strategy(symbol, strategy, start, end, initial_capital, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def backtest_compare(
    symbol: str,
    strategies: str | None = None,
    start: str | None = None,
    end: str | None = None,
    initial_capital: float = 1000000,
    source: str | None = None,
) -> str:
    """Compare multiple backtesting strategies side by side.

    Args:
        symbol: Ticker code.
        strategies: Comma-separated strategy names. If empty, runs all 12 strategies.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        initial_capital: Starting capital in JPY (default 1,000,000).
        source: Data source override.
    """
    strat_list = None
    if strategies:
        strat_list = [s.strip() for s in strategies.split(",") if s.strip()]
    result = backtest.backtest_compare(symbol, strat_list, start, end, initial_capital, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def backtest_optimize(
    symbol: str,
    strategy: str = "sma_crossover",
    param_name: str = "fast_period",
    param_values: str = "10,15,20,25,30",
    start: str | None = None,
    end: str | None = None,
    initial_capital: float = 1000000,
    source: str | None = None,
) -> str:
    """Optimize a strategy parameter by testing multiple values.

    Args:
        symbol: Ticker code.
        strategy: Strategy to optimize.
        param_name: Parameter to vary (e.g. "fast_period", "rsi_period").
        param_values: Comma-separated values to test, e.g. "10,15,20,25,30".
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        initial_capital: Starting capital.
        source: Data source override.
    """
    values = [float(v.strip()) for v in param_values.split(",") if v.strip()]
    result = backtest.backtest_optimize(symbol, strategy, param_name, values, start, end, initial_capital, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def backtest_walk_forward(
    symbol: str,
    strategy: str = "sma_crossover",
    window: int = 180,
    step: int = 30,
    start: str | None = None,
    end: str | None = None,
    initial_capital: float = 1000000,
    source: str | None = None,
) -> str:
    """Walk-forward analysis: test strategy on rolling windows for consistency.

    Args:
        symbol: Ticker code.
        strategy: Strategy to test.
        window: Window size in days (default 180).
        step: Step size in days (default 30).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        initial_capital: Starting capital.
        source: Data source override.
    """
    result = backtest.backtest_walk_forward(symbol, strategy, window, step, start, end, initial_capital, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def backtest_monte_carlo(
    symbol: str,
    strategy: str = "sma_crossover",
    num_simulations: int = 1000,
    start: str | None = None,
    end: str | None = None,
    initial_capital: float = 1000000,
    source: str | None = None,
) -> str:
    """Monte Carlo simulation for backtest strategy robustness testing.

    Randomly resamples trade returns to estimate probability distributions.

    Args:
        symbol: Ticker code.
        strategy: Strategy to simulate.
        num_simulations: Number of simulations (default 1000).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        initial_capital: Starting capital.
        source: Data source override.
    """
    result = backtest.backtest_monte_carlo(symbol, strategy, num_simulations, start, end, initial_capital, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def backtest_advanced_metrics(
    symbol: str,
    strategy: str = "sma_crossover",
    start: str | None = None,
    end: str | None = None,
    initial_capital: float = 1000000,
    source: str | None = None,
) -> str:
    """Advanced backtest metrics: Sortino, Calmar, profit factor, expectancy.

    Args:
        symbol: Ticker code.
        strategy: Strategy name.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        initial_capital: Starting capital.
        source: Data source override.
    """
    result = backtest.backtest_advanced_metrics(symbol, strategy, start, end, initial_capital, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def backtest_realistic(
    symbol: str,
    strategy: str = "sma_crossover",
    market: str = "jp",
    position_sizing: str = "atr",
    start: str | None = None,
    end: str | None = None,
    initial_capital: float = 1000000,
    source: str | None = None,
) -> str:
    """Backtest with realistic transaction costs and position sizing.

    Uses market-appropriate costs (commission, slippage, spread) and
    configurable position sizing. Also compares against zero-cost backtest.

    Args:
        symbol: Ticker code, e.g. "7203".
        strategy: Strategy name (sma_crossover, ema_crossover, etc.).
        market: Cost preset: "jp" (Japanese, default), "vn" (Vietnamese), "zero".
        position_sizing: "full" (all-in), "kelly" (Kelly Criterion),
                         "atr" (ATR-based, default), "max_loss", "fixed_fraction".
        start: Start date (YYYY-MM-DD). Defaults to 1 year ago.
        end: End date (YYYY-MM-DD).
        initial_capital: Starting capital (default 1,000,000).
        source: Data source override.
    """
    result = backtest.backtest_realistic(
        symbol, strategy, market=market, position_sizing=position_sizing,
        start=start, end=end, initial_capital=initial_capital, source=source,
    )
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def stock_history_batch(
    symbols: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    source: str | None = None,
) -> str:
    """Fetch OHLCV data for multiple symbols in parallel (much faster than sequential).

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        start: Start date (YYYY-MM-DD). Defaults to 90 days ago.
        end: End date (YYYY-MM-DD).
        interval: "1d", "1wk", "1mo".
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = core.stock_history_batch(sym_list, start, end, interval, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Portfolio Optimization Tools
# ---------------------------------------------------------------------------


@app.tool()
def portfolio_analyze(
    symbols: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Analyze a portfolio: per-stock returns, volatility, correlations.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        start: Start date (YYYY-MM-DD). Defaults to 1 year ago.
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = portfolio.portfolio_analyze(sym_list, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def portfolio_optimize(
    symbols: str,
    num_portfolios: int = 5000,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Monte Carlo portfolio optimization to find optimal weight allocation.

    Finds max Sharpe ratio, min volatility, and max return portfolios.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        num_portfolios: Number of random portfolios to simulate (default 5000).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = portfolio.portfolio_optimize(sym_list, start, end, num_portfolios, source=source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def portfolio_risk(
    symbols: str,
    weights: str | None = None,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Portfolio risk analysis: VaR, CVaR, Sortino, max drawdown.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        weights: Comma-separated weights, e.g. "0.5,0.3,0.2". Default: equal weight.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    w_list = None
    if weights:
        w_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
    result = portfolio.portfolio_risk(sym_list, w_list, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def portfolio_correlation(
    symbols: str,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Correlation and covariance matrix for a set of stocks.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = portfolio.portfolio_correlation(sym_list, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Sentiment Analysis Tools
# ---------------------------------------------------------------------------


@app.tool()
def sentiment_news(
    symbol: str,
    source: str | None = None,
) -> str:
    """Analyze sentiment from recent news headlines for a stock.

    Scores each headline from -1 (very bearish) to +1 (very bullish)
    using keyword matching (English + Japanese).

    Args:
        symbol: Ticker code, e.g. "7203" or "ACB".
        source: Data source override.
    """
    result = sentiment.sentiment_news(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def sentiment_market(
    symbols: str,
    source: str | None = None,
) -> str:
    """Batch sentiment analysis for multiple stocks.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = sentiment.sentiment_market(sym_list, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def sentiment_combined(
    symbol: str,
    source: str | None = None,
) -> str:
    """Combined technical + sentiment signal (70% TA + 30% sentiment).

    Merges multi-indicator TA score with news sentiment for a comprehensive signal.

    Args:
        symbol: Ticker code.
        source: Data source override.
    """
    result = sentiment.sentiment_combined(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def sentiment_screen(
    symbols: str,
    min_score: float = 0.0,
    source: str | None = None,
) -> str:
    """Screen stocks by news sentiment score.

    Args:
        symbols: Comma-separated ticker codes.
        min_score: Minimum sentiment score (-1 to 1). Default 0.0 (neutral+).
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = sentiment.sentiment_screen(sym_list, min_score, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ML Signal Generation Tools
# ---------------------------------------------------------------------------


@app.tool()
def ml_predict(
    symbol: str,
    horizon: int = 5,
    model_type: str = "random_forest",
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """ML price prediction: probability of price increase in next N days.

    Uses Random Forest or Gradient Boosting trained on 30+ TA features.

    Args:
        symbol: Ticker code, e.g. "7203" or "AAPL".
        horizon: Prediction horizon in trading days (default 5).
        model_type: "random_forest" (default) or "gradient_boosting".
        start: Training start date (YYYY-MM-DD). Default: 2 years ago.
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    result = ml.ml_predict(symbol, horizon, model_type, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ml_feature_importance(
    symbol: str,
    horizon: int = 5,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
) -> str:
    """Rank TA indicators by predictive power using Random Forest feature importance.

    Helps identify which indicators matter most for a specific stock.

    Args:
        symbol: Ticker code.
        horizon: Prediction horizon in days (default 5).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        source: Data source override.
    """
    result = ml.ml_feature_importance(symbol, horizon, start, end, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ml_signal(
    symbol: str,
    horizon: int = 5,
    ml_weight: float = 0.5,
    source: str | None = None,
) -> str:
    """Combined ML + TA signal with configurable weighting.

    Blends ML prediction probability with multi-indicator TA score.

    Args:
        symbol: Ticker code.
        horizon: ML prediction horizon in days (default 5).
        ml_weight: Weight for ML signal 0.0-1.0 (default 0.5). TA = 1 - ml_weight.
        source: Data source override.
    """
    result = ml.ml_signal(symbol, horizon, ml_weight, source=source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def ml_batch_predict(
    symbols: str,
    horizon: int = 5,
    model_type: str = "random_forest",
    source: str | None = None,
) -> str:
    """ML prediction for multiple symbols, sorted by probability of increase.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        horizon: Prediction horizon in days (default 5).
        model_type: "random_forest" or "gradient_boosting".
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = ml.ml_batch_predict(sym_list, horizon, model_type, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Options & Derivatives Tools
# ---------------------------------------------------------------------------


@app.tool()
def options_chain(
    symbol: str,
    expiry: str | None = None,
) -> str:
    """Fetch options chain (calls and puts) with IV, volume, open interest.

    Args:
        symbol: Ticker code (mainly US stocks, e.g. "AAPL", "MSFT").
        expiry: Specific expiry date (YYYY-MM-DD). Default: nearest expiry.
    """
    result = options.options_chain(symbol, expiry)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def options_greeks(
    symbol: str,
    expiry: str | None = None,
    option_type: str = "call",
    risk_free_rate: float = 0.05,
) -> str:
    """Calculate Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho).

    Args:
        symbol: Ticker code.
        expiry: Expiry date (YYYY-MM-DD). Default: nearest expiry.
        option_type: "call" (default) or "put".
        risk_free_rate: Annual risk-free rate (default 0.05 = 5%).
    """
    result = options.options_greeks(symbol, expiry, risk_free_rate, option_type)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def options_iv_surface(
    symbol: str,
    max_expiries: int = 6,
) -> str:
    """Build implied volatility surface (strike × expiry) with skew analysis.

    Args:
        symbol: Ticker code.
        max_expiries: Maximum expiry dates to include (default 6).
    """
    result = options.options_iv_surface(symbol, max_expiries)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def options_unusual_activity(
    symbol: str,
    volume_threshold: float = 2.0,
    expiry: str | None = None,
) -> str:
    """Detect unusual options activity (high volume/OI ratio).

    Args:
        symbol: Ticker code.
        volume_threshold: Volume/OI ratio threshold (default 2.0).
        expiry: Specific expiry date. Default: nearest expiry.
    """
    result = options.options_unusual_activity(symbol, volume_threshold, expiry)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def options_put_call_ratio(
    symbol: str,
) -> str:
    """Put/call ratio sentiment indicator across all expiries.

    P/C > 1.0 = bearish, P/C < 0.7 = bullish.

    Args:
        symbol: Ticker code.
    """
    result = options.options_put_call_ratio(symbol)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def options_max_pain(
    symbol: str,
    expiry: str | None = None,
) -> str:
    """Calculate max pain strike price (where option sellers face minimum payout).

    Args:
        symbol: Ticker code.
        expiry: Expiry date (YYYY-MM-DD). Default: nearest expiry.
    """
    result = options.options_max_pain(symbol, expiry)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Fundamental Financial Analysis Tools
# ---------------------------------------------------------------------------


@app.tool()
def financial_health(
    symbol: str,
    period: str = "annual",
) -> str:
    """Financial health assessment: Altman Z-score, Piotroski F-score, liquidity, leverage.

    Args:
        symbol: Ticker code, e.g. "7203" or "ACB".
        period: "annual" or "quarterly".
    """
    result = financial.financial_health(symbol, period)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def financial_growth(
    symbol: str,
    period: str = "annual",
) -> str:
    """Growth trend analysis: YoY revenue/earnings growth, margin trends, FCF trends.

    Args:
        symbol: Ticker code.
        period: "annual" or "quarterly".
    """
    result = financial.financial_growth(symbol, period)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def financial_valuation(
    symbol: str,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.02,
    projection_years: int = 5,
) -> str:
    """Intrinsic value estimate using DCF and relative valuation (P/E, EV/EBITDA).

    Args:
        symbol: Ticker code.
        discount_rate: WACC / required return (default 0.10 = 10%).
        terminal_growth: Perpetual growth rate (default 0.02 = 2%).
        projection_years: FCF projection horizon (default 5).
    """
    result = financial.financial_valuation(symbol, discount_rate, terminal_growth, projection_years)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def financial_peer_compare(
    symbols: str,
    period: str = "annual",
) -> str:
    """Side-by-side financial comparison of multiple stocks.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        period: "annual" or "quarterly".
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = financial.financial_peer_compare(sym_list, period)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def financial_dividend(
    symbol: str,
) -> str:
    """Dividend analysis: yield, payout ratio, growth rate, sustainability check.

    Args:
        symbol: Ticker code.
    """
    result = financial.financial_dividend(symbol)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def financial_ratios_calc(
    symbol: str,
    period: str = "annual",
) -> str:
    """Compute financial ratios from raw statements (profitability, efficiency, leverage, DuPont).

    Unlike the pre-calculated ratios from yfinance, this computes ratios directly
    from balance sheet and income statement data.

    Args:
        symbol: Ticker code.
        period: "annual" or "quarterly".
    """
    result = financial.financial_ratios_calc(symbol, period)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Stock Report Tools
# ---------------------------------------------------------------------------


@app.tool()
def stock_report(
    symbol: str,
    include_ml: bool = True,
    include_options: bool = False,
    period: str = "annual",
    source: str | None = None,
) -> str:
    """Comprehensive stock report: TA + financial health + valuation + sentiment + ML in one call.

    Args:
        symbol: Ticker code, e.g. "7203" or "ACB".
        include_ml: Include ML prediction (requires scikit-learn). Default True.
        include_options: Include options analysis. Default False.
        period: Financial period: "annual" or "quarterly".
        source: Data source override.
    """
    result = report.stock_report(symbol, include_ml, include_options, period, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def stock_report_quick(
    symbol: str,
    source: str | None = None,
) -> str:
    """Quick stock summary: price, TA signal, key ratios. Much faster than full report.

    Args:
        symbol: Ticker code.
        source: Data source override.
    """
    result = report.stock_report_quick(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def stock_report_compare(
    symbols: str,
    source: str | None = None,
) -> str:
    """Side-by-side comparison report for multiple stocks.

    Args:
        symbols: Comma-separated ticker codes, e.g. "7203,6758,9984".
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = report.stock_report_compare(sym_list, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Alert & Watchlist Tools
# ---------------------------------------------------------------------------


@app.tool()
def alert_check(
    symbol: str,
    conditions: str,
    source: str | None = None,
) -> str:
    """Evaluate custom alert conditions for a symbol.

    Args:
        symbol: Ticker code.
        conditions: JSON array of conditions, e.g. '[{"condition":"rsi_oversold","params":{"threshold":25}}]'.
        source: Data source override.
    """
    import json as json_mod
    try:
        cond_list = json_mod.loads(conditions)
    except json_mod.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for conditions"})
    result = alert.alert_check(symbol, cond_list, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def alert_price(
    symbol: str,
    above: float | None = None,
    below: float | None = None,
    source: str | None = None,
) -> str:
    """Quick price-level alert: check if price is above/below thresholds.

    Args:
        symbol: Ticker code.
        above: Alert if price above this level.
        below: Alert if price below this level.
        source: Data source override.
    """
    result = alert.alert_price(symbol, above, below, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def alert_ta(
    symbol: str,
    conditions: str | None = None,
    source: str | None = None,
) -> str:
    """Check TA-based alerts: RSI extremes, MACD cross, BB squeeze, volume spike, etc.

    Args:
        symbol: Ticker code.
        conditions: Comma-separated condition names (default: all common).
            Available: rsi_oversold, rsi_overbought, macd_bullish_cross,
            macd_bearish_cross, bb_squeeze, volume_spike, golden/death_cross.
        source: Data source override.
    """
    cond_list = None
    if conditions:
        cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    result = alert.alert_ta(symbol, cond_list, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def alert_fundamental(
    symbol: str,
    pe_below: float | None = None,
    pe_above: float | None = None,
    yield_above: float | None = None,
    roe_above: float | None = None,
    debt_to_equity_below: float | None = None,
    f_score_above: int | None = None,
) -> str:
    """Fundamental-based alerts: P/E, yield, ROE, D/E, F-score thresholds.

    Args:
        symbol: Ticker code.
        pe_below: Alert if P/E below this value.
        pe_above: Alert if P/E above this value.
        yield_above: Alert if dividend yield above this (e.g. 0.03 for 3%).
        roe_above: Alert if ROE above this threshold.
        debt_to_equity_below: Alert if D/E below this value.
        f_score_above: Alert if Piotroski F-score above this (0-9).
    """
    result = alert.alert_fundamental(
        symbol, pe_below, pe_above, yield_above,
        roe_above, debt_to_equity_below, f_score_above,
    )
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def alert_watchlist(
    symbols: str,
    conditions: str | None = None,
    source: str | None = None,
) -> str:
    """Check multiple symbols against common TA alert conditions.

    Args:
        symbols: Comma-separated ticker codes.
        conditions: Comma-separated TA condition names (default: common set).
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    cond_list = None
    if conditions:
        cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    result = alert.alert_watchlist(sym_list, cond_list, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def alert_list_conditions() -> str:
    """List all available alert conditions with descriptions and default parameters."""
    result = alert.alert_list_conditions()
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Market & Sector Analysis Tools
# ---------------------------------------------------------------------------


@app.tool()
def market_sector_performance(
    sectors: str,
    days: int = 30,
    source: str | None = None,
) -> str:
    """Compare performance across sectors. Returns ranking from best to worst.

    Args:
        sectors: JSON mapping sector→symbol lists,
            e.g. '{"Auto":["7203","7267"],"Tech":["6758","9984"]}'.
        days: Lookback period in days (default 30).
        source: Data source override.
    """
    import json as json_mod
    try:
        sector_dict = json_mod.loads(sectors)
    except json_mod.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for sectors"})
    result = market.market_sector_performance(sector_dict, days, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def market_breadth(
    symbols: str,
    days: int = 1,
    source: str | None = None,
) -> str:
    """Market breadth: advance/decline ratio, new 52w highs/lows, breadth signal.

    Args:
        symbols: Comma-separated ticker codes (market universe).
        days: Lookback period (default 1).
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = market.market_breadth(sym_list, days, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def market_top_movers(
    symbols: str,
    days: int = 1,
    top_n: int = 5,
    source: str | None = None,
) -> str:
    """Top gainers and losers from a list of symbols.

    Args:
        symbols: Comma-separated ticker codes.
        days: Lookback period (default 1).
        top_n: Number of top movers each direction (default 5).
        source: Data source override.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = market.market_top_movers(sym_list, days, top_n, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def market_regime(
    symbol: str = "^N225",
    source: str | None = None,
) -> str:
    """Detect market regime: BULL, BEAR, SIDEWAYS using SMA/momentum/volatility.

    Args:
        symbol: Index symbol. Default "^N225" (Nikkei). Others: "^GSPC" (S&P500), "^VNI" (VN-Index).
        source: Data source override.
    """
    result = market.market_regime(symbol, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def market_heatmap(
    sectors: str,
    days: int = 1,
    source: str | None = None,
) -> str:
    """Sector-grouped performance heatmap data.

    Args:
        sectors: JSON object mapping sector names to symbol lists.
        days: Lookback period (default 1).
        source: Data source override.
    """
    import json as json_mod
    try:
        sector_dict = json_mod.loads(sectors)
    except json_mod.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for sectors"})
    result = market.market_heatmap(sector_dict, days, source)
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Custom Strategy Builder Tools
# ---------------------------------------------------------------------------


@app.tool()
def strategy_evaluate(
    symbol: str,
    conditions: str,
    logic: str = "AND",
    source: str | None = None,
) -> str:
    """Evaluate custom composed conditions against a single stock.

    Combine TA + fundamental + price conditions with AND/OR logic.

    Args:
        symbol: Ticker code.
        conditions: JSON array of condition objects with type and params.
            E.g. '[{"type":"rsi_below","params":{"value":30}},{"type":"macd_bullish"}]'.
        logic: "AND" (all must pass) or "OR" (any). Default "AND".
        source: Data source override.
    """
    import json as json_mod
    try:
        cond_list = json_mod.loads(conditions)
    except json_mod.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for conditions"})
    result = strategy.strategy_evaluate(symbol, cond_list, logic, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def strategy_screen(
    symbols: str,
    conditions: str,
    logic: str = "AND",
    source: str | None = None,
) -> str:
    """Screen multiple stocks with custom composed conditions.

    Args:
        symbols: Comma-separated ticker codes.
        conditions: JSON array of conditions (same format as strategy_evaluate).
        logic: "AND" or "OR". Default "AND".
        source: Data source override.
    """
    import json as json_mod
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    try:
        cond_list = json_mod.loads(conditions)
    except json_mod.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for conditions"})
    result = strategy.strategy_screen(sym_list, cond_list, logic, source)
    return json.dumps(result, default=str, ensure_ascii=False)


@app.tool()
def strategy_list_conditions() -> str:
    """List all available strategy condition types grouped by category."""
    result = strategy.strategy_list_conditions()
    return json.dumps(result, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def run_server():
    """Start the MCP server with configured transport.

    For HTTP/SSE transports, applies :class:`AuthMiddleware` when
    ``JPSTOCK_AUTH_ENABLED=true`` or ``JPSTOCK_RATE_LIMIT_ENABLED=true``.
    """
    settings = get_settings()
    transport = settings.jpstock_mcp_transport.lower()

    if transport == "sse":
        app.run(
            transport="sse",
            host=settings.jpstock_mcp_host,
            port=settings.jpstock_mcp_port,
        )
    elif transport == "http":
        app.run(
            transport="streamable-http",
            host=settings.jpstock_mcp_host,
            port=settings.jpstock_mcp_port,
        )
    else:
        app.run(transport="stdio")


if __name__ == "__main__":
    run_server()
