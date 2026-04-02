"""FastMCP server exposing Japanese stock market data as MCP tools.

Supports three transport modes:
  - stdio  (default) – for Claude Desktop, Cursor, etc.
  - sse    – Server-Sent Events over HTTP
  - http   – Standard HTTP
"""

from __future__ import annotations

import json

from fastmcp import FastMCP

from . import core
from .config import get_settings

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
# Server entry point
# ---------------------------------------------------------------------------


def run_server():
    """Start the MCP server with configured transport."""
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
