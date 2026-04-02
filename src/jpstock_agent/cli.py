"""CLI interface for jpstock-agent.

Usage
-----
    jpstock-agent history 7203
    jpstock-agent overview 6758 --source jquants
    jpstock-agent serve --transport sse --port 8000
"""

from __future__ import annotations

import json
from functools import wraps

import click
from tabulate import tabulate

from . import core
from .config import get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_output(data, fmt: str = "table") -> str:
    """Format data as a table or JSON string."""
    if fmt == "json":
        return json.dumps(data, indent=2, default=str, ensure_ascii=False)

    if isinstance(data, dict):
        if "error" in data:
            return f"Error: {data['error']}"
        if "message" in data:
            return data["message"]
        # Dict → two-column table
        rows = [[k, v] for k, v in data.items()]
        return tabulate(rows, headers=["Field", "Value"], tablefmt="simple")

    if isinstance(data, list):
        if not data:
            return "No data returned."
        if isinstance(data[0], dict):
            return tabulate(data, headers="keys", tablefmt="simple")
        return str(data)

    return str(data)


def common_options(f):
    """Add --source and --format options to a command."""

    @click.option("--source", "-s", default=None, help="Data source: yfinance (default) or jquants")
    @click.option("--format", "-f", "fmt", default="table", help="Output format: table (default) or json")
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="jpstock-agent")
def cli():
    """JPStock Agent – Japanese stock market data CLI & MCP server."""
    pass


# ---------------------------------------------------------------------------
# Quote Commands
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("symbol")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--interval", "-i", default="1d", help="Interval: 1d, 1wk, 1mo")
@common_options
def history(symbol, start, end, interval, source, fmt):
    """Fetch OHLCV historical price data."""
    data = core.stock_history(symbol, start, end, interval, source)
    click.echo(_format_output(data, fmt))


@cli.command()
@click.argument("symbol")
@common_options
def intraday(symbol, source, fmt):
    """Fetch intraday (1-minute) price data."""
    data = core.stock_intraday(symbol, source)
    click.echo(_format_output(data, fmt))


@cli.command()
@click.argument("symbol")
@common_options
def depth(symbol, source, fmt):
    """Fetch bid/ask price depth."""
    data = core.stock_price_depth(symbol, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Company Commands
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("symbol")
@common_options
def overview(symbol, source, fmt):
    """Fetch company overview."""
    data = core.company_overview(symbol, source)
    click.echo(_format_output(data, fmt))


@cli.command()
@click.argument("symbol")
@common_options
def shareholders(symbol, source, fmt):
    """Fetch major shareholders."""
    data = core.company_shareholders(symbol, source)
    click.echo(_format_output(data, fmt))


@cli.command()
@click.argument("symbol")
@common_options
def officers(symbol, source, fmt):
    """Fetch company officers."""
    data = core.company_officers(symbol, source)
    click.echo(_format_output(data, fmt))


@cli.command()
@click.argument("symbol")
@common_options
def news(symbol, source, fmt):
    """Fetch recent company news."""
    data = core.company_news(symbol, source)
    click.echo(_format_output(data, fmt))


@cli.command()
@click.argument("symbol")
@common_options
def events(symbol, source, fmt):
    """Fetch upcoming events (earnings, dividends, splits)."""
    data = core.company_events(symbol, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Financial Commands
# ---------------------------------------------------------------------------


@cli.command(name="balance-sheet")
@click.argument("symbol")
@click.option("--period", "-p", default="annual", help="annual or quarterly")
@common_options
def balance_sheet(symbol, period, source, fmt):
    """Fetch balance sheet data."""
    data = core.financial_balance_sheet(symbol, period, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="income")
@click.argument("symbol")
@click.option("--period", "-p", default="annual", help="annual or quarterly")
@common_options
def income(symbol, period, source, fmt):
    """Fetch income statement data."""
    data = core.financial_income_statement(symbol, period, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="cashflow")
@click.argument("symbol")
@click.option("--period", "-p", default="annual", help="annual or quarterly")
@common_options
def cashflow(symbol, period, source, fmt):
    """Fetch cash flow statement data."""
    data = core.financial_cash_flow(symbol, period, source)
    click.echo(_format_output(data, fmt))


@cli.command()
@click.argument("symbol")
@common_options
def ratio(symbol, source, fmt):
    """Fetch key financial ratios."""
    data = core.financial_ratio(symbol, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Listing Commands
# ---------------------------------------------------------------------------


@cli.command(name="symbols")
@common_options
def list_symbols(source, fmt):
    """List all TSE-listed securities."""
    data = core.listing_all_symbols(source)
    click.echo(_format_output(data, fmt))


@cli.command(name="sector")
@click.argument("sector", default="")
@common_options
def list_by_sector(sector, source, fmt):
    """List symbols by sector."""
    data = core.listing_symbols_by_sector(sector, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="market")
@click.argument("market", default="Prime")
@common_options
def list_by_market(market, source, fmt):
    """List symbols by market segment (Prime, Standard, Growth)."""
    data = core.listing_symbols_by_market(market, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="sectors")
@common_options
def list_sectors(source, fmt):
    """List all TSE sector classifications."""
    data = core.listing_sectors(source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Trading Commands
# ---------------------------------------------------------------------------


@cli.command(name="board")
@click.argument("symbols")
@common_options
def price_board(symbols, source, fmt):
    """Fetch snapshot prices for multiple symbols (comma-separated)."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    data = core.trading_price_board(sym_list, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Global Market Commands
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("pair", default="USDJPY=X")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--interval", "-i", default="1d", help="Interval: 1d, 1wk, 1mo")
@click.option("--format", "-f", "fmt", default="table", help="Output format: table or json")
def fx(pair, start, end, interval, fmt):
    """Fetch forex exchange rate history."""
    data = core.fx_history(pair, start, end, interval)
    click.echo(_format_output(data, fmt))


@cli.command()
@click.argument("symbol", default="BTC-JPY")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--interval", "-i", default="1d", help="Interval: 1d, 1wk, 1mo")
@click.option("--format", "-f", "fmt", default="table", help="Output format: table or json")
def crypto(symbol, start, end, interval, fmt):
    """Fetch cryptocurrency price history in JPY."""
    data = core.crypto_history(symbol, start, end, interval)
    click.echo(_format_output(data, fmt))


@cli.command(name="index")
@click.argument("symbol", default="^N225")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--interval", "-i", default="1d", help="Interval: 1d, 1wk, 1mo")
@click.option("--format", "-f", "fmt", default="table", help="Output format: table or json")
def world_index(symbol, start, end, interval, fmt):
    """Fetch world index historical data."""
    data = core.world_index_history(symbol, start, end, interval)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# J-Quants Specific Commands
# ---------------------------------------------------------------------------


@cli.command(name="jq-statements")
@click.argument("symbol")
@click.option("--format", "-f", "fmt", default="table", help="Output format: table or json")
def jq_statements(symbol, fmt):
    """Fetch financial statements from J-Quants."""
    data = core.jquants_financial_statements(symbol)
    click.echo(_format_output(data, fmt))


@cli.command(name="jq-calendar")
@click.option("--from", "from_date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--to", "to_date", default=None, help="End date (YYYY-MM-DD)")
@click.option("--format", "-f", "fmt", default="table", help="Output format: table or json")
def jq_calendar(from_date, to_date, fmt):
    """Fetch TSE trading calendar from J-Quants."""
    data = core.jquants_trading_calendar(from_date, to_date)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Server Command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--transport", "-t", default=None, help="Transport: stdio, sse, http")
@click.option("--host", "-h", default=None, help="Server host (default: 0.0.0.0)")
@click.option("--port", "-p", default=None, type=int, help="Server port (default: 8000)")
def serve(transport, host, port):
    """Start the MCP server."""
    from .server import app
    from .config import get_settings

    settings = get_settings()
    transport = transport or settings.jpstock_mcp_transport
    host = host or settings.jpstock_mcp_host
    port = port or settings.jpstock_mcp_port

    click.echo(f"Starting JPStock MCP server ({transport}) on {host}:{port}", err=True)

    if transport == "sse":
        app.run(transport="sse", host=host, port=port)
    elif transport == "http":
        app.run(transport="streamable-http", host=host, port=port)
    else:
        app.run(transport="stdio")


if __name__ == "__main__":
    cli()
