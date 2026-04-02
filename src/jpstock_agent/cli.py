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

from . import core, ta, candlestick, backtest, portfolio, sentiment
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
# Technical Analysis Commands
# ---------------------------------------------------------------------------


@cli.command(name="ta-sma")
@click.argument("symbol")
@click.option("--period", "-p", default=20, type=int, help="SMA period (default 20)")
@common_options
def cli_ta_sma(symbol, period, source, fmt):
    """Calculate Simple Moving Average."""
    data = ta.ta_sma(symbol, period, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-ema")
@click.argument("symbol")
@click.option("--period", "-p", default=20, type=int, help="EMA period (default 20)")
@common_options
def cli_ta_ema(symbol, period, source, fmt):
    """Calculate Exponential Moving Average."""
    data = ta.ta_ema(symbol, period, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-rsi")
@click.argument("symbol")
@click.option("--period", "-p", default=14, type=int, help="RSI period (default 14)")
@common_options
def cli_ta_rsi(symbol, period, source, fmt):
    """Calculate RSI (Relative Strength Index)."""
    data = ta.ta_rsi(symbol, period, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-macd")
@click.argument("symbol")
@click.option("--fast", default=12, type=int, help="Fast period (default 12)")
@click.option("--slow", default=26, type=int, help="Slow period (default 26)")
@click.option("--signal", default=9, type=int, help="Signal period (default 9)")
@common_options
def cli_ta_macd(symbol, fast, slow, signal, source, fmt):
    """Calculate MACD indicator."""
    data = ta.ta_macd(symbol, fast, slow, signal, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-bbands")
@click.argument("symbol")
@click.option("--period", "-p", default=20, type=int, help="Period (default 20)")
@click.option("--std", default=2.0, type=float, help="Std deviation (default 2.0)")
@common_options
def cli_ta_bbands(symbol, period, std, source, fmt):
    """Calculate Bollinger Bands."""
    data = ta.ta_bbands(symbol, period, std, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-ichimoku")
@click.argument("symbol")
@common_options
def cli_ta_ichimoku(symbol, source, fmt):
    """Calculate Ichimoku Cloud."""
    data = ta.ta_ichimoku(symbol, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-stoch")
@click.argument("symbol")
@click.option("--k", "k_period", default=14, type=int, help="%K period (default 14)")
@click.option("--d", "d_period", default=3, type=int, help="%D period (default 3)")
@common_options
def cli_ta_stoch(symbol, k_period, d_period, source, fmt):
    """Calculate Stochastic Oscillator."""
    data = ta.ta_stochastic(symbol, k_period, d_period, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-atr")
@click.argument("symbol")
@click.option("--period", "-p", default=14, type=int, help="ATR period (default 14)")
@common_options
def cli_ta_atr(symbol, period, source, fmt):
    """Calculate Average True Range (ATR)."""
    data = ta.ta_atr(symbol, period, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-supertrend")
@click.argument("symbol")
@click.option("--period", "-p", default=10, type=int, help="Period (default 10)")
@click.option("--multiplier", "-m", default=3.0, type=float, help="Multiplier (default 3.0)")
@common_options
def cli_ta_supertrend(symbol, period, multiplier, source, fmt):
    """Calculate Supertrend indicator."""
    data = ta.ta_supertrend(symbol, period, multiplier, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-obv")
@click.argument("symbol")
@common_options
def cli_ta_obv(symbol, source, fmt):
    """Calculate On-Balance Volume (OBV)."""
    data = ta.ta_obv(symbol, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-vwap")
@click.argument("symbol")
@common_options
def cli_ta_vwap(symbol, source, fmt):
    """Calculate Volume Weighted Average Price (VWAP)."""
    data = ta.ta_vwap(symbol, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-fibonacci")
@click.argument("symbol")
@common_options
def cli_ta_fibonacci(symbol, source, fmt):
    """Calculate Fibonacci retracement levels."""
    data = ta.ta_fibonacci(symbol, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-support")
@click.argument("symbol")
@click.option("--window", "-w", default=20, type=int, help="Rolling window (default 20)")
@common_options
def cli_ta_support(symbol, window, source, fmt):
    """Detect support and resistance levels."""
    data = ta.ta_support_resistance(symbol, window, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-analysis")
@click.argument("symbol")
@common_options
def cli_ta_analysis(symbol, source, fmt):
    """Run comprehensive multi-indicator analysis with BUY/SELL/HOLD signal."""
    data = ta.ta_multi_indicator(symbol, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-screen")
@click.argument("symbols")
@click.option("--strategy", "-st", default="oversold",
              help="Strategy: oversold, overbought, macd_bullish, macd_bearish, bb_squeeze, "
                   "golden_cross, death_cross, volume_spike, trend_up, trend_down")
@common_options
def cli_ta_screen(symbols, strategy, source, fmt):
    """Screen multiple stocks for technical signals (comma-separated symbols)."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    data = ta.ta_screen(sym_list, strategy, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-mtf")
@click.argument("symbol")
@common_options
def cli_ta_mtf(symbol, source, fmt):
    """Multi-timeframe analysis (daily, weekly, monthly)."""
    data = ta.ta_multi_timeframe(symbol, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Candlestick Pattern Commands
# ---------------------------------------------------------------------------


@cli.command(name="ta-candle-scan")
@click.argument("symbol")
@common_options
def cli_ta_candle_scan(symbol, source, fmt):
    """Scan for all candlestick patterns in recent data."""
    data = candlestick.ta_candlestick_scan(symbol, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-candle-latest")
@click.argument("symbol")
@common_options
def cli_ta_candle_latest(symbol, source, fmt):
    """Get candlestick patterns on the most recent trading day."""
    data = candlestick.ta_candlestick_latest(symbol, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="ta-candle-screen")
@click.argument("symbols")
@click.option("--pattern", "-p", default="all",
              help="Filter: all, bullish, bearish, or pattern name (hammer, doji, etc.)")
@common_options
def cli_ta_candle_screen(symbols, pattern, source, fmt):
    """Screen multiple stocks for candlestick patterns (comma-separated)."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    data = candlestick.ta_candlestick_screen(sym_list, pattern, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Backtesting Commands
# ---------------------------------------------------------------------------


@cli.command(name="backtest")
@click.argument("symbol")
@click.option("--strategy", "-st", default="sma_crossover",
              help="Strategy: sma_crossover, ema_crossover, rsi_reversal, macd_crossover, "
                   "bollinger_bounce, supertrend, ichimoku_cloud, golden_cross, "
                   "mean_reversion, momentum, breakout, vwap_strategy")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=1000000, type=float, help="Initial capital (default 1,000,000)")
@common_options
def cli_backtest(symbol, strategy, start, end, capital, source, fmt):
    """Backtest a trading strategy on historical data."""
    data = backtest.backtest_strategy(symbol, strategy, start, end, capital, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="backtest-compare")
@click.argument("symbol")
@click.option("--strategies", default=None, help="Comma-separated strategies (default: all 12)")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=1000000, type=float, help="Initial capital")
@common_options
def cli_backtest_compare(symbol, strategies, start, end, capital, source, fmt):
    """Compare multiple backtesting strategies side by side."""
    strat_list = None
    if strategies:
        strat_list = [s.strip() for s in strategies.split(",") if s.strip()]
    data = backtest.backtest_compare(symbol, strat_list, start, end, capital, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="backtest-optimize")
@click.argument("symbol")
@click.option("--strategy", "-st", default="sma_crossover", help="Strategy to optimize")
@click.option("--param", "-p", default="fast_period", help="Parameter name to vary")
@click.option("--values", "-v", default="10,15,20,25,30", help="Comma-separated param values")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=1000000, type=float, help="Initial capital")
@common_options
def cli_backtest_optimize(symbol, strategy, param, values, start, end, capital, source, fmt):
    """Optimize strategy parameters by testing multiple values."""
    val_list = [float(v.strip()) for v in values.split(",") if v.strip()]
    data = backtest.backtest_optimize(symbol, strategy, param, val_list, start, end, capital, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="backtest-walk")
@click.argument("symbol")
@click.option("--strategy", "-st", default="sma_crossover", help="Strategy to test")
@click.option("--window", "-w", default=180, type=int, help="Window size in days (default 180)")
@click.option("--step", default=30, type=int, help="Step size in days (default 30)")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=1000000, type=float, help="Initial capital")
@common_options
def cli_backtest_walk(symbol, strategy, window, step, start, end, capital, source, fmt):
    """Walk-forward analysis on rolling windows."""
    data = backtest.backtest_walk_forward(symbol, strategy, window, step, start, end, capital, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Monte Carlo & Advanced Backtest Commands
# ---------------------------------------------------------------------------


@cli.command(name="backtest-mc")
@click.argument("symbol")
@click.option("--strategy", "-st", default="sma_crossover", help="Strategy to simulate")
@click.option("--sims", "-n", default=1000, type=int, help="Number of simulations (default 1000)")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=1000000, type=float, help="Initial capital")
@common_options
def cli_backtest_mc(symbol, strategy, sims, start, end, capital, source, fmt):
    """Monte Carlo simulation for backtest robustness."""
    data = backtest.backtest_monte_carlo(symbol, strategy, sims, start, end, capital, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="backtest-advanced")
@click.argument("symbol")
@click.option("--strategy", "-st", default="sma_crossover", help="Strategy name")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=1000000, type=float, help="Initial capital")
@common_options
def cli_backtest_advanced(symbol, strategy, start, end, capital, source, fmt):
    """Advanced backtest metrics (Sortino, Calmar, expectancy)."""
    data = backtest.backtest_advanced_metrics(symbol, strategy, start, end, capital, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Portfolio Commands
# ---------------------------------------------------------------------------


@cli.command(name="portfolio")
@click.argument("symbols")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@common_options
def cli_portfolio(symbols, start, end, source, fmt):
    """Analyze a portfolio (comma-separated symbols)."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    data = portfolio.portfolio_analyze(sym_list, start, end, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="portfolio-optimize")
@click.argument("symbols")
@click.option("--sims", "-n", default=5000, type=int, help="Portfolios to simulate (default 5000)")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@common_options
def cli_portfolio_optimize(symbols, sims, start, end, source, fmt):
    """Find optimal portfolio weights via Monte Carlo."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    data = portfolio.portfolio_optimize(sym_list, start, end, sims, source=source)
    click.echo(_format_output(data, fmt))


@cli.command(name="portfolio-risk")
@click.argument("symbols")
@click.option("--weights", "-w", default=None, help="Comma-separated weights (default: equal)")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@common_options
def cli_portfolio_risk(symbols, weights, start, end, source, fmt):
    """Portfolio risk analysis (VaR, CVaR, Sortino, drawdown)."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    w_list = None
    if weights:
        w_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
    data = portfolio.portfolio_risk(sym_list, w_list, start, end, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="portfolio-corr")
@click.argument("symbols")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@common_options
def cli_portfolio_corr(symbols, start, end, source, fmt):
    """Correlation matrix for a set of stocks."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    data = portfolio.portfolio_correlation(sym_list, start, end, source)
    click.echo(_format_output(data, fmt))


# ---------------------------------------------------------------------------
# Sentiment Commands
# ---------------------------------------------------------------------------


@cli.command(name="sentiment")
@click.argument("symbol")
@common_options
def cli_sentiment(symbol, source, fmt):
    """Analyze news sentiment for a stock."""
    data = sentiment.sentiment_news(symbol, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="sentiment-market")
@click.argument("symbols")
@common_options
def cli_sentiment_market(symbols, source, fmt):
    """Batch sentiment analysis for multiple stocks."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    data = sentiment.sentiment_market(sym_list, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="sentiment-combined")
@click.argument("symbol")
@common_options
def cli_sentiment_combined(symbol, source, fmt):
    """Combined technical + sentiment signal."""
    data = sentiment.sentiment_combined(symbol, source)
    click.echo(_format_output(data, fmt))


@cli.command(name="sentiment-screen")
@click.argument("symbols")
@click.option("--min-score", default=0.0, type=float, help="Min sentiment score (-1 to 1)")
@common_options
def cli_sentiment_screen(symbols, min_score, source, fmt):
    """Screen stocks by news sentiment."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    data = sentiment.sentiment_screen(sym_list, min_score, source)
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
