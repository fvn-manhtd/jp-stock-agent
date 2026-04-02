"""Core data-retrieval functions for Japanese and Vietnamese stock market data.

Supports three data sources:
  - **yfinance** (default) – Yahoo Finance Japan via the yfinance library
  - **jquants** – JPX official data via the J-Quants API
  - **vnstocks** – Vietnamese stock data via the vnstock library (HOSE/HNX/UPCOM)

Every public function returns ``list[dict]`` on success or
``{"error": str}`` on failure, keeping a consistent interface for both
the MCP server and the CLI.
"""

from __future__ import annotations

import contextlib
import io
import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .config import auto_detect_source, get_jquants_client, get_settings, normalize_symbol

# ---------------------------------------------------------------------------
# J-Quants client dispatch helpers
#
# jquantsapi.Client (v1) and ClientV2 expose different method names.
# These helpers call the correct method based on the client type.
# ---------------------------------------------------------------------------


def _jq_is_v2(client) -> bool:
    """Return True if *client* is a J-Quants API v2 client (ClientV2)."""
    return hasattr(client, "get_eq_bars_daily") and not hasattr(client, "get_prices_daily_quotes")


def _jq_get_prices(client, code: str, from_yyyymmdd: str, to_yyyymmdd: str):
    """Fetch daily OHLCV data using the right method for client version."""
    if _jq_is_v2(client):
        return client.get_eq_bars_daily(code=code, from_yyyymmdd=from_yyyymmdd, to_yyyymmdd=to_yyyymmdd)
    return client.get_prices_daily_quotes(code=code, from_yyyymmdd=from_yyyymmdd, to_yyyymmdd=to_yyyymmdd)


def _jq_get_listed_info(client, code: str = ""):
    """Fetch listed company info using the right method for client version."""
    if _jq_is_v2(client):
        return client.get_list(code=code)
    if code:
        return client.get_listed_info(code=code)
    return client.get_listed_info()


def _jq_get_statements(client, code: str):
    """Fetch financial statements using the right method for client version."""
    if _jq_is_v2(client):
        return client.get_fin_summary(code=code)
    return client.get_statements(code=code)


def _jq_get_trading_calendar(client, from_yyyymmdd: str, to_yyyymmdd: str):
    """Fetch trading calendar using the right method for client version."""
    if _jq_is_v2(client):
        return client.get_mkt_calendar(from_yyyymmdd=from_yyyymmdd, to_yyyymmdd=to_yyyymmdd)
    return client.get_markets_trading_calendar(from_yyyymmdd=from_yyyymmdd, to_yyyymmdd=to_yyyymmdd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_to_records(df: pd.DataFrame | pd.Series) -> list[dict]:
    """Convert a pandas DataFrame / Series to a JSON-friendly list of dicts."""
    if df is None or (isinstance(df, (pd.DataFrame, pd.Series)) and df.empty):
        return []
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # Convert timestamps to ISO strings, NaN to None
    records = df.to_dict(orient="records")
    cleaned: list[dict] = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, pd.Timestamp):
                clean_row[k] = v.isoformat()
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            else:
                clean_row[k] = v
            # Ensure column name is a string
            if not isinstance(k, str):
                clean_row[str(k)] = clean_row.pop(k)
        cleaned.append(clean_row)
    return cleaned


def _default_dates(start: str | None, end: str | None) -> tuple[str, str]:
    """Return (start, end) date strings with 90-day default lookback."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=90)
    return (
        start or start_dt.strftime("%Y-%m-%d"),
        end or end_dt.strftime("%Y-%m-%d"),
    )


def _safe_call(func, *args, **kwargs) -> Any:
    """Call *func* and return the result or an error dict."""
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@contextlib.contextmanager
def _suppress_stdout():
    """Temporarily suppress stdout (some libraries print verbose output)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# vnstock helpers
# ---------------------------------------------------------------------------

_VN_DEFAULT_DATA_SOURCE = "VCI"   # vnstock internal source: VCI, TCBS, etc.
_VN_KEY_REGISTERED = False        # register API key only once per process
_VN_INTERVAL_MAP = {
    "1d": "1D", "1D": "1D",
    "1wk": "1W", "1W": "1W",
    "1mo": "1M", "1M": "1M",
    "1h": "60", "60": "60",
    "30m": "30", "30": "30",
    "15m": "15", "15": "15",
    "5m": "5", "5": "5",
    "1m": "1", "1": "1",
}


def _vn_stock(symbol: str):
    """Return a vnstock Stock object for the given Vietnamese ticker."""
    global _VN_KEY_REGISTERED
    import logging as _log  # noqa: PLC0415

    from vnstock import Vnstock  # noqa: PLC0415
    # Suppress repeated charting-library warnings after first import
    _log.getLogger("vnstock.common.viz").setLevel(_log.ERROR)
    # Register API key once if configured (raises rate limit to 60 req/min)
    if not _VN_KEY_REGISTERED:
        api_key = get_settings().vnstock_api_key
        if api_key:
            try:
                from vnstock import register_user  # noqa: PLC0415
                register_user(api_key=api_key)
            except Exception:
                pass
        _VN_KEY_REGISTERED = True
    return Vnstock().stock(symbol=symbol, source=_VN_DEFAULT_DATA_SOURCE)


# ---------------------------------------------------------------------------
# Stock Quote Functions
# ---------------------------------------------------------------------------


def stock_history(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch OHLCV historical price data for a Japanese stock.

    Parameters
    ----------
    symbol : str
        Ticker code, e.g. "7203" (Toyota) or "7203.T".
    start, end : str, optional
        Date range in YYYY-MM-DD format. Defaults to last 90 days.
    interval : str
        Price interval – "1d", "1wk", "1mo" (yfinance) or "daily" (jquants).
    source : str, optional
        "yfinance" or "jquants". Defaults to config setting.
    """
    source = source or auto_detect_source(symbol)
    start, end = _default_dates(start, end)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured. Set JQUANTS_API_KEY or EMAIL/PASSWORD."}
        result = _safe_call(_jq_get_prices, client, sym, start.replace("-", ""), end.replace("-", ""))
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    elif source == "vnstocks":
        vn_interval = _VN_INTERVAL_MAP.get(interval, "1D")
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.quote.history, start=start, end=end, interval=vn_interval)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        result = _safe_call(ticker.history, start=start, end=end, interval=interval)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)


def stock_intraday(
    symbol: str,
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch intraday price data (today's trading).

    Uses yfinance with 1-minute interval or J-Quants intraday endpoint.
    """
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        # J-Quants doesn't have a dedicated intraday endpoint in the free tier
        # Fall back to daily quotes for today
        today = datetime.now().strftime("%Y%m%d")
        result = _safe_call(_jq_get_prices, client, sym, today, today)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    elif source == "vnstocks":
        today = datetime.now().strftime("%Y-%m-%d")
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.quote.history, start=today, end=today, interval="1")
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        result = _safe_call(ticker.history, period="1d", interval="1m")
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)


def stock_price_depth(
    symbol: str,
    source: str | None = None,
) -> dict | list[dict]:
    """Fetch order-book / bid-ask data.

    yfinance provides basic bid/ask via .info; J-Quants does not provide
    real-time order book data in most tiers.
    """
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        return {"error": "Order-book depth is not available via J-Quants API."}
    elif source == "vnstocks":
        return {"error": "Real-time order-book depth is not available via vnstocks free tier."}
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        info = _safe_call(lambda: ticker.info)
        if isinstance(info, dict) and "error" in info:
            return info
        depth_keys = ["bid", "bidSize", "ask", "askSize", "lastPrice", "volume", "regularMarketPrice"]
        return {k: info.get(k) for k in depth_keys if k in info}


# ---------------------------------------------------------------------------
# Company Information Functions
# ---------------------------------------------------------------------------


def company_overview(
    symbol: str,
    source: str | None = None,
) -> dict | list[dict]:
    """Fetch company overview / profile information.

    Returns key data like sector, industry, market cap, description.
    """
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        result = _safe_call(_jq_get_listed_info, client, sym)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    elif source == "vnstocks":
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.company.overview)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result) if isinstance(result, pd.DataFrame) else result
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        info = _safe_call(lambda: ticker.info)
        if isinstance(info, dict) and "error" in info:
            return info
        # Select the most useful fields
        overview_keys = [
            "shortName", "longName", "symbol", "exchange", "quoteType",
            "sector", "industry", "longBusinessSummary",
            "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
            "dividendYield", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "currency", "financialCurrency", "country", "city",
            "fullTimeEmployees", "website",
        ]
        return {k: info.get(k) for k in overview_keys if info.get(k) is not None}


def company_shareholders(
    symbol: str,
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch major shareholders information."""
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        return {"error": "Shareholder data is not available via J-Quants free tier."}
    elif source == "vnstocks":
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.company.shareholders)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result) if isinstance(result, pd.DataFrame) else result
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        holders = _safe_call(lambda: ticker.major_holders)
        if isinstance(holders, dict) and "error" in holders:
            return holders
        result = _df_to_records(holders)

        # Also try institutional holders
        inst = _safe_call(lambda: ticker.institutional_holders)
        if isinstance(inst, pd.DataFrame) and not inst.empty:
            result = {"major_holders": result, "institutional_holders": _df_to_records(inst)}
        return result


def company_officers(
    symbol: str,
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch company officers / key executives."""
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        return {"error": "Officer data is not available via J-Quants API."}
    elif source == "vnstocks":
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.company.officers)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result) if isinstance(result, pd.DataFrame) else result
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        info = _safe_call(lambda: ticker.info)
        if isinstance(info, dict) and "error" in info:
            return info
        officers = info.get("companyOfficers", [])
        return officers if officers else {"message": "No officer data available for this symbol."}


def company_news(
    symbol: str,
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch recent news articles for a company."""
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        return {"error": "News is not available via J-Quants API."}
    elif source == "vnstocks":
        return {"error": "News is not available via vnstocks source."}
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        news = _safe_call(lambda: ticker.news)
        if isinstance(news, dict) and "error" in news:
            return news
        return news if news else []


def company_events(
    symbol: str,
    source: str | None = None,
) -> dict | list[dict]:
    """Fetch upcoming events (earnings dates, dividends, splits)."""
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        return {"error": "Event calendar is not available via J-Quants API."}
    elif source == "vnstocks":
        return {"error": "Event calendar is not available via vnstocks source."}
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        result = {}

        calendar = _safe_call(lambda: ticker.calendar)
        if isinstance(calendar, dict) and "error" not in calendar:
            result["calendar"] = calendar
        elif isinstance(calendar, pd.DataFrame):
            result["calendar"] = _df_to_records(calendar)

        dividends = _safe_call(lambda: ticker.dividends)
        if isinstance(dividends, pd.Series) and not dividends.empty:
            result["recent_dividends"] = _df_to_records(dividends.tail(10))

        splits = _safe_call(lambda: ticker.splits)
        if isinstance(splits, pd.Series) and not splits.empty:
            result["recent_splits"] = _df_to_records(splits.tail(10))

        return result if result else {"message": "No event data available."}


# ---------------------------------------------------------------------------
# Financial Statement Functions
# ---------------------------------------------------------------------------


def financial_balance_sheet(
    symbol: str,
    period: str = "annual",
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch balance sheet data.

    Parameters
    ----------
    period : str
        "annual" or "quarterly"
    """
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        result = _safe_call(_jq_get_statements, client, sym)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    elif source == "vnstocks":
        vn_period = "quarter" if period == "quarterly" else "year"
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.finance.balance_sheet, period=vn_period, lang="en")
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        if period == "quarterly":
            bs = _safe_call(lambda: ticker.quarterly_balance_sheet)
        else:
            bs = _safe_call(lambda: ticker.balance_sheet)
        if isinstance(bs, dict) and "error" in bs:
            return bs
        return _df_to_records(bs)


def financial_income_statement(
    symbol: str,
    period: str = "annual",
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch income statement data."""
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        result = _safe_call(_jq_get_statements, client, sym)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    elif source == "vnstocks":
        vn_period = "quarter" if period == "quarterly" else "year"
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.finance.income_statement, period=vn_period, lang="en")
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        if period == "quarterly":
            stmt = _safe_call(lambda: ticker.quarterly_income_stmt)
        else:
            stmt = _safe_call(lambda: ticker.income_stmt)
        if isinstance(stmt, dict) and "error" in stmt:
            return stmt
        return _df_to_records(stmt)


def financial_cash_flow(
    symbol: str,
    period: str = "annual",
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch cash flow statement data."""
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        result = _safe_call(_jq_get_statements, client, sym)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    elif source == "vnstocks":
        vn_period = "quarter" if period == "quarterly" else "year"
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.finance.cash_flow, period=vn_period, lang="en")
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        if period == "quarterly":
            cf = _safe_call(lambda: ticker.quarterly_cashflow)
        else:
            cf = _safe_call(lambda: ticker.cashflow)
        if isinstance(cf, dict) and "error" in cf:
            return cf
        return _df_to_records(cf)


def financial_ratio(
    symbol: str,
    source: str | None = None,
) -> dict | list[dict]:
    """Fetch key financial ratios and valuation metrics."""
    source = source or auto_detect_source(symbol)
    sym = normalize_symbol(symbol, source)

    if source == "jquants":
        return {"error": "Financial ratios are not directly available via J-Quants. Use financial statements instead."}
    elif source == "vnstocks":
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, sym)
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.finance.ratio, period="year", lang="en")
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result) if isinstance(result, pd.DataFrame) else result
    else:
        import yfinance as yf

        ticker = yf.Ticker(sym)
        info = _safe_call(lambda: ticker.info)
        if isinstance(info, dict) and "error" in info:
            return info

        ratio_keys = [
            "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
            "enterpriseToRevenue", "enterpriseToEbitda", "profitMargins",
            "operatingMargins", "grossMargins", "returnOnAssets", "returnOnEquity",
            "debtToEquity", "currentRatio", "quickRatio",
            "earningsGrowth", "revenueGrowth", "dividendYield", "payoutRatio",
            "beta", "trailingEps", "forwardEps", "bookValue",
        ]
        return {k: info.get(k) for k in ratio_keys if info.get(k) is not None}


# ---------------------------------------------------------------------------
# Market Listing Functions
# ---------------------------------------------------------------------------


def listing_all_symbols(
    source: str | None = None,
) -> list[dict] | dict:
    """List all securities on the Tokyo Stock Exchange.

    J-Quants provides a comprehensive listed-info endpoint.
    yfinance does not have a native listing function, so we provide
    common indices and a note to the user.
    """
    source = source or get_settings().jpstock_default_source

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        result = _safe_call(_jq_get_listed_info, client)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    elif source == "vnstocks":
        # Return all VN stocks from HOSE, HNX, UPCOM via default symbol
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, "VNM")
        if isinstance(stock, dict) and "error" in stock:
            return stock
        result = _safe_call(stock.listing.all_symbols)
        if isinstance(result, dict) and "error" in result:
            return result
        return _df_to_records(result)
    else:
        return {
            "message": "yfinance does not provide a full listing endpoint. "
            "Use source='jquants' for complete TSE listings, or query specific symbols directly.",
            "common_indices": [
                {"symbol": "^N225", "name": "Nikkei 225"},
                {"symbol": "^TOPX", "name": "TOPIX"},
                {"symbol": "^NJX", "name": "Nikkei JASDAQ"},
            ],
        }


def listing_symbols_by_sector(
    sector: str = "",
    source: str | None = None,
) -> list[dict] | dict:
    """List symbols filtered by sector/industry.

    Primarily useful with J-Quants which has sector classification data.
    """
    source = source or get_settings().jpstock_default_source

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        result = _safe_call(_jq_get_listed_info, client)
        if isinstance(result, dict) and "error" in result:
            return result
        df = result
        if sector and isinstance(df, pd.DataFrame):
            # J-Quants uses Sector33Code / Sector17Code columns
            mask = (
                df.apply(lambda row: sector.lower() in str(row.values).lower(), axis=1)
            )
            df = df[mask]
        return _df_to_records(df)
    else:
        return {"error": "Sector-based listing requires source='jquants'."}


def listing_symbols_by_market(
    market: str = "Prime",
    source: str | None = None,
) -> list[dict] | dict:
    """List symbols by market segment (Prime, Standard, Growth).

    Parameters
    ----------
    market : str
        Market segment – "Prime", "Standard", or "Growth".
    """
    source = source or get_settings().jpstock_default_source

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        result = _safe_call(_jq_get_listed_info, client)
        if isinstance(result, dict) and "error" in result:
            return result
        df = result
        if isinstance(df, pd.DataFrame) and "MarketCode" in df.columns:
            df = df[df["MarketCode"].str.contains(market, case=False, na=False)]
        return _df_to_records(df)
    else:
        return {"error": "Market-based listing requires source='jquants'."}


def listing_sectors(
    source: str | None = None,
) -> list[dict] | dict:
    """List all available sector / industry classifications.

    J-Quants uses the TSE 33-sector and 17-sector classification systems.
    """
    source = source or get_settings().jpstock_default_source

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        result = _safe_call(_jq_get_listed_info, client)
        if isinstance(result, dict) and "error" in result:
            return result
        df = result
        if isinstance(df, pd.DataFrame):
            sector_cols = [c for c in df.columns if "Sector" in c or "sector" in c]
            if sector_cols:
                sectors = df[sector_cols].drop_duplicates().sort_values(by=sector_cols[0])
                return _df_to_records(sectors)
        return {"message": "No sector data found."}
    else:
        # Provide a static list of TSE sectors for reference
        return {
            "tse_33_sectors": [
                "Fishery, Agriculture & Forestry", "Mining", "Construction",
                "Foods", "Textiles & Apparels", "Pulp & Paper",
                "Chemicals", "Pharmaceutical", "Oil & Coal Products",
                "Rubber Products", "Glass & Ceramics Products", "Iron & Steel",
                "Nonferrous Metals", "Metal Products", "Machinery",
                "Electric Appliances", "Transportation Equipment",
                "Precision Instruments", "Other Products",
                "Electric Power & Gas", "Land Transportation",
                "Marine Transportation", "Air Transportation",
                "Warehousing & Harbor Transportation",
                "Information & Communication", "Wholesale Trade",
                "Retail Trade", "Banks", "Securities & Commodities Futures",
                "Insurance", "Other Financing Business",
                "Real Estate", "Services",
            ]
        }


# ---------------------------------------------------------------------------
# Trading / Market Data Functions
# ---------------------------------------------------------------------------


def trading_price_board(
    symbols: list[str],
    source: str | None = None,
) -> list[dict] | dict:
    """Fetch real-time snapshot prices for multiple symbols at once."""
    source = source or (auto_detect_source(symbols[0]) if symbols else get_settings().jpstock_default_source)
    results = []

    if source == "jquants":
        client = get_jquants_client()
        if client is None:
            return {"error": "J-Quants credentials not configured."}
        for sym in symbols:
            code = normalize_symbol(sym, "jquants")
            today = datetime.now().strftime("%Y%m%d")
            data = _safe_call(_jq_get_prices, client, code, today, today)
            if isinstance(data, pd.DataFrame) and not data.empty:
                rec = _df_to_records(data)
                results.extend(rec)
            elif isinstance(data, dict) and "error" in data:
                results.append({"symbol": sym, **data})
        return results
    elif source == "vnstocks":
        today = datetime.now().strftime("%Y-%m-%d")
        for sym in symbols:
            code = normalize_symbol(sym, "vnstocks")
            with _suppress_stdout():
                stock = _safe_call(_vn_stock, code)
            if isinstance(stock, dict) and "error" in stock:
                results.append({"symbol": code, **stock})
                continue
            data = _safe_call(stock.quote.history, start=today, end=today, interval="1D")
            if isinstance(data, pd.DataFrame) and not data.empty:
                rec = _df_to_records(data)
                for r in rec:
                    r["symbol"] = code
                results.extend(rec)
            elif isinstance(data, dict) and "error" in data:
                results.append({"symbol": code, **data})
        return results
    else:
        import yfinance as yf

        normalized = [normalize_symbol(s, "yfinance") for s in symbols]
        for sym in normalized:
            ticker = yf.Ticker(sym)
            info = _safe_call(lambda t=ticker: t.info)
            if isinstance(info, dict) and "error" not in info:
                results.append({
                    "symbol": sym,
                    "price": info.get("regularMarketPrice"),
                    "previousClose": info.get("previousClose"),
                    "open": info.get("regularMarketOpen"),
                    "dayHigh": info.get("regularMarketDayHigh"),
                    "dayLow": info.get("regularMarketDayLow"),
                    "volume": info.get("regularMarketVolume"),
                    "marketCap": info.get("marketCap"),
                })
            elif isinstance(info, dict):
                results.append({"symbol": sym, **info})
        return results


# ---------------------------------------------------------------------------
# Global / FX / Index Functions
# ---------------------------------------------------------------------------


def fx_history(
    pair: str = "USDJPY=X",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> list[dict] | dict:
    """Fetch forex exchange rate history via yfinance.

    Parameters
    ----------
    pair : str
        Currency pair symbol, e.g. "USDJPY=X", "EURJPY=X".
    """
    import yfinance as yf

    start, end = _default_dates(start, end)
    ticker = yf.Ticker(pair)
    result = _safe_call(ticker.history, start=start, end=end, interval=interval)
    if isinstance(result, dict) and "error" in result:
        return result
    return _df_to_records(result)


def crypto_history(
    symbol: str = "BTC-JPY",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> list[dict] | dict:
    """Fetch cryptocurrency price history in JPY via yfinance.

    Parameters
    ----------
    symbol : str
        Crypto pair, e.g. "BTC-JPY", "ETH-JPY", "BTC-USD".
    """
    import yfinance as yf

    start, end = _default_dates(start, end)
    ticker = yf.Ticker(symbol)
    result = _safe_call(ticker.history, start=start, end=end, interval=interval)
    if isinstance(result, dict) and "error" in result:
        return result
    return _df_to_records(result)


def world_index_history(
    symbol: str = "^N225",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> list[dict] | dict:
    """Fetch world index historical data via yfinance.

    Parameters
    ----------
    symbol : str
        Index symbol, e.g. "^N225" (Nikkei), "^TOPX" (TOPIX),
        "^GSPC" (S&P 500), "^DJI" (Dow Jones).
    """
    import yfinance as yf

    start, end = _default_dates(start, end)
    ticker = yf.Ticker(symbol)
    result = _safe_call(ticker.history, start=start, end=end, interval=interval)
    if isinstance(result, dict) and "error" in result:
        return result
    return _df_to_records(result)


# ---------------------------------------------------------------------------
# vnstock-specific functions
# ---------------------------------------------------------------------------


def vnstocks_listing(
    exchange: str = "HOSE",
) -> list[dict] | dict:
    """List all securities on a Vietnamese stock exchange.

    Parameters
    ----------
    exchange : str
        Exchange name – "HOSE" (Ho Chi Minh), "HNX" (Hanoi), "UPCOM", or "all".
    """
    with _suppress_stdout():
        stock = _safe_call(_vn_stock, "VNM")
    if isinstance(stock, dict) and "error" in stock:
        return stock
    result = _safe_call(stock.listing.all_symbols)
    if isinstance(result, dict) and "error" in result:
        return result
    if isinstance(result, pd.DataFrame) and exchange.upper() != "ALL" and "exchange" in result.columns:
        result = result[result["exchange"].str.upper() == exchange.upper()]
    return _df_to_records(result)


def vnstocks_price_board(
    symbols: list[str],
) -> list[dict] | dict:
    """Fetch current price board snapshot for multiple Vietnamese stocks.

    Parameters
    ----------
    symbols : list[str]
        List of Vietnamese ticker codes, e.g. ["ACB", "VNM", "VIC"].
    """
    results = []
    today = datetime.now().strftime("%Y-%m-%d")
    for sym in symbols:
        code = normalize_symbol(sym, "vnstocks")
        with _suppress_stdout():
            stock = _safe_call(_vn_stock, code)
        if isinstance(stock, dict) and "error" in stock:
            results.append({"symbol": code, **stock})
            continue
        data = _safe_call(stock.quote.history, start=today, end=today, interval="1D")
        if isinstance(data, pd.DataFrame) and not data.empty:
            rec = _df_to_records(data)
            for r in rec:
                r["symbol"] = code
            results.extend(rec)
        elif isinstance(data, dict) and "error" in data:
            results.append({"symbol": code, **data})
    return results


# ---------------------------------------------------------------------------
# J-Quants specific functions
# ---------------------------------------------------------------------------


def jquants_financial_statements(
    symbol: str,
) -> list[dict] | dict:
    """Fetch financial statements from J-Quants API.

    Returns combined financial data (income, balance sheet, cash flow
    items) from J-Quants' statements endpoint.
    """
    client = get_jquants_client()
    if client is None:
        return {"error": "J-Quants credentials not configured."}
    code = normalize_symbol(symbol, "jquants")
    result = _safe_call(_jq_get_statements, client, code)
    if isinstance(result, dict) and "error" in result:
        return result
    return _df_to_records(result)


def jquants_trading_calendar(
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[dict] | dict:
    """Fetch TSE trading calendar from J-Quants.

    Shows which days the market is open/closed.
    """
    client = get_jquants_client()
    if client is None:
        return {"error": "J-Quants credentials not configured."}
    start, end = _default_dates(from_date, to_date)
    result = _safe_call(_jq_get_trading_calendar, client, start.replace("-", ""), end.replace("-", ""))
    if isinstance(result, dict) and "error" in result:
        return result
    return _df_to_records(result)
