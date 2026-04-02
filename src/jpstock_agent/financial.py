"""Fundamental Financial Analysis Module.

Computes derived financial metrics from raw statements fetched by core.py.

Public Functions:
- financial_health: Altman Z-score, Piotroski F-score, quick/current ratio analysis
- financial_growth: YoY revenue/earnings/margin growth trends
- financial_valuation: DCF estimate, relative valuation (P/E, EV/EBITDA vs history)
- financial_peer_compare: Side-by-side comparison of multiple stocks
- financial_dividend: Dividend growth, payout sustainability, shareholder yield
- financial_ratios_calc: Compute ratios directly from raw balance sheet + income stmt

All functions return dict on success or {"error": str} on failure.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core import (
    company_overview,
    financial_balance_sheet,
    financial_cash_flow,
    financial_income_statement,
    financial_ratio,
)
from .logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round_val(v, decimals: int = 4):
    """Round a numeric value, returning None for NaN/inf."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, decimals)


def _safe_get(d: dict | list, key: str, default=None):
    """Safely get a numeric value from a dict, handling missing/non-numeric."""
    if isinstance(d, dict):
        val = d.get(key, default)
    else:
        return default
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fetch_statements(symbol: str, period: str = "annual") -> dict:
    """Fetch all three financial statements + ratios for a symbol.

    Returns dict with keys: balance_sheet, income_statement, cash_flow, ratios, error.
    """
    bs = financial_balance_sheet(symbol, period=period)
    inc = financial_income_statement(symbol, period=period)
    cf = financial_cash_flow(symbol, period=period)
    ratios = financial_ratio(symbol)

    errors = []
    if isinstance(bs, dict) and "error" in bs:
        errors.append(f"balance_sheet: {bs['error']}")
        bs = []
    if isinstance(inc, dict) and "error" in inc:
        errors.append(f"income_statement: {inc['error']}")
        inc = []
    if isinstance(cf, dict) and "error" in cf:
        errors.append(f"cash_flow: {cf['error']}")
        cf = []
    if isinstance(ratios, dict) and "error" in ratios:
        errors.append(f"ratios: {ratios['error']}")
        ratios = {}

    # Normalize: ensure list format
    if isinstance(bs, dict):
        bs = [bs]
    if isinstance(inc, dict):
        inc = [inc]
    if isinstance(cf, dict):
        cf = [cf]

    return {
        "balance_sheet": bs if isinstance(bs, list) else [],
        "income_statement": inc if isinstance(inc, list) else [],
        "cash_flow": cf if isinstance(cf, list) else [],
        "ratios": ratios if isinstance(ratios, dict) else {},
        "errors": errors,
    }


def _find_field(record: dict, *candidates: str, default=None) -> float | None:
    """Search a record for any of the candidate field names (case-insensitive)."""
    lower_map = {k.lower().replace(" ", ""): k for k in record}
    for cand in candidates:
        cand_lower = cand.lower().replace(" ", "")
        if cand_lower in lower_map:
            val = record[lower_map[cand_lower]]
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return default


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------


def _piotroski_f_score(bs: list[dict], inc: list[dict], cf: list[dict]) -> dict:
    """Calculate Piotroski F-Score (0-9) from financial statements.

    Requires at least 2 periods of data for comparison.
    """
    if len(bs) < 2 or len(inc) < 2 or len(cf) < 1:
        return {"score": None, "details": {}, "note": "Insufficient data for F-score"}

    # Current and prior period (index 0 = most recent for yfinance)
    bs_curr, bs_prev = bs[0], bs[1]
    inc_curr, inc_prev = inc[0], inc[1]
    cf_curr = cf[0]

    score = 0
    details = {}

    # --- Profitability (4 points) ---
    # 1. Positive net income
    net_income = _find_field(inc_curr, "NetIncome", "Net Income", "net_income")
    if net_income is not None and net_income > 0:
        score += 1
        details["net_income_positive"] = True
    else:
        details["net_income_positive"] = False

    # 2. Positive operating cash flow
    op_cf = _find_field(
        cf_curr, "OperatingCashFlow", "TotalCashFromOperatingActivities",
        "operating_cash_flow", "FreeCashFlow",
    )
    if op_cf is not None and op_cf > 0:
        score += 1
        details["operating_cf_positive"] = True
    else:
        details["operating_cf_positive"] = False

    # 3. ROA improvement
    total_assets_curr = _find_field(bs_curr, "TotalAssets", "Total Assets", "total_assets")
    total_assets_prev = _find_field(bs_prev, "TotalAssets", "Total Assets", "total_assets")
    net_income_prev = _find_field(inc_prev, "NetIncome", "Net Income", "net_income")
    if all(v is not None and v != 0 for v in [net_income, total_assets_curr, net_income_prev, total_assets_prev]):
        roa_curr = net_income / total_assets_curr
        roa_prev = net_income_prev / total_assets_prev
        if roa_curr > roa_prev:
            score += 1
            details["roa_improving"] = True
        else:
            details["roa_improving"] = False
    else:
        details["roa_improving"] = None

    # 4. Operating CF > Net Income (accruals quality)
    if op_cf is not None and net_income is not None and op_cf > net_income:
        score += 1
        details["cf_gt_net_income"] = True
    else:
        details["cf_gt_net_income"] = False

    # --- Leverage / Liquidity (3 points) ---
    # 5. Decreasing long-term debt ratio
    lt_debt_curr = _find_field(bs_curr, "LongTermDebt", "Long Term Debt", "long_term_debt") or 0
    lt_debt_prev = _find_field(bs_prev, "LongTermDebt", "Long Term Debt", "long_term_debt") or 0
    if total_assets_curr and total_assets_prev and total_assets_curr != 0 and total_assets_prev != 0:
        debt_ratio_curr = lt_debt_curr / total_assets_curr
        debt_ratio_prev = lt_debt_prev / total_assets_prev
        if debt_ratio_curr <= debt_ratio_prev:
            score += 1
            details["debt_ratio_decreasing"] = True
        else:
            details["debt_ratio_decreasing"] = False
    else:
        details["debt_ratio_decreasing"] = None

    # 6. Improving current ratio
    curr_assets_curr = _find_field(bs_curr, "CurrentAssets", "Total Current Assets", "current_assets") or 0
    curr_liab_curr = _find_field(
        bs_curr, "CurrentLiabilities", "Total Current Liabilities", "current_liabilities",
    ) or 1
    curr_assets_prev = _find_field(bs_prev, "CurrentAssets", "Total Current Assets", "current_assets") or 0
    curr_liab_prev = _find_field(
        bs_prev, "CurrentLiabilities", "Total Current Liabilities", "current_liabilities",
    ) or 1
    cr_curr = curr_assets_curr / curr_liab_curr if curr_liab_curr != 0 else 0
    cr_prev = curr_assets_prev / curr_liab_prev if curr_liab_prev != 0 else 0
    if cr_curr > cr_prev:
        score += 1
        details["current_ratio_improving"] = True
    else:
        details["current_ratio_improving"] = False

    # 7. No new share issuance (shares outstanding not increased)
    shares_curr = _find_field(
        bs_curr, "ShareIssued", "OrdinarySharesNumber", "CommonStock",
        "shares_outstanding", "common_stock",
    )
    shares_prev = _find_field(
        bs_prev, "ShareIssued", "OrdinarySharesNumber", "CommonStock",
        "shares_outstanding", "common_stock",
    )
    if shares_curr is not None and shares_prev is not None:
        if shares_curr <= shares_prev:
            score += 1
            details["no_dilution"] = True
        else:
            details["no_dilution"] = False
    else:
        details["no_dilution"] = None

    # --- Operating Efficiency (2 points) ---
    # 8. Improving gross margin
    gp_curr = _find_field(inc_curr, "GrossProfit", "Gross Profit", "gross_profit")
    rev_curr = _find_field(inc_curr, "TotalRevenue", "Total Revenue", "revenue")
    gp_prev = _find_field(inc_prev, "GrossProfit", "Gross Profit", "gross_profit")
    rev_prev = _find_field(inc_prev, "TotalRevenue", "Total Revenue", "revenue")
    if all(v is not None and v != 0 for v in [gp_curr, rev_curr, gp_prev, rev_prev]):
        gm_curr = gp_curr / rev_curr
        gm_prev = gp_prev / rev_prev
        if gm_curr > gm_prev:
            score += 1
            details["gross_margin_improving"] = True
        else:
            details["gross_margin_improving"] = False
    else:
        details["gross_margin_improving"] = None

    # 9. Improving asset turnover
    if all(v is not None and v != 0 for v in [rev_curr, total_assets_curr, rev_prev, total_assets_prev]):
        at_curr = rev_curr / total_assets_curr
        at_prev = rev_prev / total_assets_prev
        if at_curr > at_prev:
            score += 1
            details["asset_turnover_improving"] = True
        else:
            details["asset_turnover_improving"] = False
    else:
        details["asset_turnover_improving"] = None

    # Interpretation
    if score >= 8:
        interpretation = "STRONG"
    elif score >= 5:
        interpretation = "MODERATE"
    else:
        interpretation = "WEAK"

    return {"score": score, "max_score": 9, "interpretation": interpretation, "details": details}


# ---------------------------------------------------------------------------
# Altman Z-Score
# ---------------------------------------------------------------------------


def _altman_z_score(bs: list[dict], inc: list[dict]) -> dict:
    """Calculate Altman Z-Score for bankruptcy prediction.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap (or Equity) / Total Liabilities
    X5 = Revenue / Total Assets
    """
    if not bs or not inc:
        return {"z_score": None, "zone": None, "note": "Insufficient data"}

    curr_bs = bs[0]
    curr_inc = inc[0]

    total_assets = _find_field(curr_bs, "TotalAssets", "Total Assets", "total_assets")
    if not total_assets or total_assets == 0:
        return {"z_score": None, "zone": None, "note": "Total assets not available"}

    # X1: Working Capital / Total Assets
    curr_assets = _find_field(curr_bs, "CurrentAssets", "Total Current Assets", "current_assets") or 0
    curr_liab = _find_field(
        curr_bs, "CurrentLiabilities", "Total Current Liabilities", "current_liabilities",
    ) or 0
    x1 = (curr_assets - curr_liab) / total_assets

    # X2: Retained Earnings / Total Assets
    retained = _find_field(curr_bs, "RetainedEarnings", "Retained Earnings", "retained_earnings") or 0
    x2 = retained / total_assets

    # X3: EBIT / Total Assets
    ebit = _find_field(curr_inc, "EBIT", "ebit", "OperatingIncome", "operating_income") or 0
    x3 = ebit / total_assets

    # X4: Stockholders Equity / Total Liabilities
    equity = _find_field(
        curr_bs, "StockholdersEquity", "TotalStockholderEquity",
        "Total Stockholder Equity", "stockholders_equity",
    ) or 0
    total_liab = _find_field(
        curr_bs, "TotalLiab", "TotalLiabilitiesNetMinorityInterest",
        "Total Liabilities", "total_liabilities",
    )
    if total_liab and total_liab != 0:
        x4 = equity / total_liab
    else:
        x4 = 0

    # X5: Revenue / Total Assets
    revenue = _find_field(curr_inc, "TotalRevenue", "Total Revenue", "revenue") or 0
    x5 = revenue / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        zone = "SAFE"
    elif z > 1.81:
        zone = "GREY"
    else:
        zone = "DISTRESS"

    return {
        "z_score": _round_val(z, 2),
        "zone": zone,
        "components": {
            "x1_working_capital_ta": _round_val(x1),
            "x2_retained_earnings_ta": _round_val(x2),
            "x3_ebit_ta": _round_val(x3),
            "x4_equity_tl": _round_val(x4),
            "x5_revenue_ta": _round_val(x5),
        },
    }


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------


def financial_health(symbol: str, period: str = "annual") -> dict:
    """Comprehensive financial health assessment.

    Includes Altman Z-score, Piotroski F-score, liquidity ratios,
    and leverage metrics.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g. "7203", "ACB").
    period : str
        "annual" or "quarterly".

    Returns
    -------
    dict with keys: symbol, altman_z, piotroski_f, liquidity, leverage, health_summary.
    """
    data = _fetch_statements(symbol, period)
    bs, inc, cf = data["balance_sheet"], data["income_statement"], data["cash_flow"]

    if not bs and not inc:
        return {"error": f"No financial data available for {symbol}"}

    # Altman Z-Score
    altman = _altman_z_score(bs, inc)

    # Piotroski F-Score
    piotroski = _piotroski_f_score(bs, inc, cf)

    # Liquidity ratios from latest balance sheet
    liquidity = {}
    if bs:
        curr_bs = bs[0]
        ca = _find_field(curr_bs, "CurrentAssets", "Total Current Assets", "current_assets")
        cl = _find_field(
            curr_bs, "CurrentLiabilities", "Total Current Liabilities", "current_liabilities",
        )
        inventory = _find_field(curr_bs, "Inventory", "inventory") or 0
        cash = _find_field(
            curr_bs, "CashAndCashEquivalents", "Cash", "CashCashEquivalentsAndShortTermInvestments",
            "cash_and_equivalents",
        ) or 0

        if ca is not None and cl and cl != 0:
            liquidity["current_ratio"] = _round_val(ca / cl, 2)
            liquidity["quick_ratio"] = _round_val((ca - inventory) / cl, 2)
            liquidity["cash_ratio"] = _round_val(cash / cl, 2)

    # Leverage
    leverage = {}
    if bs:
        curr_bs = bs[0]
        total_assets = _find_field(curr_bs, "TotalAssets", "Total Assets", "total_assets")
        total_debt = _find_field(
            curr_bs, "TotalDebt", "total_debt", "LongTermDebt", "long_term_debt",
        ) or 0
        equity = _find_field(
            curr_bs, "StockholdersEquity", "TotalStockholderEquity", "stockholders_equity",
        )
        if equity and equity != 0:
            leverage["debt_to_equity"] = _round_val(total_debt / equity, 2)
        if total_assets and total_assets != 0:
            leverage["debt_to_assets"] = _round_val(total_debt / total_assets, 2)
            if equity:
                leverage["equity_multiplier"] = _round_val(total_assets / equity, 2)

    # Cash conversion (if income + balance sheet available)
    if bs and inc:
        curr_bs, curr_inc = bs[0], inc[0]
        rev = _find_field(curr_inc, "TotalRevenue", "Total Revenue", "revenue")
        cogs = _find_field(curr_inc, "CostOfRevenue", "Cost Of Revenue", "cost_of_goods_sold")
        receivables = _find_field(curr_bs, "NetReceivables", "AccountsReceivable", "receivables") or 0
        inventory_val = _find_field(curr_bs, "Inventory", "inventory") or 0
        payables = _find_field(curr_bs, "AccountsPayable", "Accounts Payable", "accounts_payable") or 0

        if rev and rev != 0:
            leverage["receivables_days"] = _round_val(receivables / (rev / 365), 1)
        if cogs and cogs != 0:
            leverage["inventory_days"] = _round_val(inventory_val / (abs(cogs) / 365), 1)
            leverage["payables_days"] = _round_val(payables / (abs(cogs) / 365), 1)
            if leverage.get("receivables_days") is not None and leverage.get("inventory_days") is not None:
                ccc = (leverage["receivables_days"] or 0) + (leverage["inventory_days"] or 0)
                ccc -= leverage.get("payables_days") or 0
                leverage["cash_conversion_cycle_days"] = _round_val(ccc, 1)

    # Health summary
    signals = []
    if altman.get("zone") == "SAFE":
        signals.append("Z-score: Safe zone")
    elif altman.get("zone") == "DISTRESS":
        signals.append("Z-score: Distress zone (bankruptcy risk)")
    if piotroski.get("score") is not None:
        if piotroski["score"] >= 7:
            signals.append(f"F-score: Strong ({piotroski['score']}/9)")
        elif piotroski["score"] <= 3:
            signals.append(f"F-score: Weak ({piotroski['score']}/9)")
    if liquidity.get("current_ratio") is not None:
        if liquidity["current_ratio"] < 1.0:
            signals.append("Liquidity: Current ratio below 1.0 (concern)")
        elif liquidity["current_ratio"] > 2.0:
            signals.append("Liquidity: Strong current ratio")

    return {
        "symbol": symbol,
        "period": period,
        "altman_z": altman,
        "piotroski_f": piotroski,
        "liquidity": liquidity,
        "leverage": leverage,
        "health_signals": signals,
    }


def financial_growth(symbol: str, period: str = "annual") -> dict:
    """Analyze revenue, earnings, and margin growth trends.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    period : str
        "annual" or "quarterly".

    Returns
    -------
    dict with keys: symbol, revenue_growth, earnings_growth, margin_trends, summary.
    """
    data = _fetch_statements(symbol, period)
    inc = data["income_statement"]
    cf = data["cash_flow"]

    if not inc or len(inc) < 2:
        return {"error": f"Need at least 2 periods of income data for {symbol}"}

    revenue_trend = []
    earnings_trend = []
    margin_trend = []

    for i, rec in enumerate(inc):
        rev = _find_field(rec, "TotalRevenue", "Total Revenue", "revenue")
        ni = _find_field(rec, "NetIncome", "Net Income", "net_income")
        gp = _find_field(rec, "GrossProfit", "Gross Profit", "gross_profit")
        oi = _find_field(rec, "OperatingIncome", "EBIT", "operating_income")

        entry = {"period_index": i}

        if rev is not None:
            entry["revenue"] = rev
            if gp is not None:
                entry["gross_margin_pct"] = _round_val(gp / rev * 100, 2)
            if oi is not None:
                entry["operating_margin_pct"] = _round_val(oi / rev * 100, 2)
            if ni is not None:
                entry["net_margin_pct"] = _round_val(ni / rev * 100, 2)

        if ni is not None:
            entry["net_income"] = ni

        margin_trend.append(entry)

    # Calculate YoY growth (compare consecutive periods)
    for i in range(len(inc) - 1):
        curr_rev = _find_field(inc[i], "TotalRevenue", "Total Revenue", "revenue")
        prev_rev = _find_field(inc[i + 1], "TotalRevenue", "Total Revenue", "revenue")
        curr_ni = _find_field(inc[i], "NetIncome", "Net Income", "net_income")
        prev_ni = _find_field(inc[i + 1], "NetIncome", "Net Income", "net_income")

        rev_growth = None
        if curr_rev is not None and prev_rev is not None and prev_rev != 0:
            rev_growth = _round_val((curr_rev - prev_rev) / abs(prev_rev) * 100, 2)
        revenue_trend.append({"period_index": i, "yoy_growth_pct": rev_growth})

        ni_growth = None
        if curr_ni is not None and prev_ni is not None and prev_ni != 0:
            ni_growth = _round_val((curr_ni - prev_ni) / abs(prev_ni) * 100, 2)
        earnings_trend.append({"period_index": i, "yoy_growth_pct": ni_growth})

    # Free cash flow trend
    fcf_trend = []
    for i, rec in enumerate(cf):
        op_cf = _find_field(
            rec, "OperatingCashFlow", "TotalCashFromOperatingActivities", "operating_cash_flow",
        )
        capex = _find_field(rec, "CapitalExpenditure", "capital_expenditure") or 0
        fcf = (op_cf - abs(capex)) if op_cf is not None else None
        fcf_trend.append({"period_index": i, "free_cash_flow": fcf})

    # Summary
    summary = []
    if revenue_trend and revenue_trend[0].get("yoy_growth_pct") is not None:
        latest_rg = revenue_trend[0]["yoy_growth_pct"]
        if latest_rg > 10:
            summary.append(f"Revenue growing strongly ({latest_rg}% YoY)")
        elif latest_rg < -10:
            summary.append(f"Revenue declining ({latest_rg}% YoY)")
    if earnings_trend and earnings_trend[0].get("yoy_growth_pct") is not None:
        latest_eg = earnings_trend[0]["yoy_growth_pct"]
        if latest_eg > 20:
            summary.append(f"Earnings surging ({latest_eg}% YoY)")
        elif latest_eg < -20:
            summary.append(f"Earnings declining sharply ({latest_eg}% YoY)")

    # Margin direction
    if len(margin_trend) >= 2:
        curr_nm = margin_trend[0].get("net_margin_pct")
        prev_nm = margin_trend[1].get("net_margin_pct")
        if curr_nm is not None and prev_nm is not None:
            if curr_nm > prev_nm:
                summary.append("Net margin expanding")
            elif curr_nm < prev_nm:
                summary.append("Net margin contracting")

    return {
        "symbol": symbol,
        "period": period,
        "revenue_growth": revenue_trend,
        "earnings_growth": earnings_trend,
        "margin_trends": margin_trend,
        "free_cash_flow_trend": fcf_trend,
        "summary": summary,
    }


def financial_valuation(
    symbol: str,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.02,
    projection_years: int = 5,
) -> dict:
    """Estimate intrinsic value using DCF and relative valuation.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    discount_rate : float
        WACC / required return (default 10%).
    terminal_growth : float
        Perpetual growth rate (default 2%).
    projection_years : int
        Number of years to project FCF (default 5).

    Returns
    -------
    dict with keys: symbol, dcf, relative_valuation, valuation_summary.
    """
    data = _fetch_statements(symbol)
    cf = data["cash_flow"]
    ratios = data["ratios"]

    overview = company_overview(symbol)
    market_cap = None
    if isinstance(overview, dict) and "error" not in overview:
        market_cap = _safe_get(overview, "marketCap")

    # --- DCF Valuation ---
    dcf = {}
    if cf and len(cf) >= 2:
        # Get recent FCFs
        fcf_list = []
        for rec in cf[:5]:
            op_cf = _find_field(
                rec, "OperatingCashFlow", "TotalCashFromOperatingActivities", "operating_cash_flow",
            )
            capex = _find_field(rec, "CapitalExpenditure", "capital_expenditure") or 0
            if op_cf is not None:
                fcf_list.append(op_cf - abs(capex))

        if fcf_list and fcf_list[0] > 0:
            latest_fcf = fcf_list[0]

            # Estimate growth from historical FCF
            if len(fcf_list) >= 2 and fcf_list[-1] != 0:
                hist_growth = (fcf_list[0] / abs(fcf_list[-1])) ** (1 / (len(fcf_list) - 1)) - 1
                hist_growth = max(min(hist_growth, 0.30), -0.10)  # Cap between -10% and 30%
            else:
                hist_growth = 0.05

            # Project future FCFs
            projected = []
            for year in range(1, projection_years + 1):
                proj_fcf = latest_fcf * (1 + hist_growth) ** year
                pv = proj_fcf / (1 + discount_rate) ** year
                projected.append({
                    "year": year,
                    "projected_fcf": _round_val(proj_fcf, 0),
                    "present_value": _round_val(pv, 0),
                })

            # Terminal value
            terminal_fcf = latest_fcf * (1 + hist_growth) ** projection_years * (1 + terminal_growth)
            if discount_rate > terminal_growth:
                terminal_value = terminal_fcf / (discount_rate - terminal_growth)
                tv_pv = terminal_value / (1 + discount_rate) ** projection_years
            else:
                terminal_value = 0
                tv_pv = 0

            total_pv = sum(p["present_value"] for p in projected if p["present_value"]) + tv_pv

            dcf = {
                "latest_fcf": _round_val(latest_fcf, 0),
                "growth_rate_used": _round_val(hist_growth * 100, 2),
                "discount_rate": _round_val(discount_rate * 100, 2),
                "terminal_growth": _round_val(terminal_growth * 100, 2),
                "projected_fcfs": projected,
                "terminal_value_pv": _round_val(tv_pv, 0),
                "intrinsic_value_estimate": _round_val(total_pv, 0),
            }

            if market_cap and total_pv:
                dcf["market_cap"] = _round_val(market_cap, 0)
                dcf["upside_pct"] = _round_val((total_pv - market_cap) / market_cap * 100, 2)
        else:
            dcf = {"note": "Negative or zero FCF — DCF not applicable"}
    else:
        dcf = {"note": "Insufficient cash flow data for DCF"}

    # --- Relative Valuation ---
    relative = {}
    if isinstance(ratios, dict) and ratios:
        for key in ["trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda",
                     "priceToSalesTrailing12Months", "enterpriseToRevenue"]:
            val = ratios.get(key)
            if val is not None:
                relative[key] = _round_val(val, 2)

    # Valuation summary
    summary = []
    if dcf.get("upside_pct") is not None:
        up = dcf["upside_pct"]
        if up > 20:
            summary.append(f"DCF suggests undervalued ({up}% upside)")
        elif up < -20:
            summary.append(f"DCF suggests overvalued ({up}% upside)")
        else:
            summary.append(f"DCF suggests fair value ({up}% upside)")

    pe = relative.get("trailingPE")
    if pe is not None:
        if pe < 10:
            summary.append(f"Low P/E ({pe}x) — potentially undervalued or low growth")
        elif pe > 30:
            summary.append(f"High P/E ({pe}x) — growth expectations or overvalued")

    return {
        "symbol": symbol,
        "dcf": dcf,
        "relative_valuation": relative,
        "valuation_summary": summary,
    }


def financial_peer_compare(
    symbols: list[str],
    period: str = "annual",
) -> dict | list[dict]:
    """Compare financial metrics across multiple stocks.

    Parameters
    ----------
    symbols : list[str]
        List of stock symbols to compare.
    period : str
        "annual" or "quarterly".

    Returns
    -------
    dict with keys: peers (list of per-symbol dicts), ranking.
    """
    if not symbols or len(symbols) < 2:
        return {"error": "Need at least 2 symbols for peer comparison"}

    def _analyze_one(sym: str) -> dict:
        data = _fetch_statements(sym, period)
        bs, inc = data["balance_sheet"], data["income_statement"]
        ratios = data["ratios"]

        result = {"symbol": sym}

        # Revenue & profit
        if inc:
            latest = inc[0]
            rev = _find_field(latest, "TotalRevenue", "Total Revenue", "revenue")
            ni = _find_field(latest, "NetIncome", "Net Income", "net_income")
            gp = _find_field(latest, "GrossProfit", "Gross Profit", "gross_profit")
            result["revenue"] = rev
            result["net_income"] = ni
            if rev and rev != 0:
                result["gross_margin_pct"] = _round_val((gp / rev * 100) if gp else None, 2)
                result["net_margin_pct"] = _round_val((ni / rev * 100) if ni else None, 2)

        # Balance sheet
        if bs:
            latest_bs = bs[0]
            ta = _find_field(latest_bs, "TotalAssets", "Total Assets", "total_assets")
            eq = _find_field(
                latest_bs, "StockholdersEquity", "TotalStockholderEquity", "stockholders_equity",
            )
            result["total_assets"] = ta
            result["equity"] = eq
            if ta and eq and ta != 0:
                result["roe_pct"] = _round_val(
                    (result.get("net_income", 0) or 0) / eq * 100, 2,
                ) if eq != 0 else None
                result["roa_pct"] = _round_val(
                    (result.get("net_income", 0) or 0) / ta * 100, 2,
                )

        # Key ratios
        if isinstance(ratios, dict):
            for k in ["trailingPE", "priceToBook", "debtToEquity", "returnOnEquity", "dividendYield"]:
                if ratios.get(k) is not None:
                    result[k] = _round_val(ratios[k], 2)

        return result

    # Fetch in parallel
    peers = []
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as executor:
        futures = {executor.submit(_analyze_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                peers.append(future.result())
            except Exception as e:
                peers.append({"symbol": futures[future], "error": str(e)})

    # Sort by symbol order
    sym_order = {s: i for i, s in enumerate(symbols)}
    peers.sort(key=lambda p: sym_order.get(p.get("symbol", ""), 999))

    # Rankings
    ranking = {}
    for metric in ["net_margin_pct", "roe_pct", "roa_pct"]:
        valid = [(p["symbol"], p[metric]) for p in peers if p.get(metric) is not None]
        if valid:
            valid.sort(key=lambda x: x[1], reverse=True)
            ranking[metric] = [{"symbol": s, "value": v} for s, v in valid]

    return {
        "peers": peers,
        "ranking": ranking,
        "symbol_count": len(symbols),
    }


def financial_dividend(symbol: str) -> dict:
    """Analyze dividend history and payout sustainability.

    Parameters
    ----------
    symbol : str
        Stock symbol.

    Returns
    -------
    dict with keys: symbol, current_yield, payout_ratio, dividend_history, sustainability.
    """
    data = _fetch_statements(symbol)
    inc = data["income_statement"]
    cf = data["cash_flow"]
    ratios = data["ratios"]

    result = {"symbol": symbol}

    # Current yield and payout from ratios
    if isinstance(ratios, dict):
        result["dividend_yield_pct"] = _round_val(
            (ratios.get("dividendYield") or 0) * 100, 2,
        )
        result["payout_ratio_pct"] = _round_val(
            (ratios.get("payoutRatio") or 0) * 100, 2,
        )

    # Dividend payments from cash flow
    div_history = []
    for i, rec in enumerate(cf):
        div_paid = _find_field(
            rec, "CashDividendsPaid", "CommonStockDividendPaid",
            "DividendsPaid", "dividends_paid", "PaymentOfDividends",
        )
        if div_paid is not None:
            div_history.append({"period_index": i, "dividends_paid": abs(div_paid)})

    result["dividend_history"] = div_history

    # Dividend growth rate
    if len(div_history) >= 2:
        latest = div_history[0]["dividends_paid"]
        oldest = div_history[-1]["dividends_paid"]
        years = len(div_history) - 1
        if oldest > 0 and years > 0:
            cagr = (latest / oldest) ** (1 / years) - 1
            result["dividend_cagr_pct"] = _round_val(cagr * 100, 2)

    # Sustainability check
    sustainability = []
    if inc and cf:
        ni = _find_field(inc[0], "NetIncome", "Net Income", "net_income")
        op_cf = _find_field(
            cf[0], "OperatingCashFlow", "TotalCashFromOperatingActivities", "operating_cash_flow",
        )
        div_paid = None
        if div_history:
            div_paid = div_history[0]["dividends_paid"]

        if ni and div_paid:
            payout = div_paid / abs(ni) * 100 if ni != 0 else None
            if payout is not None:
                if payout > 90:
                    sustainability.append(f"HIGH RISK: Payout ratio {_round_val(payout, 1)}% of earnings")
                elif payout > 60:
                    sustainability.append(f"MODERATE: Payout ratio {_round_val(payout, 1)}% of earnings")
                else:
                    sustainability.append(f"SUSTAINABLE: Payout ratio {_round_val(payout, 1)}% of earnings")

        if op_cf and div_paid:
            cf_coverage = op_cf / div_paid if div_paid != 0 else None
            if cf_coverage is not None:
                result["cf_coverage_ratio"] = _round_val(cf_coverage, 2)
                if cf_coverage < 1.0:
                    sustainability.append("WARNING: Operating CF does not cover dividends")
                elif cf_coverage > 2.0:
                    sustainability.append("Strong CF coverage for dividends")

    result["sustainability"] = sustainability
    return result


def financial_ratios_calc(symbol: str, period: str = "annual") -> dict:
    """Compute financial ratios directly from raw statements.

    Unlike financial_ratio() in core.py which fetches pre-calculated ratios
    from yfinance, this function computes ratios from actual balance sheet
    and income statement data, providing more transparency and working
    for all data sources.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    period : str
        "annual" or "quarterly".

    Returns
    -------
    dict with profitability, efficiency, leverage, and valuation ratios.
    """
    data = _fetch_statements(symbol, period)
    bs, inc = data["balance_sheet"], data["income_statement"]

    if not bs or not inc:
        return {"error": f"Insufficient financial data for {symbol}"}

    curr_bs, curr_inc = bs[0], inc[0]

    # Extract raw fields
    revenue = _find_field(curr_inc, "TotalRevenue", "Total Revenue", "revenue")
    cogs = _find_field(curr_inc, "CostOfRevenue", "Cost Of Revenue", "cost_of_goods_sold")
    gross_profit = _find_field(curr_inc, "GrossProfit", "Gross Profit", "gross_profit")
    operating_income = _find_field(curr_inc, "OperatingIncome", "EBIT", "operating_income")
    net_income = _find_field(curr_inc, "NetIncome", "Net Income", "net_income")
    interest_expense = _find_field(curr_inc, "InterestExpense", "interest_expense") or 0

    total_assets = _find_field(curr_bs, "TotalAssets", "Total Assets", "total_assets")
    current_assets = _find_field(curr_bs, "CurrentAssets", "Total Current Assets", "current_assets")
    current_liab = _find_field(
        curr_bs, "CurrentLiabilities", "Total Current Liabilities", "current_liabilities",
    )
    inventory = _find_field(curr_bs, "Inventory", "inventory") or 0
    receivables = _find_field(curr_bs, "NetReceivables", "AccountsReceivable", "receivables") or 0
    equity = _find_field(
        curr_bs, "StockholdersEquity", "TotalStockholderEquity", "stockholders_equity",
    )
    total_debt = _find_field(
        curr_bs, "TotalDebt", "total_debt", "LongTermDebt", "long_term_debt",
    ) or 0

    # -- Profitability --
    profitability = {}
    if revenue and revenue != 0:
        if gross_profit is not None:
            profitability["gross_margin_pct"] = _round_val(gross_profit / revenue * 100, 2)
        elif cogs is not None:
            profitability["gross_margin_pct"] = _round_val((revenue - abs(cogs)) / revenue * 100, 2)
        if operating_income is not None:
            profitability["operating_margin_pct"] = _round_val(operating_income / revenue * 100, 2)
        if net_income is not None:
            profitability["net_margin_pct"] = _round_val(net_income / revenue * 100, 2)

    if net_income is not None:
        if total_assets and total_assets != 0:
            profitability["roa_pct"] = _round_val(net_income / total_assets * 100, 2)
        if equity and equity != 0:
            profitability["roe_pct"] = _round_val(net_income / equity * 100, 2)

    # -- Efficiency --
    efficiency = {}
    if revenue and revenue != 0:
        if total_assets and total_assets != 0:
            efficiency["asset_turnover"] = _round_val(revenue / total_assets, 2)
        if receivables:
            efficiency["receivables_turnover"] = _round_val(revenue / receivables, 2)
            efficiency["days_sales_outstanding"] = _round_val(365 / (revenue / receivables), 1)
    if cogs and cogs != 0 and inventory:
        efficiency["inventory_turnover"] = _round_val(abs(cogs) / inventory, 2)
        efficiency["days_inventory_outstanding"] = _round_val(365 / (abs(cogs) / inventory), 1)

    # -- Leverage --
    leverage_ratios = {}
    if equity and equity != 0:
        leverage_ratios["debt_to_equity"] = _round_val(total_debt / equity, 2)
        if total_assets:
            leverage_ratios["equity_multiplier"] = _round_val(total_assets / equity, 2)
    if total_assets and total_assets != 0:
        leverage_ratios["debt_to_assets"] = _round_val(total_debt / total_assets, 2)
    if operating_income is not None and abs(interest_expense) > 0:
        leverage_ratios["interest_coverage"] = _round_val(operating_income / abs(interest_expense), 2)

    # -- Liquidity --
    liquidity = {}
    if current_assets is not None and current_liab and current_liab != 0:
        liquidity["current_ratio"] = _round_val(current_assets / current_liab, 2)
        liquidity["quick_ratio"] = _round_val((current_assets - inventory) / current_liab, 2)

    # -- DuPont Decomposition --
    dupont = {}
    if all(v is not None and v != 0 for v in [net_income, revenue, total_assets, equity]):
        npm = net_income / revenue
        at = revenue / total_assets
        em = total_assets / equity
        dupont["net_profit_margin"] = _round_val(npm, 4)
        dupont["asset_turnover"] = _round_val(at, 4)
        dupont["equity_multiplier"] = _round_val(em, 4)
        dupont["roe_decomposed_pct"] = _round_val(npm * at * em * 100, 2)

    return {
        "symbol": symbol,
        "period": period,
        "profitability": profitability,
        "efficiency": efficiency,
        "leverage": leverage_ratios,
        "liquidity": liquidity,
        "dupont_analysis": dupont,
    }
