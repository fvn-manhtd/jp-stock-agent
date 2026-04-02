"""
Tests for the fundamental financial analysis module (financial.py).

Tests cover:
- _round_val, _safe_get, _find_field: Internal helper functions
- _piotroski_f_score: F-Score calculation (0-9)
- _altman_z_score: Z-Score bankruptcy prediction
- financial_health: Combined health assessment
- financial_growth: Revenue/earnings growth trends
- financial_valuation: DCF and relative valuation
- financial_peer_compare: Multi-stock comparison
- financial_dividend: Dividend sustainability analysis
- financial_ratios_calc: Ratio computation from raw statements
- Error handling for missing/insufficient data
"""

from unittest.mock import patch

from jpstock_agent import financial
from jpstock_agent.financial import (
    _altman_z_score,
    _find_field,
    _piotroski_f_score,
    _round_val,
    _safe_get,
)

# ---------------------------------------------------------------------------
# Mock financial statement data
# ---------------------------------------------------------------------------

MOCK_BALANCE_SHEET = [
    {  # Current period
        "TotalAssets": 5_000_000_000,
        "CurrentAssets": 2_000_000_000,
        "CurrentLiabilities": 1_000_000_000,
        "Inventory": 300_000_000,
        "NetReceivables": 400_000_000,
        "CashAndCashEquivalents": 500_000_000,
        "LongTermDebt": 800_000_000,
        "TotalDebt": 1_200_000_000,
        "StockholdersEquity": 2_500_000_000,
        "RetainedEarnings": 1_500_000_000,
        "TotalLiab": 2_500_000_000,
        "AccountsPayable": 200_000_000,
        "ShareIssued": 100_000_000,
    },
    {  # Prior period
        "TotalAssets": 4_800_000_000,
        "CurrentAssets": 1_800_000_000,
        "CurrentLiabilities": 1_100_000_000,
        "Inventory": 350_000_000,
        "NetReceivables": 380_000_000,
        "CashAndCashEquivalents": 400_000_000,
        "LongTermDebt": 900_000_000,
        "TotalDebt": 1_300_000_000,
        "StockholdersEquity": 2_200_000_000,
        "RetainedEarnings": 1_200_000_000,
        "TotalLiab": 2_600_000_000,
        "AccountsPayable": 180_000_000,
        "ShareIssued": 100_000_000,
    },
]

MOCK_INCOME_STATEMENT = [
    {  # Current period
        "TotalRevenue": 3_000_000_000,
        "CostOfRevenue": 1_800_000_000,
        "GrossProfit": 1_200_000_000,
        "OperatingIncome": 600_000_000,
        "NetIncome": 400_000_000,
        "InterestExpense": 50_000_000,
    },
    {  # Prior period
        "TotalRevenue": 2_700_000_000,
        "CostOfRevenue": 1_700_000_000,
        "GrossProfit": 1_000_000_000,
        "OperatingIncome": 500_000_000,
        "NetIncome": 300_000_000,
        "InterestExpense": 55_000_000,
    },
]

MOCK_CASH_FLOW = [
    {
        "OperatingCashFlow": 700_000_000,
        "CapitalExpenditure": -200_000_000,
        "CashDividendsPaid": -100_000_000,
    },
    {
        "OperatingCashFlow": 600_000_000,
        "CapitalExpenditure": -180_000_000,
        "CashDividendsPaid": -80_000_000,
    },
]

MOCK_RATIOS = {
    "trailingPE": 15.5,
    "forwardPE": 13.2,
    "priceToBook": 2.1,
    "enterpriseToEbitda": 10.5,
    "priceToSalesTrailing12Months": 3.2,
    "dividendYield": 0.025,
    "payoutRatio": 0.35,
    "returnOnEquity": 0.16,
    "debtToEquity": 48.0,
}

MOCK_OVERVIEW = {
    "shortName": "Test Corp",
    "marketCap": 10_000_000_000,
}


def _mock_fetch_statements(symbol, period="annual"):
    """Return mock financial data."""
    return {
        "balance_sheet": MOCK_BALANCE_SHEET,
        "income_statement": MOCK_INCOME_STATEMENT,
        "cash_flow": MOCK_CASH_FLOW,
        "ratios": MOCK_RATIOS,
        "errors": [],
    }


# ============================================================================
# Helper Tests
# ============================================================================


class TestRoundVal:
    def test_normal_value(self):
        assert _round_val(3.14159, 2) == 3.14

    def test_none(self):
        assert _round_val(None) is None

    def test_nan(self):
        assert _round_val(float("nan")) is None

    def test_inf(self):
        assert _round_val(float("inf")) is None

    def test_string_number(self):
        assert _round_val("3.14", 2) == 3.14

    def test_non_numeric_string(self):
        assert _round_val("abc") is None


class TestSafeGet:
    def test_existing_key(self):
        assert _safe_get({"a": 10}, "a") == 10.0

    def test_missing_key(self):
        assert _safe_get({"a": 10}, "b") is None

    def test_default_value(self):
        assert _safe_get({"a": 10}, "b", 99) == 99

    def test_non_numeric(self):
        assert _safe_get({"a": "text"}, "a") is None

    def test_list_input(self):
        assert _safe_get([1, 2], "a") is None


class TestFindField:
    def test_first_candidate(self):
        rec = {"TotalRevenue": 1000}
        assert _find_field(rec, "TotalRevenue", "Revenue") == 1000.0

    def test_second_candidate(self):
        rec = {"Revenue": 1000}
        assert _find_field(rec, "TotalRevenue", "Revenue") == 1000.0

    def test_case_insensitive(self):
        rec = {"totalrevenue": 1000}
        assert _find_field(rec, "TotalRevenue") == 1000.0

    def test_space_insensitive(self):
        rec = {"Total Revenue": 1000}
        assert _find_field(rec, "TotalRevenue") == 1000.0

    def test_not_found(self):
        rec = {"Other": 1000}
        assert _find_field(rec, "TotalRevenue") is None

    def test_default_value(self):
        rec = {}
        assert _find_field(rec, "TotalRevenue", default=0) == 0


# ============================================================================
# Piotroski F-Score Tests
# ============================================================================


class TestPiotrokiFScore:
    def test_returns_score(self):
        result = _piotroski_f_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT, MOCK_CASH_FLOW)
        assert "score" in result
        assert result["score"] is not None
        assert 0 <= result["score"] <= 9

    def test_max_score_is_9(self):
        result = _piotroski_f_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT, MOCK_CASH_FLOW)
        assert result["max_score"] == 9

    def test_interpretation_present(self):
        result = _piotroski_f_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT, MOCK_CASH_FLOW)
        assert result["interpretation"] in ["STRONG", "MODERATE", "WEAK"]

    def test_details_dict(self):
        result = _piotroski_f_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT, MOCK_CASH_FLOW)
        assert isinstance(result["details"], dict)
        assert "net_income_positive" in result["details"]
        assert "operating_cf_positive" in result["details"]

    def test_insufficient_data(self):
        result = _piotroski_f_score([], [], [])
        assert result["score"] is None

    def test_positive_net_income_scores(self):
        """Our mock data has positive net income, so that criterion should pass."""
        result = _piotroski_f_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT, MOCK_CASH_FLOW)
        assert result["details"]["net_income_positive"] is True

    def test_operating_cf_positive_scores(self):
        """Our mock has positive operating CF."""
        result = _piotroski_f_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT, MOCK_CASH_FLOW)
        assert result["details"]["operating_cf_positive"] is True

    def test_cf_gt_net_income(self):
        """Operating CF (700M) > Net Income (400M) in mock data."""
        result = _piotroski_f_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT, MOCK_CASH_FLOW)
        assert result["details"]["cf_gt_net_income"] is True


# ============================================================================
# Altman Z-Score Tests
# ============================================================================


class TestAltmanZScore:
    def test_returns_z_score(self):
        result = _altman_z_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT)
        assert "z_score" in result
        assert result["z_score"] is not None

    def test_zone_classification(self):
        result = _altman_z_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT)
        assert result["zone"] in ["SAFE", "GREY", "DISTRESS"]

    def test_components_present(self):
        result = _altman_z_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT)
        comps = result["components"]
        assert "x1_working_capital_ta" in comps
        assert "x2_retained_earnings_ta" in comps
        assert "x3_ebit_ta" in comps
        assert "x4_equity_tl" in comps
        assert "x5_revenue_ta" in comps

    def test_empty_data(self):
        result = _altman_z_score([], [])
        assert result["z_score"] is None

    def test_healthy_company_safe_zone(self):
        """Our mock data represents a healthy company, should be SAFE or GREY."""
        result = _altman_z_score(MOCK_BALANCE_SHEET, MOCK_INCOME_STATEMENT)
        # With working capital = 1B, retained = 1.5B, EBIT = 600M, equity/liab = 1.0, rev/assets = 0.6
        # Z ≈ 1.2*0.2 + 1.4*0.3 + 3.3*0.12 + 0.6*1.0 + 1.0*0.6 = 0.24 + 0.42 + 0.396 + 0.6 + 0.6 ≈ 2.26
        assert result["zone"] in ["SAFE", "GREY"]


# ============================================================================
# Financial Health Tests
# ============================================================================


class TestFinancialHealth:
    def test_returns_dict_with_expected_keys(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_health("7203")

        assert isinstance(result, dict)
        assert result["symbol"] == "7203"
        assert "altman_z" in result
        assert "piotroski_f" in result
        assert "liquidity" in result
        assert "leverage" in result

    def test_liquidity_ratios(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_health("7203")

        liq = result["liquidity"]
        assert "current_ratio" in liq
        assert liq["current_ratio"] == 2.0  # 2B / 1B

    def test_leverage_metrics(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_health("7203")

        lev = result["leverage"]
        assert "debt_to_equity" in lev
        assert "debt_to_assets" in lev

    def test_health_signals_list(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_health("7203")

        assert isinstance(result["health_signals"], list)

    def test_no_data_returns_error(self):
        def empty_data(sym, period="annual"):
            return {"balance_sheet": [], "income_statement": [], "cash_flow": [], "ratios": {}, "errors": []}

        with patch("jpstock_agent.financial._fetch_statements", side_effect=empty_data):
            result = financial.financial_health("XXXX")

        assert "error" in result

    def test_cash_conversion_cycle(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_health("7203")

        lev = result["leverage"]
        assert "receivables_days" in lev
        assert "cash_conversion_cycle_days" in lev


# ============================================================================
# Financial Growth Tests
# ============================================================================


class TestFinancialGrowth:
    def test_returns_growth_data(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_growth("7203")

        assert result["symbol"] == "7203"
        assert "revenue_growth" in result
        assert "earnings_growth" in result
        assert "margin_trends" in result

    def test_revenue_growth_calculated(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_growth("7203")

        rg = result["revenue_growth"]
        assert len(rg) >= 1
        # Revenue: 3B vs 2.7B = 11.11% growth
        assert rg[0]["yoy_growth_pct"] is not None
        assert abs(rg[0]["yoy_growth_pct"] - 11.11) < 0.1

    def test_earnings_growth_calculated(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_growth("7203")

        eg = result["earnings_growth"]
        assert len(eg) >= 1
        # NI: 400M vs 300M = 33.33% growth
        assert eg[0]["yoy_growth_pct"] is not None
        assert abs(eg[0]["yoy_growth_pct"] - 33.33) < 0.1

    def test_margin_trends(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_growth("7203")

        mt = result["margin_trends"]
        assert len(mt) >= 2
        assert "gross_margin_pct" in mt[0]
        assert "net_margin_pct" in mt[0]

    def test_fcf_trend(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_growth("7203")

        fcf = result["free_cash_flow_trend"]
        assert len(fcf) >= 1
        # FCF = 700M - 200M = 500M
        assert fcf[0]["free_cash_flow"] == 500_000_000

    def test_insufficient_data(self):
        def one_period(sym, period="annual"):
            return {
                "balance_sheet": [MOCK_BALANCE_SHEET[0]],
                "income_statement": [MOCK_INCOME_STATEMENT[0]],
                "cash_flow": [MOCK_CASH_FLOW[0]],
                "ratios": {},
                "errors": [],
            }

        with patch("jpstock_agent.financial._fetch_statements", side_effect=one_period):
            result = financial.financial_growth("7203")

        assert "error" in result

    def test_summary_generated(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_growth("7203")

        assert isinstance(result["summary"], list)
        # 11% revenue growth and 33% earnings growth should generate summary items
        assert len(result["summary"]) > 0


# ============================================================================
# Financial Valuation Tests
# ============================================================================


class TestFinancialValuation:
    def test_returns_valuation(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements), \
             patch("jpstock_agent.financial.company_overview", return_value=MOCK_OVERVIEW):
            result = financial.financial_valuation("7203")

        assert result["symbol"] == "7203"
        assert "dcf" in result
        assert "relative_valuation" in result

    def test_dcf_with_positive_fcf(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements), \
             patch("jpstock_agent.financial.company_overview", return_value=MOCK_OVERVIEW):
            result = financial.financial_valuation("7203")

        dcf = result["dcf"]
        assert "latest_fcf" in dcf
        assert "intrinsic_value_estimate" in dcf
        assert dcf["intrinsic_value_estimate"] is not None

    def test_dcf_upside_calculated(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements), \
             patch("jpstock_agent.financial.company_overview", return_value=MOCK_OVERVIEW):
            result = financial.financial_valuation("7203")

        dcf = result["dcf"]
        assert "upside_pct" in dcf

    def test_relative_valuation_ratios(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements), \
             patch("jpstock_agent.financial.company_overview", return_value=MOCK_OVERVIEW):
            result = financial.financial_valuation("7203")

        rv = result["relative_valuation"]
        assert "trailingPE" in rv
        assert rv["trailingPE"] == 15.5

    def test_custom_discount_rate(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements), \
             patch("jpstock_agent.financial.company_overview", return_value=MOCK_OVERVIEW):
            result_low = financial.financial_valuation("7203", discount_rate=0.05)
            result_high = financial.financial_valuation("7203", discount_rate=0.15)

        # Lower discount rate → higher intrinsic value
        if result_low["dcf"].get("intrinsic_value_estimate") and result_high["dcf"].get("intrinsic_value_estimate"):
            assert result_low["dcf"]["intrinsic_value_estimate"] > result_high["dcf"]["intrinsic_value_estimate"]

    def test_negative_fcf_handled(self):
        def neg_fcf(sym, period="annual"):
            cf = [{"OperatingCashFlow": -100_000_000, "CapitalExpenditure": -200_000_000}]
            return {
                "balance_sheet": MOCK_BALANCE_SHEET,
                "income_statement": MOCK_INCOME_STATEMENT,
                "cash_flow": cf,
                "ratios": MOCK_RATIOS,
                "errors": [],
            }

        with patch("jpstock_agent.financial._fetch_statements", side_effect=neg_fcf), \
             patch("jpstock_agent.financial.company_overview", return_value=MOCK_OVERVIEW):
            result = financial.financial_valuation("7203")

        assert "note" in result["dcf"]  # Should note that DCF not applicable


# ============================================================================
# Peer Compare Tests
# ============================================================================


class TestFinancialPeerCompare:
    def test_returns_peers(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_peer_compare(["7203", "6758"])

        assert "peers" in result
        assert len(result["peers"]) == 2
        assert result["symbol_count"] == 2

    def test_ranking_present(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_peer_compare(["7203", "6758"])

        assert "ranking" in result

    def test_peer_has_metrics(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_peer_compare(["7203", "6758"])

        peer = result["peers"][0]
        assert "symbol" in peer
        assert "revenue" in peer
        assert "net_income" in peer

    def test_too_few_symbols(self):
        result = financial.financial_peer_compare(["7203"])
        assert "error" in result

    def test_empty_symbols(self):
        result = financial.financial_peer_compare([])
        assert "error" in result


# ============================================================================
# Dividend Analysis Tests
# ============================================================================


class TestFinancialDividend:
    def test_returns_dividend_data(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_dividend("7203")

        assert result["symbol"] == "7203"
        assert "dividend_yield_pct" in result
        assert "payout_ratio_pct" in result

    def test_dividend_history(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_dividend("7203")

        hist = result["dividend_history"]
        assert isinstance(hist, list)
        assert len(hist) >= 1
        assert "dividends_paid" in hist[0]

    def test_dividend_cagr(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_dividend("7203")

        # With 2 periods of dividend data, CAGR should be calculated
        assert "dividend_cagr_pct" in result

    def test_sustainability_analysis(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_dividend("7203")

        assert isinstance(result["sustainability"], list)

    def test_cf_coverage_ratio(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_dividend("7203")

        # OpCF (700M) / Div (100M) = 7.0
        assert result.get("cf_coverage_ratio") is not None
        assert result["cf_coverage_ratio"] == 7.0


# ============================================================================
# Ratios Calculation Tests
# ============================================================================


class TestFinancialRatiosCalc:
    def test_returns_ratios(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_ratios_calc("7203")

        assert result["symbol"] == "7203"
        assert "profitability" in result
        assert "efficiency" in result
        assert "leverage" in result
        assert "liquidity" in result

    def test_profitability_margins(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_ratios_calc("7203")

        prof = result["profitability"]
        # Gross margin: 1.2B / 3B = 40%
        assert abs(prof["gross_margin_pct"] - 40.0) < 0.1
        # Net margin: 400M / 3B ≈ 13.33%
        assert abs(prof["net_margin_pct"] - 13.33) < 0.1
        # ROE: 400M / 2.5B = 16%
        assert abs(prof["roe_pct"] - 16.0) < 0.1
        # ROA: 400M / 5B = 8%
        assert abs(prof["roa_pct"] - 8.0) < 0.1

    def test_efficiency_ratios(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_ratios_calc("7203")

        eff = result["efficiency"]
        assert "asset_turnover" in eff
        assert eff["asset_turnover"] == 0.6  # 3B / 5B

    def test_leverage_ratios(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_ratios_calc("7203")

        lev = result["leverage"]
        assert "debt_to_equity" in lev
        assert lev["debt_to_equity"] == 0.48  # 1.2B / 2.5B
        assert "interest_coverage" in lev
        assert lev["interest_coverage"] == 12.0  # 600M / 50M

    def test_liquidity_ratios(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_ratios_calc("7203")

        liq = result["liquidity"]
        assert liq["current_ratio"] == 2.0  # 2B / 1B
        assert liq["quick_ratio"] == 1.7  # (2B - 300M) / 1B

    def test_dupont_analysis(self):
        with patch("jpstock_agent.financial._fetch_statements", side_effect=_mock_fetch_statements):
            result = financial.financial_ratios_calc("7203")

        dp = result["dupont_analysis"]
        assert "net_profit_margin" in dp
        assert "asset_turnover" in dp
        assert "equity_multiplier" in dp
        assert "roe_decomposed_pct" in dp
        # DuPont ROE = NPM * AT * EM = (400M/3B) * (3B/5B) * (5B/2.5B)
        # = 0.1333 * 0.6 * 2.0 = 0.16 = 16%
        assert abs(dp["roe_decomposed_pct"] - 16.0) < 0.1

    def test_no_data_returns_error(self):
        def empty_data(sym, period="annual"):
            return {"balance_sheet": [], "income_statement": [], "cash_flow": [], "ratios": {}, "errors": []}

        with patch("jpstock_agent.financial._fetch_statements", side_effect=empty_data):
            result = financial.financial_ratios_calc("XXXX")

        assert "error" in result
