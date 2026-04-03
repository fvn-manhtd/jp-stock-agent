"""Tests for jpstock_agent.report module.

Comprehensive test suite for report generation functions:
- stock_report: full analysis report
- stock_report_quick: lightweight quick report
- stock_report_compare: side-by-side comparison
- Internal helpers: _safe_section, _parallel_sections, _build_executive_summary
"""

from unittest.mock import MagicMock, patch

from jpstock_agent import report
from tests.conftest import _make_ohlcv_records

# ---------------------------------------------------------------------------
# Helper: Create mock modules dict
# ---------------------------------------------------------------------------

def _mock_modules():
    """Create a complete mock modules dict matching _import_modules output."""
    return {
        "ta": MagicMock(),
        "candlestick": MagicMock(),
        "financial": MagicMock(),
        "sentiment": MagicMock(),
        "portfolio": MagicMock(),
    }


def _mock_core():
    """Create a mock core module."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Test _safe_section
# ---------------------------------------------------------------------------

class TestSafeSection:
    """Tests for _safe_section error handling."""

    def test_safe_section_returns_function_result_on_success(self):
        """Test _safe_section returns result when function succeeds."""
        def mock_func():
            return {"data": "value"}

        result = report._safe_section(mock_func, "test_section")
        assert result == {"data": "value"}

    def test_safe_section_returns_error_dict_on_exception(self):
        """Test _safe_section returns _error dict on exception."""
        def mock_func():
            raise ValueError("Test error")

        result = report._safe_section(mock_func, "test_section")
        assert isinstance(result, dict)
        assert "_error" in result
        assert "Test error" in result["_error"]

    def test_safe_section_wraps_error_in_result(self):
        """Test _safe_section converts error result to _error."""
        def mock_func():
            return {"error": "API call failed"}

        result = report._safe_section(mock_func, "test_section")
        assert result == {"_error": "API call failed"}

    def test_safe_section_logs_warning_on_failure(self):
        """Test _safe_section logs warning when section fails."""
        def mock_func():
            raise RuntimeError("Connection timeout")

        with patch("jpstock_agent.report.logger") as mock_logger:
            result = report._safe_section(mock_func, "technical")
            mock_logger.warning.assert_called_once()
            assert "_error" in result


# ---------------------------------------------------------------------------
# Test StockReport (Full Report)
# ---------------------------------------------------------------------------

class TestStockReport:
    """Tests for stock_report function."""

    def test_stock_report_returns_dict_with_symbol_and_type(self):
        """Test stock_report returns dict with symbol and report_type."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report._import_ml", return_value=None):

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.company_overview.return_value = {"symbol": "7203", "name": "Toyota"}
            mock_core_import.return_value = mock_core

            # Mock all module functions
            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {
                "signal": "BUY",
                "score": 75
            }
            mods["candlestick"].ta_candlestick_latest.return_value = {
                "patterns": ["hammer"]
            }
            mods["financial"].financial_health.return_value = {
                "altman_z": {"z_score": 2.5, "zone": "safe"}
            }
            mods["financial"].financial_growth.return_value = {"revenue_growth": 5}
            mods["financial"].financial_valuation.return_value = {"dcf": {"upside_pct": 15}}
            mods["financial"].financial_dividend.return_value = {"yield": 2.5}
            mods["sentiment"].sentiment_news.return_value = {"average_score": 0.3}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report("7203")

            assert isinstance(result, dict)
            assert result["symbol"] == "7203"
            assert result["report_type"] == "comprehensive"

    def test_stock_report_includes_price_summary(self):
        """Test stock_report includes price_summary section."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report._import_ml", return_value=None):

            mock_core = _mock_core()
            records = _make_ohlcv_records(days=30)
            mock_core.stock_history.return_value = records
            mock_core.company_overview.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            for key in ["ta", "candlestick", "financial", "sentiment"]:
                mods[key].ta_multi_indicator = MagicMock(return_value={})
                mods[key].ta_candlestick_latest = MagicMock(return_value={})
                mods[key].financial_health = MagicMock(return_value={})
                mods[key].financial_growth = MagicMock(return_value={})
                mods[key].financial_valuation = MagicMock(return_value={})
                mods[key].financial_dividend = MagicMock(return_value={})
                mods[key].sentiment_news = MagicMock(return_value={})

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report("7203")

            assert "price_summary" in result
            assert "latest_close" in result["price_summary"]
            assert "latest_date" in result["price_summary"]
            assert "data_points" in result["price_summary"]

    def test_stock_report_includes_technical_section(self):
        """Test stock_report includes technical analysis section."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report._import_ml", return_value=None):

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.company_overview.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {
                "signal": "BUY",
                "score": 85,
                "rsi": 65
            }
            # Set default empty returns for other modules
            for mod_name in ["candlestick", "financial", "sentiment"]:
                for func_name in ["ta_candlestick_latest", "financial_health", "financial_growth",
                                  "financial_valuation", "financial_dividend", "sentiment_news"]:
                    getattr(mods[mod_name], func_name, MagicMock()).return_value = {}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report("7203")

            assert "technical" in result
            assert result["technical"]["signal"] == "BUY"
            assert result["technical"]["score"] == 85

    def test_stock_report_includes_executive_summary(self):
        """Test stock_report includes executive_summary."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report._import_ml", return_value=None):

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.company_overview.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {"signal": "BUY", "score": 75}
            for mod_name in ["candlestick", "financial", "sentiment"]:
                for func_name in ["ta_candlestick_latest", "financial_health", "financial_growth",
                                  "financial_valuation", "financial_dividend", "sentiment_news"]:
                    getattr(mods[mod_name], func_name, MagicMock()).return_value = {}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report("7203")

            assert "executive_summary" in result
            assert isinstance(result["executive_summary"], list)

    def test_stock_report_includes_generation_time_ms(self):
        """Test stock_report includes generation_time_ms."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report._import_ml", return_value=None):

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.company_overview.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            for mod_name in ["ta", "candlestick", "financial", "sentiment"]:
                for func_name in ["ta_multi_indicator", "ta_candlestick_latest",
                                  "financial_health", "financial_growth",
                                  "financial_valuation", "financial_dividend", "sentiment_news"]:
                    getattr(mods[mod_name], func_name, MagicMock()).return_value = {}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report("7203")

            assert "generation_time_ms" in result
            assert isinstance(result["generation_time_ms"], (int, float))
            assert result["generation_time_ms"] >= 0

    def test_stock_report_excludes_sections_with_errors(self):
        """Test stock_report excludes sections that failed (_error)."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report._import_ml", return_value=None):

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.company_overview.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {"signal": "BUY", "score": 75}
            mods["candlestick"].ta_candlestick_latest.return_value = {"_error": "Failed to fetch"}
            for mod_name in ["financial", "sentiment"]:
                for func_name in ["financial_health", "financial_growth",
                                  "financial_valuation", "financial_dividend", "sentiment_news"]:
                    getattr(mods[mod_name], func_name, MagicMock()).return_value = {}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report("7203")

            # candlestick section should not be in result if it had _error
            if "candlestick" in result:
                assert "_error" not in result["candlestick"]

    def test_stock_report_with_ml_prediction(self):
        """Test stock_report includes ML prediction when available."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report._import_ml") as mock_ml_import:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.company_overview.return_value = {}
            mock_core_import.return_value = mock_core

            mock_ml = MagicMock()
            mock_ml.ml_signal.return_value = {"combined_signal": "BUY", "combined_score": 80}
            mock_ml_import.return_value = mock_ml

            mods = _mock_modules()
            for mod_name in ["ta", "candlestick", "financial", "sentiment"]:
                for func_name in ["ta_multi_indicator", "ta_candlestick_latest",
                                  "financial_health", "financial_growth",
                                  "financial_valuation", "financial_dividend", "sentiment_news"]:
                    getattr(mods[mod_name], func_name, MagicMock()).return_value = {}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report("7203", include_ml=True)

            assert "ml_prediction" in result
            assert result["ml_prediction"]["combined_signal"] == "BUY"


# ---------------------------------------------------------------------------
# Test StockReportQuick
# ---------------------------------------------------------------------------

class TestStockReportQuick:
    """Tests for stock_report_quick function."""

    def test_stock_report_quick_returns_dict_with_quick_type(self):
        """Test stock_report_quick returns dict with report_type='quick'."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {
                "trailingPE": 15.5,
                "forwardPE": 14.2,
                "priceToBook": 1.8,
                "dividendYield": 2.5,
                "returnOnEquity": 12.0,
                "debtToEquity": 0.5
            }
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {
                "signal": "BUY",
                "score": 75,
                "rsi": 65,
                "macd_signal": "positive"
            }

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_quick("7203")

            assert isinstance(result, dict)
            assert result["symbol"] == "7203"
            assert result["report_type"] == "quick"

    def test_stock_report_quick_includes_signal_and_score(self):
        """Test stock_report_quick includes signal and signal_score."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {
                "signal": "SELL",
                "score": 35,
                "rsi": 25,
                "macd_signal": "negative"
            }

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_quick("7203")

            assert "signal" in result
            assert result["signal"] == "SELL"
            assert "signal_score" in result
            assert result["signal_score"] == 35

    def test_stock_report_quick_includes_key_ratios(self):
        """Test stock_report_quick includes key_ratios."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {
                "trailingPE": 16.0,
                "forwardPE": 14.5,
                "priceToBook": 2.0,
                "dividendYield": 3.0,
                "returnOnEquity": 15.0,
                "debtToEquity": 0.6,
                "other_ratio": 999  # Should not be included
            }
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {"signal": "BUY", "score": 70}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_quick("7203")

            assert "key_ratios" in result
            assert result["key_ratios"]["trailingPE"] == 16.0
            assert "other_ratio" not in result["key_ratios"]

    def test_stock_report_quick_includes_quick_summary(self):
        """Test stock_report_quick includes quick_summary list."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {"trailingPE": 15.0}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {"signal": "BUY", "score": 75}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_quick("7203")

            assert "quick_summary" in result
            assert isinstance(result["quick_summary"], list)

    def test_stock_report_quick_includes_generation_time_ms(self):
        """Test stock_report_quick includes generation_time_ms."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {}

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_quick("7203")

            assert "generation_time_ms" in result
            assert isinstance(result["generation_time_ms"], (int, float))
            assert result["generation_time_ms"] >= 0


# ---------------------------------------------------------------------------
# Test StockReportCompare
# ---------------------------------------------------------------------------

class TestStockReportCompare:
    """Tests for stock_report_compare function."""

    def test_stock_report_compare_error_on_no_symbols(self):
        """Test stock_report_compare returns error when no symbols provided."""
        result = report.stock_report_compare([])
        assert isinstance(result, dict)
        assert "error" in result

    def test_stock_report_compare_error_on_single_symbol(self):
        """Test stock_report_compare returns error with less than 2 symbols."""
        result = report.stock_report_compare(["7203"])
        assert isinstance(result, dict)
        assert "error" in result
        assert "at least 2" in result["error"]

    def test_stock_report_compare_returns_dict_with_comparison_type(self):
        """Test stock_report_compare returns dict with report_type='comparison'."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report.stock_report_quick") as mock_quick:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {"signal": "BUY", "score": 75}
            mods["financial"].financial_health.return_value = {
                "altman_z": {"z_score": 2.5, "zone": "safe"},
                "piotroski_f": {"score": 7}
            }

            def mock_quick_report(symbol, source=None):
                return {
                    "symbol": symbol,
                    "report_type": "quick",
                    "price_summary": {"period_return_pct": 5.0},
                    "signal": "BUY",
                    "signal_score": 75
                }

            mock_quick.side_effect = mock_quick_report

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_compare(["7203", "6758"])

            assert isinstance(result, dict)
            assert result["report_type"] == "comparison"

    def test_stock_report_compare_includes_stocks_list(self):
        """Test stock_report_compare includes stocks list."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report.stock_report_quick") as mock_quick:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {"signal": "BUY", "score": 75}
            mods["financial"].financial_health.return_value = {
                "altman_z": {"z_score": 2.5},
                "piotroski_f": {"score": 6}
            }

            def mock_quick_report(symbol, source=None):
                return {
                    "symbol": symbol,
                    "price_summary": {"period_return_pct": 3.0},
                    "signal_score": 70
                }

            mock_quick.side_effect = mock_quick_report

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_compare(["7203", "6758"])

            assert "stocks" in result
            assert isinstance(result["stocks"], list)
            assert len(result["stocks"]) == 2

    def test_stock_report_compare_includes_rankings(self):
        """Test stock_report_compare includes rankings."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report.stock_report_quick") as mock_quick:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {"signal": "BUY", "score": 75}
            mods["financial"].financial_health.return_value = {
                "altman_z": {"z_score": 2.5},
                "piotroski_f": {"score": 6}
            }

            def mock_quick_report(symbol, source=None):
                return {
                    "symbol": symbol,
                    "price_summary": {"period_return_pct": 5.0},
                    "signal_score": 75,
                    "f_score": 6
                }

            mock_quick.side_effect = mock_quick_report

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_compare(["7203", "6758"])

            assert "rankings" in result
            assert isinstance(result["rankings"], dict)

    def test_stock_report_compare_includes_generation_time_ms(self):
        """Test stock_report_compare includes generation_time_ms."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report.stock_report_quick") as mock_quick:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {}
            mods["financial"].financial_health.return_value = {}

            def mock_quick_report(symbol, source=None):
                return {"symbol": symbol}

            mock_quick.side_effect = mock_quick_report

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_compare(["7203", "6758"])

            assert "generation_time_ms" in result
            assert isinstance(result["generation_time_ms"], (int, float))

    def test_stock_report_compare_maintains_symbol_order(self):
        """Test stock_report_compare maintains input symbol order."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report.stock_report_quick") as mock_quick:

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=30)
            mock_core.financial_ratio.return_value = {}
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {}
            mods["financial"].financial_health.return_value = {}

            def mock_quick_report(symbol, source=None):
                return {"symbol": symbol, "price_summary": {"period_return_pct": 1.0}}

            mock_quick.side_effect = mock_quick_report

            symbols = ["9984", "7203", "6758"]
            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report_compare(symbols)

            result_symbols = [s["symbol"] for s in result["stocks"]]
            assert result_symbols == symbols


# ---------------------------------------------------------------------------
# Test _build_executive_summary
# ---------------------------------------------------------------------------

class TestBuildExecutiveSummary:
    """Tests for _build_executive_summary function."""

    def test_executive_summary_returns_list(self):
        """Test _build_executive_summary returns list."""
        report_data = {"symbol": "7203"}
        result = report._build_executive_summary(report_data)
        assert isinstance(result, list)

    def test_executive_summary_includes_price_action(self):
        """Test executive_summary includes price action summary."""
        report_data = {
            "symbol": "7203",
            "price_summary": {"period_return_pct": 5.5}
        }
        result = report._build_executive_summary(report_data)
        assert any("7203" in item and "up" in item and "5.5" in item for item in result)

    def test_executive_summary_includes_technical_signal(self):
        """Test executive_summary includes TA signal."""
        report_data = {
            "symbol": "7203",
            "technical": {"signal": "BUY", "score": 85}
        }
        result = report._build_executive_summary(report_data)
        assert any("BUY" in item and "85" in item for item in result)

    def test_executive_summary_includes_financial_health(self):
        """Test executive_summary includes financial health info."""
        report_data = {
            "symbol": "7203",
            "financial_health": {
                "altman_z": {"z_score": 2.8, "zone": "safe"},
                "piotroski_f": {"score": 8, "interpretation": "strong"}
            }
        }
        result = report._build_executive_summary(report_data)
        assert any("safe" in item for item in result)

    def test_executive_summary_includes_valuation(self):
        """Test executive_summary includes valuation insight."""
        report_data = {
            "symbol": "7203",
            "valuation": {
                "dcf": {"upside_pct": 25}
            }
        }
        result = report._build_executive_summary(report_data)
        assert any("undervalued" in item or "upside" in item for item in result)

    def test_executive_summary_includes_sentiment(self):
        """Test executive_summary includes sentiment info."""
        report_data = {
            "symbol": "7203",
            "sentiment": {"average_score": 0.4}
        }
        result = report._build_executive_summary(report_data)
        assert any("Positive" in item for item in result)

    def test_executive_summary_negative_sentiment(self):
        """Test executive_summary includes negative sentiment."""
        report_data = {
            "symbol": "7203",
            "sentiment": {"average_score": -0.5}
        }
        result = report._build_executive_summary(report_data)
        assert any("Negative" in item for item in result)

    def test_executive_summary_handles_missing_sections(self):
        """Test executive_summary gracefully handles missing sections."""
        report_data = {"symbol": "7203"}
        result = report._build_executive_summary(report_data)
        assert isinstance(result, list)
        # Should not crash, just return what it can

    def test_executive_summary_includes_ml_prediction(self):
        """Test executive_summary includes ML prediction if available."""
        report_data = {
            "symbol": "7203",
            "ml_prediction": {
                "combined_signal": "STRONG_BUY",
                "combined_score": 95
            }
        }
        result = report._build_executive_summary(report_data)
        assert any("STRONG_BUY" in item and "95" in item for item in result)

    def test_executive_summary_includes_growth_insights(self):
        """Test executive_summary includes top growth insights."""
        report_data = {
            "symbol": "7203",
            "growth": {
                "summary": [
                    "Revenue growing 12% annually",
                    "EPS projected to increase 8%"
                ]
            }
        }
        result = report._build_executive_summary(report_data)
        assert any("Revenue" in item for item in result)

    def test_executive_summary_negative_return(self):
        """Test executive_summary handles negative returns."""
        report_data = {
            "symbol": "7203",
            "price_summary": {"period_return_pct": -3.2}
        }
        result = report._build_executive_summary(report_data)
        assert any("down" in item and "3.2" in item for item in result)

    def test_executive_summary_overvalued_dcf(self):
        """Test executive_summary flags overvalued DCF."""
        report_data = {
            "symbol": "7203",
            "valuation": {
                "dcf": {"upside_pct": -25}
            }
        }
        result = report._build_executive_summary(report_data)
        assert any("overvalued" in item or "downside" in item for item in result)


# ---------------------------------------------------------------------------
# Test _clean_result
# ---------------------------------------------------------------------------

class TestCleanResult:
    """Tests for _clean_result helper."""

    def test_clean_result_removes_error_sections(self):
        """Test _clean_result removes sections with _error."""
        dirty = {
            "overview": {"name": "Toyota"},
            "technical": {"_error": "Failed to fetch"},
            "financial": {"score": 85}
        }
        result = report._clean_result(dirty)
        assert "overview" in result
        assert "financial" in result
        assert "technical" not in result

    def test_clean_result_preserves_normal_sections(self):
        """Test _clean_result preserves sections without _error."""
        clean = {
            "overview": {"name": "Toyota"},
            "technical": {"signal": "BUY"},
            "financial": {"health": "good"}
        }
        result = report._clean_result(clean)
        assert result == clean

    def test_clean_result_handles_non_dict_values(self):
        """Test _clean_result handles non-dict values."""
        data = {
            "symbol": "7203",
            "timestamp": 1234567890,
            "sections": {"overview": {}}
        }
        result = report._clean_result(data)
        assert result["symbol"] == "7203"
        assert result["timestamp"] == 1234567890


# ---------------------------------------------------------------------------
# Integration-like tests
# ---------------------------------------------------------------------------

class TestReportIntegration:
    """Integration tests for report module."""

    def test_stock_report_full_workflow(self):
        """Test full stock_report workflow with multiple sections."""
        with patch("jpstock_agent.report._import_modules", return_value=_mock_modules()), \
             patch("jpstock_agent.report._import_core") as mock_core_import, \
             patch("jpstock_agent.report._import_ml", return_value=None):

            mock_core = _mock_core()
            mock_core.stock_history.return_value = _make_ohlcv_records(days=60)
            mock_core.company_overview.return_value = {
                "name": "Toyota Motor",
                "sector": "Automobiles"
            }
            mock_core_import.return_value = mock_core

            mods = _mock_modules()
            mods["ta"].ta_multi_indicator.return_value = {
                "signal": "BUY",
                "score": 72,
                "rsi": 58,
                "macd_signal": "bullish"
            }
            mods["candlestick"].ta_candlestick_latest.return_value = {
                "date": "2026-04-03",
                "patterns": ["hammer"]
            }
            mods["financial"].financial_health.return_value = {
                "altman_z": {"z_score": 2.9, "zone": "safe"},
                "piotroski_f": {"score": 7, "interpretation": "good"}
            }
            mods["financial"].financial_growth.return_value = {
                "revenue_growth": 5.2,
                "earnings_growth": 6.1
            }
            mods["financial"].financial_valuation.return_value = {
                "dcf": {"upside_pct": 18, "fair_value": 2850}
            }
            mods["financial"].financial_dividend.return_value = {
                "yield": 3.1,
                "payout_ratio": 0.35
            }
            mods["sentiment"].sentiment_news.return_value = {
                "average_score": 0.25,
                "sentiment_label": "Positive",
                "news_count": 10
            }

            with patch("jpstock_agent.report._import_modules", return_value=mods):
                result = report.stock_report("7203")

            # Verify complete report structure
            assert result["symbol"] == "7203"
            assert "price_summary" in result
            assert "technical" in result
            assert "candlestick" in result
            assert "financial_health" in result
            assert "growth" in result
            assert "valuation" in result
            assert "dividend" in result
            assert "sentiment" in result
            assert "executive_summary" in result
            assert "generation_time_ms" in result
            assert len(result["executive_summary"]) > 0
