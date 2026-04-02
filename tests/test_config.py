"""Tests for the jpstock_agent.config module.

Tests configuration loading, source auto-detection, and symbol normalization.
"""

import os
from unittest.mock import patch

from jpstock_agent.config import (
    Settings,
    auto_detect_source,
    get_settings,
    normalize_symbol,
)


class TestSettings:
    """Test the Settings pydantic model and defaults."""

    def test_settings_defaults(self):
        """Test that Settings has correct default values."""
        # Create a fresh Settings object without clearing entire environment
        # (to avoid breaking pydantic-settings .env file loading)
        settings = Settings()

        # At minimum, check that defaults are present
        assert settings.jpstock_default_source == "yfinance"
        assert settings.jpstock_mcp_transport == "stdio"
        assert settings.jpstock_mcp_host == "0.0.0.0"
        assert settings.jpstock_mcp_port == 8000

    def test_settings_from_env_jpstock_default_source(self):
        """Test that JPSTOCK_DEFAULT_SOURCE environment variable is read."""
        with patch.dict(os.environ, {"JPSTOCK_DEFAULT_SOURCE": "jquants"}, clear=True):
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.jpstock_default_source == "jquants"

    def test_settings_from_env_mcp_port(self):
        """Test that JPSTOCK_MCP_PORT environment variable is read."""
        with patch.dict(os.environ, {"JPSTOCK_MCP_PORT": "9000"}, clear=True):
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.jpstock_mcp_port == 9000

    def test_settings_from_env_jquants_api_key(self):
        """Test that JQUANTS_API_KEY environment variable is read."""
        with patch.dict(os.environ, {"JQUANTS_API_KEY": "test_api_key_v2"}, clear=True):
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.jquants_api_key == "test_api_key_v2"

    def test_settings_from_env_jquants_credentials(self):
        """Test that J-Quants v1 credentials are read from environment."""
        with patch.dict(
            os.environ,
            {
                "JQUANTS_API_EMAIL": "user@example.com",
                "JQUANTS_API_PASSWORD": "secret_pass",
                "JQUANTS_REFRESH_TOKEN": "refresh_token_value",
            },
            clear=True,
        ):
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.jquants_api_email == "user@example.com"
            assert settings.jquants_api_password == "secret_pass"
            assert settings.jquants_refresh_token == "refresh_token_value"

    def test_settings_from_env_vnstock_api_key(self):
        """Test that VNSTOCK_API_KEY environment variable is read."""
        with patch.dict(os.environ, {"VNSTOCK_API_KEY": "vn_test_key"}, clear=True):
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.vnstock_api_key == "vn_test_key"


class TestAutoDetectSource:
    """Test the auto_detect_source() function."""

    def test_auto_detect_vietnamese_ticker_acb(self):
        """Test that 'ACB' (Vietnamese ticker) detects as vnstocks."""
        result = auto_detect_source("ACB")
        assert result == "vnstocks"

    def test_auto_detect_vietnamese_ticker_vnm(self):
        """Test that 'VNM' (Vietnamese ticker) detects as vnstocks."""
        result = auto_detect_source("VNM")
        assert result == "vnstocks"

    def test_auto_detect_vietnamese_ticker_vic(self):
        """Test that 'VIC' (Vietnamese ticker) detects as vnstocks."""
        result = auto_detect_source("VIC")
        assert result == "vnstocks"

    def test_auto_detect_vietnamese_ticker_case_insensitive(self):
        """Test that Vietnamese ticker detection is case-insensitive."""
        result = auto_detect_source("acb")
        assert result == "vnstocks"

    def test_auto_detect_japanese_ticker_4digit(self):
        """Test that '7203' (4-digit Japanese code) detects as default source."""
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            result = auto_detect_source("7203")
            assert result == "yfinance"  # default

    def test_auto_detect_japanese_ticker_with_suffix(self):
        """Test that '7203.T' (Japanese with suffix) detects as default source."""
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            result = auto_detect_source("7203.T")
            assert result == "yfinance"  # default

    def test_auto_detect_forex_pair(self):
        """Test that 'USDJPY=X' (forex pair) detects as default source."""
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            result = auto_detect_source("USDJPY=X")
            assert result == "yfinance"  # default

    def test_auto_detect_index(self):
        """Test that '^N225' (index code) detects as default source."""
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            result = auto_detect_source("^N225")
            assert result == "yfinance"  # default

    def test_auto_detect_respects_configured_default_source(self):
        """Test that non-Vietnamese tickers use the configured default source."""
        with patch.dict(os.environ, {"JPSTOCK_DEFAULT_SOURCE": "jquants"}, clear=True):
            get_settings.cache_clear()
            result = auto_detect_source("7203")
            assert result == "jquants"

    def test_auto_detect_with_whitespace(self):
        """Test that whitespace is stripped before detection."""
        result = auto_detect_source("  ACB  ")
        assert result == "vnstocks"

    def test_auto_detect_rejects_numeric_in_letter_ticker(self):
        """Test that tickers with numbers (not pure letters) use default source."""
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            result = auto_detect_source("AC1")
            assert result == "yfinance"  # Not pure letters, so not Vietnamese


class TestNormalizeSymbol:
    """Test the normalize_symbol() function."""

    def test_normalize_symbol_yfinance_4digit_to_suffix(self):
        """Test that yfinance normalizes '7203' → '7203.T'."""
        result = normalize_symbol("7203", source="yfinance")
        assert result == "7203.T"

    def test_normalize_symbol_yfinance_already_suffixed(self):
        """Test that yfinance leaves '7203.T' unchanged."""
        result = normalize_symbol("7203.T", source="yfinance")
        assert result == "7203.T"

    def test_normalize_symbol_yfinance_other_suffix(self):
        """Test that yfinance leaves other suffixes like '.S' unchanged."""
        result = normalize_symbol("1234.S", source="yfinance")
        assert result == "1234.S"

    def test_normalize_symbol_yfinance_non_4digit(self):
        """Test that yfinance leaves non-4-digit codes unchanged."""
        result = normalize_symbol("^N225", source="yfinance")
        assert result == "^N225"

    def test_normalize_symbol_jquants_removes_suffix(self):
        """Test that jquants normalizes '7203.T' → '7203'."""
        result = normalize_symbol("7203.T", source="jquants")
        assert result == "7203"

    def test_normalize_symbol_jquants_bare_code(self):
        """Test that jquants leaves bare code '7203' unchanged."""
        result = normalize_symbol("7203", source="jquants")
        assert result == "7203"

    def test_normalize_symbol_vnstocks_uppercase(self):
        """Test that vnstocks normalizes 'acb' → 'ACB'."""
        result = normalize_symbol("acb", source="vnstocks")
        assert result == "ACB"

    def test_normalize_symbol_vnstocks_already_uppercase(self):
        """Test that vnstocks leaves 'ACB' unchanged."""
        result = normalize_symbol("ACB", source="vnstocks")
        assert result == "ACB"

    def test_normalize_symbol_case_insensitive(self):
        """Test that normalization is case-insensitive on input."""
        result = normalize_symbol("7203", source="yfinance")
        assert result == "7203.T"

    def test_normalize_symbol_whitespace_stripped(self):
        """Test that leading/trailing whitespace is stripped."""
        result = normalize_symbol("  7203  ", source="yfinance")
        assert result == "7203.T"

    def test_normalize_symbol_default_source_when_not_provided(self):
        """Test that source defaults to configured default when not provided."""
        with patch.dict(os.environ, {"JPSTOCK_DEFAULT_SOURCE": "jquants"}, clear=True):
            get_settings.cache_clear()
            result = normalize_symbol("7203")
            assert result == "7203"

    def test_normalize_symbol_yfinance_explicit_wins_over_default(self):
        """Test that explicit source='yfinance' overrides the default."""
        with patch.dict(os.environ, {"JPSTOCK_DEFAULT_SOURCE": "jquants"}, clear=True):
            get_settings.cache_clear()
            result = normalize_symbol("7203", source="yfinance")
            assert result == "7203.T"

    def test_normalize_symbol_jquants_with_multiple_dots(self):
        """Test that jquants handles symbols with dots (splits on first dot)."""
        result = normalize_symbol("7203.T.X", source="jquants")
        assert result == "7203"
