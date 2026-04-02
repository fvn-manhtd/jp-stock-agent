"""Configuration management for jpstock-agent.

Environment variables
---------------------
JQUANTS_API_KEY      – J-Quants API key (v2, highest priority)
JQUANTS_API_EMAIL    – J-Quants login email (v1 auth)
JQUANTS_API_PASSWORD – J-Quants login password (v1 auth)
JQUANTS_REFRESH_TOKEN – J-Quants refresh token (v1 alternative auth)
VNSTOCK_API_KEY      – vnstock API key (optional; raises rate limit from 20 to 60 req/min)
JPSTOCK_DEFAULT_SOURCE – "yfinance" (default) or "jquants"
JPSTOCK_MCP_TRANSPORT  – "stdio" (default), "sse", or "http"
JPSTOCK_MCP_HOST       – server bind address (default 0.0.0.0)
JPSTOCK_MCP_PORT       – server port (default 8000)
"""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic_settings import BaseSettings

# Vietnamese stock ticker pattern: 2-5 uppercase ASCII letters, no digits
_VN_TICKER_RE = re.compile(r"^[A-Z]{2,5}$")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # J-Quants credentials
    jquants_api_key: str = ""        # v2 API key (highest priority)
    jquants_api_email: str = ""      # v1 email/password auth
    jquants_api_password: str = ""
    jquants_refresh_token: str = ""  # v1 refresh token auth

    # vnstock credential (optional)
    vnstock_api_key: str = ""        # Community/Sponsor tier key

    # Data source: "yfinance" or "jquants" (used for non-Vietnamese tickers)
    jpstock_default_source: str = "yfinance"

    # MCP server
    jpstock_mcp_transport: str = "stdio"
    jpstock_mcp_host: str = "0.0.0.0"
    jpstock_mcp_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


def auto_detect_source(symbol: str) -> str:
    """Auto-detect the appropriate data source from the symbol format.

    Rules (applied to the cleaned, upper-cased symbol):
    - 2-5 ASCII letters only  →  "vnstocks"  (Vietnamese tickers: ACB, VNM, VIC…)
    - Everything else          →  configured ``jpstock_default_source``
                                   (Japanese 4-digit codes, forex pairs, indices, …)
    """
    clean = symbol.strip().upper()
    if _VN_TICKER_RE.match(clean):
        return "vnstocks"
    return get_settings().jpstock_default_source


def get_jquants_client():
    """Create and return a J-Quants API client.

    Auth priority: JQUANTS_API_KEY (v2) > JQUANTS_REFRESH_TOKEN (v1) > email/password (v1).
    Returns None if no credentials are configured.
    """
    s = get_settings()
    has_v1_creds = (s.jquants_api_email and s.jquants_api_password) or s.jquants_refresh_token
    if not s.jquants_api_key and not has_v1_creds:
        return None

    try:
        import jquantsapi

        if s.jquants_api_key:
            return jquantsapi.ClientV2(api_key=s.jquants_api_key)
        elif s.jquants_refresh_token:
            return jquantsapi.Client(refresh_token=s.jquants_refresh_token)
        else:
            return jquantsapi.Client(
                mail_address=s.jquants_api_email,
                password=s.jquants_api_password,
            )
    except Exception as e:
        import sys
        print(f"[jpstock-agent] Failed to initialize J-Quants client: {e}", file=sys.stderr)
        return None


def normalize_symbol(symbol: str, source: str | None = None) -> str:
    """Normalize a stock symbol for the given source.

    - yfinance: append '.T' for TSE if not already suffixed
    - jquants: use bare 4-digit code
    - vnstocks: uppercase ticker (e.g. "ACB", "VNM") – no transformation needed
    """
    source = source or get_settings().jpstock_default_source
    symbol = symbol.strip().upper()

    if source == "yfinance":
        # Common suffixes: .T (TSE Prime/Standard/Growth), .S (Sapporo), .N (Nagoya), .F (Fukuoka)
        if symbol.isdigit() and len(symbol) == 4:
            return f"{symbol}.T"
        return symbol
    elif source == "jquants":
        # J-Quants uses bare ticker codes (4 digits) or with trailing zero e.g. "72030"
        if "." in symbol:
            symbol = symbol.split(".")[0]
        return symbol
    elif source == "vnstocks":
        # Vietnamese tickers are uppercase alpha codes (e.g. "ACB", "VNM", "VIC")
        return symbol
    return symbol
