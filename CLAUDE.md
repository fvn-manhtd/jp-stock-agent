# JPStock Agent – Developer Guide

## Overview
MCP server & CLI for Japanese and Vietnamese stock market data. Wraps **yfinance**, **J-Quants API**, and **vnstock** to provide AI-friendly access to TSE/JPX and HOSE/HNX/UPCOM market data.

## Project Structure
```
src/jpstock_agent/
  __init__.py     – version
  config.py       – env-based settings (pydantic-settings)
  core.py         – all data functions (returns list[dict] | dict)
  server.py       – FastMCP tool definitions (24 tools)
  cli.py          – Click CLI commands (22 commands + serve)
```

## Key Conventions
- Every core function returns `list[dict]` on success or `{"error": str}` on failure
- Symbols auto-normalize: "7203" → "7203.T" (yfinance) or "7203" (jquants); "ACB" → "ACB" (vnstocks)
- Default lookback: 90 days
- `_safe_call()` wraps all external API calls for error handling
- Server tools return JSON strings; CLI outputs tables or JSON

## Commands
```bash
pip install -e ".[dev]"         # Install in dev mode
ruff check src/                 # Lint
pytest                          # Test
jpstock-agent history 7203      # CLI usage
jpstock-agent serve             # Start MCP server (stdio)
```

## Data Sources
- **yfinance**: Free, no auth needed. Good for prices, company info, financials, forex, crypto.
- **jquants**: Official JPX data. Needs `JQUANTS_API_EMAIL` + `JQUANTS_API_PASSWORD`. Better for listings, sectors, calendar.
- **vnstocks**: Vietnamese market data (HOSE/HNX/UPCOM) via `vnstock` library. Free, no auth needed for guest tier (20 req/min). Use 3-letter codes like "ACB", "VNM", "VIC". Internal data source defaults to "VCI".

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `JPSTOCK_DEFAULT_SOURCE` | `yfinance` | Default data source |
| `JPSTOCK_MCP_TRANSPORT` | `stdio` | MCP transport: stdio, sse, http |
| `JPSTOCK_MCP_HOST` | `0.0.0.0` | Server host |
| `JPSTOCK_MCP_PORT` | `8000` | Server port |
| `JQUANTS_API_KEY` | *(empty)* | J-Quants API key (v2, highest priority) |
| `JQUANTS_API_EMAIL` | *(empty)* | J-Quants login email (v1) |
| `JQUANTS_API_PASSWORD` | *(empty)* | J-Quants login password (v1) |
| `JQUANTS_REFRESH_TOKEN` | *(empty)* | J-Quants refresh token (v1 alternative) |
| `VNSTOCK_API_KEY` | *(empty)* | vnstock API key (optional; raises rate limit from 20→60 req/min) |

## J-Quants Auth Priority
`JQUANTS_API_KEY` (v2 ClientV2) > `JQUANTS_REFRESH_TOKEN` (v1 Client) > email/password (v1 Client)

## Auto Source Detection
When `source` is not specified, functions auto-detect based on symbol format:
- **2–5 uppercase letters only** (e.g. `ACB`, `VNM`, `VIC`) → `vnstocks`
- **Anything else** (4-digit codes, `.T` suffix, forex pairs, indices) → `jpstock_default_source`

This means `stock_history("ACB")` automatically uses vnstocks, while `stock_history("7203")` uses yfinance.
