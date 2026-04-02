# JPStock Agent 🇯🇵📈

**MCP server & CLI for Japanese stock market data** – powered by [yfinance](https://github.com/ranaroussi/yfinance) and [J-Quants API](https://jpx-jquants.com/).

Seamlessly integrate Japanese stock market data with AI assistants like **Claude** and **GPT**, or use it as a standalone CLI tool.

## Features

- **22 MCP tools** for AI integration (Claude Desktop, Cursor, etc.)
- **22 CLI commands** for terminal usage
- **Dual data sources**: yfinance (free, no auth) + J-Quants (official JPX data)
- Stock prices (OHLCV, intraday, order book)
- Company info (overview, shareholders, officers, news, events)
- Financial statements (balance sheet, income, cash flow, ratios)
- Market listings (all symbols, by sector, by market segment)
- Forex, crypto, and world index data
- Docker-ready deployment

## Quick Start

### Install

```bash
pip install jpstock-agent
```

Or from source:

```bash
git clone https://github.com/fvn-manhtd/jp-stock-agent.git
cd jpstock-agent
pip install -e ".[dev]"
```

### CLI Usage

```bash
# Stock price history (Toyota)
jpstock-agent history 7203

# Company overview (Sony)
jpstock-agent overview 6758

# Financial ratios (SoftBank)
jpstock-agent ratio 9984

# Multiple stock prices
jpstock-agent board 7203,6758,9984

# Forex rate
jpstock-agent fx USDJPY=X

# Nikkei 225 index
jpstock-agent index ^N225

# Output as JSON
jpstock-agent history 7203 --format json

# Use J-Quants as data source
jpstock-agent overview 7203 --source jquants
```

### MCP Server

```bash
# stdio mode (for Claude Desktop / Cursor)
jpstock-agent serve

# SSE mode (for web clients)
jpstock-agent serve --transport sse --port 8000

# HTTP mode
jpstock-agent serve --transport http --port 8000
```

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jpstock": {
      "command": "jpstock-agent",
      "args": ["serve"]
    }
  }
}
```

### Cursor Configuration

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "jpstock": {
      "command": "jpstock-agent",
      "args": ["serve"],
      "transport": "stdio"
    }
  }
}
```

## Data Sources

### yfinance (Default)

Free to use, no authentication required. Covers:
- Stock prices & history (via Yahoo Finance Japan)
- Company info, financials, news
- Forex, crypto, world indices

Japanese tickers use the `.T` suffix (e.g., `7203.T` for Toyota). The agent auto-appends this.

### J-Quants API

Official data from Japan Exchange Group (JPX). Requires free registration at [jpx-jquants.com](https://jpx-jquants.com/).

**Recommended (API Key v2):**

```bash
export JQUANTS_API_KEY="your_api_key"
export JPSTOCK_DEFAULT_SOURCE="jquants"
```

**Alternative (email/password v1):**

```bash
export JQUANTS_API_EMAIL="your_email@example.com"
export JQUANTS_API_PASSWORD="your_password"
export JPSTOCK_DEFAULT_SOURCE="jquants"
```

Auth priority: `JQUANTS_API_KEY` (v2) > `JQUANTS_REFRESH_TOKEN` (v1) > email/password (v1)

Provides additional data:
- Complete TSE listings
- Sector/market segment classifications
- Trading calendar
- Official financial statements

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JPSTOCK_DEFAULT_SOURCE` | `yfinance` | Default data source |
| `JPSTOCK_MCP_TRANSPORT` | `stdio` | Transport: stdio, sse, http |
| `JPSTOCK_MCP_HOST` | `0.0.0.0` | Server bind address |
| `JPSTOCK_MCP_PORT` | `8000` | Server port |
| `JQUANTS_API_KEY` | | J-Quants API key (v2, highest priority) |
| `JQUANTS_API_EMAIL` | | J-Quants login email (v1) |
| `JQUANTS_API_PASSWORD` | | J-Quants login password (v1) |
| `JQUANTS_REFRESH_TOKEN` | | J-Quants refresh token (v1 alternative) |

## Docker

```bash
docker-compose up
```

The server starts on port 8000 with SSE transport by default.

## Common Japanese Ticker Codes

| Code | Company |
|------|---------|
| 7203 | Toyota Motor |
| 6758 | Sony Group |
| 9984 | SoftBank Group |
| 6861 | Keyence |
| 8306 | Mitsubishi UFJ Financial |
| 9432 | NTT |
| 6501 | Hitachi |
| 7741 | HOYA |
| 6098 | Recruit Holdings |
| 4063 | Shin-Etsu Chemical |

## License

MIT
