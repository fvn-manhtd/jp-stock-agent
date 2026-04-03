# JPStock Agent

**MCP server & CLI for Japanese and Vietnamese stock market data** — 111 AI tools powered by [yfinance](https://github.com/ranaroussi/yfinance), [J-Quants API](https://jpx-jquants.com/), and [vnstock](https://vnstocks.com/).

Integrate real-time stock data, technical analysis, backtesting, portfolio optimization, and ML predictions directly into **Claude**, **Cursor**, or any MCP-compatible AI assistant.

## Highlights

- **111 MCP tools** + **99 CLI commands** covering quotes, TA, backtesting, ML, options, portfolio, alerts, and more
- **3 markets**: Japan (TSE/JPX), Vietnam (HOSE/HNX/UPCOM), plus forex, crypto, world indices
- **3 data sources**: yfinance (free), J-Quants (official JPX), vnstock (Vietnamese)
- **24 technical indicators**: RSI, MACD, Bollinger Bands, Ichimoku, Supertrend, VWAP, Fibonacci, etc.
- **20 candlestick patterns**: Hammer, Engulfing, Morning Star, Three White Soldiers, etc.
- **12 backtesting strategies** with Monte Carlo simulation, walk-forward analysis, cost model
- **Portfolio optimization**: Monte Carlo, efficient frontier, VaR/CVaR risk metrics
- **ML predictions**: Random Forest / Gradient Boosting with 30+ features
- **Options analysis**: Black-Scholes Greeks, IV surface, max pain, unusual activity
- **Fundamental analysis**: Altman Z-score, Piotroski F-score, DCF valuation, DuPont decomposition
- **Custom strategy builder**: 27 composable conditions with AND/OR logic
- **Auth + rate limiting**: tier-based API keys (free/pro/enterprise) with ASGI middleware
- **Usage analytics**: persistent SQLite tracking, revenue estimation

## Quick Start

### Install

```bash
pip install jpstock-agent
```

Or with ML support:

```bash
pip install "jpstock-agent[ml]"
```

### CLI Usage

```bash
# Stock price history (Toyota)
jpstock-agent history 7203

# Vietnamese stock (ACB bank)
jpstock-agent history ACB

# Technical analysis
jpstock-agent ta-rsi 7203
jpstock-agent ta-macd 6758
jpstock-agent ta-analysis 7203      # Full multi-indicator BUY/SELL/HOLD

# Screen stocks
jpstock-agent ta-screen 7203,6758,9984 --strategy oversold

# Backtesting
jpstock-agent backtest 7203 --strategy sma_crossover
jpstock-agent backtest-compare 7203   # Compare all 12 strategies

# Quick report (price + TA + key ratios)
jpstock-agent report-quick 7203

# Full comprehensive report
jpstock-agent report 7203

# Portfolio analysis
jpstock-agent portfolio-analyze 7203,6758,9984
jpstock-agent portfolio-optimize 7203,6758,9984

# Market regime
jpstock-agent market-regime

# Output as JSON
jpstock-agent history 7203 --format json
```

### MCP Server

```bash
# stdio mode (Claude Desktop / Cursor)
jpstock-agent serve

# SSE mode (web clients)
jpstock-agent serve --transport sse --port 8000

# HTTP mode
jpstock-agent serve --transport http --port 8000
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jpstock": {
      "command": "jpstock-agent",
      "args": ["serve"],
      "env": {
        "JPSTOCK_DEFAULT_SOURCE": "yfinance"
      }
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
      "transport": "stdio",
      "env": {
        "JPSTOCK_DEFAULT_SOURCE": "yfinance"
      }
    }
  }
}
```

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| Quotes | 8 | OHLCV history, intraday, batch, order book |
| Company | 5 | Overview, news, events, officers, shareholders |
| Financials | 9 | Balance sheet, income, cash flow, ratios, health, growth, valuation, dividends, DuPont |
| Technical Analysis | 24 | RSI, MACD, Bollinger, Ichimoku, Supertrend, VWAP, Fibonacci, support/resistance, multi-timeframe |
| Candlestick | 3 | Scan, latest, screen (20 patterns) |
| Backtesting | 7 | Strategy, compare, optimize, walk-forward, Monte Carlo, advanced metrics, realistic |
| Portfolio | 4 | Analyze, optimize, risk, correlation |
| Sentiment | 4 | News, market, combined TA+sentiment, screen |
| ML | 4 | Predict, feature importance, signal, batch predict |
| Options | 6 | Chain, Greeks, IV surface, unusual activity, put/call ratio, max pain |
| Reports | 3 | Comprehensive, quick, comparison |
| Alerts | 6 | Check, price, TA, fundamental, watchlist, list conditions |
| Market | 5 | Sector performance, breadth, top movers, regime, heatmap |
| Strategy | 3 | Evaluate, screen, list conditions |
| Listings | 5 | All symbols, sectors, by market, by sector, VN listing |
| Forex/Crypto | 3 | FX history, crypto history, world index |
| Auth/Usage | 6 | Auth usage, tiers, daily stats, key stats, tool stats, revenue |

## Data Sources

### yfinance (Default)

Free, no authentication required. Auto-detects Japanese tickers (4-digit codes like `7203` become `7203.T`).

### J-Quants API

Official JPX data. Register at [jpx-jquants.com](https://jpx-jquants.com/).

```bash
export JQUANTS_API_KEY="your_api_key"
export JPSTOCK_DEFAULT_SOURCE="jquants"
```

### vnstock (Vietnamese Markets)

Vietnamese market data via [vnstocks.com](https://vnstocks.com/). Auto-detected for 2-5 uppercase letter symbols (`ACB`, `VNM`, `VIC`).

```bash
export VNSTOCK_API_KEY="your_key"  # Optional, increases rate limit 20→60 req/min
```

## Authentication & Rate Limiting

For commercial deployment, enable API key authentication:

```bash
export JPSTOCK_AUTH_ENABLED=true
export JPSTOCK_MASTER_KEY="your_admin_key"  # Optional, bypasses all limits
```

### API Key Management

```bash
# Generate keys
jpstock-agent key-generate --tier pro --owner "user@example.com"

# Validate
jpstock-agent key-validate jpsk_pro_xxxx

# List all keys
jpstock-agent key-list

# View tiers
jpstock-agent auth-tiers
```

### Tiers

| Tier | Daily Limit | Access | Price |
|------|------------|--------|-------|
| Free | 50 calls | Basic tools (quotes, history, FX, crypto) | $0 |
| Pro | 1,000 calls | All 111 tools | $9/mo |
| Enterprise | 5,000 calls | All tools + priority support | $19/mo |

### Usage Analytics

```bash
jpstock-agent usage-daily              # Today's usage summary
jpstock-agent usage-key jpsk_pro_xxxx  # Per-key breakdown
jpstock-agent usage-tools              # Per-tool stats
jpstock-agent usage-revenue            # MRR/ARR estimate
jpstock-agent usage-cleanup            # Prune old records
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JPSTOCK_DEFAULT_SOURCE` | `yfinance` | Default data source |
| `JPSTOCK_MCP_TRANSPORT` | `stdio` | Transport: stdio, sse, http |
| `JPSTOCK_MCP_HOST` | `0.0.0.0` | Server host |
| `JPSTOCK_MCP_PORT` | `8000` | Server port |
| `JQUANTS_API_KEY` | | J-Quants API key (v2) |
| `JQUANTS_API_EMAIL` | | J-Quants email (v1) |
| `JQUANTS_API_PASSWORD` | | J-Quants password (v1) |
| `JQUANTS_REFRESH_TOKEN` | | J-Quants refresh token (v1) |
| `VNSTOCK_API_KEY` | | vnstock API key (optional) |
| `JPSTOCK_AUTH_ENABLED` | `false` | Require API keys |
| `JPSTOCK_AUTH_KEY_FILE` | `~/.jpstock/keys.json` | Key store path |
| `JPSTOCK_MASTER_KEY` | | Admin master key |
| `JPSTOCK_RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `JPSTOCK_BURST_PER_MINUTE` | `30` | Max calls/key/minute |

## Docker

```bash
# Quick start
docker-compose up

# With auth enabled
docker-compose -f docker-compose.yml -f docker-compose.auth.yml up
```

The server starts on port 8000 with SSE transport.

## Development

```bash
git clone https://github.com/fvn-manhtd/jp-stock-agent.git
cd jp-stock-agent
pip install -e ".[dev]"

# Lint
ruff check src/

# Test (858 tests)
pytest

# Test with coverage
pytest --cov=jpstock_agent --cov-report=term-missing
```

## Common Ticker Codes

### Japan (TSE)

| Code | Company |
|------|---------|
| 7203 | Toyota Motor |
| 6758 | Sony Group |
| 9984 | SoftBank Group |
| 6861 | Keyence |
| 8306 | Mitsubishi UFJ Financial |
| 9432 | NTT |
| 6501 | Hitachi |

### Vietnam (HOSE/HNX)

| Code | Company |
|------|---------|
| ACB | Asia Commercial Bank |
| VNM | Vinamilk |
| VIC | Vingroup |
| FPT | FPT Corporation |
| HPG | Hoa Phat Group |

## License

MIT
