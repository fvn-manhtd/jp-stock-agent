# Changelog

## [0.2.0] - 2026-04-03

### Added
- **Authentication & API Key Management** (`auth.py`)
  - Tier-based system: Free (50 calls/day), Pro ($9/mo, 1000/day), Enterprise ($19/mo, 5000/day)
  - API key generation with SHA-256 hashing, JSON file store
  - Tool-level access control (free tier = basic tools only)
  - CLI: `key-generate`, `key-list`, `key-revoke`, `key-validate`, `auth-tiers`

- **Rate Limiting** (`ratelimit.py`)
  - Sliding-window rate limiter with daily + burst (per-minute) limits
  - Per-key custom quotas, thread-safe
  - `peek()`, `usage()`, `reset()` utilities

- **ASGI Middleware** (`middleware.py`)
  - Auth + rate limiting for HTTP/SSE transports
  - Bearer token / X-API-Key header support
  - 401/429 error responses, master key bypass
  - Usage tracking integration

- **Usage Tracking & Analytics** (`usage.py`)
  - SQLite-backed persistent usage recorder (WAL mode)
  - `daily_summary`, `key_usage`, `tool_stats`, `revenue_estimate`, `cleanup`
  - CLI: `usage-daily`, `usage-key`, `usage-tools`, `usage-revenue`, `usage-cleanup`

- **Report Generator** (`report.py`)
  - `stock_report` – comprehensive report (all modules in parallel)
  - `stock_report_quick` – lightweight price + TA + ratios
  - `stock_report_compare` – multi-stock comparison with rankings

- **Alert & Watchlist** (`alert.py`)
  - 16 TA-based alert conditions (RSI, MACD, BB, golden/death cross, etc.)
  - Price and fundamental alerts
  - Parallel watchlist monitoring
  - Stateless design for scheduled tasks

- **Market Analysis** (`market.py`)
  - Sector performance comparison, market breadth (A/D ratio)
  - Top movers (gainers/losers), regime detection (bull/bear/sideways)
  - Sector heatmap data

- **Custom Strategy Builder** (`strategy.py`)
  - 27 composable conditions across 3 categories (price, TA, fundamental)
  - AND/OR logic for combining conditions
  - Parallel multi-symbol screening with per-symbol data cache

- Docker setup with auth support, healthcheck, persistent volumes
- `.env.example` with all configuration options
- LICENSE file (MIT)

### Changed
- Version bumped to 0.2.0
- `pyproject.toml` updated with full marketplace metadata, 21 keywords, 11 classifiers
- `README.md` completely rewritten for marketplace listing
- `config.py` expanded with 5 new auth/rate-limit settings
- Tool count: 48 → 106, CLI commands: 40 → 99

## [0.1.0] - 2026-03-01

### Added
- Initial release
- Core data module with yfinance, J-Quants, vnstock support
- 24 technical analysis indicators
- 20 candlestick pattern detection
- 12 backtesting strategies with Monte Carlo simulation
- Portfolio optimization with efficient frontier
- Sentiment analysis (news scoring)
- ML signal generation (Random Forest, Gradient Boosting)
- Options analysis (Black-Scholes Greeks, IV surface, max pain)
- Fundamental analysis (Altman Z, Piotroski F, DCF, DuPont)
- FastMCP server (stdio/SSE/HTTP)
- Click CLI with 40+ commands
