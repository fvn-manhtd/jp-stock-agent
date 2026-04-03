# JPStock Agent – Developer Guide

## Overview
MCP server & CLI for Japanese and Vietnamese stock market data. Wraps **yfinance**, **J-Quants API**, and **vnstock** to provide AI-friendly access to TSE/JPX and HOSE/HNX/UPCOM market data.

## Project Structure
```
src/jpstock_agent/
  __init__.py     – version
  config.py       – env-based settings (pydantic-settings)
  logging.py      – structured JSON logging, LogTimer, retry/cache observability
  core.py         – all data functions with retry + caching (returns list[dict] | dict)
  ta.py           – technical analysis module (24 TA functions + 29 screening strategies)
  candlestick.py  – candlestick pattern detection (20 patterns)
  backtest.py     – backtesting engine (12 strategies, optimizer, walk-forward, Monte Carlo, cost model, position sizing)
  portfolio.py    – portfolio optimization (Monte Carlo, risk analysis, correlation, parallel fetch)
  sentiment.py    – sentiment analysis (news scoring, combined TA+sentiment signals)
  financial.py    – fundamental analysis (Altman Z, Piotroski F, DCF, growth trends, peer compare, dividends, DuPont)
  ml.py           – ML signal generation (Random Forest, Gradient Boosting, feature importance, combined ML+TA)
  options.py      – options & derivatives (chain, Black-Scholes Greeks, IV surface, unusual activity, max pain)
  report.py       – unified stock reports (comprehensive, quick, comparison – aggregates all modules)
  alert.py        – alert & watchlist (16 TA conditions, price/fundamental alerts, watchlist monitoring)
  market.py       – sector & market analysis (breadth, regime detection, top movers, heatmap)
  strategy.py     – custom strategy builder (27 composable conditions, AND/OR logic, screening)
  auth.py         – authentication & API key management (tier system, key store, access control)
  ratelimit.py    – sliding-window rate limiter (daily + burst limits, per-key quotas, thread-safe)
  middleware.py    – ASGI middleware for HTTP/SSE auth & rate limiting
  usage.py        – persistent usage tracking & analytics (SQLite, revenue estimation)
  server.py       – FastMCP tool definitions (111 tools)
  cli.py          – Click CLI commands (99 commands + serve)
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
pip install -e ".[ml]"          # Install with ML dependencies (scikit-learn)
ruff check src/                 # Lint
pytest                          # Test
jpstock-agent history 7203      # CLI usage
jpstock-agent serve             # Start MCP server (stdio)
```

## Data Sources
- **yfinance**: Free, no auth needed. Good for prices, company info, financials, forex, crypto, options.
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
| `JPSTOCK_AUTH_ENABLED` | `false` | Set `true` to require API keys for all tool calls |
| `JPSTOCK_AUTH_KEY_FILE` | *(empty)* | Custom key store path (default `~/.jpstock/keys.json`) |
| `JPSTOCK_MASTER_KEY` | *(empty)* | Master admin key (bypasses auth + rate limits) |
| `JPSTOCK_RATE_LIMIT_ENABLED` | `true` | Enable rate limiting (works even when auth is off) |
| `JPSTOCK_BURST_PER_MINUTE` | `30` | Max calls per key per minute (burst cap) |

## J-Quants Auth Priority
`JQUANTS_API_KEY` (v2 ClientV2) > `JQUANTS_REFRESH_TOKEN` (v1 Client) > email/password (v1 Client)

## Auto Source Detection
When `source` is not specified, functions auto-detect based on symbol format:
- **2–5 uppercase letters only** (e.g. `ACB`, `VNM`, `VIC`) → `vnstocks`
- **Anything else** (4-digit codes, `.T` suffix, forex pairs, indices) → `jpstock_default_source`

This means `stock_history("ACB")` automatically uses vnstocks, while `stock_history("7203")` uses yfinance.

## Technical Analysis Module (ta.py)
24 TA functions organized into 7 categories:

### Trend Indicators
`ta_sma`, `ta_ema`, `ta_ichimoku`, `ta_supertrend`, `ta_parabolic_sar`

### Momentum Indicators
`ta_rsi`, `ta_macd`, `ta_stochastic`, `ta_williams_r`, `ta_cci`, `ta_roc`

### Volatility Indicators
`ta_bbands`, `ta_atr`, `ta_keltner`, `ta_donchian`

### Volume Indicators
`ta_obv`, `ta_vwap`, `ta_mfi`, `ta_ad`

### Composite Analysis
`ta_fibonacci`, `ta_support_resistance`, `ta_multi_indicator` (generates BUY/SELL/HOLD signals with -100 to +100 score)

### Screening
`ta_screen` – scan multiple stocks with 29 strategies:
- **Original 10**: `oversold`, `overbought`, `macd_bullish`, `macd_bearish`, `bb_squeeze`, `golden_cross`, `death_cross`, `volume_spike`, `trend_up`, `trend_down`
- **New 19**: `rsi_divergence_bull`, `rsi_divergence_bear`, `gap_up`, `gap_down`, `inside_bar`, `outside_bar`, `new_high_52w`, `new_low_52w`, `breakout_up`, `breakout_down`, `ema_bullish_cross`, `ema_bearish_cross`, `mfi_oversold`, `mfi_overbought`, `bb_breakout_up`, `bb_breakout_down`, `supertrend_bullish`, `supertrend_bearish`, `high_volume_gain`

### Multi-Timeframe
`ta_multi_timeframe` – analyze across daily/weekly/monthly timeframes

## Candlestick Pattern Detection (candlestick.py)
20 Japanese candlestick patterns in 5 categories:

### Single Candle - Bullish
Hammer, Inverted Hammer, Dragonfly Doji, Bullish Marubozu

### Single Candle - Bearish
Hanging Man, Shooting Star, Gravestone Doji, Bearish Marubozu

### Single Candle - Neutral
Doji, Spinning Top, High Wave

### Two Candle Patterns
Bullish Engulfing, Bearish Engulfing, Tweezer Top, Tweezer Bottom, Piercing Line

### Three Candle Patterns
Morning Star, Evening Star, Three White Soldiers, Three Black Crows

### Functions
- `ta_candlestick_scan` – scan all patterns in recent data
- `ta_candlestick_latest` – patterns on most recent day only
- `ta_candlestick_screen` – screen multiple stocks for patterns

## Backtesting Engine (backtest.py)
12 trading strategies with full performance analytics:

### Strategies
`sma_crossover`, `ema_crossover`, `rsi_reversal`, `macd_crossover`, `bollinger_bounce`, `supertrend`, `ichimoku_cloud`, `golden_cross`, `mean_reversion`, `momentum`, `breakout`, `vwap_strategy`

### Functions
- `backtest_strategy` – run single strategy backtest (returns: total return, Sharpe, drawdown, win rate, alpha)
- `backtest_compare` – compare all 12 strategies side by side
- `backtest_optimize` – parameter optimization (test ranges of values)
- `backtest_walk_forward` – rolling window consistency analysis
- `backtest_monte_carlo` – Monte Carlo simulation (probability distributions, confidence intervals)
- `backtest_advanced_metrics` – advanced metrics (Sortino, Calmar, profit factor, expectancy, risk/reward)
- `backtest_realistic` – realistic backtesting with transaction costs + position sizing

### Cost Model & Position Sizing
- `CostModel`: commission_pct, slippage_pct, spread_pct, min_commission
- Market presets: `JP_MARKET_COSTS`, `VN_MARKET_COSTS`, `NO_COSTS`
- 5 position sizing strategies: `full`, `kelly` (half-Kelly), `atr`, `max_loss`, `fixed_fraction`

## Portfolio Optimization (portfolio.py)
Modern portfolio theory with Monte Carlo simulation:

### Functions
- `portfolio_analyze` – per-stock returns, volatility, Sharpe, correlation matrix
- `portfolio_optimize` – Monte Carlo optimization (max Sharpe, min volatility, efficient frontier)
- `portfolio_risk` – risk metrics (VaR 95%, CVaR, Sortino, max drawdown, beta)
- `portfolio_correlation` – correlation & covariance matrices, most/least correlated pairs

### Performance
- Parallel data fetching via ThreadPoolExecutor for multi-symbol portfolios

## Sentiment Analysis (sentiment.py)
News-based sentiment scoring with English + Japanese keyword matching:

### Functions
- `sentiment_news` – analyze news headlines for a stock (-1 to +1 score)
- `sentiment_market` – batch sentiment for multiple stocks
- `sentiment_combined` – combined TA (70%) + sentiment (30%) signal
- `sentiment_screen` – screen stocks by sentiment threshold

## Fundamental Financial Analysis (financial.py)
Comprehensive financial analysis computed from raw statements:

### Functions
- `financial_health` – Altman Z-score (bankruptcy prediction), Piotroski F-score (0-9), liquidity ratios, leverage metrics, cash conversion cycle
- `financial_growth` – YoY revenue/earnings growth, margin trends (gross/operating/net), free cash flow trends
- `financial_valuation` – DCF valuation (projected FCF, terminal value, intrinsic value), relative valuation (P/E, EV/EBITDA, P/B)
- `financial_peer_compare` – side-by-side multi-stock comparison with parallel fetching, ranked by ROE/ROA/margin
- `financial_dividend` – dividend yield, payout ratio, CAGR, CF coverage, sustainability assessment
- `financial_ratios_calc` – compute ratios from raw statements + DuPont decomposition (NPM × AT × EM = ROE)

## ML Signal Generation (ml.py)
Machine learning predictions using scikit-learn (optional dependency):

### Functions
- `ml_predict` – probability of price increase using Random Forest or Gradient Boosting (30+ TA features)
- `ml_feature_importance` – rank TA indicators by predictive power with category breakdown
- `ml_signal` – combined ML + TA signal (configurable weight blend, default 50/50)
- `ml_batch_predict` – batch predictions for multiple symbols sorted by probability

### Features (30+)
Trend (SMA/EMA ratios), Momentum (RSI, MACD, Stochastic, ROC, Williams %R), Volatility (ATR, Bollinger %), Volume (ratios, OBV), Price Action (returns, gaps)

## Options & Derivatives (options.py)
Options analysis with pure Python Black-Scholes (no scipy dependency):

### Functions
- `options_chain` – calls & puts with strike, bid/ask, volume, OI, IV, summary
- `options_greeks` – Delta, Gamma, Theta, Vega, Rho for all options at an expiry
- `options_iv_surface` – IV surface data, skew summary, term structure
- `options_unusual_activity` – high volume/OI ratio detection with alert levels
- `options_put_call_ratio` – P/C ratio with sentiment classification (BEARISH/NEUTRAL/BULLISH)
- `options_max_pain` – max pain strike calculation with pain-by-strike breakdown

## Stock Report Generator (report.py)
Unified reports aggregating all analysis modules in a single call:

### Functions
- `stock_report` – comprehensive report: overview + price + TA + candlestick + financial health + growth + valuation + dividend + sentiment + ML (all sections run in parallel, includes executive summary)
- `stock_report_quick` – lightweight quick report: price summary + TA signal + key ratios (much faster)
- `stock_report_compare` – side-by-side comparison for multiple stocks with rankings by return, TA score, F-score

## Alert & Watchlist (alert.py)
Stateless condition-based monitoring — evaluates at call time, designed for scheduled tasks:

### Functions
- `alert_check` – evaluate custom alert conditions for a symbol
- `alert_price` – quick price-level alert (above/below thresholds)
- `alert_ta` – 16 TA-based alerts: rsi_oversold, rsi_overbought, macd_bullish_cross, macd_bearish_cross, bb_squeeze, bb_breakout_upper, bb_breakout_lower, golden_cross, death_cross, volume_spike, price_above_sma, price_below_sma, supertrend_bullish, supertrend_bearish, new_high_52w, new_low_52w
- `alert_fundamental` – fundamental alerts: P/E, dividend yield, ROE, D/E, Piotroski F-score thresholds
- `alert_watchlist` – check multiple symbols against conditions in parallel
- `alert_list_conditions` – list all available alert conditions

## Sector & Market Analysis (market.py)
Market-level analysis beyond individual stocks:

### Functions
- `market_sector_performance` – compare performance across sectors with ranking
- `market_breadth` – advance/decline ratio, 52w new highs/lows, breadth signal (STRONG_BULLISH to STRONG_BEARISH)
- `market_top_movers` – top N gainers and losers from a symbol universe
- `market_regime` – detect bull/bear/sideways regime from index (SMA trend + momentum + volatility scoring)
- `market_heatmap` – sector-grouped performance data for visualization

## Custom Strategy Builder (strategy.py)
Compose multiple conditions into custom screening strategies with AND/OR logic:

### Condition Types (27 total)
- **Price (5)**: price_above, price_below, return_above, return_below, volume_above_avg
- **TA (15)**: rsi_below, rsi_above, rsi_between, macd_bullish, macd_bearish, price_above_sma, price_below_sma, bb_above_upper, bb_below_lower, supertrend_bullish, supertrend_bearish, ta_signal_buy, ta_signal_sell, ta_score_above, ta_score_below
- **Fundamental (7)**: pe_below, pe_above, pb_below, dividend_yield_above, roe_above, debt_to_equity_below, f_score_above

### Functions
- `strategy_evaluate` – evaluate composed conditions against a single symbol (AND/OR logic)
- `strategy_screen` – screen multiple symbols with custom strategy in parallel
- `strategy_list_conditions` – list all condition types grouped by category

### Performance
- Per-symbol data cache prevents redundant API calls when multiple conditions use the same data
- Parallel screening via ThreadPoolExecutor

## Infrastructure (logging.py + core.py)

### Structured Logging
- `get_logger(name)` – returns JSON-formatted logger (outputs to stderr)
- `LogTimer(logger, operation)` – context manager for timing + logging operations
- All log entries include: timestamp, level, logger, message, and optional fields (symbol, source, duration_ms, error, retry_attempt, cache_hit)

### Retry Logic
- `_safe_call_with_retry(func, *args, max_attempts=3)` – exponential backoff retry
- Retries on: `ConnectionError`, `TimeoutError`, `OSError`
- Backoff delays: 0.5s → 1.0s → 2.0s
- Non-retryable exceptions (ValueError, KeyError, etc.) fail immediately

### Caching
- `_TTLCache(maxsize=128, ttl=300)` – LRU cache with 5-minute TTL
- `cache_clear()` – clear all cached entries
- `@with_cache` decorator applied to: `stock_history`, `company_overview`, `company_news`, `financial_ratio`, `fx_history`, `crypto_history`, `world_index_history`
- Cache key = hash of (function_name, args, kwargs)

### Parallel Data Fetching
- `fetch_parallel(func, symbols, max_workers=8)` – generic parallel fetcher via ThreadPoolExecutor
- `stock_history_batch()` – batch price history for multiple symbols
- `company_overview_batch()` – batch company overview

## Authentication & API Key Management (auth.py)
Tier-based API key system for commercial MCP marketplace deployment.

### Tiers
| Tier | Daily Limit | Tools | Price |
|------|------------|-------|-------|
| **free** | 50 | basic (quotes, history, overview, FX, crypto) | $0 |
| **pro** | 1,000 | all 105 tools | $9/mo |
| **enterprise** | 5,000 | all + priority support | $19/mo |

### API Key Format
`jpsk_{tier}_{random_hex}` e.g. `jpsk_pro_a1b2c3d4e5f6`

### Functions
- `KeyStore.generate_key` – create new API key (raw key shown once, only hash stored)
- `KeyStore.validate` – validate key, return tier/owner
- `KeyStore.revoke` – deactivate a key
- `KeyStore.list_keys` – list keys (hashes only)
- `check_tool_access` – check if tier allows a specific tool
- `get_daily_limit` – return daily call limit for a tier

### Storage
JSON file at `~/.jpstock/keys.json` (configurable). SHA-256 hashed keys.

### CLI Commands
`key-generate`, `key-list`, `key-revoke`, `key-validate`, `auth-tiers`

## Rate Limiting (ratelimit.py)
In-memory sliding-window rate limiter, thread-safe.

### Features
- **Daily window** (86,400s) – enforces per-tier daily limits
- **Burst window** (60s) – prevents API abuse spikes (default 30/min)
- Per-key custom limits via `set_limit()`
- `peek()` – check quota without consuming a call
- `usage()` – detailed stats (daily/minute used, remaining, utilization %)
- `reset()` / `reset_all()` – clear state

### Thread Safety
All public methods acquire `threading.Lock` for concurrent FastMCP handlers.

### Production Note
For multi-process deployment, swap with Redis-backed implementation (same interface).

## ASGI Middleware (middleware.py)
Auth + rate limiting middleware for HTTP/SSE transports.

### How It Works
- **stdio transport**: No auth (local subprocess, user is owner)
- **HTTP/SSE transport**: `AuthMiddleware` wraps the ASGI app
- API key via `Authorization: Bearer <key>` or `X-API-Key` header
- Returns `401` for missing/invalid keys, `429` for rate limit exceeded
- Master key bypasses all checks
- Injects `scope["auth"]` context for downstream use

## Usage Tracking & Analytics (usage.py)
SQLite-backed persistent usage recorder and analytics engine.

### Storage
Default: `~/.jpstock/usage.db` (WAL mode for concurrent writes).
Single `calls` table with indexes on `(key_hash, date_str)`, `(tool, date_str)`, `(date_str)`.

### Functions
- `record` – log a tool call (key_hash, tier, tool, latency, status)
- `daily_summary` – aggregate stats for a day (total calls, unique keys, by tier, top tools, error rate, avg latency)
- `key_usage` – per-key breakdown over N days (daily counts, top tools, errors)
- `tool_stats` – per-tool stats (call counts, avg latency, error rates)
- `revenue_estimate` – MRR/ARR estimate from active keys per tier (free=$0, pro=$9, enterprise=$19)
- `cleanup` – delete records older than N days

### Thread Safety
Each thread gets its own SQLite connection via thread-local storage.

### CLI Commands
`usage-daily`, `usage-key`, `usage-tools`, `usage-revenue`, `usage-cleanup`

## Test Suite
- 23 test files, 858 tests (3 skipped for ta library compat)
- Overall coverage: 79%
- Key module coverage: core 87%, backtest 88%, candlestick 91%, sentiment 97%, logging 98%, financial 95%+
- CI/CD: GitHub Actions with Python 3.10/3.11/3.12, ruff lint, pytest-cov (threshold 75%)
