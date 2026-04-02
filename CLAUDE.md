# JPStock Agent – Developer Guide

## Overview
MCP server & CLI for Japanese and Vietnamese stock market data. Wraps **yfinance**, **J-Quants API**, and **vnstock** to provide AI-friendly access to TSE/JPX and HOSE/HNX/UPCOM market data.

## Project Structure
```
src/jpstock_agent/
  __init__.py     – version
  config.py       – env-based settings (pydantic-settings)
  core.py         – all data functions (returns list[dict] | dict)
  ta.py           – technical analysis module (24 TA functions + 29 screening strategies)
  candlestick.py  – candlestick pattern detection (20 patterns)
  backtest.py     – backtesting engine (12 strategies, optimizer, walk-forward)
  server.py       – FastMCP tool definitions (55 tools: 24 data + 24 TA + 3 candlestick + 4 backtest)
  cli.py          – Click CLI commands (50+ commands + serve)
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
