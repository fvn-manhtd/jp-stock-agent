"""
Backtesting engine for testing trading strategies on historical stock data.

Provides functions to:
- Run single strategies on historical data
- Compare multiple strategies
- Optimize strategy parameters
- Walk-forward analysis for consistency checking
"""

from datetime import datetime, timedelta
from typing import Optional, Union
import pandas as pd
import numpy as np

from .core import _safe_call
from .ta import _get_ohlcv_df, _round_val


# ============================================================================
# Main API Functions
# ============================================================================


def backtest_strategy(
    symbol: str,
    strategy: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 1_000_000,
    source: Optional[str] = None,
    **params,
) -> Union[dict, list]:
    """
    Backtest a trading strategy on historical data.

    Args:
        symbol: Stock ticker code
        strategy: Strategy name (sma_crossover, ema_crossover, rsi_reversal, etc.)
        start: Start date (YYYY-MM-DD), default 365 days ago
        end: End date (YYYY-MM-DD), default today
        initial_capital: Starting capital in JPY (default 1,000,000)
        source: Data source (yfinance, jquants, vnstocks)
        **params: Additional strategy parameters (overrides defaults)

    Returns:
        dict with backtest results including:
        - strategy, symbol, period, initial_capital
        - final_capital, total_return_pct, annual_return_pct
        - total_trades, winning_trades, losing_trades, win_rate_pct
        - max_drawdown_pct, sharpe_ratio
        - avg_trade_return_pct, best_trade_pct, worst_trade_pct
        - buy_hold_return_pct, alpha_pct
        - trades: list of individual trades
    """

    def _backtest():
        # Set default date range (365 days lookback)
        end_date = end or datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            start_date = start

        # Fetch OHLCV data
        df = _get_ohlcv_df(symbol, start=start_date, end=end_date, source=source)
        if df is None or df.empty:
            return {"error": f"Failed to fetch data for {symbol}"}

        # Generate trading signals
        signals_df = _generate_signals(df, strategy, **params)
        if signals_df is None:
            return {"error": f"Failed to generate signals for strategy {strategy}"}

        # Execute trades based on signals
        trades, final_capital = _execute_trades(signals_df, initial_capital)

        # Calculate metrics
        metrics = _calculate_metrics(
            df,
            trades,
            initial_capital,
            final_capital,
            strategy,
            symbol,
            start_date,
            end_date,
        )
        return metrics

    return _safe_call(_backtest)


def backtest_compare(
    symbol: str,
    strategies: Optional[list] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 1_000_000,
    source: Optional[str] = None,
) -> list:
    """
    Compare multiple strategies on the same symbol.

    Args:
        symbol: Stock ticker code
        strategies: List of strategy names. If None, run all 12 default strategies
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        source: Data source

    Returns:
        List of dicts (one per strategy), sorted by total_return_pct descending
    """

    if strategies is None:
        strategies = [
            "sma_crossover",
            "ema_crossover",
            "rsi_reversal",
            "macd_crossover",
            "bollinger_bounce",
            "supertrend",
            "ichimoku_cloud",
            "golden_cross",
            "mean_reversion",
            "momentum",
            "breakout",
            "vwap_strategy",
        ]

    results = []
    for strat in strategies:
        result = backtest_strategy(
            symbol,
            strat,
            start=start,
            end=end,
            initial_capital=initial_capital,
            source=source,
        )
        # Remove trades list for comparison view
        if isinstance(result, dict) and "error" not in result:
            result_copy = {k: v for k, v in result.items() if k != "trades"}
            results.append(result_copy)
        elif isinstance(result, dict) and "error" in result:
            results.append(result)

    # Sort by total_return_pct descending
    results.sort(
        key=lambda x: x.get("total_return_pct", float("-inf")), reverse=True
    )
    return results


def backtest_optimize(
    symbol: str,
    strategy: str,
    param_name: str,
    param_range: list,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 1_000_000,
    source: Optional[str] = None,
) -> list:
    """
    Optimize a strategy parameter by testing multiple values.

    Args:
        symbol: Stock ticker code
        strategy: Strategy name to optimize
        param_name: Parameter name (e.g. "fast_period", "slow_period")
        param_range: List of values to test
        start: Start date
        end: End date
        initial_capital: Starting capital
        source: Data source

    Returns:
        List of backtest results sorted by total_return_pct descending
    """

    results = []
    for param_value in param_range:
        result = backtest_strategy(
            symbol,
            strategy,
            start=start,
            end=end,
            initial_capital=initial_capital,
            source=source,
            **{param_name: param_value},
        )
        if isinstance(result, dict) and "error" not in result:
            # Remove trades list, add param info
            result_copy = {k: v for k, v in result.items() if k != "trades"}
            result_copy[param_name] = param_value
            results.append(result_copy)

    # Sort by total_return_pct descending
    results.sort(
        key=lambda x: x.get("total_return_pct", float("-inf")), reverse=True
    )
    return results


def backtest_walk_forward(
    symbol: str,
    strategy: str,
    window: int = 180,
    step: int = 30,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 1_000_000,
    source: Optional[str] = None,
) -> dict:
    """
    Walk-forward analysis: test strategy on rolling windows for consistency.

    Args:
        symbol: Stock ticker code
        strategy: Strategy name
        window: Training/test window in days (default 180)
        step: Step forward in days (default 30)
        start: Start date
        end: End date
        initial_capital: Starting capital
        source: Data source

    Returns:
        dict with overall metrics and per-window results
    """

    def _walk_forward():
        # Set default date range
        end_date = end or datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        else:
            start_date = start

        # Fetch full data range
        df = _get_ohlcv_df(symbol, start=start_date, end=end_date, source=source)
        if df is None or df.empty:
            return {"error": f"Failed to fetch data for {symbol}"}

        window_results = []
        all_trades = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        current = start_dt
        while current + timedelta(days=window) <= end_dt:
            window_start = current.strftime("%Y-%m-%d")
            window_end = (current + timedelta(days=window)).strftime("%Y-%m-%d")

            # Get data for this window
            window_df = df[
                (df.index >= window_start) & (df.index <= window_end)
            ].copy()
            if len(window_df) < 20:  # Need minimum data
                current += timedelta(days=step)
                continue

            # Run backtest on this window
            signals_df = _generate_signals(window_df, strategy)
            if signals_df is None:
                current += timedelta(days=step)
                continue

            trades, final_capital = _execute_trades(signals_df, initial_capital)
            all_trades.extend(trades)

            window_results.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "total_return_pct": _round_val(
                        ((final_capital - initial_capital) / initial_capital) * 100,
                        2,
                    ),
                    "trades": len(trades),
                }
            )

            current += timedelta(days=step)

        if not window_results:
            return {"error": "Not enough data for walk-forward analysis"}

        # Calculate overall metrics
        final_capital = initial_capital
        for wr in window_results:
            ret = wr["total_return_pct"]
            final_capital = final_capital * (1 + ret / 100)

        overall_return = _round_val(
            ((final_capital - initial_capital) / initial_capital) * 100, 2
        )
        avg_window_return = _round_val(
            sum(wr["total_return_pct"] for wr in window_results) / len(window_results),
            2,
        )

        return {
            "strategy": strategy,
            "symbol": symbol,
            "window_days": window,
            "step_days": step,
            "total_windows": len(window_results),
            "overall_return_pct": overall_return,
            "avg_window_return_pct": avg_window_return,
            "window_results": window_results,
        }

    return _safe_call(_walk_forward)


# ============================================================================
# Internal Helpers
# ============================================================================


def _generate_signals(
    df: pd.DataFrame, strategy: str, **params
) -> Optional[pd.DataFrame]:
    """
    Generate trading signals for a given strategy.

    Args:
        df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
        strategy: Strategy name
        **params: Strategy-specific parameters

    Returns:
        DataFrame with added 'signal' column: 1=buy, -1=sell, 0=hold
    """

    signals_df = df.copy()
    signals_df["signal"] = 0

    try:
        if strategy == "sma_crossover":
            fast_period = params.get("fast_period", 20)
            slow_period = params.get("slow_period", 50)
            signals_df["sma_fast"] = signals_df["close"].rolling(fast_period).mean()
            signals_df["sma_slow"] = signals_df["close"].rolling(slow_period).mean()
            signals_df["signal"] = np.where(
                signals_df["sma_fast"] > signals_df["sma_slow"], 1, -1
            )

        elif strategy == "ema_crossover":
            fast_period = params.get("fast_period", 9)
            slow_period = params.get("slow_period", 21)
            signals_df["ema_fast"] = signals_df["close"].ewm(span=fast_period).mean()
            signals_df["ema_slow"] = signals_df["close"].ewm(span=slow_period).mean()
            signals_df["signal"] = np.where(
                signals_df["ema_fast"] > signals_df["ema_slow"], 1, -1
            )

        elif strategy == "rsi_reversal":
            period = params.get("rsi_period", 14)
            oversold = params.get("oversold", 30)
            overbought = params.get("overbought", 70)
            signals_df["rsi"] = _calculate_rsi(signals_df["close"], period)
            signals_df["signal"] = np.where(
                signals_df["rsi"] < oversold,
                1,
                np.where(signals_df["rsi"] > overbought, -1, 0),
            )

        elif strategy == "macd_crossover":
            fast = params.get("fast_period", 12)
            slow = params.get("slow_period", 26)
            signal_period = params.get("signal_period", 9)
            ema_fast = signals_df["close"].ewm(span=fast).mean()
            ema_slow = signals_df["close"].ewm(span=slow).mean()
            signals_df["macd"] = ema_fast - ema_slow
            signals_df["macd_signal"] = signals_df["macd"].ewm(span=signal_period).mean()
            signals_df["signal"] = np.where(
                signals_df["macd"] > signals_df["macd_signal"], 1, -1
            )

        elif strategy == "bollinger_bounce":
            period = params.get("period", 20)
            std_dev = params.get("std_dev", 2.0)
            sma = signals_df["close"].rolling(period).mean()
            std = signals_df["close"].rolling(period).std()
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            signals_df["signal"] = np.where(
                signals_df["close"] < lower_band,
                1,
                np.where(signals_df["close"] > upper_band, -1, 0),
            )

        elif strategy == "supertrend":
            period = params.get("period", 10)
            multiplier = params.get("multiplier", 3.0)
            signals_df["atr"] = _calculate_atr(signals_df, period)
            hl_avg = (signals_df["high"] + signals_df["low"]) / 2
            signals_df["basic_ub"] = hl_avg + multiplier * signals_df["atr"]
            signals_df["basic_lb"] = hl_avg - multiplier * signals_df["atr"]
            signals_df["final_ub"] = signals_df["basic_ub"]
            signals_df["final_lb"] = signals_df["basic_lb"]

            for i in range(1, len(signals_df)):
                signals_df.iloc[i, signals_df.columns.get_loc("final_ub")] = (
                    min(
                        signals_df.iloc[i, signals_df.columns.get_loc("basic_ub")],
                        signals_df.iloc[i - 1, signals_df.columns.get_loc("final_ub")],
                    )
                    if signals_df.iloc[i]["close"] > signals_df.iloc[i - 1]["final_ub"]
                    else signals_df.iloc[i, signals_df.columns.get_loc("basic_ub")]
                )
                signals_df.iloc[i, signals_df.columns.get_loc("final_lb")] = (
                    max(
                        signals_df.iloc[i, signals_df.columns.get_loc("basic_lb")],
                        signals_df.iloc[i - 1, signals_df.columns.get_loc("final_lb")],
                    )
                    if signals_df.iloc[i]["close"] < signals_df.iloc[i - 1]["final_lb"]
                    else signals_df.iloc[i, signals_df.columns.get_loc("basic_lb")]
                )

            signals_df["signal"] = np.where(
                signals_df["close"] > signals_df["final_ub"], 1, -1
            )

        elif strategy == "ichimoku_cloud":
            tenkan = params.get("tenkan", 9)
            kijun = params.get("kijun", 26)
            signals_df["tenkan_sen"] = (
                signals_df["high"].rolling(tenkan).max()
                + signals_df["low"].rolling(tenkan).min()
            ) / 2
            signals_df["kijun_sen"] = (
                signals_df["high"].rolling(kijun).max()
                + signals_df["low"].rolling(kijun).min()
            ) / 2
            signals_df["signal"] = np.where(
                signals_df["close"] > signals_df["tenkan_sen"], 1, -1
            )

        elif strategy == "golden_cross":
            sma_50 = signals_df["close"].rolling(50).mean()
            sma_200 = signals_df["close"].rolling(200).mean()
            signals_df["signal"] = np.where(sma_50 > sma_200, 1, -1)

        elif strategy == "mean_reversion":
            period = params.get("period", 20)
            sma = signals_df["close"].rolling(period).mean()
            std = signals_df["close"].rolling(period).std()
            lower_band = sma - (2 * std)
            upper_band = sma + std
            signals_df["signal"] = np.where(
                signals_df["close"] < lower_band,
                1,
                np.where(signals_df["close"] > upper_band, -1, 0),
            )

        elif strategy == "momentum":
            roc_period = params.get("roc_period", 12)
            rsi_period = params.get("rsi_period", 14)
            signals_df["roc"] = _calculate_roc(signals_df["close"], roc_period)
            signals_df["rsi"] = _calculate_rsi(signals_df["close"], rsi_period)
            signals_df["signal"] = np.where(
                (signals_df["roc"] > 0) & (signals_df["rsi"] < 70),
                1,
                np.where((signals_df["roc"] < 0) | (signals_df["rsi"] > 80), -1, 0),
            )

        elif strategy == "breakout":
            period = params.get("period", 20)
            high_band = signals_df["high"].rolling(period).max()
            low_band = signals_df["low"].rolling(period).min()
            signals_df["signal"] = np.where(
                signals_df["close"] > high_band,
                1,
                np.where(signals_df["close"] < low_band, -1, 0),
            )

        elif strategy == "vwap_strategy":
            signals_df["vwap"] = _calculate_vwap(signals_df)
            signals_df["signal"] = np.where(
                signals_df["close"] > signals_df["vwap"], 1, -1
            )

        else:
            return None

        return signals_df

    except Exception:
        return None


def _execute_trades(
    signals_df: pd.DataFrame, initial_capital: float
) -> tuple[list, float]:
    """
    Execute trades based on signals.

    Args:
        signals_df: DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
        initial_capital: Starting capital

    Returns:
        tuple: (trades list, final capital)
    """

    trades = []
    capital = initial_capital
    position = None  # None or (shares, entry_price, entry_date)

    for i, row in signals_df.iterrows():
        signal = row["signal"]
        price = row["close"]

        if signal == 1 and position is None:
            # Buy signal and not in position
            shares = capital / price
            position = (shares, price, i)
            trades.append(
                {
                    "date": str(i.date()),
                    "action": "BUY",
                    "price": _round_val(price, 2),
                    "shares": _round_val(shares, 4),
                    "capital": _round_val(capital, 2),
                }
            )

        elif signal == -1 and position is not None:
            # Sell signal and in position
            shares, entry_price, entry_date = position
            capital = shares * price
            trade_return = ((price - entry_price) / entry_price) * 100
            trades.append(
                {
                    "date": str(i.date()),
                    "action": "SELL",
                    "price": _round_val(price, 2),
                    "shares": _round_val(shares, 4),
                    "capital": _round_val(capital, 2),
                    "return_pct": _round_val(trade_return, 2),
                }
            )
            position = None

    # Close any remaining position at the last price
    if position is not None:
        shares, entry_price, entry_date = position
        last_price = signals_df.iloc[-1]["close"]
        capital = shares * last_price
        trade_return = ((last_price - entry_price) / entry_price) * 100
        trades.append(
            {
                "date": str(signals_df.index[-1].date()),
                "action": "SELL",
                "price": _round_val(last_price, 2),
                "shares": _round_val(shares, 4),
                "capital": _round_val(capital, 2),
                "return_pct": _round_val(trade_return, 2),
            }
        )

    return trades, capital


def _calculate_metrics(
    df: pd.DataFrame,
    trades: list,
    initial_capital: float,
    final_capital: float,
    strategy: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Calculate performance metrics from backtest results.
    """

    # Total return
    total_return = final_capital - initial_capital
    total_return_pct = _round_val((total_return / initial_capital) * 100, 2)

    # Annual return (annualized)
    trading_days = len(df)
    if trading_days > 0:
        annual_return = _round_val(
            ((final_capital / initial_capital) ** (252 / trading_days) - 1) * 100, 2
        )
    else:
        annual_return = 0.0

    # Trade statistics
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    winning_trades = len([t for t in sell_trades if t.get("return_pct", 0) > 0])
    losing_trades = len([t for t in sell_trades if t.get("return_pct", 0) < 0])
    total_trades = len(trades)
    win_rate = _round_val(
        (winning_trades / len(sell_trades) * 100) if sell_trades else 0, 2
    )

    # Average trade return
    if sell_trades:
        avg_trade_return = _round_val(
            sum(t.get("return_pct", 0) for t in sell_trades) / len(sell_trades), 2
        )
        best_trade = _round_val(
            max(t.get("return_pct", 0) for t in sell_trades), 2
        )
        worst_trade = _round_val(
            min(t.get("return_pct", 0) for t in sell_trades), 2
        )
    else:
        avg_trade_return = 0.0
        best_trade = 0.0
        worst_trade = 0.0

    # Sharpe ratio (annualized, risk-free=0)
    if trading_days > 1:
        daily_returns = df["close"].pct_change().dropna()
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        if std_daily_return > 0:
            sharpe_ratio = _round_val(
                (mean_daily_return / std_daily_return) * np.sqrt(252), 2
            )
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumulative = (1 + df["close"].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = _round_val(drawdown.min() * 100, 2)

    # Buy & hold benchmark
    buy_hold_return = _round_val(
        ((df.iloc[-1]["close"] / df.iloc[0]["close"]) - 1) * 100, 2
    )

    # Alpha (strategy return - buy & hold)
    alpha = _round_val(total_return_pct - buy_hold_return, 2)

    return {
        "strategy": strategy,
        "symbol": symbol,
        "period": f"{start_date} to {end_date}",
        "initial_capital": _round_val(initial_capital, 2),
        "final_capital": _round_val(final_capital, 2),
        "total_return_pct": total_return_pct,
        "annual_return_pct": annual_return,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "avg_trade_return_pct": avg_trade_return,
        "best_trade_pct": best_trade,
        "worst_trade_pct": worst_trade,
        "buy_hold_return_pct": buy_hold_return,
        "alpha_pct": alpha,
        "trades": trades,
    }


# ============================================================================
# TA Indicator Helpers (simple implementations)
# ============================================================================


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR (Average True Range)."""
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def _calculate_roc(prices: pd.Series, period: int = 12) -> pd.Series:
    """Calculate ROC (Rate of Change)."""
    roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
    return roc


def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP (Volume Weighted Average Price)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    return vwap
