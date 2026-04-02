"""
Portfolio Optimization Module

Provides portfolio analysis, optimization, and risk assessment tools.
Supports Japanese (TSE/JPX) and Vietnamese (HOSE/HNX/UPCOM) stock markets.

Public Functions:
- portfolio_analyze: Analyze multi-stock portfolio performance and correlations
- portfolio_optimize: Monte Carlo portfolio optimization (random weight simulation)
- portfolio_risk: Risk metrics for a portfolio with given or equal weights
- portfolio_correlation: Correlation and covariance matrices

All functions return dict on success or {"error": str} on failure.
Default lookback: 365 days.
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

from .ta import _get_ohlcv_df, _round_val
from .core import _safe_call


def _get_returns_df(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: Optional[str] = None,
) -> Union[pd.DataFrame, Dict[str, str]]:
    """
    Fetch OHLCV data for all symbols and return aligned daily returns DataFrame.

    Columns: symbol names
    Index: date (daily)
    Values: daily returns (pct_change)

    Symbols that fail to fetch are dropped with a warning.
    Returns {"error": str} if no valid symbols remain.
    """
    if not symbols or len(symbols) == 0:
        return {"error": "No symbols provided"}

    # Default lookback: 365 days if not specified
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

    dfs_list = []
    valid_symbols = []
    failed_symbols = []

    for symbol in symbols:
        result = _get_ohlcv_df(symbol, start=start, end=end, source=source)

        if isinstance(result, dict) and "error" in result:
            failed_symbols.append(symbol)
            continue

        if result is None or len(result) == 0:
            failed_symbols.append(symbol)
            continue

        # Extract 'close' column and rename to symbol
        closes = result[["close"]].copy()
        closes.columns = [symbol]
        dfs_list.append(closes)
        valid_symbols.append(symbol)

    if failed_symbols:
        warnings.warn(f"Failed to fetch data for symbols: {', '.join(failed_symbols)}")

    if not valid_symbols or not dfs_list:
        return {"error": f"No valid symbols. Failed: {', '.join(failed_symbols)}"}

    # Align all dataframes on date index
    df_combined = pd.concat(dfs_list, axis=1, join="inner")

    if len(df_combined) < 2:
        return {"error": "Insufficient data points after alignment"}

    # Calculate daily returns (pct_change)
    returns_df = df_combined.pct_change().dropna()

    if len(returns_df) == 0:
        return {"error": "No valid returns calculated"}

    return returns_df


def _annualize_return(daily_returns_series: pd.Series) -> float:
    """
    Annualize a daily returns series.
    Formula: mean_daily_return * 252
    """
    if len(daily_returns_series) == 0:
        return 0.0
    return float(daily_returns_series.mean() * 252)


def _annualize_volatility(daily_returns_series: pd.Series) -> float:
    """
    Annualize a daily returns series volatility.
    Formula: std_daily_return * sqrt(252)
    """
    if len(daily_returns_series) == 0:
        return 0.0
    return float(daily_returns_series.std() * np.sqrt(252))


def portfolio_analyze(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    """
    Analyze a portfolio of stocks.

    Returns:
    {
        "symbols": [str],
        "period": {"start": str, "end": str},
        "stocks": {
            "SYM": {
                "return_pct": float,
                "volatility_pct": float,
                "sharpe_ratio": float
            }
        },
        "correlation_matrix": {
            "SYM1": {"SYM2": float, ...},
            ...
        },
        "portfolio_equal_weight": {
            "return_pct": float,
            "volatility_pct": float
        },
        "best_performer": {"symbol": str, "return_pct": float},
        "worst_performer": {"symbol": str, "return_pct": float}
    }
    """

    def _run():
        returns_df = _get_returns_df(symbols, start=start, end=end, source=source)

        if isinstance(returns_df, dict) and "error" in returns_df:
            return returns_df

        # Extract actual start/end dates from data
        actual_start = returns_df.index[0].strftime("%Y-%m-%d")
        actual_end = returns_df.index[-1].strftime("%Y-%m-%d")

        # Per-stock metrics
        stocks_metrics = {}
        for symbol in returns_df.columns:
            ret_series = returns_df[symbol]
            ann_return = _annualize_return(ret_series)
            ann_vol = _annualize_volatility(ret_series)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

            stocks_metrics[symbol] = {
                "return_pct": _round_val(ann_return * 100),
                "volatility_pct": _round_val(ann_vol * 100),
                "sharpe_ratio": _round_val(sharpe),
            }

        # Correlation matrix
        corr_matrix = returns_df.corr().to_dict()

        # Equal-weight portfolio
        n = len(returns_df.columns)
        equal_weights = np.ones(n) / n
        portfolio_daily_returns = (returns_df * equal_weights).sum(axis=1)
        eq_return = _annualize_return(portfolio_daily_returns)
        eq_vol = _annualize_volatility(portfolio_daily_returns)

        # Best and worst performers
        returns_dict = {symbol: stocks_metrics[symbol]["return_pct"] for symbol in returns_df.columns}
        best_sym = max(returns_dict, key=returns_dict.get)
        worst_sym = min(returns_dict, key=returns_dict.get)

        return {
            "symbols": list(returns_df.columns),
            "period": {"start": actual_start, "end": actual_end},
            "stocks": stocks_metrics,
            "correlation_matrix": corr_matrix,
            "portfolio_equal_weight": {
                "return_pct": _round_val(eq_return * 100),
                "volatility_pct": _round_val(eq_vol * 100),
            },
            "best_performer": {
                "symbol": best_sym,
                "return_pct": _round_val(returns_dict[best_sym]),
            },
            "worst_performer": {
                "symbol": worst_sym,
                "return_pct": _round_val(returns_dict[worst_sym]),
            },
        }

    return _safe_call(_run)


def portfolio_optimize(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    num_portfolios: int = 5000,
    risk_free_rate: float = 0.0,
    source: Optional[str] = None,
) -> dict:
    """
    Monte Carlo portfolio optimization via random weight simulation.

    Generates num_portfolios random long-only weight combinations and identifies:
    - max_sharpe_portfolio: Highest Sharpe ratio
    - min_volatility_portfolio: Lowest volatility
    - max_return_portfolio: Highest expected return
    - efficient_frontier: Top 20 portfolios by Sharpe ratio

    All weights sum to 1.0, no negative weights (long-only).

    Returns:
    {
        "num_portfolios": int,
        "max_sharpe_portfolio": {
            "weights": {"SYM": weight, ...},
            "return_pct": float,
            "volatility_pct": float,
            "sharpe_ratio": float
        },
        "min_volatility_portfolio": {...},
        "max_return_portfolio": {...},
        "efficient_frontier": [
            {"weights": {...}, "return_pct": float, "volatility_pct": float, "sharpe_ratio": float},
            ...  (top 20)
        ]
    }
    """

    def _run():
        returns_df = _get_returns_df(symbols, start=start, end=end, source=source)

        if isinstance(returns_df, dict) and "error" in returns_df:
            return returns_df

        symbols_list = list(returns_df.columns)
        n_assets = len(symbols_list)

        # Pre-compute annualized return and covariance
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252

        results = []

        # Monte Carlo simulation
        np.random.seed(42)  # for reproducibility
        for _ in range(num_portfolios):
            # Generate random weights (dirichlet ensures sum=1, all positive)
            weights = np.random.dirichlet(np.ones(n_assets))

            # Portfolio return
            port_return = np.dot(weights, mean_returns)

            # Portfolio volatility
            port_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            # Sharpe ratio
            sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0.0

            results.append({
                "weights": weights,
                "return": port_return,
                "volatility": port_volatility,
                "sharpe": sharpe,
            })

        # Find optimal portfolios
        results_sorted_sharpe = sorted(results, key=lambda x: x["sharpe"], reverse=True)
        max_sharpe = results_sorted_sharpe[0]

        results_sorted_vol = sorted(results, key=lambda x: x["volatility"])
        min_vol = results_sorted_vol[0]

        results_sorted_ret = sorted(results, key=lambda x: x["return"], reverse=True)
        max_ret = results_sorted_ret[0]

        # Efficient frontier (top 20 by Sharpe)
        frontier = results_sorted_sharpe[:20]

        def _format_portfolio(portfolio_data):
            weights_dict = {sym: _round_val(float(w)) for sym, w in zip(symbols_list, portfolio_data["weights"])}
            return {
                "weights": weights_dict,
                "return_pct": _round_val(float(portfolio_data["return"]) * 100),
                "volatility_pct": _round_val(float(portfolio_data["volatility"]) * 100),
                "sharpe_ratio": _round_val(float(portfolio_data["sharpe"])),
            }

        efficient_frontier_formatted = [_format_portfolio(p) for p in frontier]

        return {
            "num_portfolios": num_portfolios,
            "max_sharpe_portfolio": _format_portfolio(max_sharpe),
            "min_volatility_portfolio": _format_portfolio(min_vol),
            "max_return_portfolio": _format_portfolio(max_ret),
            "efficient_frontier": efficient_frontier_formatted,
        }

    return _safe_call(_run)


def portfolio_risk(
    symbols: List[str],
    weights: Optional[Union[List[float], Dict[str, float]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    """
    Risk analysis for a portfolio with given weights (default: equal weight).

    weights: Optional list (must match symbol order) or dict {symbol: weight}
             If None, equal weight is used.

    Returns:
    {
        "portfolio_return_pct": float (annualized),
        "portfolio_volatility_pct": float (annualized),
        "sharpe_ratio": float,
        "sortino_ratio": float (using downside deviation),
        "max_drawdown_pct": float,
        "var_95_pct": float (Value at Risk at 95% confidence),
        "cvar_95_pct": float (Conditional VaR / Expected Shortfall),
        "beta": float,
        "weights_used": {"SYM": weight, ...}
    }
    """

    def _run():
        returns_df = _get_returns_df(symbols, start=start, end=end, source=source)

        if isinstance(returns_df, dict) and "error" in returns_df:
            return returns_df

        symbols_list = list(returns_df.columns)
        n = len(symbols_list)

        # Process weights
        if weights is None:
            # Equal weight
            portfolio_weights = np.ones(n) / n
            weights_dict = {sym: _round_val(1.0 / n) for sym in symbols_list}
        elif isinstance(weights, dict):
            # Dict format
            portfolio_weights = np.array([weights.get(sym, 0) for sym in symbols_list])
            weights_dict = {sym: _round_val(float(weights.get(sym, 0))) for sym in symbols_list}
        else:
            # List format
            if len(weights) != n:
                return {"error": f"weights length {len(weights)} != symbols length {n}"}
            portfolio_weights = np.array(weights)
            weights_dict = {sym: _round_val(float(w)) for sym, w in zip(symbols_list, weights)}

        # Validate weights sum to 1
        weight_sum = float(np.sum(portfolio_weights))
        if not (0.99 < weight_sum < 1.01):
            return {"error": f"Weights do not sum to 1.0 (sum={weight_sum})"}

        # Portfolio daily returns
        portfolio_daily_returns = (returns_df * portfolio_weights).sum(axis=1)

        # Annualized return and volatility
        port_return = _annualize_return(portfolio_daily_returns)
        port_volatility = _annualize_volatility(portfolio_daily_returns)

        # Sharpe ratio (risk_free_rate = 0)
        sharpe = port_return / port_volatility if port_volatility > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std()
            sortino = (port_return / (downside_std * np.sqrt(252))) if downside_std > 0 else 0.0
        else:
            sortino = 0.0

        # Max drawdown
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_dd = float(drawdowns.min())

        # VaR 95% (historical method, 5th percentile)
        var_95 = np.percentile(portfolio_daily_returns, 5)

        # CVaR 95% (mean of returns below VaR)
        cvar_95 = portfolio_daily_returns[portfolio_daily_returns <= var_95].mean()

        # Beta (vs equal-weight benchmark if >1 stock, else vs self=1)
        if n > 1:
            equal_weight = np.ones(n) / n
            benchmark_returns = (returns_df * equal_weight).sum(axis=1)
            covariance = portfolio_daily_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        else:
            beta = 1.0

        return {
            "portfolio_return_pct": _round_val(port_return * 100),
            "portfolio_volatility_pct": _round_val(port_volatility * 100),
            "sharpe_ratio": _round_val(sharpe),
            "sortino_ratio": _round_val(sortino),
            "max_drawdown_pct": _round_val(max_dd * 100),
            "var_95_pct": _round_val(var_95 * 100),
            "cvar_95_pct": _round_val(cvar_95 * 100),
            "beta": _round_val(beta),
            "weights_used": weights_dict,
        }

    return _safe_call(_run)


def portfolio_correlation(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    """
    Return correlation matrix and covariance matrix as nested dicts.

    Also returns:
    - most_correlated_pair: {symbols: [str, str], correlation: float}
    - least_correlated_pair: {symbols: [str, str], correlation: float}

    Returns:
    {
        "correlation_matrix": {"SYM1": {"SYM2": float, ...}, ...},
        "covariance_matrix": {"SYM1": {"SYM2": float, ...}, ...},
        "most_correlated_pair": {"symbols": [str, str], "correlation": float},
        "least_correlated_pair": {"symbols": [str, str], "correlation": float}
    }
    """

    def _run():
        returns_df = _get_returns_df(symbols, start=start, end=end, source=source)

        if isinstance(returns_df, dict) and "error" in returns_df:
            return returns_df

        # Correlation matrix
        corr_matrix = returns_df.corr().to_dict()

        # Covariance matrix (annualized)
        cov_matrix_annual = (returns_df.cov() * 252).to_dict()

        # Find most and least correlated pairs
        # Extract upper triangle (avoid duplicates and self-correlations)
        symbols_list = list(returns_df.columns)
        corr_pairs = []

        for i, sym1 in enumerate(symbols_list):
            for j, sym2 in enumerate(symbols_list):
                if i < j:  # Upper triangle only
                    corr_value = returns_df[sym1].corr(returns_df[sym2])
                    corr_pairs.append({
                        "symbols": [sym1, sym2],
                        "correlation": float(corr_value),
                    })

        if corr_pairs:
            most_corr = max(corr_pairs, key=lambda x: x["correlation"])
            least_corr = min(corr_pairs, key=lambda x: x["correlation"])
        else:
            most_corr = {"symbols": [], "correlation": 0.0}
            least_corr = {"symbols": [], "correlation": 0.0}

        # Round correlation and covariance values
        def _round_dict(d):
            return {k: {k2: _round_val(v2) for k2, v2 in v.items()} for k, v in d.items()}

        return {
            "correlation_matrix": _round_dict(corr_matrix),
            "covariance_matrix": _round_dict(cov_matrix_annual),
            "most_correlated_pair": {
                "symbols": most_corr["symbols"],
                "correlation": _round_val(most_corr["correlation"]),
            },
            "least_correlated_pair": {
                "symbols": least_corr["symbols"],
                "correlation": _round_val(least_corr["correlation"]),
            },
        }

    return _safe_call(_run)
