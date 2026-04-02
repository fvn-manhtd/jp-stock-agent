"""Options & Derivatives Data Module.

Provides options chain data, implied volatility analysis, Greeks
calculation, and IV surface visualization data using yfinance.

Public Functions:
- options_chain: Fetch options chain (calls & puts) for a symbol
- options_greeks: Calculate Greeks (Delta, Gamma, Theta, Vega, Rho) via Black-Scholes
- options_iv_surface: Build implied volatility surface (strike vs expiry)
- options_unusual_activity: Detect unusual options volume/OI
- options_put_call_ratio: Calculate put/call ratio as sentiment indicator
- options_max_pain: Calculate max pain strike price

All functions return dict or list[dict] on success, or {"error": str} on failure.
Uses yfinance for options chain data. Only works with yfinance source
(J-Quants and vnstocks do not provide options data).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .core import _safe_call
from .ta import _round_val

# ---------------------------------------------------------------------------
# Black-Scholes Greeks calculation
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (no scipy needed)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _black_scholes_greeks(
    S: float,      # Current stock price
    K: float,      # Strike price
    T: float,      # Time to expiry in years
    r: float,      # Risk-free rate (annualized)
    sigma: float,  # Implied volatility (annualized)
    option_type: str = "call",
) -> dict:
    """Calculate Black-Scholes Greeks for a single option.

    Returns dict with: delta, gamma, theta, vega, rho, theoretical_price.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {
            "delta": 0.0, "gamma": 0.0, "theta": 0.0,
            "vega": 0.0, "rho": 0.0, "theoretical_price": 0.0,
        }

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    n_neg_d1 = _norm_cdf(-d1)
    n_neg_d2 = _norm_cdf(-d2)
    npd1 = _norm_pdf(d1)

    if option_type == "call":
        price = S * nd1 - K * math.exp(-r * T) * nd2
        delta = nd1
        theta = (
            -(S * npd1 * sigma) / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * nd2
        ) / 365  # Per day
        rho = K * T * math.exp(-r * T) * nd2 / 100  # Per 1% change
    else:
        price = K * math.exp(-r * T) * n_neg_d2 - S * n_neg_d1
        delta = nd1 - 1
        theta = (
            -(S * npd1 * sigma) / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * n_neg_d2
        ) / 365
        rho = -K * T * math.exp(-r * T) * n_neg_d2 / 100

    gamma = npd1 / (S * sigma * sqrt_T)
    vega = S * npd1 * sqrt_T / 100  # Per 1% change in IV

    return {
        "delta": _round_val(delta, 4),
        "gamma": _round_val(gamma, 6),
        "theta": _round_val(theta, 4),
        "vega": _round_val(vega, 4),
        "rho": _round_val(rho, 4),
        "theoretical_price": _round_val(price, 2),
    }


# ---------------------------------------------------------------------------
# yfinance Options helpers
# ---------------------------------------------------------------------------

def _get_ticker(symbol: str):
    """Get a yfinance Ticker object."""
    import yfinance as yf
    # Normalize Japanese symbol
    if symbol.isdigit() and len(symbol) == 4:
        symbol = f"{symbol}.T"
    return yf.Ticker(symbol)


def _get_current_price(ticker) -> float:
    """Get current stock price from ticker."""
    info = ticker.info
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    return float(price) if price else 0.0


def _days_to_years(expiry_str: str) -> float:
    """Convert expiry date string to years from now."""
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
    now = datetime.now()
    days = (expiry - now).days
    return max(days, 1) / 365.0


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def options_chain(
    symbol: str,
    expiry: Optional[str] = None,
) -> dict:
    """Fetch options chain (calls and puts) for a symbol.

    Args:
        symbol: Stock ticker code (e.g. "AAPL", "7203").
        expiry: Specific expiry date (YYYY-MM-DD). If None, uses nearest expiry.

    Returns:
        dict with:
        - symbol, expiry, current_price
        - available_expiries: list of all available expiry dates
        - calls: list of call options with strike, bid, ask, volume, OI, IV
        - puts: list of put options
        - summary: call/put count, total volume, total OI
    """
    def _run():
        ticker = _get_ticker(symbol)
        current_price = _get_current_price(ticker)

        # Get available expiry dates
        try:
            expiries = list(ticker.options)
        except Exception:
            return {"error": f"No options data available for {symbol}. Options are mainly available for US stocks."}

        if not expiries:
            return {"error": f"No options expiry dates found for {symbol}"}

        # Select expiry
        selected_expiry = expiry if expiry and expiry in expiries else expiries[0]

        # Fetch option chain
        opt = ticker.option_chain(selected_expiry)

        def _process_chain(chain_df: pd.DataFrame, opt_type: str) -> list[dict]:
            records = []
            for _, row in chain_df.iterrows():
                rec = {
                    "type": opt_type,
                    "strike": _round_val(float(row.get("strike", 0)), 2),
                    "last_price": _round_val(float(row.get("lastPrice", 0)), 2),
                    "bid": _round_val(float(row.get("bid", 0)), 2),
                    "ask": _round_val(float(row.get("ask", 0)), 2),
                    "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                    "open_interest": int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0,
                    "implied_volatility": _round_val(float(row.get("impliedVolatility", 0)) * 100, 2),
                    "in_the_money": bool(row.get("inTheMoney", False)),
                }
                records.append(rec)
            return records

        calls = _process_chain(opt.calls, "call")
        puts = _process_chain(opt.puts, "put")

        total_call_vol = sum(c["volume"] for c in calls)
        total_put_vol = sum(p["volume"] for p in puts)
        total_call_oi = sum(c["open_interest"] for c in calls)
        total_put_oi = sum(p["open_interest"] for p in puts)

        return {
            "symbol": symbol,
            "current_price": _round_val(current_price, 2),
            "expiry": selected_expiry,
            "available_expiries": expiries,
            "calls": calls,
            "puts": puts,
            "summary": {
                "total_calls": len(calls),
                "total_puts": len(puts),
                "total_call_volume": total_call_vol,
                "total_put_volume": total_put_vol,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "put_call_volume_ratio": _round_val(
                    total_put_vol / total_call_vol if total_call_vol > 0 else 0, 4
                ),
                "put_call_oi_ratio": _round_val(
                    total_put_oi / total_call_oi if total_call_oi > 0 else 0, 4
                ),
            },
        }

    return _safe_call(_run)


def options_greeks(
    symbol: str,
    expiry: Optional[str] = None,
    risk_free_rate: float = 0.05,
    option_type: str = "call",
) -> dict:
    """Calculate Black-Scholes Greeks for all options at a given expiry.

    Args:
        symbol: Stock ticker code.
        expiry: Expiry date (YYYY-MM-DD). If None, uses nearest expiry.
        risk_free_rate: Annual risk-free rate (default 0.05 = 5%).
        option_type: "call" or "put" (default "call").

    Returns:
        dict with:
        - symbol, expiry, current_price, option_type
        - options: list of {strike, market_price, IV, delta, gamma, theta, vega, rho, theoretical_price}
        - atm_greeks: Greeks for the at-the-money option
    """
    def _run():
        ticker = _get_ticker(symbol)
        current_price = _get_current_price(ticker)

        try:
            expiries = list(ticker.options)
        except Exception:
            return {"error": f"No options data available for {symbol}"}

        if not expiries:
            return {"error": f"No options expiry dates found for {symbol}"}

        selected_expiry = expiry if expiry and expiry in expiries else expiries[0]
        T = _days_to_years(selected_expiry)

        opt = ticker.option_chain(selected_expiry)
        chain = opt.calls if option_type == "call" else opt.puts

        options_list = []
        atm_greeks = None
        min_atm_diff = float("inf")

        for _, row in chain.iterrows():
            strike = float(row.get("strike", 0))
            iv = float(row.get("impliedVolatility", 0))
            market_price = float(row.get("lastPrice", 0))

            if iv <= 0 or strike <= 0:
                continue

            greeks = _black_scholes_greeks(current_price, strike, T, risk_free_rate, iv, option_type)

            option_data = {
                "strike": _round_val(strike, 2),
                "market_price": _round_val(market_price, 2),
                "implied_volatility_pct": _round_val(iv * 100, 2),
                **greeks,
            }
            options_list.append(option_data)

            # Track ATM option
            diff = abs(strike - current_price)
            if diff < min_atm_diff:
                min_atm_diff = diff
                atm_greeks = option_data

        return {
            "symbol": symbol,
            "current_price": _round_val(current_price, 2),
            "expiry": selected_expiry,
            "time_to_expiry_years": _round_val(T, 4),
            "risk_free_rate": risk_free_rate,
            "option_type": option_type,
            "options": options_list,
            "atm_greeks": atm_greeks,
        }

    return _safe_call(_run)


def options_iv_surface(
    symbol: str,
    max_expiries: int = 6,
) -> dict:
    """Build implied volatility surface data (strike × expiry).

    Returns IV data across multiple expiry dates and strike prices,
    useful for understanding volatility skew and term structure.

    Args:
        symbol: Stock ticker code.
        max_expiries: Maximum number of expiry dates to include (default 6).

    Returns:
        dict with:
        - symbol, current_price
        - expiries: list of expiry dates used
        - surface: list of {expiry, strike, iv_pct, moneyness, days_to_expiry}
        - skew_summary: IV skew metrics per expiry
        - term_structure: ATM IV across expiries
    """
    def _run():
        ticker = _get_ticker(symbol)
        current_price = _get_current_price(ticker)

        try:
            expiries = list(ticker.options)
        except Exception:
            return {"error": f"No options data available for {symbol}"}

        if not expiries:
            return {"error": f"No options expiry dates found for {symbol}"}

        selected_expiries = expiries[:max_expiries]

        surface_data = []
        term_structure = []
        skew_summary = []

        for exp in selected_expiries:
            try:
                opt = ticker.option_chain(exp)
            except Exception:
                continue

            days = max((datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days, 1)
            expiry_ivs = []

            for _, row in opt.calls.iterrows():
                strike = float(row.get("strike", 0))
                iv = float(row.get("impliedVolatility", 0))

                if iv <= 0 or strike <= 0:
                    continue

                moneyness = strike / current_price if current_price > 0 else 0

                surface_data.append({
                    "expiry": exp,
                    "strike": _round_val(strike, 2),
                    "iv_pct": _round_val(iv * 100, 2),
                    "moneyness": _round_val(moneyness, 4),
                    "days_to_expiry": days,
                })
                expiry_ivs.append((moneyness, iv))

            if expiry_ivs:
                ivs = [x[1] for x in expiry_ivs]
                atm_candidates = [(abs(m - 1.0), iv_val) for m, iv_val in expiry_ivs]
                atm_iv = min(atm_candidates, key=lambda x: x[0])[1]
                otm_puts = [iv_val for m, iv_val in expiry_ivs if m < 0.95]
                otm_calls = [iv_val for m, iv_val in expiry_ivs if m > 1.05]

                term_structure.append({
                    "expiry": exp,
                    "days_to_expiry": days,
                    "atm_iv_pct": _round_val(atm_iv * 100, 2),
                })

                skew_summary.append({
                    "expiry": exp,
                    "days_to_expiry": days,
                    "atm_iv_pct": _round_val(atm_iv * 100, 2),
                    "min_iv_pct": _round_val(min(ivs) * 100, 2),
                    "max_iv_pct": _round_val(max(ivs) * 100, 2),
                    "avg_otm_put_iv_pct": _round_val(
                        np.mean(otm_puts) * 100, 2
                    ) if otm_puts else None,
                    "avg_otm_call_iv_pct": _round_val(
                        np.mean(otm_calls) * 100, 2
                    ) if otm_calls else None,
                    "skew": _round_val(
                        (np.mean(otm_puts) - np.mean(otm_calls)) * 100, 2
                    ) if otm_puts and otm_calls else None,
                })

        return {
            "symbol": symbol,
            "current_price": _round_val(current_price, 2),
            "expiries": selected_expiries,
            "surface_points": len(surface_data),
            "surface": surface_data,
            "term_structure": term_structure,
            "skew_summary": skew_summary,
        }

    return _safe_call(_run)


def options_unusual_activity(
    symbol: str,
    volume_threshold: float = 2.0,
    expiry: Optional[str] = None,
) -> dict:
    """Detect unusual options activity (high volume relative to open interest).

    Unusual activity often signals institutional positioning or
    anticipated price moves.

    Args:
        symbol: Stock ticker code.
        volume_threshold: Volume/OI ratio threshold (default 2.0 = 2x normal).
        expiry: Specific expiry date. If None, checks nearest expiry.

    Returns:
        dict with:
        - symbol, expiry, current_price
        - unusual_calls: list of calls with high volume/OI
        - unusual_puts: list of puts with high volume/OI
        - alert_level: "HIGH", "MODERATE", or "LOW"
    """
    def _run():
        ticker = _get_ticker(symbol)
        current_price = _get_current_price(ticker)

        try:
            expiries = list(ticker.options)
        except Exception:
            return {"error": f"No options data available for {symbol}"}

        if not expiries:
            return {"error": f"No options expiry dates found for {symbol}"}

        selected_expiry = expiry if expiry and expiry in expiries else expiries[0]
        opt = ticker.option_chain(selected_expiry)

        def _find_unusual(chain_df, opt_type):
            unusual = []
            for _, row in chain_df.iterrows():
                vol = int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0
                oi = int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0

                if oi > 0 and vol > 0:
                    ratio = vol / oi
                    if ratio >= volume_threshold:
                        unusual.append({
                            "type": opt_type,
                            "strike": _round_val(float(row.get("strike", 0)), 2),
                            "volume": vol,
                            "open_interest": oi,
                            "volume_oi_ratio": _round_val(ratio, 2),
                            "implied_volatility_pct": _round_val(
                                float(row.get("impliedVolatility", 0)) * 100, 2
                            ),
                            "last_price": _round_val(float(row.get("lastPrice", 0)), 2),
                            "in_the_money": bool(row.get("inTheMoney", False)),
                        })
            unusual.sort(key=lambda x: x["volume_oi_ratio"], reverse=True)
            return unusual

        unusual_calls = _find_unusual(opt.calls, "call")
        unusual_puts = _find_unusual(opt.puts, "put")

        total_unusual = len(unusual_calls) + len(unusual_puts)
        if total_unusual >= 5:
            alert_level = "HIGH"
        elif total_unusual >= 2:
            alert_level = "MODERATE"
        else:
            alert_level = "LOW"

        return {
            "symbol": symbol,
            "current_price": _round_val(current_price, 2),
            "expiry": selected_expiry,
            "volume_threshold": volume_threshold,
            "unusual_calls": unusual_calls,
            "unusual_puts": unusual_puts,
            "total_unusual_options": total_unusual,
            "alert_level": alert_level,
        }

    return _safe_call(_run)


def options_put_call_ratio(
    symbol: str,
) -> dict:
    """Calculate put/call ratio across all expiries as sentiment indicator.

    P/C ratio > 1.0 = bearish sentiment (more puts), < 0.7 = bullish (more calls).

    Args:
        symbol: Stock ticker code.

    Returns:
        dict with:
        - volume_ratio: put/call volume ratio
        - oi_ratio: put/call open interest ratio
        - sentiment: "BEARISH", "NEUTRAL", or "BULLISH"
        - per_expiry: breakdown by expiry date
    """
    def _run():
        ticker = _get_ticker(symbol)
        current_price = _get_current_price(ticker)

        try:
            expiries = list(ticker.options)
        except Exception:
            return {"error": f"No options data available for {symbol}"}

        if not expiries:
            return {"error": f"No options expiry dates found for {symbol}"}

        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        per_expiry = []

        for exp in expiries[:8]:  # Limit to 8 nearest expiries
            try:
                opt = ticker.option_chain(exp)
            except Exception:
                continue

            call_vol = int(opt.calls["volume"].sum()) if "volume" in opt.calls else 0
            put_vol = int(opt.puts["volume"].sum()) if "volume" in opt.puts else 0
            call_oi = int(opt.calls["openInterest"].sum()) if "openInterest" in opt.calls else 0
            put_oi = int(opt.puts["openInterest"].sum()) if "openInterest" in opt.puts else 0

            total_call_vol += call_vol
            total_put_vol += put_vol
            total_call_oi += call_oi
            total_put_oi += put_oi

            per_expiry.append({
                "expiry": exp,
                "call_volume": call_vol,
                "put_volume": put_vol,
                "call_oi": call_oi,
                "put_oi": put_oi,
                "pc_volume_ratio": _round_val(put_vol / call_vol if call_vol > 0 else 0, 4),
                "pc_oi_ratio": _round_val(put_oi / call_oi if call_oi > 0 else 0, 4),
            })

        vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        if vol_ratio > 1.0:
            sentiment = "BEARISH"
        elif vol_ratio < 0.7:
            sentiment = "BULLISH"
        else:
            sentiment = "NEUTRAL"

        return {
            "symbol": symbol,
            "current_price": _round_val(current_price, 2),
            "volume_put_call_ratio": _round_val(vol_ratio, 4),
            "oi_put_call_ratio": _round_val(oi_ratio, 4),
            "sentiment": sentiment,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "per_expiry": per_expiry,
        }

    return _safe_call(_run)


def options_max_pain(
    symbol: str,
    expiry: Optional[str] = None,
) -> dict:
    """Calculate max pain strike price.

    Max pain is the strike price where option sellers (writers) face
    minimum total payout — a commonly watched level for price magnets.

    Args:
        symbol: Stock ticker code.
        expiry: Specific expiry date. If None, uses nearest expiry.

    Returns:
        dict with:
        - max_pain_strike: The strike price with minimum total payout
        - current_price, distance_pct: How far current price is from max pain
        - pain_by_strike: List of {strike, total_pain} for visualization
    """
    def _run():
        ticker = _get_ticker(symbol)
        current_price = _get_current_price(ticker)

        try:
            expiries = list(ticker.options)
        except Exception:
            return {"error": f"No options data available for {symbol}"}

        if not expiries:
            return {"error": f"No options expiry dates found for {symbol}"}

        selected_expiry = expiry if expiry and expiry in expiries else expiries[0]
        opt = ticker.option_chain(selected_expiry)

        # Build OI by strike
        call_oi = {}
        put_oi = {}

        for _, row in opt.calls.iterrows():
            strike = float(row.get("strike", 0))
            oi = int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0
            call_oi[strike] = oi

        for _, row in opt.puts.iterrows():
            strike = float(row.get("strike", 0))
            oi = int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0
            put_oi[strike] = oi

        all_strikes = sorted(set(list(call_oi.keys()) + list(put_oi.keys())))

        if not all_strikes:
            return {"error": "No strikes with open interest found"}

        # Calculate pain at each strike (total $ value of ITM options at expiry)
        pain_by_strike = []
        min_pain = float("inf")
        max_pain_strike = all_strikes[0]

        for test_price in all_strikes:
            total_pain = 0.0

            # Call pain: for each call strike < test_price, calls are ITM
            for strike, oi in call_oi.items():
                if test_price > strike:
                    total_pain += (test_price - strike) * oi

            # Put pain: for each put strike > test_price, puts are ITM
            for strike, oi in put_oi.items():
                if test_price < strike:
                    total_pain += (strike - test_price) * oi

            pain_by_strike.append({
                "strike": _round_val(test_price, 2),
                "total_pain": _round_val(total_pain, 0),
            })

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_price

        distance_pct = ((current_price - max_pain_strike) / max_pain_strike * 100) if max_pain_strike > 0 else 0

        return {
            "symbol": symbol,
            "current_price": _round_val(current_price, 2),
            "expiry": selected_expiry,
            "max_pain_strike": _round_val(max_pain_strike, 2),
            "distance_from_max_pain_pct": _round_val(distance_pct, 2),
            "pain_by_strike": pain_by_strike,
        }

    return _safe_call(_run)
