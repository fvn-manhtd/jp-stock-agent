"""Machine Learning Signal Generation Module.

Provides ML-based price prediction and feature importance analysis
using scikit-learn (Random Forest, Gradient Boosting) built on top
of the 24 existing TA indicators.

Public Functions:
- ml_predict: Predict probability of price increase in next N days
- ml_feature_importance: Rank TA indicators by predictive power
- ml_signal: Combined ML + TA signal (configurable weight blend)
- ml_batch_predict: Predict for multiple symbols in parallel

All functions return dict on success or {"error": str} on failure.

Dependencies: scikit-learn (sklearn) — installed lazily to keep base
install lightweight. If sklearn is missing, functions return a clear
error message telling the user to ``pip install scikit-learn``.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .core import _safe_call
from .ta import _get_ohlcv_df, _round_val

# ---------------------------------------------------------------------------
# Lazy sklearn import
# ---------------------------------------------------------------------------

def _check_sklearn():
    """Check if scikit-learn is available; return error dict if not."""
    try:
        import sklearn  # noqa: F401
        return None
    except ImportError:
        return {"error": "scikit-learn is required for ML features. Install with: pip install scikit-learn"}


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from OHLCV data using TA indicators.

    Creates ~30 features from price/volume data including:
    - Trend: SMA, EMA ratios, Ichimoku-like
    - Momentum: RSI, MACD, Stochastic, ROC, Williams %R
    - Volatility: ATR, Bollinger Band width, Keltner width
    - Volume: OBV slope, VWAP ratio, volume MA ratio
    - Price action: returns, high-low range, gap

    Returns DataFrame with feature columns added. NaN rows are NOT dropped
    so the caller can decide how to handle them.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Trend features ---
    for p in [5, 10, 20, 50]:
        sma = close.rolling(p).mean()
        feat[f"sma_ratio_{p}"] = close / sma

    for p in [9, 21]:
        ema = close.ewm(span=p, adjust=False).mean()
        feat[f"ema_ratio_{p}"] = close / ema

    # SMA cross signal (fast vs slow)
    feat["sma_cross_20_50"] = (close.rolling(20).mean() - close.rolling(50).mean()) / close

    # --- Momentum features ---
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_histogram"] = macd - macd_signal

    # Stochastic %K
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    feat["stoch_k"] = ((close - low_14) / (high_14 - low_14)) * 100

    # Rate of Change
    for p in [5, 10, 20]:
        feat[f"roc_{p}"] = ((close - close.shift(p)) / close.shift(p)) * 100

    # Williams %R
    feat["williams_r"] = ((high_14 - close) / (high_14 - low_14)) * -100

    # --- Volatility features ---
    # ATR normalized
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feat["atr_ratio"] = atr / close

    # Bollinger Band width
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_width"] = (4 * std20) / sma20  # Normalized width
    feat["bb_position"] = (close - (sma20 - 2 * std20)) / (4 * std20)  # Position in band [0,1]

    # Daily range
    feat["daily_range"] = (high - low) / close

    # --- Volume features ---
    vol_ma = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / vol_ma

    # OBV slope (normalized)
    obv = (np.sign(close.diff()) * volume).cumsum()
    feat["obv_slope_10"] = obv.diff(10) / (volume.rolling(10).mean() * 10)

    # VWAP ratio
    typical = (high + low + close) / 3
    vwap = (typical * volume).cumsum() / volume.cumsum()
    feat["vwap_ratio"] = close / vwap

    # --- Price action features ---
    feat["return_1d"] = close.pct_change(1)
    feat["return_5d"] = close.pct_change(5)
    feat["gap"] = (df["open"] - close.shift(1)) / close.shift(1)

    return feat


def _build_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Build binary target: 1 if price increases over next `horizon` days, else 0."""
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    return (future_return > 0).astype(int)


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def ml_predict(
    symbol: str,
    horizon: int = 5,
    model_type: str = "random_forest",
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: Optional[str] = None,
    lookback_days: int = 730,
) -> dict:
    """Predict probability of price increase in next N days.

    Uses historical TA features to train a classifier and predict
    the probability of a price increase over the given horizon.

    Args:
        symbol: Stock ticker code.
        horizon: Prediction horizon in trading days (default 5).
        model_type: "random_forest" or "gradient_boosting" (default "random_forest").
        start: Training start date (YYYY-MM-DD). Defaults to lookback_days ago.
        end: End date (YYYY-MM-DD). Defaults to today.
        source: Data source override.
        lookback_days: Days of historical data for training (default 730 = ~2 years).

    Returns:
        dict with:
        - symbol, horizon, model_type
        - probability_up: Probability of price increase (0.0 - 1.0)
        - probability_down: Probability of price decrease (0.0 - 1.0)
        - signal: "BUY", "SELL", or "HOLD"
        - confidence: "HIGH", "MEDIUM", or "LOW"
        - model_accuracy: Cross-validated accuracy on training data
        - training_samples: Number of samples used for training
        - top_features: Top 5 most important features
    """
    def _run():
        err = _check_sklearn()
        if err:
            return err

        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Fetch data
        end_date = end or datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        else:
            start_date = start

        df = _get_ohlcv_df(symbol, start=start_date, end=end_date, source=source)
        if isinstance(df, dict):
            return df
        if len(df) < 100:
            return {"error": f"Insufficient data for ML ({len(df)} rows, need >= 100)"}

        # Build features and target
        features = _build_features(df)
        target = _build_target(df, horizon)

        # Combine and drop NaN
        combined = features.copy()
        combined["_target"] = target
        combined = combined.dropna()

        if len(combined) < 60:
            return {"error": f"Insufficient valid samples after feature engineering ({len(combined)}, need >= 60)"}

        X = combined.drop(columns=["_target"])
        y = combined["_target"]

        # Split: all but last row for training, last row for prediction
        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        X_latest = X.iloc[[-1]]

        if len(X_train) < 50:
            return {"error": f"Insufficient training samples ({len(X_train)}, need >= 50)"}

        # Select model
        if model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, min_samples_leaf=10,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42,
                min_samples_leaf=10, n_jobs=-1,
            )

        # Cross-validated accuracy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
            cv_accuracy = float(cv_scores.mean())

        # Train final model on all training data
        model.fit(X_train, y_train)

        # Predict probability for latest data point
        prob = model.predict_proba(X_latest)[0]
        # prob[0] = P(down), prob[1] = P(up) — assuming classes are [0, 1]
        classes = list(model.classes_)
        if 1 in classes:
            prob_up = float(prob[classes.index(1)])
        else:
            prob_up = 0.5
        prob_down = 1.0 - prob_up

        # Generate signal
        if prob_up >= 0.65:
            signal = "BUY"
        elif prob_up <= 0.35:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence level
        max_prob = max(prob_up, prob_down)
        if max_prob >= 0.75:
            confidence = "HIGH"
        elif max_prob >= 0.60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Feature importance
        importances = model.feature_importances_
        feature_names = list(X.columns)
        feat_imp = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )
        top_features = [
            {"feature": name, "importance": _round_val(float(imp), 4)}
            for name, imp in feat_imp[:5]
        ]

        return {
            "symbol": symbol,
            "horizon_days": horizon,
            "model_type": model_type,
            "probability_up": _round_val(prob_up, 4),
            "probability_down": _round_val(prob_down, 4),
            "signal": signal,
            "confidence": confidence,
            "model_accuracy": _round_val(cv_accuracy, 4),
            "training_samples": len(X_train),
            "top_features": top_features,
            "prediction_date": (
                df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])
            ),
        }

    return _safe_call(_run)


def ml_feature_importance(
    symbol: str,
    horizon: int = 5,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: Optional[str] = None,
    lookback_days: int = 730,
) -> dict:
    """Rank all TA features by predictive power using Random Forest.

    Trains a Random Forest on historical data and returns feature importance
    rankings, helping identify which indicators matter most for a given stock.

    Args:
        symbol: Stock ticker code.
        horizon: Prediction horizon in days (default 5).
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        source: Data source override.
        lookback_days: Days of historical data (default 730).

    Returns:
        dict with:
        - symbol, horizon
        - features: list of {feature, importance, category} sorted by importance
        - category_importance: aggregated importance by category
        - model_accuracy: cross-validated accuracy
    """
    def _run():
        err = _check_sklearn()
        if err:
            return err

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        end_date = end or datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        else:
            start_date = start

        df = _get_ohlcv_df(symbol, start=start_date, end=end_date, source=source)
        if isinstance(df, dict):
            return df
        if len(df) < 100:
            return {"error": f"Insufficient data for feature analysis ({len(df)} rows, need >= 100)"}

        features = _build_features(df)
        target = _build_target(df, horizon)

        combined = features.copy()
        combined["_target"] = target
        combined = combined.dropna()

        if len(combined) < 60:
            return {"error": f"Insufficient valid samples ({len(combined)}, need >= 60)"}

        X = combined.drop(columns=["_target"])
        y = combined["_target"]

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42,
            min_samples_leaf=10, n_jobs=-1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

        model.fit(X, y)

        # Feature categorization
        categories = {
            "trend": ["sma_ratio_", "ema_ratio_", "sma_cross_"],
            "momentum": ["rsi_", "macd_", "stoch_", "roc_", "williams_"],
            "volatility": ["atr_", "bb_", "daily_range"],
            "volume": ["volume_ratio", "obv_", "vwap_"],
            "price_action": ["return_", "gap"],
        }

        def _categorize(name):
            for cat, prefixes in categories.items():
                for prefix in prefixes:
                    if name.startswith(prefix):
                        return cat
            return "other"

        importances = model.feature_importances_
        feature_names = list(X.columns)

        features_list = []
        category_totals = {}
        for name, imp in zip(feature_names, importances):
            cat = _categorize(name)
            features_list.append({
                "feature": name,
                "importance": _round_val(float(imp), 4),
                "category": cat,
            })
            category_totals[cat] = category_totals.get(cat, 0) + float(imp)

        features_list.sort(key=lambda x: x["importance"], reverse=True)

        # Normalize category importance
        total_imp = sum(category_totals.values())
        category_importance = {
            cat: _round_val(val / total_imp * 100, 2) if total_imp > 0 else 0
            for cat, val in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        }

        return {
            "symbol": symbol,
            "horizon_days": horizon,
            "model_accuracy": _round_val(float(cv_scores.mean()), 4),
            "features": features_list,
            "category_importance_pct": category_importance,
            "total_features": len(features_list),
        }

    return _safe_call(_run)


def ml_signal(
    symbol: str,
    horizon: int = 5,
    ml_weight: float = 0.5,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    """Combined ML + TA signal with configurable weighting.

    Blends the ML prediction probability with the multi-indicator TA score
    to produce a unified signal. Default weight: 50% ML + 50% TA.

    Args:
        symbol: Stock ticker code.
        horizon: ML prediction horizon in days (default 5).
        ml_weight: Weight for ML signal (0.0-1.0, default 0.5). TA weight = 1 - ml_weight.
        start: Start date for ML training.
        end: End date.
        source: Data source override.

    Returns:
        dict with:
        - ml_probability_up, ml_signal, ml_confidence
        - ta_score (-100 to +100), ta_signal
        - combined_score (-100 to +100), combined_signal
        - weights_used: {ml: float, ta: float}
    """
    def _run():
        from .ta import ta_multi_indicator

        ml_weight_clamped = max(0.0, min(1.0, ml_weight))
        ta_weight = 1.0 - ml_weight_clamped

        # Get ML prediction
        ml_result = ml_predict(symbol, horizon=horizon, start=start, end=end, source=source)
        if isinstance(ml_result, dict) and "error" in ml_result:
            return ml_result

        # Get TA multi-indicator signal
        ta_result = ta_multi_indicator(symbol, start=start, end=end, source=source)
        if isinstance(ta_result, dict) and "error" in ta_result:
            return ta_result

        # Extract scores
        ml_prob_up = ml_result.get("probability_up", 0.5)
        ml_score = (ml_prob_up - 0.5) * 200  # Convert 0-1 probability to -100 to +100

        ta_score = ta_result.get("overall_score", 0)

        # Combine
        combined_score = ml_weight_clamped * ml_score + ta_weight * ta_score
        combined_score = max(-100, min(100, combined_score))

        # Generate combined signal
        if combined_score >= 30:
            combined_signal = "BUY"
        elif combined_score <= -30:
            combined_signal = "SELL"
        else:
            combined_signal = "HOLD"

        return {
            "symbol": symbol,
            "ml_probability_up": ml_result.get("probability_up"),
            "ml_signal": ml_result.get("signal"),
            "ml_confidence": ml_result.get("confidence"),
            "ml_model_accuracy": ml_result.get("model_accuracy"),
            "ta_score": _round_val(ta_score, 2),
            "ta_signal": ta_result.get("signal"),
            "combined_score": _round_val(combined_score, 2),
            "combined_signal": combined_signal,
            "weights_used": {
                "ml": _round_val(ml_weight_clamped, 2),
                "ta": _round_val(ta_weight, 2),
            },
            "horizon_days": horizon,
        }

    return _safe_call(_run)


def ml_batch_predict(
    symbols: list[str],
    horizon: int = 5,
    model_type: str = "random_forest",
    source: Optional[str] = None,
) -> list[dict]:
    """Predict price direction for multiple symbols (parallel).

    Args:
        symbols: List of ticker codes.
        horizon: Prediction horizon in days (default 5).
        model_type: "random_forest" or "gradient_boosting".
        source: Data source override.

    Returns:
        list of prediction dicts, sorted by probability_up descending.
    """
    def _run():
        err = _check_sklearn()
        if err:
            return err

        results = []
        for sym in symbols:
            result = ml_predict(sym, horizon=horizon, model_type=model_type, source=source)
            if isinstance(result, dict) and "error" not in result:
                results.append(result)

        results.sort(key=lambda x: x.get("probability_up", 0), reverse=True)
        return results

    return _safe_call(_run)
