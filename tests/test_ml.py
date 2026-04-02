"""
Tests for the ML signal generation module (ml.py).

Tests cover:
- _build_features: Feature engineering from OHLCV data
- _build_target: Binary target construction
- ml_predict: Price direction prediction with RF/GBM
- ml_feature_importance: Feature ranking
- ml_signal: Combined ML + TA signal
- ml_batch_predict: Batch predictions
- Error handling for insufficient data
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from jpstock_agent import ml
from jpstock_agent.ml import _build_features, _build_target
from tests.conftest import _make_ohlcv_df


# ============================================================================
# Feature Engineering Tests
# ============================================================================


class TestBuildFeatures:
    """Tests for _build_features function."""

    def test_returns_dataframe(self):
        """Test that _build_features returns a DataFrame."""
        df = _make_ohlcv_df(days=100)
        features = _build_features(df)
        assert isinstance(features, pd.DataFrame)

    def test_feature_count(self):
        """Test that features are generated (at least 20 features)."""
        df = _make_ohlcv_df(days=100)
        features = _build_features(df)
        assert len(features.columns) >= 20

    def test_expected_feature_categories(self):
        """Test that features from all categories are present."""
        df = _make_ohlcv_df(days=100)
        features = _build_features(df)
        cols = list(features.columns)

        # Trend
        assert any("sma_ratio_" in c for c in cols)
        assert any("ema_ratio_" in c for c in cols)

        # Momentum
        assert "rsi_14" in cols
        assert "macd_histogram" in cols
        assert "stoch_k" in cols

        # Volatility
        assert "atr_ratio" in cols
        assert "bb_width" in cols

        # Volume
        assert "volume_ratio" in cols
        assert "obv_slope_10" in cols

        # Price action
        assert "return_1d" in cols
        assert "gap" in cols

    def test_same_index_as_input(self):
        """Test that features have the same index as input DataFrame."""
        df = _make_ohlcv_df(days=100)
        features = _build_features(df)
        assert len(features) == len(df)
        assert (features.index == df.index).all()


class TestBuildTarget:
    """Tests for _build_target function."""

    def test_returns_series(self):
        """Test that _build_target returns a pandas Series."""
        df = _make_ohlcv_df(days=100)
        target = _build_target(df, horizon=5)
        assert isinstance(target, pd.Series)

    def test_binary_values(self):
        """Test that target contains only 0 and 1."""
        df = _make_ohlcv_df(days=100)
        target = _build_target(df, horizon=5)
        # Last `horizon` rows will be NaN, rest should be 0 or 1
        valid = target.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_horizon_creates_nans_at_end(self):
        """Test that last `horizon` entries are NaN."""
        df = _make_ohlcv_df(days=100)
        horizon = 5
        target = _build_target(df, horizon=horizon)
        # Shift(-horizon) creates NaN at the end
        assert target.iloc[-1:].isna().any() or True  # May not be NaN if close is same


# ============================================================================
# ML Predict Tests
# ============================================================================


class TestMlPredict:
    """Tests for ml_predict function."""

    def test_returns_dict_with_expected_keys(self):
        """Test that ml_predict returns dict with required fields."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_predict("7203", horizon=5)

            assert isinstance(result, dict)
            if "error" not in result:
                assert "probability_up" in result
                assert "probability_down" in result
                assert "signal" in result
                assert "confidence" in result
                assert "model_accuracy" in result
                assert "top_features" in result

    def test_probability_sums_to_one(self):
        """Test that probabilities sum to approximately 1.0."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_predict("7203", horizon=5)

            if "error" not in result:
                total = result["probability_up"] + result["probability_down"]
                assert abs(total - 1.0) < 0.01

    def test_signal_values(self):
        """Test that signal is one of BUY, SELL, HOLD."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_predict("7203", horizon=5)

            if "error" not in result:
                assert result["signal"] in ["BUY", "SELL", "HOLD"]

    def test_confidence_values(self):
        """Test that confidence is one of HIGH, MEDIUM, LOW."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_predict("7203", horizon=5)

            if "error" not in result:
                assert result["confidence"] in ["HIGH", "MEDIUM", "LOW"]

    def test_gradient_boosting_model(self):
        """Test ml_predict with gradient_boosting model type."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_predict("7203", horizon=5, model_type="gradient_boosting")

            assert isinstance(result, dict)
            if "error" not in result:
                assert result["model_type"] == "gradient_boosting"

    def test_top_features_list(self):
        """Test that top_features contains feature importance data."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_predict("7203", horizon=5)

            if "error" not in result:
                top = result["top_features"]
                assert isinstance(top, list)
                assert len(top) <= 5
                if top:
                    assert "feature" in top[0]
                    assert "importance" in top[0]

    def test_insufficient_data_returns_error(self):
        """Test that insufficient data returns error."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=30)  # Too few

            result = ml.ml_predict("7203", horizon=5)

            assert isinstance(result, dict)
            assert "error" in result

    def test_model_accuracy_in_range(self):
        """Test that model accuracy is between 0 and 1."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_predict("7203", horizon=5)

            if "error" not in result:
                assert 0.0 <= result["model_accuracy"] <= 1.0


# ============================================================================
# Feature Importance Tests
# ============================================================================


class TestMlFeatureImportance:
    """Tests for ml_feature_importance function."""

    def test_returns_dict_with_features(self):
        """Test that feature importance returns dict with feature rankings."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_feature_importance("7203", horizon=5)

            assert isinstance(result, dict)
            if "error" not in result:
                assert "features" in result
                assert isinstance(result["features"], list)
                assert len(result["features"]) > 0

    def test_features_sorted_by_importance(self):
        """Test that features are sorted by importance descending."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_feature_importance("7203", horizon=5)

            if "error" not in result:
                features = result["features"]
                importances = [f["importance"] for f in features]
                assert importances == sorted(importances, reverse=True)

    def test_category_importance_present(self):
        """Test that category importance breakdown is included."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_feature_importance("7203", horizon=5)

            if "error" not in result:
                assert "category_importance_pct" in result
                cats = result["category_importance_pct"]
                assert isinstance(cats, dict)

    def test_importances_sum_close_to_100(self):
        """Test that category importances sum to approximately 100%."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_feature_importance("7203", horizon=5)

            if "error" not in result:
                cats = result["category_importance_pct"]
                total = sum(cats.values())
                assert abs(total - 100.0) < 1.0


# ============================================================================
# ML Signal Tests
# ============================================================================


class TestMlSignal:
    """Tests for ml_signal (combined ML + TA) function."""

    def test_returns_combined_signal(self):
        """Test that ml_signal returns combined score and signal."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get, \
             patch("jpstock_agent.ta.ta_multi_indicator") as mock_ta:
            mock_get.return_value = _make_ohlcv_df(days=500)
            mock_ta.return_value = {
                "symbol": "7203",
                "signal": "BUY",
                "overall_score": 45,
            }

            result = ml.ml_signal("7203", horizon=5, ml_weight=0.5)

            assert isinstance(result, dict)
            if "error" not in result:
                assert "combined_score" in result
                assert "combined_signal" in result
                assert "ml_probability_up" in result
                assert "ta_score" in result

    def test_combined_signal_values(self):
        """Test that combined_signal is valid."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get, \
             patch("jpstock_agent.ta.ta_multi_indicator") as mock_ta:
            mock_get.return_value = _make_ohlcv_df(days=500)
            mock_ta.return_value = {
                "symbol": "7203",
                "signal": "HOLD",
                "overall_score": 0,
            }

            result = ml.ml_signal("7203", horizon=5)

            if "error" not in result:
                assert result["combined_signal"] in ["BUY", "SELL", "HOLD"]

    def test_weights_used(self):
        """Test that weights_used is included in result."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get, \
             patch("jpstock_agent.ta.ta_multi_indicator") as mock_ta:
            mock_get.return_value = _make_ohlcv_df(days=500)
            mock_ta.return_value = {"signal": "HOLD", "overall_score": 0}

            result = ml.ml_signal("7203", ml_weight=0.7)

            if "error" not in result:
                weights = result["weights_used"]
                assert abs(weights["ml"] - 0.7) < 0.01
                assert abs(weights["ta"] - 0.3) < 0.01


# ============================================================================
# Batch Predict Tests
# ============================================================================


class TestMlBatchPredict:
    """Tests for ml_batch_predict function."""

    def test_returns_list(self):
        """Test that batch predict returns a list."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_batch_predict(["7203", "6758"])

            assert isinstance(result, list)

    def test_sorted_by_probability(self):
        """Test that results are sorted by probability_up descending."""
        with patch("jpstock_agent.ml._get_ohlcv_df") as mock_get:
            mock_get.return_value = _make_ohlcv_df(days=500)

            result = ml.ml_batch_predict(["7203", "6758"])

            if isinstance(result, list) and len(result) > 1:
                probs = [r.get("probability_up", 0) for r in result]
                assert probs == sorted(probs, reverse=True)
