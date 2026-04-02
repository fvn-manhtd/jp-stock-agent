"""Shared fixtures for jpstock-agent tests.

Provides mock data fixtures to avoid hitting real APIs during testing.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Sample OHLCV data (60 days of realistic stock data)
# ---------------------------------------------------------------------------

def _make_ohlcv_df(days=60, base_price=2500.0, seed=42):
    """Generate realistic OHLCV DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B", tz="UTC")
    prices = [base_price]
    for _ in range(days - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        "high": prices * (1 + np.random.uniform(0.005, 0.02, days)),
        "low": prices * (1 - np.random.uniform(0.005, 0.02, days)),
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, days).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


def _make_ohlcv_records(days=60, base_price=2500.0, seed=42):
    """Generate OHLCV as list[dict] (what core.stock_history returns)."""
    df = _make_ohlcv_df(days, base_price, seed)
    df = df.reset_index()
    df["date"] = df["date"].astype(str)
    return df.to_dict("records")


@pytest.fixture
def sample_ohlcv_df():
    """60-day OHLCV DataFrame with DatetimeIndex."""
    return _make_ohlcv_df()


@pytest.fixture
def sample_ohlcv_records():
    """60-day OHLCV as list[dict]."""
    return _make_ohlcv_records()


@pytest.fixture
def sample_news():
    """Sample news data as returned by core.company_news."""
    return [
        {"title": "Toyota reports record profit growth for Q4", "published": "2026-03-30"},
        {"title": "Company announces strong revenue surge", "published": "2026-03-29"},
        {"title": "Stock price decline amid market concerns", "published": "2026-03-28"},
        {"title": "New partnership deal announced", "published": "2026-03-27"},
        {"title": "Analyst downgrade warning on weak outlook", "published": "2026-03-26"},
        {"title": "Quarterly earnings beat expectations", "published": "2026-03-25"},
        {"title": "CEO discusses expansion plans at conference", "published": "2026-03-24"},
        {"title": "Industry faces recession risk concerns", "published": "2026-03-23"},
        {"title": "Company buyback program approved", "published": "2026-03-22"},
        {"title": "Regular trading day with no significant news", "published": "2026-03-21"},
    ]


@pytest.fixture
def mock_stock_history(sample_ohlcv_records):
    """Patch core.stock_history to return sample data."""
    with patch("jpstock_agent.core.stock_history", return_value=sample_ohlcv_records):
        yield sample_ohlcv_records


@pytest.fixture
def mock_get_ohlcv_df(sample_ohlcv_df):
    """Patch ta._get_ohlcv_df to return sample DataFrame."""
    with patch("jpstock_agent.ta._get_ohlcv_df", return_value=sample_ohlcv_df):
        yield sample_ohlcv_df
