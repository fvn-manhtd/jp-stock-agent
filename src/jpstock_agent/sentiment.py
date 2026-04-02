"""Sentiment Analysis module for jpstock-agent.

Provides keyword-based sentiment analysis for stock market news,
combining sentiment signals with technical analysis for trading decisions.

Uses simple keyword matching (no external NLP libraries) to analyze
news headlines for positive/negative sentiment across Japanese and
Vietnamese stocks.

Every public function returns ``dict`` or ``list[dict]`` on success,
or ``{"error": str}`` on failure.
"""

from __future__ import annotations

from typing import Any

from .core import _safe_call, company_news
from .ta import _round_val, ta_multi_indicator

# ---------------------------------------------------------------------------
# Sentiment Analysis Keywords
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS_EN = {
    "profit", "growth", "surge", "gain", "beat", "upgrade", "bullish",
    "record", "strong", "rise", "high", "up", "positive", "exceed",
    "outperform", "rally", "boom", "dividend", "buyback", "expand",
    "improve", "optimistic", "recovery", "strength", "upside", "boom",
    "exceptional", "excellent", "outstanding", "soar", "breakthrough",
}

NEGATIVE_KEYWORDS_EN = {
    "loss", "decline", "drop", "fall", "miss", "downgrade", "bearish",
    "weak", "low", "down", "negative", "cut", "layoff", "lawsuit",
    "debt", "warning", "crash", "recession", "risk", "concern", "sell",
    "pessimistic", "default", "bankruptcy", "weakness", "downside",
    "slump", "plunge", "tumble", "collapse", "distress",
}

# Japanese keywords for sentiment
POSITIVE_KEYWORDS_JP = {
    "上昇", "増益", "好調", "最高", "成長", "増配", "回復", "強い", "買い",
    "高い", "上げ", "好況", "躍進", "拡大", "堅調", "期待", "買われる",
}

NEGATIVE_KEYWORDS_JP = {
    "下落", "減益", "低迷", "損失", "赤字", "減配", "悪化", "弱い", "売り",
    "低い", "下げ", "不況", "停滞", "縮小", "軟調", "懸念", "売られる",
}


# ---------------------------------------------------------------------------
# Sentiment Analysis Functions
# ---------------------------------------------------------------------------


def sentiment_news(symbol: str, source: str | None = None) -> dict:
    """Analyze sentiment from recent news headlines for a stock.

    Fetches news using `core.company_news()` and scores each headline
    using keyword matching (English and Japanese keywords).

    Parameters
    ----------
    symbol : str
        Stock ticker code, e.g. "7203" (Toyota).
    source : str, optional
        Data source ("yfinance", "jquants", "vnstocks").

    Returns
    -------
    dict
        Contains:
        - symbol : str
        - news_count : int (number of headlines analyzed)
        - overall_sentiment_score : float (-1.0 to +1.0)
        - sentiment_label : str ("Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish")
        - positive_count : int (number of positive headlines)
        - negative_count : int (number of negative headlines)
        - neutral_count : int (number of neutral headlines)
        - headlines : list[dict] with {title, sentiment_score, sentiment_label, published}
    """
    # Fetch news
    news_result = _safe_call(company_news, symbol, source)
    if isinstance(news_result, dict) and "error" in news_result:
        # Gracefully handle news fetch failure - return neutral sentiment
        return {
            "symbol": symbol,
            "news_count": 0,
            "overall_sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "headlines": [],
        }

    if not news_result:
        return {
            "symbol": symbol,
            "news_count": 0,
            "overall_sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "headlines": [],
        }

    # Analyze each headline
    headlines = []
    sentiment_scores = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for news_item in news_result:
        title = news_item.get("title", "")
        published = news_item.get("providerPublishTime") or news_item.get("published")

        # Score sentiment for this headline
        score = _score_headline(title)
        sentiment_scores.append(score)

        # Classify headline
        if score > 0.15:
            label = "Positive"
            positive_count += 1
        elif score < -0.15:
            label = "Negative"
            negative_count += 1
        else:
            label = "Neutral"
            neutral_count += 1

        headlines.append({
            "title": title,
            "sentiment_score": _round_val(score, decimals=3),
            "sentiment_label": label,
            "published": published,
        })

    # Calculate overall sentiment
    if sentiment_scores:
        overall_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
    else:
        overall_sentiment_score = 0.0

    # Determine overall sentiment label
    if overall_sentiment_score > 0.5:
        sentiment_label = "Very Bullish"
    elif overall_sentiment_score > 0.15:
        sentiment_label = "Bullish"
    elif overall_sentiment_score < -0.5:
        sentiment_label = "Very Bearish"
    elif overall_sentiment_score < -0.15:
        sentiment_label = "Bearish"
    else:
        sentiment_label = "Neutral"

    return {
        "symbol": symbol,
        "news_count": len(headlines),
        "overall_sentiment_score": _round_val(overall_sentiment_score, decimals=3),
        "sentiment_label": sentiment_label,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "headlines": headlines,
    }


def sentiment_market(symbols: list[str], source: str | None = None) -> list[dict]:
    """Batch sentiment analysis for multiple stocks.

    Runs sentiment_news() for each symbol and returns results
    sorted by overall_sentiment_score descending.

    Parameters
    ----------
    symbols : list[str]
        List of ticker codes.
    source : str, optional
        Data source.

    Returns
    -------
    list[dict]
        Each item: {symbol, sentiment_score, sentiment_label, news_count, positive_count, negative_count}
    """
    results = []
    for symbol in symbols:
        result = sentiment_news(symbol, source)
        # Extract summary info only (not full headlines)
        results.append({
            "symbol": result["symbol"],
            "sentiment_score": result["overall_sentiment_score"],
            "sentiment_label": result["sentiment_label"],
            "news_count": result["news_count"],
            "positive_count": result["positive_count"],
            "negative_count": result["negative_count"],
        })

    # Sort by sentiment score descending
    results.sort(key=lambda x: x["sentiment_score"], reverse=True)
    return results


def sentiment_combined(symbol: str, source: str | None = None) -> dict:
    """Combined technical + sentiment signal.

    Fetches news sentiment and technical indicators, then combines
    them with weighted average (70% technical, 30% sentiment).

    Parameters
    ----------
    symbol : str
        Stock ticker code.
    source : str, optional
        Data source.

    Returns
    -------
    dict
        Contains:
        - symbol : str
        - technical_score : float (-100 to +100) from ta_multi_indicator
        - technical_signal : str ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")
        - technical_summary : list of signal descriptions
        - sentiment_score : float (-100 to +100, scaled from -1 to +1)
        - sentiment_label : str
        - sentiment_summary : dict with counts and overall score
        - combined_score : float (-100 to +100, weighted average)
        - combined_signal : str ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")
    """
    # Fetch sentiment
    sentiment_result = sentiment_news(symbol, source)
    if isinstance(sentiment_result, dict) and "error" in sentiment_result:
        return sentiment_result

    # Fetch technical analysis
    technical_result = _safe_call(ta_multi_indicator, symbol, source=source)
    if isinstance(technical_result, dict) and "error" in technical_result:
        return technical_result

    # Extract values
    sentiment_raw = sentiment_result["overall_sentiment_score"]  # -1 to +1
    sentiment_scaled = sentiment_raw * 100  # Scale to -100 to +100
    technical_score = technical_result.get("signal_score", 0)

    # Calculate combined score (70% technical, 30% sentiment)
    combined_score = (technical_score * 0.7) + (sentiment_scaled * 0.3)
    combined_score = max(-100, min(100, combined_score))

    # Determine combined signal
    if combined_score >= 40:
        combined_signal = "STRONG BUY"
    elif combined_score >= 15:
        combined_signal = "BUY"
    elif combined_score <= -40:
        combined_signal = "STRONG SELL"
    elif combined_score <= -15:
        combined_signal = "SELL"
    else:
        combined_signal = "HOLD"

    return {
        "symbol": symbol,
        "technical_score": _round_val(technical_score, decimals=2),
        "technical_signal": technical_result.get("overall_signal", "HOLD"),
        "technical_summary": technical_result.get("signals", []),
        "sentiment_score": _round_val(sentiment_scaled, decimals=2),
        "sentiment_label": sentiment_result["sentiment_label"],
        "sentiment_summary": {
            "overall_score": _round_val(sentiment_raw, decimals=3),
            "positive_headlines": sentiment_result["positive_count"],
            "negative_headlines": sentiment_result["negative_count"],
            "neutral_headlines": sentiment_result["neutral_count"],
            "total_news": sentiment_result["news_count"],
        },
        "combined_score": _round_val(combined_score, decimals=2),
        "combined_signal": combined_signal,
    }


def sentiment_screen(symbols: list[str], min_score: float = 0.0, source: str | None = None) -> list[dict]:
    """Screen stocks by sentiment.

    Returns only stocks with sentiment_score >= min_score,
    sorted by sentiment_score descending.

    Parameters
    ----------
    symbols : list[str]
        List of ticker codes.
    min_score : float, optional
        Minimum sentiment score threshold (-1.0 to +1.0).
    source : str, optional
        Data source.

    Returns
    -------
    list[dict]
        Filtered and sorted results from sentiment_market().
    """
    results = sentiment_market(symbols, source)
    # Filter by min_score (note: sentiment_score in results is -100 to +100 scale)
    min_score_scaled = min_score * 100
    filtered = [r for r in results if r["sentiment_score"] >= min_score_scaled]
    return filtered


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _score_headline(headline: str) -> float:
    """Score a headline for sentiment.

    Uses keyword matching in English and Japanese.
    Returns normalized score from -1.0 to +1.0.

    Parameters
    ----------
    headline : str
        News headline text.

    Returns
    -------
    float
        Sentiment score (-1.0 to +1.0).
    """
    headline_lower = headline.lower()

    # Count English keywords
    pos_count_en = sum(1 for kw in POSITIVE_KEYWORDS_EN if kw in headline_lower)
    neg_count_en = sum(1 for kw in NEGATIVE_KEYWORDS_EN if kw in headline_lower)

    # Count Japanese keywords
    pos_count_jp = sum(1 for kw in POSITIVE_KEYWORDS_JP if kw in headline)
    neg_count_jp = sum(1 for kw in NEGATIVE_KEYWORDS_JP if kw in headline)

    # Combine counts
    pos_count = pos_count_en + pos_count_jp
    neg_count = neg_count_en + neg_count_jp
    total_count = pos_count + neg_count

    # Calculate normalized score
    if total_count == 0:
        return 0.0

    score = (pos_count - neg_count) / total_count
    # Clamp to -1.0 to +1.0 range
    return max(-1.0, min(1.0, score))
