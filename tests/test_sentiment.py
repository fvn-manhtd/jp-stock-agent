"""
Tests for the sentiment analysis module.

Tests cover:
- sentiment_news: news sentiment analysis
- sentiment_market: batch sentiment for multiple stocks
- sentiment_combined: technical + sentiment signals
- sentiment_screen: filtering by sentiment threshold
- Keyword detection for positive/negative/neutral headlines
- Error handling and graceful degradation
"""

from unittest.mock import patch

from jpstock_agent import sentiment


class TestSentimentNews:
    """Tests for sentiment_news function."""

    def test_sentiment_news_returns_dict_with_expected_keys(self, sample_news):
        """Test that sentiment_news returns dict with all expected keys."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_news("7203")

            assert isinstance(result, dict)
            assert "symbol" in result
            assert "overall_sentiment_score" in result
            assert "sentiment_label" in result
            assert "headlines" in result
            assert "news_count" in result
            assert "positive_count" in result
            assert "negative_count" in result
            assert "neutral_count" in result

    def test_sentiment_news_score_range(self, sample_news):
        """Test that overall_sentiment_score is in valid range."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_news("7203")

            score = result["overall_sentiment_score"]
            assert -1.0 <= score <= 1.0

    def test_sentiment_news_positive_headline_scores_positive(self, sample_news):
        """Test that positive headline returns positive sentiment score."""
        positive_news = [
            {"title": "record profit growth"},
            {"title": "strong revenue surge"},
            {"title": "company beat expectations"},
        ]

        with patch("jpstock_agent.sentiment.company_news", return_value=positive_news):
            result = sentiment.sentiment_news("7203")

            assert result["overall_sentiment_score"] > 0
            assert result["sentiment_label"] in ["Bullish", "Very Bullish"]

    def test_sentiment_news_negative_headline_scores_negative(self, sample_news):
        """Test that negative headline returns negative sentiment score."""
        negative_news = [
            {"title": "stock decline warning"},
            {"title": "crash warning concerning"},
            {"title": "company losses bankruptcy"},
        ]

        with patch("jpstock_agent.sentiment.company_news", return_value=negative_news):
            result = sentiment.sentiment_news("7203")

            assert result["overall_sentiment_score"] < 0
            assert result["sentiment_label"] in ["Bearish", "Very Bearish"]

    def test_sentiment_news_neutral_headline_scores_neutral(self, sample_news):
        """Test that neutral headline returns neutral sentiment score."""
        neutral_news = [
            {"title": "regular meeting scheduled"},
            {"title": "quarterly conference announcement"},
            {"title": "new office location opening"},
        ]

        with patch("jpstock_agent.sentiment.company_news", return_value=neutral_news):
            result = sentiment.sentiment_news("7203")

            score = result["overall_sentiment_score"]
            assert -0.2 <= score <= 0.2
            assert result["sentiment_label"] == "Neutral"

    def test_sentiment_news_headlines_have_scores(self, sample_news):
        """Test that each headline has sentiment_score and sentiment_label."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_news("7203")

            for headline in result["headlines"]:
                assert "title" in headline
                assert "sentiment_score" in headline
                assert "sentiment_label" in headline
                # Sentiment label should be one of the valid values
                assert headline["sentiment_label"] in ["Positive", "Negative", "Neutral"]

    def test_sentiment_news_counts_sum_to_total(self, sample_news):
        """Test that positive + negative + neutral count = news_count."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_news("7203")

            total = (
                result["positive_count"]
                + result["negative_count"]
                + result["neutral_count"]
            )
            assert total == result["news_count"]

    def test_sentiment_news_empty_news_returns_neutral(self):
        """Test that empty news list returns neutral sentiment."""
        with patch("jpstock_agent.sentiment.company_news", return_value=[]):
            result = sentiment.sentiment_news("7203")

            assert result["overall_sentiment_score"] == 0.0
            assert result["sentiment_label"] == "Neutral"
            assert result["news_count"] == 0

    def test_sentiment_news_failed_news_fetch_graceful_degradation(self):
        """Test that failed news fetch returns neutral, not error."""
        with patch("jpstock_agent.sentiment.company_news") as mock_news:
            mock_news.return_value = {"error": "Failed to fetch"}

            result = sentiment.sentiment_news("7203")

            # Should gracefully degrade to neutral
            assert isinstance(result, dict)
            assert "error" not in result
            assert result["sentiment_label"] == "Neutral"
            assert result["overall_sentiment_score"] == 0.0


class TestSentimentMarket:
    """Tests for sentiment_market function."""

    def test_sentiment_market_returns_list(self, sample_news):
        """Test that sentiment_market returns list."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_market(["7203", "6758", "9984"])

            assert isinstance(result, list)
            assert len(result) == 3

    def test_sentiment_market_items_have_expected_structure(self, sample_news):
        """Test that each item in result has expected keys."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_market(["7203", "6758"])

            for item in result:
                assert "symbol" in item
                assert "sentiment_score" in item
                assert "sentiment_label" in item
                assert "news_count" in item
                assert "positive_count" in item
                assert "negative_count" in item

    def test_sentiment_market_sorted_by_score_descending(self, sample_news):
        """Test that results are sorted by sentiment_score descending."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_market(["7203", "6758", "9984"])

            scores = [item["sentiment_score"] for item in result]
            assert scores == sorted(scores, reverse=True)

    def test_sentiment_market_with_single_symbol(self, sample_news):
        """Test sentiment_market with single symbol."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_market(["7203"])

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["symbol"] == "7203"


class TestSentimentCombined:
    """Tests for sentiment_combined function."""

    def test_sentiment_combined_returns_dict_with_expected_keys(self, sample_news):
        """Test that sentiment_combined returns dict with all expected keys."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            with patch(
                "jpstock_agent.sentiment.ta_multi_indicator"
            ) as mock_ta:
                mock_ta.return_value = {
                    "signal_score": 50,
                    "overall_signal": "BUY",
                    "signals": [],
                }

                result = sentiment.sentiment_combined("7203")

                assert isinstance(result, dict)
                assert "symbol" in result
                assert "technical_score" in result
                assert "technical_signal" in result
                assert "sentiment_score" in result
                assert "sentiment_label" in result
                assert "combined_score" in result
                assert "combined_signal" in result

    def test_sentiment_combined_score_range(self, sample_news):
        """Test that combined_score is in -100 to 100 range."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            with patch(
                "jpstock_agent.sentiment.ta_multi_indicator"
            ) as mock_ta:
                mock_ta.return_value = {
                    "signal_score": 50,
                    "overall_signal": "BUY",
                    "signals": [],
                }

                result = sentiment.sentiment_combined("7203")

                combined = result["combined_score"]
                assert -100 <= combined <= 100

    def test_sentiment_combined_signal_strong_buy(self, sample_news):
        """Test that high combined score produces STRONG BUY signal."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            with patch(
                "jpstock_agent.sentiment.ta_multi_indicator"
            ) as mock_ta:
                # High technical score with positive sentiment
                mock_ta.return_value = {
                    "signal_score": 80,
                    "overall_signal": "STRONG BUY",
                    "signals": [],
                }

                result = sentiment.sentiment_combined("7203")

                # With high technical and positive sentiment, should be STRONG BUY or BUY
                assert result["combined_signal"] in ["STRONG BUY", "BUY"]

    def test_sentiment_combined_signal_strong_sell(self):
        """Test that low combined score produces STRONG SELL signal."""
        negative_news = [
            {"title": "company crash bankruptcy loss"},
        ]

        with patch("jpstock_agent.sentiment.company_news", return_value=negative_news):
            with patch(
                "jpstock_agent.sentiment.ta_multi_indicator"
            ) as mock_ta:
                # Low technical score with negative sentiment
                mock_ta.return_value = {
                    "signal_score": -80,
                    "overall_signal": "STRONG SELL",
                    "signals": [],
                }

                result = sentiment.sentiment_combined("7203")

                # With low technical and negative sentiment, should be STRONG SELL or SELL
                assert result["combined_signal"] in ["STRONG SELL", "SELL"]

    def test_sentiment_combined_includes_sentiment_summary(self, sample_news):
        """Test that sentiment_summary is included."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            with patch(
                "jpstock_agent.sentiment.ta_multi_indicator"
            ) as mock_ta:
                mock_ta.return_value = {
                    "signal_score": 0,
                    "overall_signal": "HOLD",
                    "signals": [],
                }

                result = sentiment.sentiment_combined("7203")

                assert "sentiment_summary" in result
                summary = result["sentiment_summary"]
                assert "overall_score" in summary
                assert "positive_headlines" in summary
                assert "negative_headlines" in summary
                assert "total_news" in summary


class TestSentimentScreen:
    """Tests for sentiment_screen function."""

    def test_sentiment_screen_returns_list(self, sample_news):
        """Test that sentiment_screen returns list."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_screen(["7203", "6758", "9984"], min_score=0.0)

            assert isinstance(result, list)

    def test_sentiment_screen_filters_by_min_score(self, sample_news):
        """Test that filtering by min_score works correctly."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_screen(
                ["7203", "6758", "9984"], min_score=0.2
            )

            # All results should have sentiment_score >= min_score (scaled to -100 to 100)
            min_score_scaled = 0.2 * 100  # 20
            for item in result:
                assert item["sentiment_score"] >= min_score_scaled

    def test_sentiment_screen_negative_min_score(self, sample_news):
        """Test sentiment_screen with negative min_score."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_screen(
                ["7203", "6758"], min_score=-0.5
            )

            # Should return items with score >= -50
            min_score_scaled = -0.5 * 100
            for item in result:
                assert item["sentiment_score"] >= min_score_scaled

    def test_sentiment_screen_sorted_descending(self, sample_news):
        """Test that screen results are sorted by sentiment_score descending."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result = sentiment.sentiment_screen(
                ["7203", "6758", "9984"], min_score=-1.0
            )

            if len(result) > 1:
                scores = [item["sentiment_score"] for item in result]
                assert scores == sorted(scores, reverse=True)

    def test_sentiment_screen_high_threshold_returns_fewer(self, sample_news):
        """Test that higher threshold returns fewer results."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            result_low = sentiment.sentiment_screen(
                ["7203", "6758", "9984"], min_score=-1.0
            )
            result_high = sentiment.sentiment_screen(
                ["7203", "6758", "9984"], min_score=0.5
            )

            # Higher threshold should return fewer or equal results
            assert len(result_high) <= len(result_low)


class TestSentimentKeywordDetection:
    """Tests for keyword detection in sentiment scoring."""

    def test_score_headline_positive_keywords(self):
        """Test that positive keywords are detected."""
        headlines = [
            "record profit growth",
            "strong revenue surge",
            "company beat expectations",
        ]

        for headline in headlines:
            score = sentiment._score_headline(headline)
            assert score > 0, f"Headline '{headline}' should have positive score"

    def test_score_headline_negative_keywords(self):
        """Test that negative keywords are detected."""
        headlines = [
            "stock decline warning",
            "crash warning concerning",
            "company losses bankruptcy",
        ]

        for headline in headlines:
            score = sentiment._score_headline(headline)
            assert score < 0, f"Headline '{headline}' should have negative score"

    def test_score_headline_neutral_keywords(self):
        """Test that neutral headlines score near zero."""
        headlines = [
            "regular meeting scheduled",
            "quarterly conference call",
            "office location opening",
        ]

        for headline in headlines:
            score = sentiment._score_headline(headline)
            assert -0.25 <= score <= 0.25, f"Headline '{headline}' should be neutral"

    def test_score_headline_mixed_sentiment(self):
        """Test headlines with mixed positive and negative keywords."""
        headline = "company growth but weak profit"
        score = sentiment._score_headline(headline)
        # With both positive and negative, should be closer to neutral
        assert -0.5 <= score <= 0.5

    def test_score_headline_range(self):
        """Test that _score_headline returns value in -1 to 1 range."""
        test_headlines = [
            "extremely profitable growth surge",
            "major loss crash bankruptcy",
            "regular meeting",
        ]

        for headline in test_headlines:
            score = sentiment._score_headline(headline)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for '{headline}'"


class TestSentimentIntegration:
    """Integration tests for sentiment functions."""

    def test_sentiment_workflow_news_to_screen(self, sample_news):
        """Test complete workflow: news -> market -> screen."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            # Get market sentiment
            market_result = sentiment.sentiment_market(["7203", "6758", "9984"])

            # Screen by positive threshold
            with patch("jpstock_agent.sentiment.sentiment_market", return_value=market_result):
                screen_result = sentiment.sentiment_screen(
                    ["7203", "6758", "9984"], min_score=0.0
                )

            # All screened items should have positive sentiment
            for item in screen_result:
                assert item["sentiment_score"] >= 0

    def test_sentiment_combined_with_technical_correlation(self, sample_news):
        """Test that combined signal reflects both technical and sentiment."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            with patch(
                "jpstock_agent.sentiment.ta_multi_indicator"
            ) as mock_ta:
                # Case 1: Both positive
                mock_ta.return_value = {
                    "signal_score": 70,
                    "overall_signal": "BUY",
                    "signals": [],
                }

                result_positive = sentiment.sentiment_combined("7203")
                assert result_positive["combined_score"] > 0

                # Case 2: Technical negative, sentiment positive
                mock_ta.return_value = {
                    "signal_score": -60,
                    "overall_signal": "SELL",
                    "signals": [],
                }

                result_negative_tech = sentiment.sentiment_combined("7203")
                # Combined should be lower but not as negative due to 30% sentiment weight
                assert result_negative_tech["combined_score"] < result_positive[
                    "combined_score"
                ]


class TestSentimentErrorHandling:
    """Tests for error handling in sentiment module."""

    def test_sentiment_news_error_handling(self):
        """Test that sentiment_news handles errors gracefully."""
        with patch("jpstock_agent.sentiment.company_news") as mock_news:
            mock_news.return_value = {"error": "Network error"}

            result = sentiment.sentiment_news("7203")

            # Should return dict with neutral sentiment, not error
            assert isinstance(result, dict)
            assert "error" not in result
            assert result["sentiment_label"] == "Neutral"

    def test_sentiment_combined_with_missing_ta_data(self, sample_news):
        """Test sentiment_combined when ta_multi_indicator fails."""
        with patch("jpstock_agent.sentiment.company_news", return_value=sample_news):
            with patch(
                "jpstock_agent.sentiment.ta_multi_indicator"
            ) as mock_ta:
                mock_ta.return_value = {"error": "Failed to calculate"}

                result = sentiment.sentiment_combined("7203")

                # Should return error
                assert isinstance(result, dict)
                assert "error" in result

    def test_sentiment_market_with_empty_list(self):
        """Test sentiment_market with empty symbols list."""
        result = sentiment.sentiment_market([])

        assert isinstance(result, list)
        assert len(result) == 0

    def test_sentiment_screen_with_empty_list(self):
        """Test sentiment_screen with empty symbols list."""
        result = sentiment.sentiment_screen([])

        assert isinstance(result, list)
        assert len(result) == 0
