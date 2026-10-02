"""Unit tests for the deterministic stock sentiment tool."""

from typing import Any
from unittest.mock import patch

import pytest

from tools.sentiment_tool import (
    calculate_technical_signals,
    compute_sentiment_score,
    determine_sentiment_label,
    get_stock_sentiment,
)


def _generate_artificial_history(
    prices: list[float], volumes: list[int] | None = None
) -> list[dict[str, Any]]:
    """Helper to construct deterministic historical price bars."""
    bars: list[dict[str, Any]] = []
    for i, p in enumerate(prices):
        vol = volumes[i] if volumes and i < len(volumes) else 1000000
        bars.append(
            {
                "date": f"2026-01-{i + 1:02d}",
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "volume": vol,
            }
        )
    return bars


class TestDeterministicSentiment:
    """Test suite for deterministic sentiment analysis using fixed artificial price histories."""

    def test_bullish_data(self) -> None:
        """Test with steady upward trend: price > 20 DMA > 50 DMA, positive returns."""
        # 60 days climbing from 100.0 to 159.0 (+1.0 per day)
        prices = [100.0 + i for i in range(60)]
        volumes = [1000000] * 59 + [2500000]  # Surge on the last day
        history = _generate_artificial_history(prices, volumes)

        raw, structured = calculate_technical_signals(history)
        score = compute_sentiment_score(raw)
        sentiment = determine_sentiment_label(score)

        assert raw["return_1d"] > 0.5
        assert raw["return_5d"] > 1.0
        assert raw["dma_20"] is not None
        assert raw["dma_50"] is not None
        assert raw["price_vs_20_dma"] > 0
        assert raw["price_vs_50_dma"] > 0
        assert raw["dma_20"] > raw["dma_50"]

        assert score > 50.0
        assert sentiment == "Bullish"

        # End-to-end with provider injection
        res = get_stock_sentiment("TCS", history_provider=lambda s, period: history)
        assert res["symbol"] == "TCS"
        assert res["sentiment"] == "Bullish"
        assert res["score"] == score
        assert len(res["signals"]) == 8

    def test_bearish_data(self) -> None:
        """Test with steady downward trend: price < 20 DMA < 50 DMA, negative returns."""
        # 60 days descending from 160.0 to 101.0 (-1.0 per day)
        prices = [160.0 - i for i in range(60)]
        volumes = [1000000] * 59 + [2500000]  # Surge volume on down day
        history = _generate_artificial_history(prices, volumes)

        raw, structured = calculate_technical_signals(history)
        score = compute_sentiment_score(raw)
        sentiment = determine_sentiment_label(score)

        assert raw["return_1d"] < -0.5
        assert raw["return_5d"] < -1.0
        assert raw["price_vs_20_dma"] < 0
        assert raw["price_vs_50_dma"] < 0
        assert raw["dma_20"] < raw["dma_50"]

        assert score < -50.0
        assert sentiment == "Bearish"

        res = get_stock_sentiment("TCS", history_provider=lambda s, period: history)
        assert res["symbol"] == "TCS"
        assert res["sentiment"] == "Bearish"
        assert res["score"] == score

    def test_neutral_data(self) -> None:
        """Test with flat/oscillating prices around moving averages."""
        # 60 days alternating slightly around 100.0 (negligible return)
        prices = [100.0 if i % 2 == 0 else 100.05 for i in range(60)]
        history = _generate_artificial_history(prices)

        raw, structured = calculate_technical_signals(history)
        score = compute_sentiment_score(raw)
        sentiment = determine_sentiment_label(score)

        assert -20.0 <= score <= 20.0
        assert sentiment == "Neutral"

        res = get_stock_sentiment("TCS", history_provider=lambda s, period: history)
        assert res["sentiment"] == "Neutral"

    def test_insufficient_history(self) -> None:
        """Test with only 3 bars of history where DMAs cannot be computed."""
        prices = [100.0, 101.0, 102.0]
        history = _generate_artificial_history(prices)

        raw, structured = calculate_technical_signals(history)
        score = compute_sentiment_score(raw)
        sentiment = determine_sentiment_label(score)

        assert raw["dma_20"] is None
        assert raw["dma_50"] is None
        assert raw["price_vs_20_dma"] is None
        assert raw["price_vs_50_dma"] is None
        assert raw["return_1d"] is not None

        res = get_stock_sentiment("TCS", history_provider=lambda s, period: history)
        assert res["symbol"] == "TCS"
        assert isinstance(res["score"], (int, float))
        assert res["sentiment"] in ("Bullish", "Neutral", "Bearish")

    def test_missing_volume(self) -> None:
        """Test history where volume is missing or None across all bars."""
        bars: list[dict[str, Any]] = []
        for i in range(60):
            p = 100.0 + i
            bars.append(
                {
                    "date": f"2026-01-{i + 1:02d}",
                    "open": p,
                    "high": p + 1.0,
                    "low": p - 1.0,
                    "close": p,
                    "volume": None,
                }
            )

        raw, structured = calculate_technical_signals(bars)
        assert raw["current_volume"] is None
        assert raw["average_volume"] is None

        score = compute_sentiment_score(raw)
        assert isinstance(score, float)

        res = get_stock_sentiment("TCS", history_provider=lambda s, period: bars)
        assert res["symbol"] == "TCS"
        assert res["sentiment"] == "Bullish"

    def test_data_provider_failure(self) -> None:
        """Test graceful fallback when price history provider raises an exception."""
        def failing_provider(symbol: str, period: str):
            raise Exception("Remote provider failure")

        res = get_stock_sentiment("TCS", history_provider=failing_provider)
        assert res["symbol"] == "TCS"
        assert res["score"] == 0.0
        assert res["sentiment"] == "Neutral"
        assert res["signals"] == []

    def test_empty_history(self) -> None:
        """Test graceful handling when provider returns an empty list."""
        res = get_stock_sentiment("TCS", history_provider=lambda s, period: [])
        assert res["symbol"] == "TCS"
        assert res["score"] == 0.0
        assert res["sentiment"] == "Neutral"
        assert res["signals"] == []

    def test_invalid_symbol(self) -> None:
        """Test invalid symbol inputs return safe default dictionary."""
        res = get_stock_sentiment("INVALID$$$")
        assert res["symbol"] is None
        assert res["score"] == 0.0
        assert res["sentiment"] == "Neutral"
        assert res["signals"] == []
