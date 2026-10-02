"""Deterministic technical sentiment analysis tool for NSE equities."""

import logging
from typing import Any, Callable, Optional

from tools.market_tool import get_price_history
from utils.helpers import is_valid_nse_symbol, normalize_nse_symbol

logger = logging.getLogger(__name__)


def calculate_technical_signals(
    history: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract and calculate technical signals from historical OHLCV data.

    Returns:
        Tuple of (raw_signals_dict, structured_signals_list).
    """
    closes = [
        float(bar["close"])
        for bar in history
        if bar.get("close") is not None
    ]
    volumes = [
        float(bar["volume"])
        for bar in history
        if bar.get("volume") is not None and bar.get("volume") > 0
    ]

    n = len(closes)
    raw: dict[str, Any] = {
        "current_price": None,
        "return_1d": None,
        "return_5d": None,
        "dma_20": None,
        "dma_50": None,
        "price_vs_20_dma": None,
        "price_vs_50_dma": None,
        "current_volume": None,
        "average_volume": None,
    }

    if n == 0:
        return raw, []

    current_price = closes[-1]
    raw["current_price"] = current_price

    # 1-day return
    if n >= 2 and closes[-2] > 0:
        raw["return_1d"] = round(((closes[-1] - closes[-2]) / closes[-2]) * 100, 2)

    # 5-day return
    if n >= 6 and closes[-6] > 0:
        raw["return_5d"] = round(((closes[-1] - closes[-6]) / closes[-6]) * 100, 2)
    elif n >= 5 and closes[0] > 0:
        raw["return_5d"] = round(((closes[-1] - closes[0]) / closes[0]) * 100, 2)

    # 20 DMA
    if n >= 20:
        dma_20 = round(sum(closes[-20:]) / 20.0, 2)
        raw["dma_20"] = dma_20
        if dma_20 > 0:
            raw["price_vs_20_dma"] = round(
                ((current_price - dma_20) / dma_20) * 100, 2
            )

    # 50 DMA
    if n >= 50:
        dma_50 = round(sum(closes[-50:]) / 50.0, 2)
        raw["dma_50"] = dma_50
        if dma_50 > 0:
            raw["price_vs_50_dma"] = round(
                ((current_price - dma_50) / dma_50) * 100, 2
            )

    # Volumes
    if len(volumes) >= 1:
        raw["current_volume"] = int(volumes[-1])

    if len(volumes) >= 20:
        raw["average_volume"] = int(round(sum(volumes[-20:]) / 20.0))
    elif len(volumes) >= 5:
        raw["average_volume"] = int(round(sum(volumes) / len(volumes)))

    # Construct structured signal list
    structured = [
        {
            "name": "1-day return",
            "value": raw["return_1d"],
            "unit": "%",
            "status": (
                "bullish"
                if raw["return_1d"] is not None and raw["return_1d"] > 0
                else "bearish"
                if raw["return_1d"] is not None and raw["return_1d"] < 0
                else "neutral"
            ),
        },
        {
            "name": "5-day return",
            "value": raw["return_5d"],
            "unit": "%",
            "status": (
                "bullish"
                if raw["return_5d"] is not None and raw["return_5d"] > 0
                else "bearish"
                if raw["return_5d"] is not None and raw["return_5d"] < 0
                else "neutral"
            ),
        },
        {
            "name": "20 DMA",
            "value": raw["dma_20"],
            "unit": "INR",
            "status": "available" if raw["dma_20"] is not None else "insufficient_data",
        },
        {
            "name": "50 DMA",
            "value": raw["dma_50"],
            "unit": "INR",
            "status": "available" if raw["dma_50"] is not None else "insufficient_data",
        },
        {
            "name": "price vs 20 DMA",
            "value": raw["price_vs_20_dma"],
            "unit": "%",
            "status": (
                "bullish"
                if raw["price_vs_20_dma"] is not None and raw["price_vs_20_dma"] > 0
                else "bearish"
                if raw["price_vs_20_dma"] is not None and raw["price_vs_20_dma"] < 0
                else "neutral"
            ),
        },
        {
            "name": "price vs 50 DMA",
            "value": raw["price_vs_50_dma"],
            "unit": "%",
            "status": (
                "bullish"
                if raw["price_vs_50_dma"] is not None and raw["price_vs_50_dma"] > 0
                else "bearish"
                if raw["price_vs_50_dma"] is not None and raw["price_vs_50_dma"] < 0
                else "neutral"
            ),
        },
        {
            "name": "current volume",
            "value": raw["current_volume"],
            "unit": "shares",
            "status": "available" if raw["current_volume"] is not None else "missing",
        },
        {
            "name": "average volume",
            "value": raw["average_volume"],
            "unit": "shares",
            "status": "available" if raw["average_volume"] is not None else "missing",
        },
    ]

    return raw, structured


def compute_sentiment_score(raw_signals: dict[str, Any]) -> float:
    """Compute deterministic sentiment score between -100.0 and +100.0 based on technical signals.

    Independent from natural language explanations.
    """
    points = 0.0
    evaluated_weight = 0.0

    # 1. Price vs 20 DMA (Weight: 25.0)
    p_vs_20 = raw_signals.get("price_vs_20_dma")
    if p_vs_20 is not None:
        evaluated_weight += 25.0
        if p_vs_20 > 0.5:
            points += 25.0
        elif p_vs_20 < -0.5:
            points -= 25.0

    # 2. Price vs 50 DMA (Weight: 25.0)
    p_vs_50 = raw_signals.get("price_vs_50_dma")
    if p_vs_50 is not None:
        evaluated_weight += 25.0
        if p_vs_50 > 0.5:
            points += 25.0
        elif p_vs_50 < -0.5:
            points -= 25.0

    # 3. 20 DMA vs 50 DMA trend alignment (Weight: 20.0)
    dma_20 = raw_signals.get("dma_20")
    dma_50 = raw_signals.get("dma_50")
    if dma_20 is not None and dma_50 is not None and dma_50 > 0:
        evaluated_weight += 20.0
        diff_pct = ((dma_20 - dma_50) / dma_50) * 100.0
        if diff_pct > 0.25:
            points += 20.0
        elif diff_pct < -0.25:
            points -= 20.0

    # 4. 5-day Return (Weight: 15.0)
    ret_5d = raw_signals.get("return_5d")
    if ret_5d is not None:
        evaluated_weight += 15.0
        if ret_5d > 1.0:
            points += 15.0
        elif ret_5d < -1.0:
            points -= 15.0

    # 5. 1-day Return (Weight: 15.0)
    ret_1d = raw_signals.get("return_1d")
    if ret_1d is not None:
        evaluated_weight += 15.0
        if ret_1d > 0.5:
            points += 15.0
        elif ret_1d < -0.5:
            points -= 15.0

    # 6. Volume confirmation (Weight: 10.0)
    curr_vol = raw_signals.get("current_volume")
    avg_vol = raw_signals.get("average_volume")
    if curr_vol is not None and avg_vol is not None and avg_vol > 0:
        evaluated_weight += 10.0
        if curr_vol > avg_vol:
            if ret_1d is not None and ret_1d > 0:
                points += 10.0
            elif ret_1d is not None and ret_1d < 0:
                points -= 10.0

    if evaluated_weight == 0.0:
        return 0.0

    score = round((points / evaluated_weight) * 100.0, 2)
    return max(-100.0, min(100.0, score))


def determine_sentiment_label(score: float) -> str:
    """Classify numeric score into deterministic sentiment category."""
    if score >= 20.0:
        return "Bullish"
    elif score <= -20.0:
        return "Bearish"
    return "Neutral"


def get_stock_sentiment(
    symbol: str, history_provider: Optional[Callable[..., Any]] = None
) -> dict[str, Any]:
    """Calculate deterministic stock sentiment and technical signals.

    Args:
        symbol: NSE stock symbol (e.g. 'TCS', 'RELIANCE').
        history_provider: Optional callable returning price history (for testing).

    Returns:
        Dictionary containing:
            - symbol: Stock ticker
            - score: Number between -100.0 and +100.0
            - sentiment: 'Bullish' | 'Neutral' | 'Bearish'
            - signals: List of individual signal dictionaries
    """
    if isinstance(symbol, str):
        try:
            from services.symbol_resolver import resolve_nse_symbol
            res = resolve_nse_symbol(symbol)
            if res.get("symbol"):
                symbol = res["symbol"]
        except Exception:
            pass

    if not isinstance(symbol, str) or not is_valid_nse_symbol(symbol):
        logger.warning("Invalid symbol for sentiment: '%s'", symbol)
        return {
            "symbol": None,
            "score": 0.0,
            "sentiment": "Neutral",
            "signals": [],
        }

    clean_symbol = normalize_nse_symbol(symbol, target_format="clean")

    # Fetch 6-month historical data to ensure sufficient bars for 50 DMA
    try:
        if history_provider is not None:
            history = history_provider(clean_symbol, period="6mo")
        else:
            history = get_price_history(clean_symbol, period="6mo")
    except Exception as exc:
        logger.warning(
            "Failed to retrieve price history for sentiment on '%s': %s",
            clean_symbol,
            exc,
        )
        return {
            "symbol": clean_symbol,
            "score": 0.0,
            "sentiment": "Neutral",
            "signals": [],
        }

    if not history:
        return {
            "symbol": clean_symbol,
            "score": 0.0,
            "sentiment": "Neutral",
            "signals": [],
        }

    raw_signals, structured_signals = calculate_technical_signals(history)
    score = compute_sentiment_score(raw_signals)
    sentiment_label = determine_sentiment_label(score)

    return {
        "symbol": clean_symbol,
        "score": score,
        "sentiment": sentiment_label,
        "signals": structured_signals,
    }
