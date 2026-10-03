"""Intent detection, query normalization, and parameter extraction service for NSE Market Assistant.

Deterministically routes natural language questions across market intents without requiring
an external LLM API key.
"""

from __future__ import annotations

import enum
import re
from typing import Any, Optional


class QueryScope(str, enum.Enum):
    """Scope of market query (single stock, multi-stock, market-wide, sector, etc.)."""

    SINGLE_STOCK = "SINGLE_STOCK"
    MULTI_STOCK = "MULTI_STOCK"
    MARKET_WIDE = "MARKET_WIDE"
    SECTOR = "SECTOR"
    INDEX = "INDEX"
    AMBIGUOUS = "AMBIGUOUS"
    GENERAL = "GENERAL"


class MarketIntent(str, enum.Enum):
    """Enumeration of supported natural language stock market intents."""

    # Stock-specific intents
    STOCK_OVERVIEW = "STOCK_OVERVIEW"
    STOCK_PRICE = "STOCK_PRICE"
    STOCK_FUNDAMENTALS = "STOCK_FUNDAMENTALS"
    STOCK_TECHNICALS = "STOCK_TECHNICALS"
    STOCK_NEWS = "STOCK_NEWS"
    STOCK_RESULTS = "STOCK_RESULTS"
    STOCK_52_WEEK_HIGH_LOW = "STOCK_52_WEEK_HIGH_LOW"
    STOCK_DECISION = "STOCK_DECISION"

    # Market-wide intents
    MARKET_OVERVIEW = "MARKET_OVERVIEW"
    TOP_GAINERS = "TOP_GAINERS"
    TOP_LOSERS = "TOP_LOSERS"
    MOST_ACTIVE = "MOST_ACTIVE"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    FIFTY_TWO_WEEK_HIGH_STOCKS = "52_WEEK_HIGH_STOCKS"
    FIFTY_TWO_WEEK_LOW_STOCKS = "52_WEEK_LOW_STOCKS"
    SHORT_TERM_OPPORTUNITIES = "SHORT_TERM_OPPORTUNITIES"

    # Multi-stock & Sector intents
    STOCK_COMPARISON = "STOCK_COMPARISON"
    SECTOR_PERFORMANCE = "SECTOR_PERFORMANCE"
    STOCKS_BY_SECTOR = "STOCKS_BY_SECTOR"

    # News & Catalyst intents
    POSITIVE_NEWS_STOCKS = "POSITIVE_NEWS_STOCKS"
    NEGATIVE_NEWS_STOCKS = "NEGATIVE_NEWS_STOCKS"
    RECENT_RESULTS = "RECENT_RESULTS"
    CORPORATE_ANNOUNCEMENTS = "CORPORATE_ANNOUNCEMENTS"

    # Ambiguity & Fallback
    AMBIGUOUS_STOCK = "AMBIGUOUS_STOCK"
    FALLBACK = "FALLBACK"


# Known sector synonyms for STOCKS_BY_SECTOR queries
SECTOR_KEYWORDS: dict[str, str] = {
    "it": "Information Technology",
    "tech": "Information Technology",
    "technology": "Information Technology",
    "software": "Information Technology",
    "bank": "Banking",
    "banking": "Banking",
    "banks": "Banking",
    "auto": "Auto",
    "automobile": "Auto",
    "automobiles": "Auto",
    "pharma": "Pharma & Healthcare",
    "pharmaceutical": "Pharma & Healthcare",
    "healthcare": "Pharma & Healthcare",
    "fmcg": "FMCG",
    "consumer": "FMCG",
    "metal": "Metals & Mining",
    "metals": "Metals & Mining",
    "mining": "Metals & Mining",
    "steel": "Metals & Mining",
    "energy": "Energy & Oil & Gas",
    "oil": "Energy & Oil & Gas",
    "gas": "Energy & Oil & Gas",
    "power": "Power & Utilities",
    "utilities": "Power & Utilities",
    "infra": "Infrastructure & Capital Goods",
    "infrastructure": "Infrastructure & Capital Goods",
    "realty": "Real Estate",
    "real estate": "Real Estate",
}


def normalize_query(query: str) -> str:
    """Normalize user question by stripping extra punctuation and whitespace."""
    if not query:
        return ""
    # Retain alphanumeric characters, hyphens, slashes, apostrophes, and whitespace
    clean = re.sub(r"[^\w\s\-\&%\.,/'’]", " ", query)
    return " ".join(clean.split()).strip()


def detect_intent(
    query: str,
    symbols: list[str],
    has_ambiguous_conglomerate: bool = False,
    context: Optional[dict[str, Any]] = None,
) -> tuple[MarketIntent, dict[str, Any]]:
    """Detect market intent and extract parameters from normalized query and symbols.

    Args:
        query: Raw or normalized user input string.
        symbols: List of extracted canonical NSE symbols.
        has_ambiguous_conglomerate: True if query mentions ambiguous group (e.g. 'Tata').
        context: Optional conversation context (e.g. previous focused symbol).

    Returns:
        Tuple of (MarketIntent, parameters_dict).
    """
    q_norm = normalize_query(query).lower()
    params: dict[str, Any] = {}

    # Check if user specifically requested explanation of catalysts ("why", "reason", "catalyst")
    include_catalysts = any(
        w in q_norm
        for w in (
            "why",
            "reason",
            "reasons",
            "catalyst",
            "catalysts",
            "what happened",
            "what caused",
            "driving",
            "moving",
            "driver",
            "drivers",
            "factor",
            "factors",
        )
    )
    params["include_catalysts"] = include_catalysts

    # 1. Conglomerate ambiguity check
    if has_ambiguous_conglomerate and not symbols:
        params["scope"] = QueryScope.AMBIGUOUS.value
        return MarketIntent.AMBIGUOUS_STOCK, params

    # 2. 52-Week High / Low market scans (Market-wide vs Single Stock)
    is_52w_high_phrase = any(
        phrase in q_norm
        for phrase in (
            "52 week high",
            "52-week high",
            "52w high",
            "52 week highs",
            "52 weeks high",
            "52 weeks highs",
            "52 week high stock",
            "52 week high stocks",
            "52weeks high",
            "52weeks high stocks",
            "52 wk high",
            "52 wk high stocks",
            "52wk high",
            "52wk high stocks",
            "at 52wk high",
            "yearly high",
            "yearly highs",
            "yearly high stocks",
            "shares at yearly highs",
            "year high",
            "year highs",
            "year high stocks",
            "annual high",
            "annual high stocks",
            "near annual high",
            "close to one-year high",
            "near their 1 year high",
            "all time high",
            "all time highs",
            "all time high stocks",
            "all-time high",
            "all-time highs",
            "all-time high stocks",
            "ath",
            "ath stocks",
            "lifetime high",
            "lifetime highs",
            "near high",
            "near 52w high",
            "52w high shares",
            "close to 52 week high",
            "touched 52 week high",
            "touched a 52-week high",
            "touching 52 week high",
            "at 52 week high",
            "making new highs",
            "making fresh highs",
            "making fresh 52-week highs",
            "making new 52 week highs",
            "stocks making new highs",
            "breakout stocks",
            "52 week breakout",
            "breakout names",
        )
    )

    is_52w_low_phrase = any(
        phrase in q_norm
        for phrase in (
            "52 week low",
            "52-week low",
            "52w low",
            "52 week lows",
            "52 weeks low",
            "52 weeks lows",
            "52 week low stock",
            "52 week low stocks",
            "52weeks low",
            "52weeks low stocks",
            "52 wk low",
            "52 wk low stocks",
            "52wk low",
            "52wk low stocks",
            "at 52wk low",
            "yearly low",
            "yearly lows",
            "yearly low stocks",
            "shares at yearly lows",
            "year low",
            "year lows",
            "year low stocks",
            "annual low",
            "annual low stocks",
            "near annual low",
            "close to one-year low",
            "near their 1 year low",
            "all time low",
            "all time lows",
            "all time low stocks",
            "all-time low",
            "all-time low stocks",
            "atl",
            "atl stocks",
            "lifetime low",
            "lifetime lows",
            "near low",
            "near 52w low",
            "52w low shares",
            "close to 52 week low",
            "touched 52 week low",
            "touched a 52-week low",
            "touching 52 week low",
            "at 52 week low",
            "making new lows",
            "making fresh lows",
            "making fresh 52-week lows",
            "making new 52 week lows",
            "stocks making new lows",
            "breakdown stocks",
            "52 week breakdown",
            "weakest versus their 52 week low",
        )
    )

    has_low_word = any(w in q_norm for w in ("low", "lows", "down", "breakdown", "weakest", "bottom"))
    has_high_word = any(w in q_norm for w in ("high", "highs", "up", "breakout", "strongest", "top", "ath", "peak"))
    is_multi_stock = len(symbols) >= 2 or (("compare" in q_norm or " vs " in q_norm or ("versus" in q_norm and "versus their" not in q_norm)) and not ("52" in q_norm and not symbols))

    if not is_multi_stock and (is_52w_low_phrase or (has_low_word and ("52" in q_norm or "yearly" in q_norm or "annual" in q_norm) and not has_high_word)):
        is_plural_req = any(w in q_norm for w in ("stocks", "shares", "companies", "equities", "which", "list", "top", "all"))
        has_pronoun_focus = any(re.search(pat, q_norm) for pat in (r"\b(it|its|this|that|the stock|the company)\b",))

        if symbols and len(symbols) == 1 and not (is_plural_req and not has_pronoun_focus):
            params["scope"] = QueryScope.SINGLE_STOCK.value
            return MarketIntent.STOCK_52_WEEK_HIGH_LOW, params
        else:
            params["scope"] = QueryScope.MARKET_WIDE.value
            return MarketIntent.FIFTY_TWO_WEEK_LOW_STOCKS, params

    if not is_multi_stock and (is_52w_high_phrase or (has_high_word and ("52" in q_norm or "yearly" in q_norm or "annual" in q_norm) and not has_low_word)):
        is_plural_req = any(w in q_norm for w in ("stocks", "shares", "companies", "equities", "which", "list", "top", "all"))
        has_pronoun_focus = any(re.search(pat, q_norm) for pat in (r"\b(it|its|this|that|the stock|the company)\b",))

        if symbols and len(symbols) == 1 and not (is_plural_req and not has_pronoun_focus):
            params["scope"] = QueryScope.SINGLE_STOCK.value
            return MarketIntent.STOCK_52_WEEK_HIGH_LOW, params
        else:
            params["scope"] = QueryScope.MARKET_WIDE.value
            return MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS, params

    # 2b. 2-Day Trading Opportunities & Technical Setups
    if any(
        kw in q_norm
        for kw in (
            "2-day",
            "2 day",
            "trading opportunit",
            "short-term opportunit",
            "short term opportunit",
            "stocks to watch",
            "trading setup",
            "short-term setup",
            "short term setup",
            "breakout names",
            "oversold stocks",
            "overbought stocks",
            "bullish momentum stocks",
            "bearish momentum stocks",
            "breakout stocks today",
            "breakdown stocks today",
            "moving average",
            "50 day",
            "50-day",
            "200 day",
            "200-day",
            "dma",
            "rsi above",
            "rsi below",
            "technical breakout",
            "technical screen",
        )
    ) and not symbols:
        return MarketIntent.SHORT_TERM_OPPORTUNITIES, params

    # 2c. Sector Performance vs Stock Gainers
    if any(
        phrase in q_norm
        for phrase in (
            "top sector",
            "top sectors",
            "best sector",
            "best sectors",
            "best performing sector",
            "best performing sectors",
            "leading sector",
            "leading sectors",
            "sector performance",
            "sectors performance",
            "sector ranking",
            "sector rotation",
            "how are sectors doing",
            "which sector is hot",
            "sector is hot",
            "which sector is weak",
            "sector is weak",
            "sectors leading",
            "sectors weak",
            "sectors are leading",
            "sectors are weak",
            "what sectors are leading",
            "what sectors are weak",
            "leading today",
            "weak today",
            "rank sectors",
            "rank sectors by today's performance",
            "ranking sectors",
            "strongest sectors",
            "weakest sectors",
            "which sector is leading",
            "which sector is falling",
        )
    ) and not symbols:
        return MarketIntent.SECTOR_PERFORMANCE, params

    # 3. Top Gainers & Losers
    if any(
        phrase in q_norm
        for phrase in (
            "gainer",
            "gainers",
            "top gainer",
            "top gainers",
            "best performing",
            "best stock",
            "best stocks",
            "going up",
            "surging",
            "biggest gainer",
            "biggest gainers",
            "biggest nse gainer",
            "biggest nse gainers",
            "highest gain",
            "top performers",
            "rallied today",
            "rallying",
            "rising the most",
            "stocks up the most",
            "green stocks",
            "gained the most",
            "winners today",
            "nse winners",
            "leading the market",
            "strongest stocks",
            "highest return",
            "biggest movers on the upside",
            "maximum gain",
            "up sharply",
            "up more than",
            "top 10 gainers",
            "top five nse gainers",
            "what are the gainers",
            "who's winning",
            "who s winning",
            "whos winning",
            "who is winning",
            "winning today",
        )
    ) and not any(kw in q_norm for kw in ("it stocks", "banking stocks", "pharma stocks", "auto stocks", "sector", "sectors")):
        return MarketIntent.TOP_GAINERS, params

    if any(
        phrase in q_norm
        for phrase in (
            "loser",
            "losers",
            "top loser",
            "top losers",
            "worst performing",
            "falling today",
            "going down",
            "biggest loser",
            "biggest losers",
            "biggest nse loser",
            "biggest nse losers",
            "top fallers",
            "dragged down",
            "biggest drop",
            "falling the most",
            "stocks down the most",
            "red stocks",
            "lost the most",
            "dragging the market",
            "weakest stocks",
            "lowest return",
            "biggest movers on the downside",
            "maximum fall",
            "down sharply",
            "down more than",
            "top 10 losers",
            "top five nse losers",
            "what are the losers",
            "who's getting hit",
            "who s getting hit",
            "whos getting hit",
            "who is getting hit",
            "getting hit today",
        )
    ):
        return MarketIntent.TOP_LOSERS, params

    # 4. Volume Spikes & Most Active
    if any(
        phrase in q_norm
        for phrase in (
            "high volume",
            "volume spike",
            "volume surge",
            "volume breakout",
            "unusual volume",
            "unusually high volume",
            "volume shocker",
            "volume shockers",
            "heavy volume",
            "most active",
            "most traded",
            "top traded",
            "top traded stocks",
            "highest volume",
            "high turnover",
            "most action",
            "shares have the most action",
            "buzzing",
            "sudden volume",
            "sudden volume increase",
            "delivery volume",
            "high delivery",
            "price and volume",
            "strong price and volume",
            "traders watching",
            "movers and shakers",
            "heavy trading",
            "volume leaders",
            "volume movers",
            "above average volume",
            "2x normal volume",
            "active stocks by turnover",
            "top stocks by traded value",
            "abnormal volume",
            "most liquid",
            "most liquid stocks",
            "activity leaders",
        )
    ) and not symbols:
        return MarketIntent.VOLUME_SPIKE, params

    # 4b. Market Overview & Broad Sentiment
    if any(
        kw in q_norm
        for kw in (
            "bullish or bearish",
            "market trend",
            "nifty trend",
            "market sentiment",
            "market movers",
            "moving the market",
            "what's moving the market",
            "whats moving the market",
            "important nse moves",
            "nse action",
            "index performance",
            "market overview",
            "market breadth",
            "market status",
            "market today",
            "how is the market",
            "how is market doing",
            "summarize today's nse action",
            "summarize todays nse action",
        )
    ) and not symbols:
        return MarketIntent.MARKET_OVERVIEW, params

    # 5. Market Indices / Overview
    # e.g. "How is Nifty doing?", "How is Sensex?", "How is market today?", "Bank Nifty status"
    if any(
        phrase in q_norm
        for phrase in (
            "nifty",
            "sensex",
            "banknifty",
            "bank nifty",
            "market overview",
            "market today",
            "how is market",
            "market doing",
            "how is the market",
            "overall market",
            "indices today",
            "market trend",
            "what moved the market",
            "what's moving the market",
            "whats moving the market",
            "market mood",
            "market summary",
            "what changed in nse",
            "intraday market snapshot",
            "market breadth",
            "advances and declines",
            "how many stocks are up versus down",
            "is today's market bullish or bearish",
            "is todays market bullish or bearish",
            "indian market status",
            "how did nse open",
            "why is the market down",
            "why is the market up",
            "biggest market stories",
            "summarize today's nse action",
            "summarize todays nse action",
            "nifty snapshot",
            "bank nifty snapshot",
            "is the market up or down",
            "market status",
        )
    ) and not symbols:
        return MarketIntent.MARKET_OVERVIEW, params

    # 6. Stocks by Sector / Sector Performance
    # e.g., "Which IT stocks are doing well today?", "Show banking stocks", "top auto stocks"
    detected_sector = None
    for kw, sec_name in SECTOR_KEYWORDS.items():
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, q_norm):
            detected_sector = sec_name
            params["sector"] = sec_name
            break

    if detected_sector and not symbols:
        # Specific sector stock screen
        return MarketIntent.STOCKS_BY_SECTOR, params

    if any(
        phrase in q_norm
        for phrase in (
            "top sector",
            "top sectors",
            "sector performance",
            "sectors performance",
            "best sector",
            "best sectors",
            "leading sector",
            "leading sectors",
            "sector ranking",
            "sector rotation",
            "how are sectors doing",
        )
    ):
        return MarketIntent.SECTOR_PERFORMANCE, params

    # 7. Positive / Negative News Stocks
    if any(
        phrase in q_norm
        for phrase in (
            "positive news",
            "good news",
            "bullish news",
            "stocks with good news",
            "positive catalyst",
            "positive catalysts",
        )
    ) and not symbols:
        return MarketIntent.POSITIVE_NEWS_STOCKS, params

    if any(
        phrase in q_norm
        for phrase in (
            "negative news",
            "bad news",
            "bearish news",
            "stocks with bad news",
            "negative catalyst",
            "negative catalysts",
        )
    ) and not symbols:
        return MarketIntent.NEGATIVE_NEWS_STOCKS, params

    # 8. Corporate Announcements / Recent Results (Market-wide)
    if any(
        phrase in q_norm
        for phrase in (
            "corporate announcement",
            "corporate announcements",
            "board meetings",
            "dividends declared",
            "upcoming dividend",
            "dividend record date",
        )
    ) and not symbols:
        return MarketIntent.CORPORATE_ANNOUNCEMENTS, params

    if any(
        phrase in q_norm
        for phrase in (
            "recent results",
            "latest results",
            "quarterly results declared",
            "earnings declared",
            "results today",
        )
    ) and not symbols:
        return MarketIntent.RECENT_RESULTS, params

    # 9. Stock Decision (Buy / Hold / Sell / Exit / Should I buy...)
    if any(
        phrase in q_norm
        for phrase in (
            "should i buy",
            "should i sell",
            "should i hold",
            "should i exit",
            "should i reduce",
            "should i average",
            "should i keep",
            "good time to buy",
            "good to buy",
            "hold or sell",
            "buy or sell",
            "can i hold",
            "can i buy",
            "bought at",
            "i own",
            "already own",
            "good investment",
            "wait for correction",
            "decision on",
        )
    ):
        return MarketIntent.STOCK_DECISION, params

    # 10. Multi-Stock Comparison
    # e.g. "Compare TCS and Infosys", "Which has better PE?", "Compare SBI and HDFC Bank"
    is_comparison_query = len(symbols) >= 2 or any(
        w in q_norm
        for w in (
            "compare",
            "comparison",
            "vs",
            "versus",
            "between both",
            "which is better",
            "which has better",
            "which has lower",
            "which has higher",
            "which one has",
            "better pe",
            "cheaper valuation",
        )
    )
    if is_comparison_query:
        # Check if specific metric comparison requested
        if re.search(r"\b(pe|p/e|valuation)\b", q_norm):
            params["metric"] = "pe"
        elif re.search(r"\broe\b", q_norm):
            params["metric"] = "roe"
        elif re.search(r"\broce\b", q_norm):
            params["metric"] = "roce"
        elif "market cap" in q_norm or "size" in q_norm:
            params["metric"] = "market_cap"
        elif "dividend" in q_norm or "yield" in q_norm:
            params["metric"] = "dividend"
        return MarketIntent.STOCK_COMPARISON, params

    # 11. Single Stock Specific Intents
    if symbols and len(symbols) == 1:
        # Results / Earnings
        if any(re.search(r"\b" + re.escape(w) + r"\b", q_norm) for w in ("result", "results", "quarterly", "q1", "q2", "q3", "q4", "earnings", "net profit", "revenue", "ebitda")):
            return MarketIntent.STOCK_RESULTS, params

        # News & Catalysts
        if any(re.search(r"\b" + re.escape(w) + r"\b", q_norm) for w in (
            "news", "headline", "headlines", "latest update", "announcement", "announcements", "updates",
            "what happened to", "what happened", "why is it down", "why is it up", "why is down", "why is up",
            "why down", "why up", "why is", "happened to", "happened",
        )):
            return MarketIntent.STOCK_NEWS, params

        # Fundamentals / Valuation (using word boundaries so 'pe' does not match 'happened')
        if any(
            re.search(r"\b" + re.escape(w) + r"\b", q_norm)
            for w in (
                "pe",
                "p/e",
                "pe ratio",
                "p/e ratio",
                "valuation",
                "market cap",
                "market capitalization",
                "market capitalisation",
                "financial health",
                "financial position",
                "pb",
                "p/b",
                "roe",
                "roce",
                "debt",
                "debt to equity",
                "fundamental",
                "fundamentals",
                "promoter",
                "shareholding",
                "dividend",
                "eps",
                "book value",
            )
        ):
            if re.search(r"\b(pe|p/e|pe ratio|p/e ratio)\b", q_norm):
                params["metric"] = "pe"
            elif re.search(r"\broe\b", q_norm):
                params["metric"] = "roe"
            elif "market cap" in q_norm or "market capital" in q_norm:
                params["metric"] = "market_cap"
            return MarketIntent.STOCK_FUNDAMENTALS, params

        # Technicals
        if any(
            re.search(r"\b" + re.escape(w) + r"\b", q_norm)
            for w in (
                "technical",
                "technicals",
                "rsi",
                "dma",
                "moving average",
                "support",
                "resistance",
                "trend",
                "chart",
                "breakout",
                "atr",
                "sentiment",
            )
        ):
            return MarketIntent.STOCK_TECHNICALS, params

        # 52-week High/Low for stock
        if any(w in q_norm for w in ("52 week", "52-week", "52w", "high low", "all time high", "yearly high")):
            return MarketIntent.STOCK_52_WEEK_HIGH_LOW, params

        # Price / What happened today
        if any(
            re.search(r"\b" + re.escape(w) + r"\b", q_norm)
            for w in (
                "price",
                "rate",
                "quote",
                "trading at",
                "how much is",
                "current price",
                "down today",
                "up today",
                "today change",
            )
        ):
            return MarketIntent.STOCK_PRICE, params

        # Default for a single stock is comprehensive overview
        return MarketIntent.STOCK_OVERVIEW, params

    # 12. Fallback
    return MarketIntent.FALLBACK, params
