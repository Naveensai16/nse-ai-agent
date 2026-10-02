"""Tools package for the NSE AI Agent application.

Provides tools for stock market data retrieval, company fundamentals,
market index tracking, historical price series, market movers (gainers/losers),
news retrieval, and technical sentiment analysis.
"""

from tools.company_tool import get_company_info
from tools.market_tool import (
    INDEX_MAPPING,
    SUPPORTED_PERIODS,
    GainersLosersProvider,
    PriceHistory,
    get_market_index,
    get_price_history,
    get_top_gainers,
    get_top_losers,
)
from tools.news_tool import (
    deduplicate_stories,
    get_market_news,
    normalize_news_query,
    parse_news_date,
)
from tools.sentiment_tool import (
    calculate_technical_signals,
    compute_sentiment_score,
    determine_sentiment_label,
    get_stock_sentiment,
)
from tools.decision_tool import get_stock_decision
from tools.opportunity_tool import get_short_term_opportunities
from tools.sector_tool import get_top_sectors_and_companies
from tools.stock_tool import get_stock_price

__all__ = [
    "get_stock_price",
    "get_company_info",
    "get_market_index",
    "get_price_history",
    "get_top_gainers",
    "get_top_losers",
    "GainersLosersProvider",
    "get_market_news",
    "parse_news_date",
    "deduplicate_stories",
    "normalize_news_query",
    "get_stock_sentiment",
    "calculate_technical_signals",
    "compute_sentiment_score",
    "determine_sentiment_label",
    "get_short_term_opportunities",
    "get_top_sectors_and_companies",
    "get_stock_decision",
    "INDEX_MAPPING",
    "SUPPORTED_PERIODS",
    "PriceHistory",
]
