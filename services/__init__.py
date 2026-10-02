"""Services package for the NSE AI Agent application.

Provides persistent storage, conversation management, LLM model initialization,
and AI-callable LangChain tool wrappers.
"""

from services.database_service import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    initialize_database,
    list_conversations,
    update_conversation_title,
)
from services.llm_service import (
    ALL_TOOLS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROVIDER,
    LLMConfigurationError,
    company_info_tool,
    get_all_tools,
    get_company_info_tool,
    get_llm,
    get_llm_with_tools,
    get_market_index_tool,
    get_market_news_tool,
    get_stock_price_tool,
    get_stock_sentiment_tool,
    get_tool_by_name,
    get_tools,
    get_top_gainers_tool,
    get_top_losers_tool,
    market_index_tool,
    market_news_tool,
    stock_price_tool,
    stock_sentiment_tool,
    top_gainers_tool,
    top_losers_tool,
)

__all__ = [
    # Database Service
    "initialize_database",
    "create_conversation",
    "list_conversations",
    "get_conversation",
    "add_message",
    "get_messages",
    "update_conversation_title",
    "delete_conversation",
    # LLM Service & Model Factory
    "get_llm",
    "get_llm_with_tools",
    "LLMConfigurationError",
    "DEFAULT_PROVIDER",
    "DEFAULT_OPENAI_MODEL",
    # Tool Registry & Tools
    "get_all_tools",
    "get_tools",
    "get_tool_by_name",
    "ALL_TOOLS",
    "get_stock_price_tool",
    "get_company_info_tool",
    "get_market_news_tool",
    "get_market_index_tool",
    "get_top_gainers_tool",
    "get_top_losers_tool",
    "get_stock_sentiment_tool",
    "stock_price_tool",
    "company_info_tool",
    "market_news_tool",
    "market_index_tool",
    "top_gainers_tool",
    "top_losers_tool",
    "stock_sentiment_tool",
]
