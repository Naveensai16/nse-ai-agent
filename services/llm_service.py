"""LLM Service and AI-callable tool wrappers for the NSE AI Agent application.

Provides LangChain-compatible model initialization and structured tools
wrapping existing market analysis, stock quote, news, fundamentals,
sentiment, and index services.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from tools.analysis_tool import get_company_analysis as _get_company_analysis
from tools.company_tool import get_company_info as _get_company_info
from tools.financials_tool import (
    get_cash_flow as _get_cash_flow,
    get_debt_metrics as _get_debt_metrics,
    get_quarterly_financials as _get_quarterly_financials,
)
from tools.governance_tool import (
    get_board_meetings as _get_board_meetings,
    get_corporate_actions as _get_corporate_actions,
    get_shareholding_pattern as _get_shareholding_pattern,
)
from tools.decision_tool import get_stock_decision as _get_stock_decision
from tools.market_tool import (
    get_market_index as _get_market_index,
    get_top_gainers as _get_top_gainers,
    get_top_losers as _get_top_losers,
)
from tools.news_tool import get_market_news as _get_market_news
from tools.opportunity_tool import get_short_term_opportunities as _get_short_term_opportunities
from tools.sector_tool import get_top_sectors_and_companies as _get_top_sectors_and_companies
from tools.sentiment_tool import get_stock_sentiment as _get_stock_sentiment
from tools.stock_tool import get_stock_price as _get_stock_price

logger = logging.getLogger(__name__)


class LLMConfigurationError(ValueError):
    """Raised when LLM configuration or credentials are missing, invalid, or unsupported."""

    pass


DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
SUPPORTED_PROVIDERS = ("openai",)


# ==============================================================================
# AI-Callable Tools (Wrapping Existing Domain Logic with Audit Logging)
# ==============================================================================


@tool("get_stock_price")
def get_stock_price_tool(symbol: str) -> dict:
    """Fetch current real-time or latest stock price and trading summary for an NSE stock.

    Retrieves current market price, previous close, price change, percentage change,
    52-week high, 52-week low, and currency for any equity listed on the National
    Stock Exchange of India (NSE).

    Args:
        symbol: The NSE stock ticker symbol (e.g., 'TCS', 'INFY', 'RELIANCE', 'HDFCBANK').

    Returns:
        dict: Market price metrics including current_price, previous_close,
        change, change_percent, 52_week_high, 52_week_low, currency, company name, and symbol.
    """
    logger.info("Executing tool 'get_stock_price' with symbol=%r", symbol)
    try:
        result = _get_stock_price(symbol)
        logger.debug("Tool 'get_stock_price' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_stock_price' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_company_info")
def get_company_info_tool(symbol: str) -> dict:
    """Retrieve comprehensive company fundamentals and background for an NSE stock.

    Provides business overview, sector, industry, market capitalization, valuation metrics
    including trailing PE, forward PE, EPS, dividend yield, 52-week range, and official website.

    Args:
        symbol: The NSE stock ticker symbol (e.g., 'TCS', 'INFY', 'RELIANCE').

    Returns:
        dict: Company fundamentals and business profile dictionary.
    """
    logger.info("Executing tool 'get_company_info' with symbol=%r", symbol)
    try:
        result = _get_company_info(symbol)
        logger.debug("Tool 'get_company_info' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_company_info' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_market_news")
def get_market_news_tool(company_or_symbol: str, limit: int = 5) -> list[dict]:
    """Fetch recent news articles and market updates for an Indian company, stock, or index.

    Retrieves verified, deduplicated news stories sorted by recency from reliable financial feeds.
    Includes article title, source publication, published timestamp, URL, and summary.

    Args:
        company_or_symbol: Company name, NSE symbol (e.g., 'TCS', 'RELIANCE'), or index (e.g., 'NIFTY').
        limit: Maximum number of recent articles to return (default is 5, max 10).

    Returns:
        list[dict]: List of news story dictionaries with title, source, published_at, url, and summary.
    """
    logger.info("Executing tool 'get_market_news' with query=%r, limit=%d", company_or_symbol, limit)
    try:
        result = _get_market_news(company_or_symbol, limit=limit)
        logger.debug("Tool 'get_market_news' completed successfully (%d stories retrieved)", len(result))
        return result
    except Exception as exc:
        logger.error("Tool 'get_market_news' failed for query=%r: %s", company_or_symbol, exc)
        raise


@tool("get_market_index")
def get_market_index_tool(index_name: str) -> dict:
    """Fetch current value and daily performance for major Indian market indices.

    Supports major benchmark and sectoral indices including NIFTY 50 (NIFTY),
    BANK NIFTY (BANKNIFTY), SENSEX, and NIFTY IT.

    Args:
        index_name: Name of the index (e.g., 'NIFTY', 'NIFTY 50', 'BANK NIFTY', 'BANKNIFTY', 'SENSEX', 'NIFTY IT').

    Returns:
        dict: Index data containing index name, ticker, current_value, previous_close, change, and change_percent.
    """
    logger.info("Executing tool 'get_market_index' with index_name=%r", index_name)
    try:
        result = _get_market_index(index_name)
        logger.debug("Tool 'get_market_index' completed successfully for %r", index_name)
        return result
    except Exception as exc:
        logger.error("Tool 'get_market_index' failed for %r: %s", index_name, exc)
        raise


@tool("get_top_gainers")
def get_top_gainers_tool() -> list[dict]:
    """Fetch the current top gainers in the Indian stock market (NSE).

    Returns the top performing NSE liquid stocks sorted by highest percentage price gain,
    including current price, previous close, absolute change, and percentage change.

    Returns:
        list[dict]: List of top gaining stocks ranked by percentage gain.
    """
    logger.info("Executing tool 'get_top_gainers'")
    try:
        result = _get_top_gainers()
        logger.debug("Tool 'get_top_gainers' completed successfully (%d gainers found)", len(result))
        return result
    except Exception as exc:
        logger.error("Tool 'get_top_gainers' failed: %s", exc)
        raise


@tool("get_top_losers")
def get_top_losers_tool() -> list[dict]:
    """Fetch the current top losers in the Indian stock market (NSE).

    Returns the worst performing NSE liquid stocks sorted by highest percentage price decline,
    including current price, previous close, absolute change, and percentage change.

    Returns:
        list[dict]: List of top losing stocks ranked by percentage decline.
    """
    logger.info("Executing tool 'get_top_losers'")
    try:
        result = _get_top_losers()
        logger.debug("Tool 'get_top_losers' completed successfully (%d losers found)", len(result))
        return result
    except Exception as exc:
        logger.error("Tool 'get_top_losers' failed: %s", exc)
        raise


@tool("get_stock_sentiment")
def get_stock_sentiment_tool(symbol: str) -> dict:
    """Calculate deterministic quantitative technical sentiment for an NSE stock.

    Evaluates technical price signals (1-day return, 5-day return, 20 DMA, 50 DMA,
    price vs moving averages, and volume vs average volume) using fixed scoring rules.
    Does NOT hallucinate or guess sentiment; computes a mathematical score.

    Args:
        symbol: The NSE stock ticker symbol (e.g., 'TCS', 'INFY', 'RELIANCE').

    Returns:
        dict: Sentiment result containing symbol, numeric score (-1.0 to 1.0),
        sentiment classification ('Bullish', 'Neutral', 'Bearish'), and detailed signals list.
    """
    logger.info("Executing tool 'get_stock_sentiment' with symbol=%r", symbol)
    try:
        result = _get_stock_sentiment(symbol)
        logger.debug("Tool 'get_stock_sentiment' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_stock_sentiment' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_quarterly_financials")
def get_quarterly_financials_tool(symbol: str) -> dict:
    """Fetch the latest four quarters of financial statements (Revenue, Expenses, Operating Profit, OPM %, Net Profit, EPS) and growth trends.

    Args:
        symbol: NSE stock symbol (e.g. 'TCS', 'INFY', 'RELIANCE').

    Returns:
        dict: Financial performance across recent quarters with QoQ and YoY growth rates.
    """
    logger.info("Executing tool 'get_quarterly_financials' with symbol=%r", symbol)
    try:
        result = _get_quarterly_financials(symbol)
        logger.debug("Tool 'get_quarterly_financials' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_quarterly_financials' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_cash_flow")
def get_cash_flow_tool(symbol: str) -> dict:
    """Retrieve Operating Cash Flow (OCF), annual OCF history, and evaluate cash conversion ratio (OCF / Net Profit).

    Args:
        symbol: NSE stock symbol.

    Returns:
        dict: Cash flow metrics and earnings cash-conversion quality assessment.
    """
    logger.info("Executing tool 'get_cash_flow' with symbol=%r", symbol)
    try:
        result = _get_cash_flow(symbol)
        logger.debug("Tool 'get_cash_flow' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_cash_flow' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_debt_metrics")
def get_debt_metrics_tool(symbol: str) -> dict:
    """Retrieve total debt, cash holdings, net debt, and leverage metrics.

    Identifies banks/NBFCs where standard debt rules do not apply.

    Args:
        symbol: NSE stock symbol.

    Returns:
        dict: Debt metrics, net debt, and leverage assessment.
    """
    logger.info("Executing tool 'get_debt_metrics' with symbol=%r", symbol)
    try:
        result = _get_debt_metrics(symbol)
        logger.debug("Tool 'get_debt_metrics' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_debt_metrics' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_shareholding_pattern")
def get_shareholding_pattern_tool(symbol: str) -> dict:
    """Retrieve latest official shareholding pattern (Promoter %, FII %, DII %, Government %, Public %), disclosed promoter entities, and trends.

    Args:
        symbol: NSE stock symbol.

    Returns:
        dict: Ownership breakdown and disclosed promoter entities.
    """
    logger.info("Executing tool 'get_shareholding_pattern' with symbol=%r", symbol)
    try:
        result = _get_shareholding_pattern(symbol)
        logger.debug("Tool 'get_shareholding_pattern' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_shareholding_pattern' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_board_meetings")
def get_board_meetings_tool(symbol: str) -> list[dict]:
    """Retrieve the last two publicly announced or scheduled board meetings, purpose, outcome, and sources.

    Args:
        symbol: NSE stock symbol.

    Returns:
        list[dict]: List of board meetings with date, purpose, outcome, and filing link.
    """
    logger.info("Executing tool 'get_board_meetings' with symbol=%r", symbol)
    try:
        result = _get_board_meetings(symbol)
        logger.debug("Tool 'get_board_meetings' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_board_meetings' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_corporate_actions")
def get_corporate_actions_tool(symbol: str) -> list[dict]:
    """Retrieve the latest 5 corporate actions (Dividends, Splits, Bonus issues) with dates and amounts.

    Args:
        symbol: NSE stock symbol.

    Returns:
        list[dict]: List of up to 5 corporate actions.
    """
    logger.info("Executing tool 'get_corporate_actions' with symbol=%r", symbol)
    try:
        result = _get_corporate_actions(symbol)
        logger.debug("Tool 'get_corporate_actions' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_corporate_actions' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_company_analysis")
def get_company_analysis_tool(symbol: str) -> dict:
    """Execute a comprehensive fundamental, technical, financial, and governance analysis for an NSE company.

    Generates a full structured Markdown report with Snapshot, Valuation (P/E, ROCE, ROE),
    Technical Sentiment, 4-quarter financials, Cash Flow & Debt, Financial Health Scoring,
    Shareholding Pattern, Board Meetings, Corporate Actions, and News.

    Args:
        symbol: NSE stock symbol (e.g. 'TCS', 'INFY', 'RELIANCE', 'HDFCBANK').

    Returns:
        dict: Complete analysis including 'report_markdown' and structured data.
    """
    logger.info("Executing tool 'get_company_analysis' with symbol=%r", symbol)
    try:
        result = _get_company_analysis(symbol)
        logger.debug("Tool 'get_company_analysis' completed successfully for symbol=%r", symbol)
        return result
    except Exception as exc:
        logger.error("Tool 'get_company_analysis' failed for symbol=%r: %s", symbol, exc)
        raise


@tool("get_short_term_opportunities")
def get_short_term_opportunities_tool(limit: int = 5, universe: str = "NIFTY 200") -> dict:
    """Scan and rank NSE stocks showing favorable short-term momentum and catalyst setups for next 1–2 trading sessions.

    Applies deterministic multi-factor screening across price action, 20/50 DMA alignment,
    volume surge, market regime, verified corporate catalysts/results, and overextension risk checks.

    Args:
        limit: Number of top watchlist candidates to return (default: 5, max: 20).
        universe: Scanned universe ('NIFTY 50', 'NIFTY NEXT 50', 'NIFTY 100', 'NIFTY 200').

    Returns:
        dict: Opportunity scan result including 'report_markdown', 'market_regime', and ranked candidates.
    """
    logger.info("Executing tool 'get_short_term_opportunities' with limit=%r, universe=%r", limit, universe)
    try:
        result = _get_short_term_opportunities(limit=limit, universe=universe)
        logger.debug("Tool 'get_short_term_opportunities' completed successfully")
        return result
    except Exception as exc:
        logger.error("Tool 'get_short_term_opportunities' failed: %s", exc)
        raise


@tool("get_top_sectors_and_companies")
def get_top_sectors_and_companies_tool(
    sector_name: str = "",
    timeframe: str = "1 Day",
    sort_by: str = "Momentum",
    top_n: int = 10,
) -> dict:
    """Identify current Top 5 performing sectors in Indian stock market and rank Top 10 companies in a sector.

    Calculates sector performance, 1-day and 1-week changes, trends, institutional context,
    and multi-factor ranked constituent companies.

    Args:
        sector_name: Optional sector (e.g. 'Banking', 'Information Technology', 'Auto', etc.)
        timeframe: '1 Day', '1 Week', or '1 Month'.
        sort_by: 'Performance', 'Volume', 'Market Cap', 'RSI', or 'Momentum'.
        top_n: Number of ranked companies to retrieve (default: 10).

    Returns:
        dict: Top sectors result with 'report_markdown', 'top_sectors', and 'top_companies'.
    """
    logger.info("Executing tool 'get_top_sectors_and_companies' with sector=%r, timeframe=%r", sector_name, timeframe)
    try:
        sec_arg = sector_name if sector_name and sector_name.strip() else None
        result = _get_top_sectors_and_companies(sector_name=sec_arg, timeframe=timeframe, sort_by=sort_by, top_n=top_n)
        logger.debug("Tool 'get_top_sectors_and_companies' completed successfully")
        return result
    except Exception as exc:
        logger.error("Tool 'get_top_sectors_and_companies' failed: %s", exc)
        raise


@tool("get_stock_decision")
def get_stock_decision_tool(
    symbol: str,
    intent: str = "new_investment",
    horizon: str = "1 Year",
    purchase_price: float = 0.0,
    quantity: int = 0,
) -> dict:
    """Analyze whether an investor should consider BUY | HOLD | REDUCE | EXIT | WAIT for an Indian stock.

    Performs complete company, sector, valuation, financial, technical, news, risk, and peer analysis
    based on intended holding period (e.g. '1 Month', '3 Months', '6 Months', '1 Year', '2 Years', '3+ Years').

    Args:
        symbol: NSE stock ticker symbol or company name (e.g. 'Tata Power', 'TCS', 'India Cements', 'HDFCBANK').
        intent: 'new_investment' (thinking of buying) or 'existing_investment' (already own).
        horizon: Holding period ('1 Month', '3 Months', '6 Months', '1 Year', '2 Years', '3+ Years').
        purchase_price: Optional historical purchase price if stock is already owned.
        quantity: Optional number of shares held.

    Returns:
        dict: Complete structured decision result including 'decision_indicator', 'snapshot', and 'report_markdown'.
    """
    logger.info("Executing tool 'get_stock_decision' with symbol=%r, intent=%r, horizon=%r", symbol, intent, horizon)
    try:
        p_price = float(purchase_price) if purchase_price and purchase_price > 0 else None
        qty = int(quantity) if quantity and quantity > 0 else None
        result = _get_stock_decision(symbol_or_name=symbol, intent=intent, horizon=horizon, purchase_price=p_price, quantity=qty)
        logger.debug("Tool 'get_stock_decision' completed successfully")
        return result
    except Exception as exc:
        logger.error("Tool 'get_stock_decision' failed: %s", exc)
        raise


# Convenient references
stock_price_tool = get_stock_price_tool
company_info_tool = get_company_info_tool
market_news_tool = get_market_news_tool
market_index_tool = get_market_index_tool
top_gainers_tool = get_top_gainers_tool
top_losers_tool = get_top_losers_tool
stock_sentiment_tool = get_stock_sentiment_tool
quarterly_financials_tool = get_quarterly_financials_tool
cash_flow_tool = get_cash_flow_tool
debt_metrics_tool = get_debt_metrics_tool
shareholding_pattern_tool = get_shareholding_pattern_tool
board_meetings_tool = get_board_meetings_tool
corporate_actions_tool = get_corporate_actions_tool
company_analysis_tool = get_company_analysis_tool
short_term_opportunities_tool = get_short_term_opportunities_tool
top_sectors_tool = get_top_sectors_and_companies_tool
stock_decision_tool = get_stock_decision_tool


def get_all_tools(include_extended: bool = False) -> list[BaseTool]:
    """Return the list of AI-callable LangChain tools.

    Args:
        include_extended: When True, includes comprehensive company analysis, opportunities, and governance tools.
                          Defaults to False to preserve standard core tool registry.

    Returns:
        list[BaseTool]: List of LangChain tools.
    """
    core_tools = [
        get_stock_price_tool,
        get_company_info_tool,
        get_market_news_tool,
        get_market_index_tool,
        get_top_gainers_tool,
        get_top_losers_tool,
        get_stock_sentiment_tool,
    ]
    if not include_extended:
        return core_tools

    return core_tools + [
        get_quarterly_financials_tool,
        get_cash_flow_tool,
        get_debt_metrics_tool,
        get_shareholding_pattern_tool,
        get_board_meetings_tool,
        get_corporate_actions_tool,
        get_company_analysis_tool,
        get_short_term_opportunities_tool,
        get_top_sectors_and_companies_tool,
        get_stock_decision_tool,
    ]


def get_extended_tools() -> list[BaseTool]:
    """Return the complete list of core and extended AI-callable LangChain tools."""
    return get_all_tools(include_extended=True)


get_tools = get_all_tools
ALL_TOOLS: list[BaseTool] = get_all_tools()
EXTENDED_TOOLS: list[BaseTool] = get_all_tools(include_extended=True)


def get_tool_by_name(name: str) -> BaseTool:
    """Retrieve a tool instance by its registered name.

    Args:
        name: Canonical tool name (e.g. 'get_stock_price', 'get_company_analysis').

    Returns:
        BaseTool: Matching tool instance.

    Raises:
        KeyError: If tool name is not recognized.
    """
    tools_map = {t.name: t for t in get_all_tools(include_extended=True)}
    if name not in tools_map:
        raise KeyError(f"Tool '{name}' not found. Available tools: {list(tools_map.keys())}")
    return tools_map[name]


# ==============================================================================
# LLM Initialization and Factory
# ==============================================================================


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    request_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs: Any,
) -> ChatOpenAI:
    """Instantiate and configure a supported LangChain chat model.

    Configuration is resolved with precedence:
    1. Explicit function arguments (`provider`, `model`, `api_key`)
    2. Environment variables (`LLM_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`)
    3. Sensible defaults (`openai`, `gpt-4o-mini`, `temperature=0.0`, `timeout=30s`)

    Args:
        provider: LLM provider name (e.g. 'openai'). Defaults to env LLM_PROVIDER or 'openai'.
        model: Model identifier (e.g. 'gpt-4o-mini'). Defaults to env OPENAI_MODEL or 'gpt-4o-mini'.
        api_key: Provider API key. Defaults to env OPENAI_API_KEY.
        temperature: Sampling temperature (default 0.0 for deterministic tool usage).
        request_timeout: Timeout in seconds for remote API requests.
        max_retries: Maximum automatic retries on transient network errors.
        **kwargs: Additional parameters forwarded to the model constructor.

    Returns:
        ChatOpenAI: Configured chat model instance.

    Raises:
        LLMConfigurationError: If the provider is unsupported or required credentials are missing.
    """
    load_dotenv()

    selected_provider = (provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)).strip().lower()

    if selected_provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: '{selected_provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    selected_model = (model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)).strip()
    resolved_api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()

    if not resolved_api_key:
        raise LLMConfigurationError(
            "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable "
            "in your .env file or system environment."
        )

    return ChatOpenAI(
        model=selected_model,
        api_key=resolved_api_key,
        temperature=temperature,
        timeout=request_timeout,
        max_retries=max_retries,
        **kwargs,
    )


def get_llm_with_tools(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    tools: list[BaseTool] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a chat model instance with AI-callable tools bound for tool calling.

    Args:
        provider: LLM provider name.
        model: Model identifier.
        api_key: Provider API key.
        temperature: Sampling temperature.
        tools: List of BaseTool instances to bind. Defaults to get_all_tools().
        **kwargs: Additional parameters forwarded to get_llm.

    Returns:
        Runnable: Model with tools bound.
    """
    llm = get_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
    tools_to_bind = tools if tools is not None else get_all_tools()
    return llm.bind_tools(tools_to_bind)
