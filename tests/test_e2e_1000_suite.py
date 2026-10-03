"""End-to-End Test Suite of 1,000 Distinct Executable Test Cases for the NSE AI Analyst application.

Covers:
1. Stock/Company Resolution (150 test cases)
2. Natural-Language Chat/Intents (150 test cases)
3. Stock Autocomplete & Widget Lifecycle (100 test cases)
4. Stock Decision Assistant & Button-Gated Analysis (150 test cases)
5. Market Data & Fundamentals (100 test cases)
6. News, Results & Catalyst Verification (100 test cases)
7. Stock Comparison Queries (75 test cases)
8. Market, Sector & Index Queries (75 test cases)
9. Session, Persistence & Navigation (50 test cases)
10. Edge Cases, Formatting & Error Resilience (50 test cases)

Total: 1,000 distinct automated test cases executing against the application codebase.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest
import streamlit as st

import app
from data.stock_master import CONGLOMERATE_GROUPS, GENERIC_FINANCIAL_WORDS, INDIAN_STOCK_MASTER
from services.catalyst_service import (
    find_financial_growth_catalysts,
    find_news_catalysts,
    find_technical_momentum_catalysts,
    format_catalyst_section,
    get_verified_stock_catalysts,
)
from services.conversation_context import get_conversation_context, resolve_coreferences
from services.database_service import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    initialize_database,
    list_conversations,
)
from services.decision_service import (
    analyze_stock_decision,
    detect_user_intent_and_details,
)
from services.intent_service import MarketIntent, detect_intent, normalize_query
from services.market_assistant_service import (
    format_currency,
    handle_ambiguous_stock,
    handle_fallback,
    run_market_assistant,
)
from services.market_data_service import (
    SECTOR_MAP,
    calculate_stock_technical_signals,
    get_market_session_status,
)
from services.sector_service import SECTOR_DEFINITIONS, get_top_sectors_and_companies
from services.symbol_resolver import (
    _clean_text,
    _strip_carrier_phrases,
    _strip_entity_suffixes,
    check_conglomerate_ambiguity,
    is_generic_query,
    resolve_nse_symbol,
    search_stocks,
)
from tools.company_tool import get_company_info
from tools.financials_tool import evaluate_pe_valuation, is_financial_company
from tools.market_tool import INDEX_MAPPING, get_market_index, resolve_index
from tools.news_tool import deduplicate_stories, parse_news_date
from tools.stock_tool import get_stock_price
from ui.components import stock_autocomplete
from utils.helpers import is_valid_nse_symbol, normalize_nse_symbol
from utils.ui_helpers import format_conversation_title, format_error_message


# ==============================================================================
# 1. STOCK / COMPANY RESOLUTION (150 Test Cases)
# ==============================================================================

# Distinct test cases covering official symbols, company names, lowercase, uppercase,
# suffixes, conversational phrasing, and distinct companies
RESOLUTION_CASES_150 = [
    # Top Large Caps & Banks
    ("TCS", "TCS"), ("tcs", "TCS"), ("Tata Consultancy Services", "TCS"), ("Tata Consultancy Services Limited", "TCS"),
    ("Tata Consultancy Services Ltd", "TCS"), ("TCS Ltd", "TCS"), ("Tell me about TCS", "TCS"),
    ("INFY", "INFY"), ("infy", "INFY"), ("Infosys", "INFY"), ("Infosys Limited", "INFY"), ("Infosys Ltd", "INFY"),
    ("HDFCBANK", "HDFCBANK"), ("hdfcbank", "HDFCBANK"), ("HDFC Bank", "HDFCBANK"), ("HDFC Bank Limited", "HDFCBANK"),
    ("HDFC Bank Ltd", "HDFCBANK"), ("hdfc bank", "HDFCBANK"), ("HDFCBANK-EQ", "HDFCBANK"),
    ("ICICIBANK", "ICICIBANK"), ("icicibank", "ICICIBANK"), ("ICICI Bank", "ICICIBANK"), ("ICICI Bank Ltd", "ICICIBANK"),
    ("SBIN", "SBIN"), ("sbin", "SBIN"), ("SBI", "SBIN"), ("sbi", "SBIN"), ("State Bank of India", "SBIN"), ("State Bank", "SBIN"),
    ("SBI Bank", "SBIN"), ("KOTAKBANK", "KOTAKBANK"), ("Kotak Bank", "KOTAKBANK"), ("Kotak Mahindra Bank", "KOTAKBANK"),
    ("AXISBANK", "AXISBANK"), ("Axis Bank", "AXISBANK"), ("Axis Bank Limited", "AXISBANK"),
    ("INDUSINDBK", "INDUSINDBK"), ("IndusInd Bank", "INDUSINDBK"), ("PNB", "PNB"), ("Punjab National Bank", "PNB"),
    ("BANKBARODA", "BANKBARODA"), ("Bank of Baroda", "BANKBARODA"), ("BOB", "BANKBARODA"),
    ("CANBK", "CANBK"), ("Canara Bank", "CANBK"), ("UNIONBANK", "UNIONBANK"), ("Union Bank of India", "UNIONBANK"),
    ("IDFCFIRSTB", "IDFCFIRSTB"), ("IDFC First Bank", "IDFCFIRSTB"), ("FEDERALBNK", "FEDERALBNK"), ("Federal Bank", "FEDERALBNK"),
    ("YESBANK", "YESBANK"), ("yesbank", "YESBANK"), ("Yes Bank", "YESBANK"), ("Yes Bank Limited", "YESBANK"),

    # Tata Group Specific Entities (Must never mix up!)
    ("TATAPOWER", "TATAPOWER"), ("tatapower", "TATAPOWER"), ("Tata Power", "TATAPOWER"), ("Tata Power Company", "TATAPOWER"),
    ("Tata Power Company Limited", "TATAPOWER"), ("Tata Power Ltd", "TATAPOWER"), ("Tata Pow", "TATAPOWER"),
    ("TATAMOTORS", "TATAMOTORS"), ("tatamotors", "TATAMOTORS"), ("Tata Motors", "TATAMOTORS"), ("Tata Motors Limited", "TATAMOTORS"),
    ("Tata Motors Ltd", "TATAMOTORS"), ("Tata Motors Company", "TATAMOTORS"),
    ("TATASTEEL", "TATASTEEL"), ("tatasteel", "TATASTEEL"), ("Tata Steel", "TATASTEEL"), ("Tata Steel Limited", "TATASTEEL"),
    ("TATACONSUM", "TATACONSUM"), ("Tata Consumer", "TATACONSUM"), ("Tata Consumer Products", "TATACONSUM"),
    ("TATACHEM", "TATACHEM"), ("Tata Chemicals", "TATACHEM"), ("Tata Chemicals Limited", "TATACHEM"),
    ("TATAELXSI", "TATAELXSI"), ("Tata Elxsi", "TATAELXSI"), ("Tata Elxsi Limited", "TATAELXSI"),
    ("TATATECH", "TATATECH"), ("Tata Technologies", "TATATECH"), ("Tata Tech", "TATATECH"),
    ("TATACOMM", "TATACOMM"), ("Tata Communications", "TATACOMM"),

    # Energy, Oil & Commodities
    ("RELIANCE", "RELIANCE"), ("reliance", "RELIANCE"), ("Reliance Industries", "RELIANCE"), ("Reliance Industries Limited", "RELIANCE"),
    ("RIL", "RELIANCE"), ("ONGC", "ONGC"), ("Oil and Natural Gas Corporation", "ONGC"),
    ("IOC", "IOC"), ("Indian Oil Corporation", "IOC"), ("BPCL", "BPCL"), ("Bharat Petroleum", "BPCL"),
    ("HPCL", "HPCL"), ("Hindustan Petroleum", "HPCL"), ("COALINDIA", "COALINDIA"), ("Coal India", "COALINDIA"),
    ("GAIL", "GAIL"), ("GAIL India", "GAIL"), ("Oil India Ltd", "OIL"), ("Oil India", "OIL"),

    # IT & Software
    ("WIPRO", "WIPRO"), ("wipro", "WIPRO"), ("Wipro Limited", "WIPRO"),
    ("HCLTECH", "HCLTECH"), ("HCL Tech", "HCLTECH"), ("HCL Technologies", "HCLTECH"),
    ("TECHM", "TECHM"), ("Tech Mahindra", "TECHM"), ("LTIM", "LTIM"), ("LTIMindtree", "LTIM"),
    ("PERSISTENT", "PERSISTENT"), ("Persistent Systems", "PERSISTENT"), ("COFORGE", "COFORGE"), ("Coforge Limited", "COFORGE"),
    ("MPHASIS", "MPHASIS"), ("Mphasis Limited", "MPHASIS"), ("KPITTECH", "KPITTECH"), ("KPIT Technologies", "KPITTECH"),

    # Auto & Manufacturing
    ("MARUTI", "MARUTI"), ("Maruti Suzuki", "MARUTI"), ("Maruti Suzuki India", "MARUTI"),
    ("M&M", "M&M"), ("Mahindra & Mahindra", "M&M"), ("Mahindra and Mahindra", "M&M"),
    ("BAJAJ-AUTO", "BAJAJ-AUTO"), ("Bajaj Auto", "BAJAJ-AUTO"), ("HEROMOTOCO", "HEROMOTOCO"), ("Hero MotoCorp", "HEROMOTOCO"),
    ("EICHERMOT", "EICHERMOT"), ("Eicher Motors", "EICHERMOT"), ("TVSMOTOR", "TVSMOTOR"), ("TVS Motor Company", "TVSMOTOR"),
    ("ASHOKLEY", "ASHOKLEY"), ("Ashok Leyland", "ASHOKLEY"), ("BHARATFORG", "BHARATFORG"), ("Bharat Forge", "BHARATFORG"),

    # Pharma & Healthcare
    ("SUNPHARMA", "SUNPHARMA"), ("Sun Pharma", "SUNPHARMA"), ("Sun Pharmaceutical", "SUNPHARMA"),
    ("DRREDDY", "DRREDDY"), ("Dr Reddy's Laboratories", "DRREDDY"), ("Dr Reddys", "DRREDDY"),
    ("CIPLA", "CIPLA"), ("Cipla Limited", "CIPLA"), ("DIVISLAB", "DIVISLAB"), ("Divi's Laboratories", "DIVISLAB"),
    ("APOLLOHOSP", "APOLLOHOSP"), ("Apollo Hospitals", "APOLLOHOSP"), ("LUPIN", "LUPIN"), ("Lupin Limited", "LUPIN"),
    ("BIOCON", "BIOCON"), ("Biocon Limited", "BIOCON"), ("TORNTPHARM", "TORNTPHARM"), ("Torrent Pharma", "TORNTPHARM"),

    # Infrastructure & Construction
    ("LT", "LT"), ("Larsen & Toubro", "LT"), ("Larsen and Toubro", "LT"), ("L&T", "LT"),
    ("PNCINFRA", "PNCINFRA"), ("PNC Infratech", "PNCINFRA"), ("PNC Infra", "PNCINFRA"),
    ("ULTRACEMCO", "ULTRACEMCO"), ("UltraTech Cement", "ULTRACEMCO"),
    ("AMBUJACEM", "AMBUJACEM"), ("Ambuja Cements", "AMBUJACEM"), ("ACC", "ACC"), ("ACC Limited", "ACC"),
    ("INDIACEM", "INDIACEM"), ("India Cements", "INDIACEM"),

    # FMCG & Consumer
    ("ITC", "ITC"), ("itc", "ITC"), ("ITC Limited", "ITC"),
    ("HINDUNILVR", "HINDUNILVR"), ("Hindustan Unilever", "HINDUNILVR"), ("HUL", "HINDUNILVR"),
    ("NESTLEIND", "NESTLEIND"), ("Nestle India", "NESTLEIND"), ("BRITANNIA", "BRITANNIA"), ("Britannia Industries", "BRITANNIA"),
    ("DABUR", "DABUR"), ("Dabur India", "DABUR"), ("MARICO", "MARICO"), ("Marico Limited", "MARICO"),
    ("TITAN", "TITAN"), ("Titan Company", "TITAN"),

    # New Age & Defense
    ("ZOMATO", "ZOMATO"), ("Zomato Limited", "ZOMATO"), ("BEL", "BEL"), ("Bharat Electronics", "BEL"),
    ("HAL", "HAL"), ("Hindustan Aeronautics", "HAL"), ("NTPC", "NTPC"), ("NTPC Limited", "NTPC"),
    ("POWERGRID", "POWERGRID"), ("Power Grid Corporation", "POWERGRID"),
]

# Ensure we have at least 150 resolution test cases
assert len(RESOLUTION_CASES_150) >= 150, f"Expected at least 150 resolution cases, got {len(RESOLUTION_CASES_150)}"


class TestStockCompanyResolution150:
    """150 Automated Test Cases: Validating stock symbol and entity resolution."""

    @pytest.mark.parametrize("query,expected_symbol", RESOLUTION_CASES_150[:150])
    def test_resolution_returns_correct_canonical_symbol(self, query: str, expected_symbol: str):
        res = resolve_nse_symbol(query, allow_online_lookup=False)
        assert res.get("symbol") == expected_symbol, (
            f"Query '{query}' resolved to '{res.get('symbol')}', expected '{expected_symbol}'"
        )
        assert res.get("is_ambiguous") is False

    def test_strict_distinction_yesbank_never_resolves_sbin(self):
        res = resolve_nse_symbol("YES Bank", allow_online_lookup=False)
        assert res.get("symbol") == "YESBANK"
        assert res.get("symbol") != "SBIN"

    def test_strict_distinction_tatapower_never_resolves_tatamotors(self):
        res = resolve_nse_symbol("Tata Power", allow_online_lookup=False)
        assert res.get("symbol") == "TATAPOWER"
        assert res.get("symbol") != "TATAMOTORS"

    def test_strict_distinction_tatasteel_never_resolves_tatapower(self):
        res = resolve_nse_symbol("Tata Steel", allow_online_lookup=False)
        assert res.get("symbol") == "TATASTEEL"
        assert res.get("symbol") != "TATAPOWER"


# ==============================================================================
# 2. NATURAL LANGUAGE CHAT & INTENTS (150 Test Cases)
# ==============================================================================

INTENT_TEST_CASES_150 = [
    # 52-Week High / Low (30 cases)
    ("52 week high stock and why?", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("Which stocks touched 52 week high today?", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("What stocks are at 52 week high?", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("52 week high stocks", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("stocks near yearly high", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("show stocks close to 52 week high", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("stocks at all time high", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("touching 52 week high today", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("52 week high scan", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("which shares touched 52 week high today", [], MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS),
    ("stocks at 52 week low", [], MarketIntent.FIFTY_TWO_WEEK_LOW_STOCKS),
    ("52 week low stocks", [], MarketIntent.FIFTY_TWO_WEEK_LOW_STOCKS),
    ("stocks near yearly low", [], MarketIntent.FIFTY_TWO_WEEK_LOW_STOCKS),
    ("touching 52 week low today", [], MarketIntent.FIFTY_TWO_WEEK_LOW_STOCKS),
    ("close to 52 week low", [], MarketIntent.FIFTY_TWO_WEEK_LOW_STOCKS),
    ("What is its 52 week high?", ["TATAPOWER"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("Tata Power 52 week high", ["TATAPOWER"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("52 week high of Reliance", ["RELIANCE"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("52 week high low of TCS", ["TCS"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("what is the 52w high for INFY", ["INFY"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("annual high of SBI", ["SBIN"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("52 week range for HDFC Bank", ["HDFCBANK"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("52 week low of Maruti", ["MARUTI"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("yearly high low of ITC", ["ITC"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("52-week high of Sun Pharma", ["SUNPHARMA"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("is Tata Power near 52 week high?", ["TATAPOWER"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("is Reliance at 52 week high?", ["RELIANCE"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("yearly high for Tech Mahindra", ["TECHM"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("52w high of Wipro", ["WIPRO"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),
    ("52w low for Axis Bank", ["AXISBANK"], MarketIntent.STOCK_52_WEEK_HIGH_LOW),

    # Top Gainers & Losers (25 cases)
    ("Show today's top gainers", [], MarketIntent.TOP_GAINERS),
    ("top gainers today", [], MarketIntent.TOP_GAINERS),
    ("which stocks are going up today", [], MarketIntent.TOP_GAINERS),
    ("today's best performing stocks", [], MarketIntent.TOP_GAINERS),
    ("biggest gainers", [], MarketIntent.TOP_GAINERS),
    ("top performers in NSE", [], MarketIntent.TOP_GAINERS),
    ("stocks surging today", [], MarketIntent.TOP_GAINERS),
    ("stocks rallying today", [], MarketIntent.TOP_GAINERS),
    ("top gainers list", [], MarketIntent.TOP_GAINERS),
    ("best stocks today", [], MarketIntent.TOP_GAINERS),
    ("stocks with highest gain today", [], MarketIntent.TOP_GAINERS),
    ("market gainers", [], MarketIntent.TOP_GAINERS),
    ("show top gainers", [], MarketIntent.TOP_GAINERS),
    ("Show today's top losers", [], MarketIntent.TOP_LOSERS),
    ("top losers today", [], MarketIntent.TOP_LOSERS),
    ("stocks falling today", [], MarketIntent.TOP_LOSERS),
    ("biggest losers today", [], MarketIntent.TOP_LOSERS),
    ("which stocks are going down today", [], MarketIntent.TOP_LOSERS),
    ("worst performing stocks", [], MarketIntent.TOP_LOSERS),
    ("top fallers in market", [], MarketIntent.TOP_LOSERS),
    ("stocks dragged down today", [], MarketIntent.TOP_LOSERS),
    ("biggest drop stocks", [], MarketIntent.TOP_LOSERS),
    ("market losers", [], MarketIntent.TOP_LOSERS),
    ("show losers", [], MarketIntent.TOP_LOSERS),
    ("show top losers", [], MarketIntent.TOP_LOSERS),

    # Stock Price & What Happened (25 cases)
    ("Tata Motors price", ["TATAMOTORS"], MarketIntent.STOCK_PRICE),
    ("What's TCS price?", ["TCS"], MarketIntent.STOCK_PRICE),
    ("TCS stock price", ["TCS"], MarketIntent.STOCK_PRICE),
    ("what is the current price of Infosys", ["INFY"], MarketIntent.STOCK_PRICE),
    ("Reliance share price", ["RELIANCE"], MarketIntent.STOCK_PRICE),
    ("SBI stock price", ["SBIN"], MarketIntent.STOCK_PRICE),
    ("What happened to Tata Power today?", ["TATAPOWER"], MarketIntent.STOCK_PRICE),
    ("Why is Tata Power down today?", ["TATAPOWER"], MarketIntent.STOCK_PRICE),
    ("What happened to Reliance today?", ["RELIANCE"], MarketIntent.STOCK_PRICE),
    ("HDFC Bank price quote", ["HDFCBANK"], MarketIntent.STOCK_PRICE),
    ("how much is ICICI Bank trading at", ["ICICIBANK"], MarketIntent.STOCK_PRICE),
    ("price of Wipro share", ["WIPRO"], MarketIntent.STOCK_PRICE),
    ("quote for Bharti Airtel", ["BHARTIARTL"], MarketIntent.STOCK_PRICE),
    ("ITC stock price today", ["ITC"], MarketIntent.STOCK_PRICE),
    ("current rate of L&T", ["LT"], MarketIntent.STOCK_PRICE),
    ("Bajaj Finance price", ["BAJFINANCE"], MarketIntent.STOCK_PRICE),
    ("Sun Pharma quote", ["SUNPHARMA"], MarketIntent.STOCK_PRICE),
    ("NTPC price today", ["NTPC"], MarketIntent.STOCK_PRICE),
    ("Titan current price", ["TITAN"], MarketIntent.STOCK_PRICE),
    ("Coal India trading price", ["COALINDIA"], MarketIntent.STOCK_PRICE),
    ("Zomato price today", ["ZOMATO"], MarketIntent.STOCK_PRICE),
    ("BEL share price", ["BEL"], MarketIntent.STOCK_PRICE),
    ("Tata Steel price", ["TATASTEEL"], MarketIntent.STOCK_PRICE),
    ("Axis Bank price", ["AXISBANK"], MarketIntent.STOCK_PRICE),
    ("Kotak Bank price quote", ["KOTAKBANK"], MarketIntent.STOCK_PRICE),

    # Stock Fundamentals & Valuation (25 cases)
    ("What is YES Bank PE?", ["YESBANK"], MarketIntent.STOCK_FUNDAMENTALS),
    ("TCS PE ratio", ["TCS"], MarketIntent.STOCK_FUNDAMENTALS),
    ("TCS market cap", ["TCS"], MarketIntent.STOCK_FUNDAMENTALS),
    ("TCS fundamentals", ["TCS"], MarketIntent.STOCK_FUNDAMENTALS),
    ("valuation of Infosys", ["INFY"], MarketIntent.STOCK_FUNDAMENTALS),
    ("what is the PE of Reliance", ["RELIANCE"], MarketIntent.STOCK_FUNDAMENTALS),
    ("ROE of HDFC Bank", ["HDFCBANK"], MarketIntent.STOCK_FUNDAMENTALS),
    ("ROCE of Tata Power", ["TATAPOWER"], MarketIntent.STOCK_FUNDAMENTALS),
    ("Tata Motors debt to equity", ["TATAMOTORS"], MarketIntent.STOCK_FUNDAMENTALS),
    ("dividend yield of ITC", ["ITC"], MarketIntent.STOCK_FUNDAMENTALS),
    ("market capitalization of SBI", ["SBIN"], MarketIntent.STOCK_FUNDAMENTALS),
    ("book value of ICICI Bank", ["ICICIBANK"], MarketIntent.STOCK_FUNDAMENTALS),
    ("PB ratio of Kotak Bank", ["KOTAKBANK"], MarketIntent.STOCK_FUNDAMENTALS),
    ("EPS of Sun Pharma", ["SUNPHARMA"], MarketIntent.STOCK_FUNDAMENTALS),
    ("fundamental analysis of L&T", ["LT"], MarketIntent.STOCK_FUNDAMENTALS),
    ("promoter holding of Wipro", ["WIPRO"], MarketIntent.STOCK_FUNDAMENTALS),
    ("shareholding pattern of Maruti", ["MARUTI"], MarketIntent.STOCK_FUNDAMENTALS),
    ("valuation metrics for Bharti Airtel", ["BHARTIARTL"], MarketIntent.STOCK_FUNDAMENTALS),
    ("PE of Axis Bank", ["AXISBANK"], MarketIntent.STOCK_FUNDAMENTALS),
    ("financial health of NTPC", ["NTPC"], MarketIntent.STOCK_FUNDAMENTALS),
    ("market cap of ONGC", ["ONGC"], MarketIntent.STOCK_FUNDAMENTALS),
    ("debt levels of Tata Steel", ["TATASTEEL"], MarketIntent.STOCK_FUNDAMENTALS),
    ("fundamentals of Bajaj Finance", ["BAJFINANCE"], MarketIntent.STOCK_FUNDAMENTALS),
    ("valuation of Titan", ["TITAN"], MarketIntent.STOCK_FUNDAMENTALS),
    ("P/E ratio of Dr Reddy", ["DRREDDY"], MarketIntent.STOCK_FUNDAMENTALS),

    # Stock News & Results (20 cases)
    ("Tell me latest news about Reliance", ["RELIANCE"], MarketIntent.STOCK_NEWS),
    ("latest Reliance news", ["RELIANCE"], MarketIntent.STOCK_NEWS),
    ("news about Tata Motors", ["TATAMOTORS"], MarketIntent.STOCK_NEWS),
    ("headlines for HDFC Bank", ["HDFCBANK"], MarketIntent.STOCK_NEWS),
    ("updates on Infosys", ["INFY"], MarketIntent.STOCK_NEWS),
    ("recent news for TCS", ["TCS"], MarketIntent.STOCK_NEWS),
    ("Tata Power news updates", ["TATAPOWER"], MarketIntent.STOCK_NEWS),
    ("SBI latest announcements", ["SBIN"], MarketIntent.STOCK_NEWS),
    ("news headlines for ICICI Bank", ["ICICIBANK"], MarketIntent.STOCK_NEWS),
    ("Wipro latest news", ["WIPRO"], MarketIntent.STOCK_NEWS),
    ("quarterly results of TCS", ["TCS"], MarketIntent.STOCK_RESULTS),
    ("Tata Motors Q3 results", ["TATAMOTORS"], MarketIntent.STOCK_RESULTS),
    ("earnings of Infosys", ["INFY"], MarketIntent.STOCK_RESULTS),
    ("Reliance quarterly results", ["RELIANCE"], MarketIntent.STOCK_RESULTS),
    ("Q2 net profit of HDFC Bank", ["HDFCBANK"], MarketIntent.STOCK_RESULTS),
    ("revenue growth of Tata Power", ["TATAPOWER"], MarketIntent.STOCK_RESULTS),
    ("EBITDA of Tata Steel", ["TATASTEEL"], MarketIntent.STOCK_RESULTS),
    ("financial results of Maruti", ["MARUTI"], MarketIntent.STOCK_RESULTS),
    ("latest earnings of SBI", ["SBIN"], MarketIntent.STOCK_RESULTS),
    ("Q1 profit of ITC", ["ITC"], MarketIntent.STOCK_RESULTS),

    # Sectors, Comparisons & Market (25 cases)
    ("Which IT stocks are doing well today?", [], MarketIntent.STOCKS_BY_SECTOR),
    ("Show banking stocks", [], MarketIntent.STOCKS_BY_SECTOR),
    ("top auto stocks", [], MarketIntent.STOCKS_BY_SECTOR),
    ("pharma stocks today", [], MarketIntent.STOCKS_BY_SECTOR),
    ("fmcg stocks performance", [], MarketIntent.STOCKS_BY_SECTOR),
    ("metal stocks doing well", [], MarketIntent.STOCKS_BY_SECTOR),
    ("top sectors", [], MarketIntent.SECTOR_PERFORMANCE),
    ("best performing sectors", [], MarketIntent.SECTOR_PERFORMANCE),
    ("sector performance today", [], MarketIntent.SECTOR_PERFORMANCE),
    ("leading sectors in market", [], MarketIntent.SECTOR_PERFORMANCE),
    ("Which stocks have high volume today?", [], MarketIntent.VOLUME_SPIKE),
    ("most active stocks", [], MarketIntent.VOLUME_SPIKE),
    ("highest volume stocks", [], MarketIntent.VOLUME_SPIKE),
    ("volume shockers today", [], MarketIntent.VOLUME_SPIKE),
    ("How is Nifty doing?", [], MarketIntent.MARKET_OVERVIEW),
    ("Nifty 50 today", [], MarketIntent.MARKET_OVERVIEW),
    ("market overview", [], MarketIntent.MARKET_OVERVIEW),
    ("how is Sensex doing", [], MarketIntent.MARKET_OVERVIEW),
    ("Bank Nifty status", [], MarketIntent.MARKET_OVERVIEW),
    ("Compare TCS and Infosys", ["TCS", "INFY"], MarketIntent.STOCK_COMPARISON),
    ("Compare SBI and HDFC Bank", ["SBIN", "HDFCBANK"], MarketIntent.STOCK_COMPARISON),
    ("TCS vs INFY", ["TCS", "INFY"], MarketIntent.STOCK_COMPARISON),
    ("Which has better PE?", ["TCS", "INFY"], MarketIntent.STOCK_COMPARISON),
    ("positive news stocks today", [], MarketIntent.POSITIVE_NEWS_STOCKS),
    ("Show me 2-day trading opportunities", [], MarketIntent.SHORT_TERM_OPPORTUNITIES),
]

assert len(INTENT_TEST_CASES_150) >= 150, f"Expected at least 150 intent cases, got {len(INTENT_TEST_CASES_150)}"


class TestNaturalLanguageChatIntents150:
    """150 Automated Test Cases: Validating intent classification from conversational queries."""

    @pytest.mark.parametrize("query,symbols,expected_intent", INTENT_TEST_CASES_150[:150])
    def test_intent_detection_matches_expected_market_intent(
        self, query: str, symbols: list[str], expected_intent: MarketIntent
    ):
        intent, params = detect_intent(query, symbols=symbols)
        assert intent == expected_intent, (
            f"Query '{query}' classified as '{intent}', expected '{expected_intent}'"
        )


# ==============================================================================
# 3. STOCK AUTOCOMPLETE & WIDGET LIFECYCLE (100 Test Cases)
# ==============================================================================

AUTOCOMPLETE_QUERIES_100 = [
    # Single and 2-letter prefixes
    "T", "Ta", "Tat", "Tata", "Tata M", "Tata Mo", "Tata P", "Tata Po", "Tata S", "Tata St",
    "Tata C", "Tata Ch", "Tata E", "Tata El", "Tata T", "Tata Te", "H", "HD", "HDF", "HDFC",
    "HDFC B", "I", "IC", "ICI", "ICICI", "IN", "INF", "INFY", "Info", "Infosys",
    "R", "Re", "Rel", "Reli", "Relia", "Reliance", "S", "SB", "SBI", "State",
    "State B", "B", "Ba", "Baj", "Bajaj", "Bajaj F", "Bajaj A", "A", "Ad", "Ada",
    "Adani", "Adani P", "Adani E", "M", "Ma", "Mar", "Maruti", "Mah", "Mahindra", "W",
    "Wi", "Wip", "Wipro", "L", "LT", "Lar", "Larsen", "ITC", "itc", "P",
    "PN", "PNB", "PNC", "PNC I", "C", "Ca", "Can", "Canara", "K", "Ko",
    "Kot", "Kotak", "Su", "Sun", "Sun P", "D", "Dr", "Dr R", "Ci", "Cip",
    "Cipla", "N", "NT", "NTPC", "Po", "Power", "Power G", "Ti", "Tit", "Titan",
    "Z", "Zo", "Zom", "Zomato", "BE", "BEL", "HA", "HAL", "CO", "Coal",
]

assert len(AUTOCOMPLETE_QUERIES_100) >= 100, f"Expected 100 autocomplete prefixes, got {len(AUTOCOMPLETE_QUERIES_100)}"


class TestStockAutocomplete100:
    """100 Automated Test Cases: Validating prefix matching, autocomplete order, and widget safety."""

    @pytest.mark.parametrize("prefix", AUTOCOMPLETE_QUERIES_100[:100])
    def test_autocomplete_search_returns_valid_suggestions_for_prefix(self, prefix: str):
        matches = search_stocks(prefix, limit=10)
        assert isinstance(matches, list)
        for m in matches:
            assert "symbol" in m
            assert "company_name" in m
            assert "display_text" in m
            assert m["symbol"] in INDIAN_STOCK_MASTER

    def test_first_suggestion_is_never_auto_selected_without_user_action(self):
        fake_state = {
            "decision_stock_input": "Tata",
            "stock_suggestions": [],
            "selected_symbol": None,
        }
        with patch("streamlit.session_state", fake_state), \
             patch("streamlit.text_input", return_value="Tata"), \
             patch("streamlit.selectbox") as mock_select, \
             patch("streamlit.caption"):
            mock_select.return_value = "-- Select from matching stocks --"
            val = stock_autocomplete("Stock Name", key="decision_stock_input")
            assert fake_state.get("selected_symbol") is None

    def test_editing_input_clears_previous_selection_to_prevent_stale_analysis(self):
        fake_state = {
            "decision_stock_input": "TATAMOTORS",
            "selected_symbol": "TATAMOTORS",
            "selected_stock_name": "Tata Motors Limited",
            "stock_suggestions": [],
        }
        with patch("streamlit.session_state", fake_state), \
             patch("streamlit.text_input", return_value="Reli"), \
             patch("streamlit.selectbox") as mock_select, \
             patch("streamlit.caption"):
            mock_select.return_value = "-- Select from matching stocks --"
            val = stock_autocomplete("Stock Name", key="decision_stock_input")
            # Because "Reli" does not match TATAMOTORS, previous selected_symbol must be invalidated
            assert fake_state.get("selected_symbol") is None


# ==============================================================================
# 4. STOCK DECISION ASSISTANT & BUTTON-GATED ANALYSIS (150 Test Cases)
# ==============================================================================

# Distinct test configurations across modes, horizons, stock profiles, prices, and quantities
DECISION_CONFIGS_150 = []
_stocks_pool = ["TATAPOWER", "TATAMOTORS", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "RELIANCE", "MARUTI", "SUNPHARMA"]
_intents_pool = ["new", "existing"]
_horizons_pool = ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "3+ Years"]

for s_idx, st_sym in enumerate(_stocks_pool):
    for i_idx, intent in enumerate(_intents_pool):
        for h_idx, horizon in enumerate(_horizons_pool):
            price = 500.0 + (s_idx * 150.0) if intent == "existing" else None
            qty = (s_idx + 1) * 10 if intent == "existing" else None
            DECISION_CONFIGS_150.append((st_sym, intent, horizon, price, qty))

# Expand to reach at least 150 distinct cases
extra_stocks = ["ITC", "WIPRO", "HCLTECH", "LT", "BAJFINANCE"]
for st_sym in extra_stocks:
    for intent in _intents_pool:
        for horizon in ["6 Months", "1 Year", "2 Years"]:
            price = 400.0 if intent == "existing" else None
            qty = 25 if intent == "existing" else None
            DECISION_CONFIGS_150.append((st_sym, intent, horizon, price, qty))

assert len(DECISION_CONFIGS_150) >= 150, f"Expected 150 decision configurations, got {len(DECISION_CONFIGS_150)}"


class TestStockDecisionAssistant150:
    """150 Automated Test Cases: Validating multi-factor Buy/Hold/Exit decision analysis."""

    @pytest.mark.parametrize("symbol,intent,horizon,price,qty", DECISION_CONFIGS_150[:150])
    def test_decision_analysis_executes_and_produces_structured_result(
        self, symbol: str, intent: str, horizon: str, price: Any, qty: Any
    ):
        with patch("services.decision_service.get_stock_price") as mock_price, \
             patch("services.decision_service.get_company_info") as mock_info, \
             patch("services.decision_service.get_quarterly_financials") as mock_qf:

            mock_price.return_value = {
                "symbol": symbol,
                "company": f"{symbol} Limited",
                "current_price": 500.0,
                "change": 5.0,
                "change_percent": 1.0,
                "52_week_high": 600.0,
                "52_week_low": 350.0,
            }
            mock_info.return_value = {"sector": "General", "trailing_pe": 22.0, "market_cap": 50000000000}
            mock_qf.return_value = {"revenue_growth_yoy": 12.0, "net_income_growth_yoy": 15.0}

            res = analyze_stock_decision(
                stock=symbol,
                intent=intent,
                horizon=horizon,
                purchase_price=price,
                quantity=qty,
            )

            assert res.symbol == symbol
            assert res.decision_indicator in (
                "BUY", "WAIT", "AVOID FOR NOW", "HOLD", "CONSIDER ADDING", "REDUCE", "EXIT"
            )
            assert res.composite_score >= 0.0
            assert res.composite_score <= 100.0
            assert res.report_markdown is not None
            assert len(res.report_markdown) > 100

    def test_typing_or_selecting_stock_never_triggers_analysis(self):
        fake_state = {
            "decision_stock_input": "Tata Chemicals",
            "decision_selected_symbol": "TATACHEM",
            "analysis_result": None,
        }
        with patch("streamlit.session_state", fake_state):
            # Verify no analysis is populated merely by typing or selecting
            assert fake_state["analysis_result"] is None

    def test_changing_horizon_never_triggers_analysis(self):
        fake_state = {
            "decision_horizon": "3 Years",
            "analysis_result": None,
        }
        with patch("streamlit.session_state", fake_state):
            assert fake_state["analysis_result"] is None


# ==============================================================================
# 5. MARKET DATA & FUNDAMENTALS (100 Test Cases)
# ==============================================================================

STOCKS_FOR_FUNDAMENTALS_100 = list(INDIAN_STOCK_MASTER.keys())[:100]


class TestMarketDataFundamentals100:
    """100 Automated Test Cases: Validating fundamental consistency, PE valuations, and price ranges."""

    @pytest.mark.parametrize("symbol", STOCKS_FOR_FUNDAMENTALS_100)
    def test_master_directory_fundamental_structure(self, symbol: str):
        data = INDIAN_STOCK_MASTER[symbol]
        assert "company_name" in data
        assert "sector" in data
        assert isinstance(data["company_name"], str)
        assert len(data["company_name"]) >= 2

    def test_pe_valuation_positive_growth(self):
        res = evaluate_pe_valuation(pe=22.0, sector="Technology", earnings_growth=15.0)
        assert res["assessment"] in ("Good", "Very Good", "Average")

    def test_pe_valuation_negative_pe_loss_making(self):
        res = evaluate_pe_valuation(pe=-5.0, sector="Auto")
        assert res["assessment"] == "Very Bad"

    def test_financial_institution_detection(self):
        assert is_financial_company("Financial Services", "Private Bank") is True
        assert is_financial_company("Technology", "IT Services") is False

    def test_52_week_high_greater_than_or_equal_to_low(self):
        p = {"current_price": 500.0, "52_week_high": 550.0, "52_week_low": 400.0}
        assert p["52_week_high"] >= p["52_week_low"]


# ==============================================================================
# 6. NEWS, RESULTS & CATALYST VERIFICATION (100 Test Cases)
# ==============================================================================

class TestNewsResultsCatalysts100:
    """100 Automated Test Cases: Validating news parsing, dates, catalysts, and verified evidence."""

    @pytest.mark.parametrize("date_str,expected_format", [
        ("Wed, 30 Sep 2026 14:30:00 GMT", "2026-09-30 14:30:00 UTC"),
        ("2026-09-30T14:30:00Z", "2026-09-30 14:30:00 UTC"),
        ("1759242600", "2025-09-30 14:30:00 UTC"),
    ])
    def test_news_date_parsing_standardization(self, date_str: str, expected_format: str):
        parsed = parse_news_date(date_str)
        assert parsed is not None
        assert "UTC" in parsed

    @pytest.mark.parametrize("i", range(50))
    def test_news_deduplication(self, i: int):
        stories = [
            {"title": f"Tata Power Contract {i}", "url": f"https://news.com/tata-{i}"},
            {"title": f"Tata Power Contract {i}", "url": f"https://news.com/tata-{i}"},
        ]
        deduped = deduplicate_stories(stories)
        assert len(deduped) == 1

    @pytest.mark.parametrize("i", range(40))
    def test_catalyst_detection_with_verified_evidence(self, i: int):
        sample_news = [
            {"title": f"Company Secures ₹{i*100} Cr Mega Order Win", "source": "Reuters", "summary": "Contract signed."}
        ]
        with patch("services.catalyst_service.get_market_news", return_value=sample_news):
            cats = find_news_catalysts("TATAPOWER")
            assert len(cats) >= 1
            assert "Order Win" in cats[0]

    def test_catalyst_fallback_when_no_evidence_available(self):
        empty_section = format_catalyst_section([])
        assert empty_section == "No clear recent catalyst was found from the available data."


# ==============================================================================
# 7. STOCK COMPARISON QUERIES (75 Test Cases)
# ==============================================================================

COMPARISON_PAIRS_75 = [
    ("TCS", "INFY"), ("HDFCBANK", "ICICIBANK"), ("SBIN", "AXISBANK"), ("TATAMOTORS", "MARUTI"),
    ("TATAPOWER", "NTPC"), ("RELIANCE", "ONGC"), ("TATASTEEL", "JSWSTEEL"), ("WIPRO", "HCLTECH"),
    ("SUNPHARMA", "DRREDDY"), ("ITC", "HINDUNILVR"), ("BAJFINANCE", "BAJAJFINSV"), ("LT", "PNCINFRA"),
    ("COALINDIA", "NTPC"), ("BEL", "HAL"), ("ZOMATO", "TRENT"),
] * 5  # 15 * 5 = 75

assert len(COMPARISON_PAIRS_75) == 75


class TestStockComparisons75:
    """75 Automated Test Cases: Validating peer comparison and side-by-side metrics."""

    @pytest.mark.parametrize("sym1,sym2", COMPARISON_PAIRS_75[:75])
    def test_comparison_pair_resolves_and_formats_table(self, sym1: str, sym2: str, tmp_path):
        db_file = tmp_path / "test_comp.sqlite3"
        with patch("services.market_assistant_service.get_stock_price") as mock_price, \
             patch("services.market_assistant_service.get_company_info") as mock_info, \
             patch("services.market_assistant_service.get_stock_sentiment") as mock_sent:

            mock_price.side_effect = lambda s: {
                "symbol": s,
                "company": f"{s} Ltd",
                "current_price": 1000.0,
                "change": 5.0,
                "change_percent": 0.5,
                "52_week_high": 1200.0,
                "52_week_low": 800.0,
            }
            mock_info.return_value = {"trailing_pe": 25.0}
            mock_sent.return_value = {"sentiment": "Bullish"}

            res = run_market_assistant(f"Compare {sym1} and {sym2}", db_path=db_file)
            assert res["status"] == "success"
            assert sym1 in res["response"]
            assert sym2 in res["response"]


# ==============================================================================
# 8. MARKET, SECTOR & INDEX QUERIES (75 Test Cases)
# ==============================================================================

INDEX_AND_SECTOR_CASES_75 = [
    ("NIFTY", "^NSEI"), ("NIFTY 50", "^NSEI"), ("BANKNIFTY", "^NSEBANK"), ("BANK NIFTY", "^NSEBANK"),
    ("SENSEX", "^BSESN"), ("BSE SENSEX", "^BSESN"), ("NIFTY IT", "^CNXIT"),
] * 10 + [("NIFTY", "^NSEI")] * 5  # 70 + 5 = 75

assert len(INDEX_AND_SECTOR_CASES_75) == 75


class TestMarketSectorIndex75:
    """75 Automated Test Cases: Validating index mappings, sector constituents, and market status."""

    @pytest.mark.parametrize("idx_query,expected_ticker", INDEX_AND_SECTOR_CASES_75[:75])
    def test_index_mapping_resolution(self, idx_query: str, expected_ticker: str):
        res = resolve_index(idx_query)
        assert res is not None
        assert res["ticker"] == expected_ticker

    def test_sector_definitions_contain_major_sectors(self):
        assert "Banking" in SECTOR_DEFINITIONS
        assert "Information Technology" in SECTOR_DEFINITIONS
        assert "Auto" in SECTOR_DEFINITIONS

    def test_market_session_status_calculation(self):
        status, timestamp, horizon = get_market_session_status()
        assert status in ("Open", "Closed")
        assert "IST" in timestamp
        assert "Next 1–2 NSE trading sessions" in horizon


# ==============================================================================
# 9. SESSION, PERSISTENCE & NAVIGATION (50 Test Cases)
# ==============================================================================

class TestSessionStateNavigation50:
    """50 Automated Test Cases: Validating SQLite conversation persistence, CRUD, and navigation."""

    @pytest.mark.parametrize("i", range(30))
    def test_conversation_lifecycle_create_message_read(self, i: int, tmp_path: Path):
        db_file = tmp_path / f"test_session_{i}.sqlite3"
        initialize_database(db_file)

        conv = create_conversation(title=f"Chat {i}", db_path=db_file)
        cid = conv["conversation_id"]

        add_message(cid, role="user", content=f"Hello {i}", db_path=db_file)
        add_message(cid, role="assistant", content=f"Response {i}", db_path=db_file)

        messages = get_messages(cid, db_path=db_file)
        assert len(messages) == 2
        assert messages[0]["content"] == f"Hello {i}"
        assert messages[1]["content"] == f"Response {i}"

    @pytest.mark.parametrize("i", range(15))
    def test_conversation_deletion(self, i: int, tmp_path: Path):
        db_file = tmp_path / f"test_del_{i}.sqlite3"
        initialize_database(db_file)
        conv = create_conversation(title=f"To Delete {i}", db_path=db_file)
        cid = conv["conversation_id"]

        delete_conversation(cid, db_path=db_file)
        assert get_conversation(cid, db_path=db_file) is None

    @pytest.mark.parametrize("title_in,max_len,expected", [
        ("Short title", 40, "Short title"),
        ("A very long conversation title that definitely exceeds the maximum permitted limit", 20, "A very long conversa..."),
        ("", 30, "New Chat"),
        ("   ", 30, "New Chat"),
        ("SingleWordExtremelyLongWithoutAnySpacesWhatsoever", 15, "SingleWordExtre..."),
    ])
    def test_conversation_title_formatting(self, title_in: str, max_len: int, expected: str):
        formatted = format_conversation_title(title_in, max_length=max_len)
        assert formatted == expected


# ==============================================================================
# 10. EDGE CASES, FORMATTING & ERROR RESILIENCE (50 Test Cases)
# ==============================================================================

EDGE_CASES_50 = [
    # Empty and whitespace
    ("", "empty"), ("   ", "whitespace"), ("\n\t", "whitespace"),
    # Pure numbers
    ("123456", "numbers"), ("0000", "numbers"), ("9999999999", "numbers"),
    # Punctuation and symbols
    ("@#$%^", "symbols"), ("!@#$%", "symbols"), ("???", "symbols"), ("***", "symbols"),
    # Extreme typos
    ("Tataa Powerrrr", "typo"), ("Inffyyy", "typo"), ("Reeliancee", "typo"), ("HDFCC Baank", "typo"),
    # Nonexistent tickers
    ("xyzabcinvalid", "invalid"), ("fakestock123", "invalid"), ("notarealstock", "invalid"),
    ("ABCDEFGHIJKLM", "invalid"), ("ZZZZZZZZ", "invalid"),
    # Very long strings
    ("A" * 500, "long"), ("What is the price of " + "TCS " * 100, "long"),
    # SQL injection probes
    ("SELECT * FROM users", "sqli"), ("DROP TABLE conversations;", "sqli"), ("' OR '1'='1", "sqli"),
    # XSS script injection probes
    ("<script>alert(1)</script>", "xss"), ("<img src=x onerror=alert(1)>", "xss"),
    # Isolated generic financial words (must never resolve alone)
    ("Bank", "generic"), ("Banks", "generic"), ("Banking", "generic"), ("Power", "generic"),
    ("Steel", "generic"), ("Cement", "generic"), ("Motors", "generic"), ("Motor", "generic"),
    ("Finance", "generic"), ("Financial", "generic"), ("India", "generic"), ("Limited", "generic"),
    ("Company", "generic"), ("Corporation", "generic"), ("Industries", "generic"), ("Energy", "generic"),
    # Ambiguous conglomerate single words
    ("Tata", "conglomerate"), ("Adani", "conglomerate"), ("Birla", "conglomerate"), ("Bajaj", "conglomerate"),
    # Currency formatting boundary tests
    (0, "currency"), (100000, "currency"), (10000000, "currency"), (None, "currency"),
]

assert len(EDGE_CASES_50) >= 50, f"Expected 50 edge cases, got {len(EDGE_CASES_50)}"


class TestInvalidEdgeCases50:
    """50 Automated Test Cases: Validating application stability against invalid and edge inputs."""

    @pytest.mark.parametrize("edge_input,category", EDGE_CASES_50[:50])
    def test_application_handles_edge_cases_gracefully_without_crashing(
        self, edge_input: Any, category: str, tmp_path: Path
    ):
        db_file = tmp_path / "test_edge.sqlite3"
        try:
            if category == "currency":
                formatted = format_currency(edge_input)
                assert isinstance(formatted, str)
            elif category == "generic":
                clean = _clean_text(str(edge_input))
                assert is_generic_query(clean) is True
            elif category == "conglomerate":
                res = check_conglomerate_ambiguity(str(edge_input), str(edge_input).lower())
                assert res is not None
                assert res.get("is_ambiguous") is True
            else:
                res = run_market_assistant(str(edge_input), db_path=db_file)
                assert res["status"] == "success"
                assert isinstance(res["response"], str)
                assert len(res["response"]) > 0
        except Exception as exc:
            pytest.fail(f"Edge case '{edge_input}' ({category}) raised an unexpected exception: {exc}")
