"""Comprehensive test suite validating natural language intent understanding, entity resolution,
catalyst discovery, coreference resolution, and fallback behavior for the built-in NSE Market Assistant.

Verifies zero dependency on OpenAI API key.
"""

from unittest.mock import MagicMock, patch
import pytest

from agent.demo_agent import run_demo_agent
from services.intent_service import MarketIntent, detect_intent, normalize_query
from services.market_assistant_service import run_market_assistant


class TestMarketAssistantIntents:
    """Validate deterministic natural language intent detection across diverse user queries."""

    def test_52_week_high_intent_and_why(self):
        intent, params = detect_intent("52 week high stock and why?", symbols=[])
        assert intent == MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS
        assert params.get("include_catalysts") is True

    def test_which_stocks_touched_52_week_high_today(self):
        intent, params = detect_intent("Which stocks touched 52 week high today?", symbols=[])
        assert intent == MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS

    def test_top_gainers_intent(self):
        intent, params = detect_intent("Show today's top gainers", symbols=[])
        assert intent == MarketIntent.TOP_GAINERS

    def test_top_losers_intent(self):
        intent, params = detect_intent("Show today's top losers", symbols=[])
        assert intent == MarketIntent.TOP_LOSERS

    def test_tell_me_about_stock(self):
        intent, params = detect_intent("Tell me about Tata Power", symbols=["TATAPOWER"])
        assert intent == MarketIntent.STOCK_OVERVIEW

    def test_stock_price_intent(self):
        intent, params = detect_intent("Tata Motors price", symbols=["TATAMOTORS"])
        assert intent == MarketIntent.STOCK_PRICE

    def test_stock_pe_intent(self):
        intent, params = detect_intent("What is YES Bank PE?", symbols=["YESBANK"])
        assert intent == MarketIntent.STOCK_FUNDAMENTALS
        assert params.get("metric") == "pe"

    def test_stock_news_intent(self):
        intent, params = detect_intent("Tell me latest news about Reliance", symbols=["RELIANCE"])
        assert intent == MarketIntent.STOCK_NEWS

    def test_stock_comparison_intent(self):
        intent, params = detect_intent("Compare TCS and Infosys", symbols=["TCS", "INFY"])
        assert intent == MarketIntent.STOCK_COMPARISON

    def test_stock_comparison_sbi_hdfc(self):
        intent, params = detect_intent("Compare SBI and HDFC Bank", symbols=["SBIN", "HDFCBANK"])
        assert intent == MarketIntent.STOCK_COMPARISON

    def test_stocks_by_sector_it(self):
        intent, params = detect_intent("Which IT stocks are doing well today?", symbols=[])
        assert intent == MarketIntent.STOCKS_BY_SECTOR
        assert params.get("sector") == "Information Technology"

    def test_stocks_by_sector_banking(self):
        intent, params = detect_intent("Show banking stocks", symbols=[])
        assert intent == MarketIntent.STOCKS_BY_SECTOR
        assert params.get("sector") == "Banking"

    def test_volume_spike_intent(self):
        intent, params = detect_intent("Which stocks have high volume today?", symbols=[])
        assert intent in (MarketIntent.VOLUME_SPIKE, MarketIntent.MOST_ACTIVE)

    def test_what_happened_today_intent(self):
        intent, params = detect_intent("What happened to Tata Power today?", symbols=["TATAPOWER"])
        assert intent == MarketIntent.STOCK_PRICE

    def test_how_is_nifty_doing_intent(self):
        intent, params = detect_intent("How is Nifty doing?", symbols=[])
        assert intent == MarketIntent.MARKET_OVERVIEW

    def test_ambiguous_conglomerate_tata(self):
        intent, params = detect_intent("Tata", symbols=[], has_ambiguous_conglomerate=True)
        assert intent == MarketIntent.AMBIGUOUS_STOCK


class TestMarketAssistantExecution:
    """Validate full end-to-end execution of target queries without requiring an OpenAI key."""

    @patch("services.market_assistant_service.get_stock_price")
    @patch("services.catalyst_service.get_market_news")
    def test_query_52_week_high_stock_and_why(self, mock_news, mock_price, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_price.side_effect = lambda sym: {
            "symbol": sym,
            "company": f"{sym} Ltd",
            "current_price": 1000.0,
            "52_week_high": 1010.0,
            "52_week_low": 600.0,
            "change": 15.0,
            "change_percent": 1.5,
        }
        mock_news.return_value = [
            {"title": "Company Bags Mega Order", "source": "Mint", "summary": "Secured major infrastructure project."}
        ]

        res = run_market_assistant("52 week high stock and why?", db_path=db_file)
        assert res["status"] == "success"
        response = res["response"]
        assert "52-Week High" in response
        assert "Why it may be moving" in response
        assert "OpenAI" not in response
        assert "API key" not in response

    @patch("services.market_assistant_service.get_top_gainers")
    def test_query_show_today_top_gainers(self, mock_gainers, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_gainers.return_value = [
            {"symbol": "TATAMOTORS", "company": "Tata Motors Ltd", "price": 950.0, "change_percent": 3.8, "previous_close": 915.0},
            {"symbol": "SBIN", "company": "State Bank of India", "price": 820.0, "change_percent": 2.4, "previous_close": 801.0},
        ]
        res = run_market_assistant("Show today's top gainers", db_path=db_file)
        assert res["status"] == "success"
        response = res["response"]
        assert "Top Market Gainers" in response
        assert "TATAMOTORS" in response
        assert "SBIN" in response
        assert "OpenAI" not in response

    @patch("services.market_assistant_service.get_top_losers")
    def test_query_show_today_top_losers(self, mock_losers, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_losers.return_value = [
            {"symbol": "INFY", "company": "Infosys Ltd", "price": 1850.0, "change_percent": -2.1, "previous_close": 1890.0},
        ]
        res = run_market_assistant("Show today's top losers", db_path=db_file)
        assert res["status"] == "success"
        assert "Top Market Losers" in res["response"]
        assert "INFY" in res["response"]

    @patch("services.market_assistant_service.get_stock_price")
    @patch("services.market_assistant_service.get_company_info")
    def test_query_tell_me_about_tata_power(self, mock_info, mock_price, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_price.return_value = {
            "symbol": "TATAPOWER",
            "company": "Tata Power Company Limited",
            "current_price": 430.0,
            "change": 5.0,
            "change_percent": 1.18,
            "52_week_high": 460.0,
            "52_week_low": 280.0,
        }
        mock_info.return_value = {
            "symbol": "TATAPOWER",
            "company": "Tata Power Company Limited",
            "sector": "Utilities",
            "trailing_pe": 34.0,
        }
        res = run_market_assistant("Tell me about Tata Power", db_path=db_file)
        assert res["status"] == "success"
        assert "TATAPOWER" in res["response"]
        assert "Tata Power" in res["response"]

    @patch("services.market_assistant_service.get_stock_price")
    def test_query_tata_motors_price(self, mock_price, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_price.return_value = {
            "symbol": "TATAMOTORS",
            "company": "Tata Motors Limited",
            "current_price": 960.0,
            "previous_close": 945.0,
            "change": 15.0,
            "change_percent": 1.59,
            "52_week_high": 1179.0,
            "52_week_low": 650.0,
        }
        res = run_market_assistant("Tata Motors price", db_path=db_file)
        assert res["status"] == "success"
        assert "₹960.00" in res["response"]
        assert "TATAMOTORS" in res["response"]

    @patch("services.market_assistant_service.get_company_info")
    @patch("services.market_assistant_service.get_stock_price")
    def test_query_what_is_yes_bank_pe(self, mock_price, mock_info, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_info.return_value = {
            "symbol": "YESBANK",
            "company": "Yes Bank Limited",
            "trailing_pe": 16.5,
            "sector": "Financial Services",
        }
        mock_price.return_value = {"symbol": "YESBANK", "current_price": 22.0}
        res = run_market_assistant("What is YES Bank PE?", db_path=db_file)
        assert res["status"] == "success"
        assert "YESBANK" in res["response"]
        assert "16.50x" in res["response"]

    @patch("services.market_assistant_service.get_market_news")
    def test_query_tell_me_latest_news_about_reliance(self, mock_news, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_news.return_value = [
            {"title": "Reliance Q2 Results Announced", "source": "Bloomberg", "summary": "Retail and Jio drive operating revenue."}
        ]
        res = run_market_assistant("Tell me latest news about Reliance", db_path=db_file)
        assert res["status"] == "success"
        assert "RELIANCE" in res["response"]
        assert "Reliance Q2 Results" in res["response"]

    @patch("services.market_assistant_service.get_stock_price")
    @patch("services.market_assistant_service.get_company_info")
    @patch("services.market_assistant_service.get_stock_sentiment")
    def test_query_compare_tcs_and_infosys(self, mock_sent, mock_info, mock_price, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_price.side_effect = lambda s: {
            "symbol": s,
            "company": f"{s} Ltd",
            "current_price": 3000.0,
            "change": 10.0,
            "change_percent": 0.33,
            "52_week_high": 3500.0,
            "52_week_low": 2500.0,
        }
        mock_info.return_value = {"trailing_pe": 28.0}
        mock_sent.return_value = {"sentiment": "Bullish"}

        res = run_market_assistant("Compare TCS and Infosys", db_path=db_file)
        assert res["status"] == "success"
        assert "TCS vs INFY" in res["response"]

    @patch("services.market_assistant_service.get_stock_price")
    def test_query_which_it_stocks_are_doing_well_today(self, mock_price, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_price.side_effect = lambda s: {
            "symbol": s,
            "company": f"{s} Ltd",
            "current_price": 2000.0,
            "change_percent": 2.5 if s == "TCS" else 0.5,
        }
        res = run_market_assistant("Which IT stocks are doing well today?", db_path=db_file)
        assert res["status"] == "success"
        assert "Information Technology Sector" in res["response"]
        assert "TCS" in res["response"]

    @patch("services.market_assistant_service.get_stock_price")
    def test_query_show_banking_stocks(self, mock_price, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_price.side_effect = lambda s: {
            "symbol": s,
            "company": f"{s} Ltd",
            "current_price": 1000.0,
            "change_percent": 1.2,
        }
        res = run_market_assistant("Show banking stocks", db_path=db_file)
        assert res["status"] == "success"
        assert "Banking Sector Equities" in res["response"]
        assert "HDFCBANK" in res["response"] or "SBIN" in res["response"]

    @patch("services.market_assistant_service.get_market_index")
    def test_query_how_is_nifty_doing(self, mock_index, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        mock_index.side_effect = lambda idx: {
            "name": idx,
            "current_value": 25000.0,
            "change": 120.0,
            "change_percent": 0.48,
        }
        res = run_market_assistant("How is Nifty doing?", db_path=db_file)
        assert res["status"] == "success"
        assert "NIFTY 50" in res["response"]
        assert "25,000.00" in res["response"]

    def test_ambiguous_group_tata_shows_candidates_and_does_not_assume_power(self, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        res = run_market_assistant("Tata", db_path=db_file)
        assert res["status"] == "success"
        resp = res["response"]
        assert "TATAMOTORS" in resp
        assert "TATASTEEL" in resp
        assert "TATAPOWER" in resp
        assert "Which one would you like to analyze?" in resp


class TestConversationContextCoreference:
    """Validate follow-up questions and pronouns ('its', 'which has better PE')."""

    @patch("services.market_assistant_service.get_stock_price")
    @patch("services.market_assistant_service.get_company_info")
    def test_follow_up_its_52_week_high(self, mock_info, mock_price, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        cid = "conv-follow-up-1"

        mock_price.return_value = {
            "symbol": "TATAPOWER",
            "company": "Tata Power Company Limited",
            "current_price": 435.0,
            "change": 4.0,
            "change_percent": 0.93,
            "52_week_high": 460.0,
            "52_week_low": 280.0,
        }
        mock_info.return_value = {"sector": "Power", "trailing_pe": 33.0}

        # Turn 1: User asks about Tata Power
        res1 = run_market_assistant("Tell me about Tata Power", conversation_id=cid, db_path=db_file)
        assert "TATAPOWER" in res1["response"]

        # Turn 2: User asks follow-up: "What is its 52 week high?"
        res2 = run_market_assistant("What is its 52 week high?", conversation_id=cid, db_path=db_file)
        assert "TATAPOWER" in res2["response"]
        assert "52-Week High" in res2["response"]
        assert "₹460.00" in res2["response"]

    @patch("services.market_assistant_service.get_stock_price")
    @patch("services.market_assistant_service.get_company_info")
    @patch("services.market_assistant_service.get_stock_sentiment")
    def test_follow_up_which_has_better_pe(self, mock_sent, mock_info, mock_price, tmp_path):
        db_file = tmp_path / "test_chat.sqlite3"
        cid = "conv-follow-up-2"

        mock_price.side_effect = lambda s: {
            "symbol": s,
            "company": f"{s} Ltd",
            "current_price": 3000.0,
            "change": 10.0,
            "change_percent": 0.33,
            "52_week_high": 3500.0,
            "52_week_low": 2500.0,
        }
        mock_info.side_effect = lambda s: {
            "trailing_pe": 28.0 if s == "TCS" else 22.0
        }
        mock_sent.return_value = {"sentiment": "Bullish"}

        # Turn 1: Compare TCS and Infosys
        res1 = run_market_assistant("Compare TCS and Infosys", conversation_id=cid, db_path=db_file)
        assert "TCS vs INFY" in res1["response"]

        # Turn 2: Which has better PE?
        res2 = run_market_assistant("Which has better PE?", conversation_id=cid, db_path=db_file)
        assert "Valuation Comparison: P/E Ratio" in res2["response"]
        assert "INFY" in res2["response"]
