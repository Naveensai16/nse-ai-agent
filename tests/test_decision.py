"""Unit tests for Stock Decision Assistant service, intent parser, evaluation engine, and tool."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.demo_agent import run_demo_agent
from models.decision import DecisionResult
from services.decision_service import (
    analyze_stock_decision,
    detect_user_intent_and_details,
    format_decision_report_markdown,
)
from services.llm_service import get_all_tools, get_stock_decision_tool
from tools.decision_tool import get_stock_decision


def make_dummy_history(symbol: str, trend: str = "bullish") -> pd.DataFrame:
    """Generate deterministic OHLCV DataFrame for testing."""
    dates = pd.date_range(end="2026-10-02", periods=60, freq="B")
    n = len(dates)

    if trend == "bullish":
        base = 200.0
        prices = [base + i * 2.0 for i in range(n)]
        volumes = [150000 + i * 2000 for i in range(n)]
    elif trend == "bearish":
        base = 350.0
        prices = [base - i * 2.0 for i in range(n)]
        volumes = [80000 + i * 1000 for i in range(n)]
    else:
        base = 250.0
        prices = [base + (i % 3 - 1) * 2.0 for i in range(n)]
        volumes = [100000 for _ in range(n)]

    return pd.DataFrame(
        {
            "Open": [p * 0.99 for p in prices],
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )


def dummy_bullish_profile(symbol: str) -> dict:
    """Deterministic strong company profile."""
    return {
        "company": f"{symbol} India Ltd",
        "market_cap": 2_500_000_000_000,
        "trailing_pe": 16.5,
        "forward_pe": 14.2,
        "price_to_book": 2.8,
        "dividend_yield": 2.1,
        "52_week_high": 350.0,
        "52_week_low": 200.0,
    }


def dummy_bearish_profile(symbol: str) -> dict:
    """Deterministic weak/expensive company profile."""
    return {
        "company": f"{symbol} Ltd",
        "market_cap": 80_000_000_000,
        "trailing_pe": 85.0,
        "forward_pe": 78.0,
        "price_to_book": 9.5,
        "dividend_yield": 0.1,
        "52_week_high": 500.0,
        "52_week_low": 200.0,
    }


def dummy_bullish_financials(symbol: str) -> dict:
    """Deterministic strong financial metrics."""
    return {
        "operating_margin": 24.5,
        "roe": 22.0,
        "roce": 25.5,
        "debt_to_equity": 0.35,
        "free_cash_flow": 15000000000,
        "yoy_revenue_growth": 18.5,
        "yoy_profit_growth": 22.4,
        "qoq_revenue_growth": 5.2,
        "qoq_profit_growth": 8.1,
        "quarters": [
            {"date": "Q3 FY26", "revenue": 60000, "operating_margin": 25.0, "net_profit": 14000},
            {"date": "Q2 FY26", "revenue": 57000, "operating_margin": 24.5, "net_profit": 13000},
            {"date": "Q1 FY26", "revenue": 54000, "operating_margin": 24.0, "net_profit": 12000},
            {"date": "Q4 FY25", "revenue": 51000, "operating_margin": 23.5, "net_profit": 11000},
        ],
    }


def dummy_bearish_financials(symbol: str) -> dict:
    """Deterministic deteriorating financial metrics."""
    return {
        "operating_margin": 4.5,
        "roe": 3.2,
        "roce": 4.1,
        "debt_to_equity": 2.8,
        "free_cash_flow": -5000000000,
        "yoy_revenue_growth": -8.2,
        "yoy_profit_growth": -25.4,
        "qoq_revenue_growth": -3.1,
        "qoq_profit_growth": -12.5,
        "quarters": [
            {"date": "Q3 FY26", "revenue": 10000, "operating_margin": 4.0, "net_profit": 200},
            {"date": "Q2 FY26", "revenue": 11000, "operating_margin": 5.0, "net_profit": 400},
            {"date": "Q1 FY26", "revenue": 12000, "operating_margin": 6.0, "net_profit": 600},
            {"date": "Q4 FY25", "revenue": 13000, "operating_margin": 7.0, "net_profit": 800},
        ],
    }


def dummy_positive_news(symbol: str, count: int = 5) -> list[dict]:
    """Deterministic positive news catalysts."""
    return [
        {
            "title": f"{symbol} Secures ₹5,000 Crore Infrastructure Mega Order",
            "source": "Economic Times",
            "published_date": "2026-10-01",
            "url": f"https://economictimes.indiatimes.com/{symbol.lower()}-order",
            "summary": "Record order inflow bolsters forward revenue visibility for the next 3 years.",
        },
        {
            "title": f"{symbol} Reports Stellar Q3 Earnings, Margins Expand 250 bps",
            "source": "CNBC TV18",
            "published_date": "2026-09-28",
            "url": f"https://cnbctv18.com/{symbol.lower()}-earnings",
            "summary": "Management raises full-year guidance on operational efficiency and export growth.",
        },
    ]


def dummy_negative_news(symbol: str, count: int = 5) -> list[dict]:
    """Deterministic negative news catalysts."""
    return [
        {
            "title": f"{symbol} Faces Regulatory Probe Over Environmental Clearance",
            "source": "Business Standard",
            "published_date": "2026-10-01",
            "url": f"https://business-standard.com/{symbol.lower()}-probe",
            "summary": "Operations at primary manufacturing unit suspended pending compliance review.",
        },
        {
            "title": f"{symbol} Profit Plummets 25% Amid Soaring Raw Material Costs",
            "source": "Livemint",
            "published_date": "2026-09-25",
            "url": f"https://livemint.com/{symbol.lower()}-profit-fall",
            "summary": "Mounting interest expenses and debt burden compress net profit margins.",
        },
    ]


# -------------------------------------------------------------------------
# Test Intent and Parameter Parsing
# -------------------------------------------------------------------------


class TestIntentDetection:
    """Test natural language intent and detail extraction."""

    def test_detect_new_investment_buy_question(self):
        query = "Should I buy India Cements?"
        intent, stock, horizon, price, qty = detect_user_intent_and_details(query)
        assert intent == "new"
        assert stock in ("India Cements", "INDIACEM")
        assert horizon == "1 Year"  # Default
        assert price is None
        assert qty is None

    def test_detect_new_investment_good_time(self):
        query = "Is this a good time to buy Tata Power?"
        intent, stock, horizon, price, qty = detect_user_intent_and_details(query)
        assert intent == "new"
        assert stock in ("Tata Power", "TATAPOWER")

    def test_detect_existing_investment_hold_or_sell(self):
        query = "I already own Tata Motors. Should I hold or sell?"
        intent, stock, horizon, price, qty = detect_user_intent_and_details(query)
        assert intent == "existing"
        assert stock in ("Tata Motors", "TATAMOTORS")

    def test_detect_existing_investment_with_price(self):
        query = "I bought HDFC Bank at ₹1,650. Should I keep it?"
        intent, stock, horizon, price, qty = detect_user_intent_and_details(query)
        assert intent == "existing"
        assert stock in ("HDFC Bank", "HDFCBANK")
        assert price == 1650.0

    def test_detect_existing_investment_with_quantity_and_horizon(self):
        query = "I own 50 shares of Reliance and want to hold it for another 1 year. Should I continue?"
        intent, stock, horizon, price, qty = detect_user_intent_and_details(query)
        assert intent == "existing"
        assert stock in ("Reliance", "RELIANCE")
        assert horizon == "1 Year"
        assert qty == 50

    def test_detect_exit_question(self):
        query = "Should I exit PNC Infratech?"
        intent, stock, horizon, price, qty = detect_user_intent_and_details(query)
        assert intent == "existing"
        assert stock in ("PNC Infratech", "PNCINFRA")

    def test_detect_down_average_or_sell(self):
        query = "I am down 12% in this stock. Should I average or sell?"
        intent, stock, horizon, price, qty = detect_user_intent_and_details(query)
        assert intent == "existing"

    def test_detect_horizon_phrases(self):
        assert detect_user_intent_and_details("Can I hold TCS for the next 6 months?")[2] == "6 Months"
        assert detect_user_intent_and_details("Is Tata Power good for a 3-year investment?")[2] == "3+ Years"
        assert detect_user_intent_and_details("Buy INFY for 1 month")[2] == "1 Month"
        assert detect_user_intent_and_details("Hold RELIANCE for 2 years")[2] == "2 Years"


# -------------------------------------------------------------------------
# Test Decision Engine Logic
# -------------------------------------------------------------------------


class TestDecisionEngine:
    """Test deterministic evaluation and scoring across horizons and indicators."""

    def test_bullish_stock_new_investment_yields_buy(self):
        res = analyze_stock_decision(
            stock="TATAPOWER",
            intent="new",
            horizon="1 Year",
            history_provider=lambda s: make_dummy_history(s, "bullish"),
            company_info_provider=dummy_bullish_profile,
            financials_provider=dummy_bullish_financials,
            news_provider=dummy_positive_news,
        )
        assert isinstance(res, DecisionResult)
        assert res.resolved_symbol == "TATAPOWER"
        assert res.decision_indicator in ("BUY", "CONSIDER ADDING")
        assert res.fundamental_trend == "Improving"
        assert res.quarterly_results_quality in ("Very Strong", "Strong")
        assert res.valuation_assessment in ("Attractive", "Fair", "Very Attractive")
        assert len(res.positive_catalysts) >= 1
        assert len(res.key_reasons) >= 3
        assert len(res.what_makes_view_more_positive) >= 1
        assert len(res.what_makes_view_more_negative) >= 1

    def test_bearish_stock_new_investment_yields_avoid(self):
        res = analyze_stock_decision(
            stock="INDIACEM",
            intent="new",
            horizon="1 Year",
            history_provider=lambda s: make_dummy_history(s, "bearish"),
            company_info_provider=dummy_bearish_profile,
            financials_provider=dummy_bearish_financials,
            news_provider=dummy_negative_news,
        )
        assert isinstance(res, DecisionResult)
        assert res.resolved_symbol == "INDIACEM"
        assert res.decision_indicator in ("AVOID FOR NOW", "WAIT")
        assert res.fundamental_trend == "Deteriorating"
        assert res.valuation_assessment in ("Expensive", "Very Expensive")
        assert res.risk_level in ("High", "Very High")
        assert len(res.negative_catalysts) >= 1

    def test_existing_investment_indicators_vocabulary(self):
        # Existing bullish should be HOLD or CONSIDER ADDING (not BUY)
        res_bullish = analyze_stock_decision(
            stock="TCS",
            intent="existing",
            horizon="1 Year",
            history_provider=lambda s: make_dummy_history(s, "bullish"),
            company_info_provider=dummy_bullish_profile,
            financials_provider=dummy_bullish_financials,
            news_provider=dummy_positive_news,
        )
        assert res_bullish.decision_indicator in ("HOLD", "CONSIDER ADDING")

        # Existing bearish should be REDUCE or EXIT (not AVOID FOR NOW)
        res_bearish = analyze_stock_decision(
            stock="INDIACEM",
            intent="existing",
            horizon="1 Year",
            history_provider=lambda s: make_dummy_history(s, "bearish"),
            company_info_provider=dummy_bearish_profile,
            financials_provider=dummy_bearish_financials,
            news_provider=dummy_negative_news,
        )
        assert res_bearish.decision_indicator in ("REDUCE", "EXIT")

    def test_purchase_price_does_not_bias_decision(self):
        """Past purchase price must not cause sunk-cost bias."""
        res_high_price = analyze_stock_decision(
            stock="TATAPOWER",
            intent="existing",
            horizon="1 Year",
            purchase_price=9999.0,  # Huge loss
            history_provider=lambda s: make_dummy_history(s, "bullish"),
            company_info_provider=dummy_bullish_profile,
            financials_provider=dummy_bullish_financials,
            news_provider=dummy_positive_news,
        )
        res_low_price = analyze_stock_decision(
            stock="TATAPOWER",
            intent="existing",
            horizon="1 Year",
            purchase_price=10.0,  # Huge gain
            history_provider=lambda s: make_dummy_history(s, "bullish"),
            company_info_provider=dummy_bullish_profile,
            financials_provider=dummy_bullish_financials,
            news_provider=dummy_positive_news,
        )
        assert res_high_price.decision_indicator == res_low_price.decision_indicator
        assert res_high_price.purchase_price == 9999.0
        assert res_low_price.purchase_price == 10.0

    def test_short_vs_long_horizon_weighting(self):
        """Short-term horizon weights technicals heavily; long-term weights fundamentals."""
        res_short = analyze_stock_decision(
            stock="TATAPOWER",
            intent="new",
            horizon="1 Month",
            history_provider=lambda s: make_dummy_history(s, "bullish"),
            company_info_provider=dummy_bearish_profile,  # Expensive valuation
            financials_provider=dummy_bearish_financials,  # Weak financials
            news_provider=dummy_positive_news,
        )
        res_long = analyze_stock_decision(
            stock="TATAPOWER",
            intent="new",
            horizon="3+ Years",
            history_provider=lambda s: make_dummy_history(s, "bullish"),
            company_info_provider=dummy_bearish_profile,
            financials_provider=dummy_bearish_financials,
            news_provider=dummy_positive_news,
        )
        # Because of strong technicals, short-term score is higher than long-term score
        assert res_short.composite_score > res_long.composite_score


# -------------------------------------------------------------------------
# Test Report Formatting and LLM Tool
# -------------------------------------------------------------------------


class TestDecisionReportAndTool:
    """Test Markdown formatting and tool interfaces."""

    def test_markdown_report_structure(self):
        res = analyze_stock_decision(
            stock="TATAPOWER",
            intent="new",
            horizon="1 Year",
            history_provider=lambda s: make_dummy_history(s, "bullish"),
            company_info_provider=dummy_bullish_profile,
            financials_provider=dummy_bullish_financials,
            news_provider=dummy_positive_news,
        )
        md = res.report_markdown
        assert "Stock Decision Analysis" in md
        assert "Decision Indicator" in md
        assert "Overall Factor Snapshot" in md
        assert "Market & Sector Environment" in md
        assert "Company Fundamentals & Latest Results" in md
        assert "Valuation & Peer Benchmarking" in md
        assert "Technical Setup & Key Levels" in md
        assert "Catalysts & Verified News" in md
        assert "What Would Change This View?" in md
        assert "Decision Support & Compliance Notice" in md

    def test_get_stock_decision_tool_execution(self):
        output = get_stock_decision(stock="Tata Power", intent="new", horizon="1 Year")
        assert isinstance(output, dict)
        assert output["status"] == "success"
        assert output["symbol"] == "TATAPOWER"
        assert output["decision_indicator"] in ("BUY", "WAIT", "AVOID FOR NOW", "CONSIDER ADDING", "HOLD")
        assert "report_markdown" in output
        assert len(output["why_decision"]) >= 3

    def test_llm_tool_registration(self):
        all_tools = get_all_tools(include_extended=True)
        tool_names = [getattr(t, "name", str(t)) for t in all_tools]
        assert "get_stock_decision" in tool_names


# -------------------------------------------------------------------------
# Test Demo Agent Routing
# -------------------------------------------------------------------------


class TestDemoAgentDecisionRouting:
    """Test demo agent routing of decision queries without LLM API key."""

    def test_demo_agent_routes_should_i_buy(self):
        res = run_demo_agent("Should I buy India Cements?")
        resp = res.get("response", "")
        tool_calls = res.get("tool_calls", [])
        assert any(tc.get("name") == "get_stock_decision" for tc in tool_calls)
        assert "Stock Decision Analysis" in resp
        assert "Decision Indicator" in resp

    def test_demo_agent_routes_existing_hold_or_sell(self):
        res = run_demo_agent("I own Tata Motors. Should I hold or sell?")
        resp = res.get("response", "")
        tool_calls = res.get("tool_calls", [])
        assert any(tc.get("name") == "get_stock_decision" for tc in tool_calls)
        assert "TATAMOTORS" in resp or "Tata Motors" in resp

    def test_demo_agent_routes_bought_at_price(self):
        res = run_demo_agent("I bought HDFC Bank at ₹1,650. Should I keep it?")
        resp = res.get("response", "")
        tool_calls = res.get("tool_calls", [])
        assert any(tc.get("name") == "get_stock_decision" for tc in tool_calls)
        assert "HDFCBANK" in resp
