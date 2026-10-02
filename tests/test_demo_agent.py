"""Unit tests for the demo agent, multi-stock extraction, typo correction, and N-company comparisons."""

from unittest.mock import MagicMock, patch
import pytest

from agent.demo_agent import (
    extract_symbols_from_query,
    format_currency,
    format_news_section,
    run_demo_agent,
    run_demo_comparison,
    run_demo_single_stock,
)


class TestDemoAgentExtraction:
    """Test suite for robust symbol extraction and typo tolerance."""

    def test_extract_five_companies_with_typos_and_globals(self):
        """Verify extraction of 5 companies including typos ('Wirpo') and overseas stocks ('capgemini', 'IBM')."""
        query = "Compare this 5 compaines Tcs, infosys, Wirpo,capgemini , IBM"
        symbols = extract_symbols_from_query(query)
        assert "TCS" in symbols
        assert "INFY" in symbols
        assert "WIPRO" in symbols
        assert "CAPGEMINI" in symbols
        assert "IBM" in symbols
        assert len(symbols) == 5

    def test_extract_n_companies_banking_sector(self):
        """Verify extraction of arbitrary N companies across a sector."""
        query = "Compare SBI, HDFC Bank, ICICI Bank, Axis Bank, Kotak Bank, IndusInd Bank, PNB"
        symbols = extract_symbols_from_query(query)
        assert len(symbols) >= 6
        assert "SBIN" in symbols
        assert "HDFCBANK" in symbols
        assert "ICICIBANK" in symbols
        assert "AXISBANK" in symbols
        assert "KOTAKBANK" in symbols

    def test_extract_natural_company_names_and_variations(self):
        """Verify extraction of natural company names with suffixes like 'Tata Power' and 'Tata Motors company'."""
        assert extract_symbols_from_query("Tell me about Tata Power") == ["TATAPOWER"]
        assert extract_symbols_from_query("Tell me about TataPower company") == ["TATAPOWER"]
        assert extract_symbols_from_query("Tell me about Tata Motors") == ["TATAMOTORS"]
        assert extract_symbols_from_query("Tell me about Tata Motors company") == ["TATAMOTORS"]
        assert extract_symbols_from_query("Tell me about Tata Motors Ltd") == ["TATAMOTORS"]
        assert extract_symbols_from_query("Tell me about Tata Power Company Limited") == ["TATAPOWER"]
        assert extract_symbols_from_query("Tell me about TCS") == ["TCS"]
        assert extract_symbols_from_query("Tell me about Infosys") == ["INFY"]
        assert extract_symbols_from_query("Tell me about Reliance") == ["RELIANCE"]

    def test_format_currency_multi_currency(self):
        """Verify formatting with various international currencies."""
        assert format_currency(3500.5, "INR") == "₹3,500.50"
        assert format_currency(225.62, "USD") == "$225.62"
        assert format_currency(112.10, "EUR") == "€112.10"
        assert format_currency(None) == "N/A"


class TestDemoComparisonExecution:
    """Test execution of N-company comparisons and error resilience."""

    @patch("agent.demo_agent.get_stock_price")
    @patch("agent.demo_agent.get_company_info")
    @patch("agent.demo_agent.get_stock_sentiment")
    def test_compare_three_nse_companies(self, mock_sent, mock_info, mock_price):
        """Verify comparison table generation for 3 NSE companies."""
        mock_price.side_effect = lambda sym: {
            "symbol": sym,
            "company": f"{sym} Ltd",
            "current_price": 1000.0,
            "change": 10.0,
            "change_percent": 1.0,
            "currency": "INR",
            "52_week_high": 1200.0,
            "52_week_low": 800.0,
        }
        mock_info.return_value = {"trailing_pe": 25.0}
        mock_sent.return_value = {"sentiment": "Bullish"}

        report, tool_calls = run_demo_comparison(["TCS", "INFY", "WIPRO"])
        assert "TCS vs INFY vs WIPRO" in report
        assert "| **Company** |" in report
        assert "TCS Ltd" in report
        assert "INFY Ltd" in report
        assert "WIPRO Ltd" in report
        assert "🟢 Bullish" in report

    @patch("agent.demo_agent.get_stock_price")
    @patch("agent.demo_agent.get_company_info")
    @patch("agent.demo_agent.get_stock_sentiment")
    def test_compare_with_unlisted_resilience(self, mock_sent, mock_info, mock_price):
        """Verify that an unlisted or missing stock does not break comparison of valid stocks."""
        def price_side_effect(sym):
            if sym == "NOTREAL":
                return {"symbol": sym, "current_price": None, "company": None}
            return {
                "symbol": sym,
                "company": f"{sym} Ltd",
                "current_price": 500.0,
                "change": 5.0,
                "change_percent": 1.0,
                "currency": "INR",
            }

        mock_price.side_effect = price_side_effect
        mock_info.return_value = {}
        mock_sent.return_value = {"sentiment": "Neutral"}

        report, tool_calls = run_demo_comparison(["TCS", "NOTREAL"])
        assert "TCS vs NOTREAL" in report
        assert "TCS Ltd" in report
        assert "Not listed" in report or "No live market price" in report


class TestDemoNewsFormatting:
    """Test suite for news executive summaries and external links in demo mode."""

    def test_format_news_section_with_full_summaries_and_external_links(self) -> None:
        """Verify that news items format executive summaries and direct clickable links."""
        sample_news = [
            {
                "title": "HDFC Bank Appoints New MD & CEO",
                "source": "Livemint",
                "published_at": "2026-10-01 12:00:00 UTC",
                "url": "https://www.livemint.com/market/hdfc-bank-ceo.html",
                "summary": "Anup Bagchi appointed as MD & CEO. Brokerages see 32% upside potential.",
            },
            {
                "title": "HDFC Bank Q2 Preview",
                "source": "Moneycontrol",
                "published_at": "2026-10-01 10:00:00 UTC",
                "url": "https://www.moneycontrol.com/news/q2-preview.html",
                "summary": "Deposits and margins remain key monitorables for the upcoming quarter.",
            },
        ]

        lines = format_news_section(sample_news, company_name="HDFC Bank", symbol="HDFCBANK")
        rendered = "\n".join(lines)

        # Verify executive summaries are present directly
        assert "Recent Market News & Executive Summaries" in rendered
        assert "HDFC Bank Appoints New MD & CEO" in rendered
        assert "Anup Bagchi appointed as MD & CEO" in rendered
        assert "Deposits and margins remain key monitorables" in rendered

        # Verify external links are present with publisher attribution
        assert "🔗 [Read full article on Livemint ↗](https://www.livemint.com/market/hdfc-bank-ceo.html)" in rendered
        assert "🔗 [Read full article on Moneycontrol ↗](https://www.moneycontrol.com/news/q2-preview.html)" in rendered

    def test_format_news_section_fallback_summary(self) -> None:
        """Verify that missing summary gets a clear contextual synopsis."""
        sample_news = [
            {
                "title": "Infosys Signs Deal",
                "source": "Reuters",
                "published_at": None,
                "url": "https://reuters.com/tech/1",
                "summary": None,
            }
        ]

        lines = format_news_section(sample_news, company_name="Infosys Ltd", symbol="INFY")
        rendered = "\n".join(lines)

        assert "Executive Summary:" in rendered
        assert "Infosys Ltd (INFY)" in rendered
        assert "🔗 [Read full article on Reuters ↗](https://reuters.com/tech/1)" in rendered

    @patch("agent.demo_agent.get_stock_price")
    @patch("agent.demo_agent.get_company_info")
    @patch("agent.demo_agent.get_stock_sentiment")
    @patch("agent.demo_agent.get_market_news")
    def test_run_demo_single_stock_renders_news_and_links(
        self, mock_news, mock_sent, mock_info, mock_price
    ) -> None:
        """Verify that single stock analysis incorporates news summaries and external links."""
        mock_price.return_value = {
            "symbol": "HDFCBANK",
            "company": "HDFC Bank Limited",
            "current_price": 1700.0,
            "change": 15.0,
            "change_percent": 0.89,
            "52_week_high": 1800.0,
            "52_week_low": 1400.0,
        }
        mock_info.return_value = {"sector": "Financial Services", "industry": "Private Bank"}
        mock_sent.return_value = {"sentiment": "Bullish", "score": 8.0}
        mock_news.return_value = [
            {
                "title": "HDFC Bank Nomura Target Upside",
                "source": "The Financial Express",
                "published_at": "2026-10-01 14:00:00 UTC",
                "url": "https://financialexpress.com/hdfc-target",
                "summary": "Nomura reiterates Buy rating on HDFC Bank citing strong deposit traction.",
            }
        ]

        report, tool_calls = run_demo_single_stock("HDFCBANK")
        assert "HDFC Bank Limited (HDFCBANK) — Market Overview" in report
        assert "Nomura reiterates Buy rating on HDFC Bank" in report
        assert "🔗 [Read full article on The Financial Express ↗](https://financialexpress.com/hdfc-target)" in report
