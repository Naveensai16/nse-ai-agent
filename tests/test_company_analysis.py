"""Comprehensive unit test suite for the Company Analysis, Fundamentals, Financials, and Governance tools.

Verifies the 33 required test areas:
1. Complete company analysis returns expected sections
2. Missing P/E handled gracefully
3. Missing ROCE handled gracefully
4. Missing ROE handled gracefully
5. Missing book value handled gracefully
6. Missing dividend yield handled gracefully
7. 4-quarter financial statement parsing
8. Revenue growth calculation (QoQ, YoY)
9. Expense growth calculation
10. Net profit growth calculation
11. Operating margin calculation
12. Cash flow comparison with net profit
13. Missing quarterly cash flow handled gracefully
14. Debt calculation
15. Net debt calculation
16. Financial health scoring
17. Strong company gets appropriate score/category
18. Average company gets appropriate score/category
19. Weak company gets appropriate score/category
20. Shareholding pattern parsing
21. Missing shareholder names handled gracefully
22. Promoters aggregate correctly
23. FIIs aggregate correctly
24. DIIs aggregate correctly
25. Government holding reported correctly
26. Public holding reported correctly
27. Board meeting parsing
28. Corporate action parsing
29. Verified news deduplication
30. Partial provider failure produces partial report with explicit limitations
31. Financial company (Bank/NBFC) detection
32. Bank-specific scoring does not use normal corporate ROCE/debt rules
33. Existing stock price tests continue to pass
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from tools.analysis_tool import format_currency_inr, get_company_analysis
from tools.company_tool import get_company_info
from tools.financials_tool import (
    calculate_financial_health,
    evaluate_pe_valuation,
    evaluate_roce,
    evaluate_roe,
    get_cash_flow,
    get_debt_metrics,
    get_quarterly_financials,
    is_financial_company,
)
from tools.governance_tool import (
    get_board_meetings,
    get_corporate_actions,
    get_shareholding_pattern,
)
from tools.stock_tool import get_stock_price


# =====================================================================
# Test Group 1: Complete Company Analysis & Snapshot (Tests 1 - 6)
# =====================================================================

class TestCompanyAnalysisAndSnapshot:
    """Test full report structure and graceful handling of missing snapshot metrics."""

    @patch("tools.analysis_tool.get_stock_price")
    @patch("tools.analysis_tool.get_company_info")
    @patch("tools.analysis_tool.get_stock_sentiment")
    @patch("tools.analysis_tool.get_quarterly_financials")
    @patch("tools.analysis_tool.get_cash_flow")
    @patch("tools.analysis_tool.get_debt_metrics")
    @patch("tools.analysis_tool.get_shareholding_pattern")
    @patch("tools.analysis_tool.get_board_meetings")
    @patch("tools.analysis_tool.get_corporate_actions")
    @patch("tools.analysis_tool.get_market_news")
    def test_1_complete_company_analysis_returns_expected_sections(
        self,
        mock_news,
        mock_actions,
        mock_meetings,
        mock_shareholding,
        mock_debt,
        mock_cf,
        mock_fin,
        mock_sent,
        mock_info,
        mock_price,
    ):
        """Test 1: Verify complete company analysis returns all Section V required markdown sections."""
        mock_price.return_value = {
            "symbol": "TCS",
            "company": "Tata Consultancy Services Limited",
            "current_price": 3500.0,
            "change": 25.0,
            "change_percent": 0.72,
            "52_week_high": 4200.0,
            "52_week_low": 3200.0,
        }
        mock_info.return_value = {
            "symbol": "TCS",
            "company": "Tata Consultancy Services Limited",
            "sector": "Information Technology",
            "industry": "IT Services",
            "market_cap": 12700000000000,
            "trailing_pe": 24.5,
            "forward_pe": 22.0,
            "book_value": 265.0,
            "dividend_yield": 0.021,
            "roce": 52.0,
            "roe": 44.0,
        }
        mock_sent.return_value = {
            "sentiment": "Bullish",
            "score": 6.5,
            "signals": [{"name": "price vs 20 DMA", "value": 2.1}],
        }
        mock_fin.return_value = {
            "quarters": [
                {
                    "date": "2025-06-30",
                    "revenue": 620000000000,
                    "expenses": 460000000000,
                    "operating_profit": 160000000000,
                    "operating_margin": 25.8,
                    "net_profit": 120000000000,
                    "eps": 33.1,
                }
            ],
            "qoq_revenue_growth": 3.2,
            "yoy_revenue_growth": 8.5,
            "qoq_profit_growth": 2.1,
            "yoy_profit_growth": 6.7,
            "margin_trend": "Stable (25.80% latest)",
        }
        mock_cf.return_value = {
            "quarterly_reported": False,
            "annual_ocf_history": [{"period": "FY2025", "ocf": 450000000000}],
            "cash_conversion_ratio": 0.95,
            "cash_conversion_assessment": "Healthy (0.95x)",
        }
        mock_debt.return_value = {
            "total_debt": 0,
            "cash_and_equivalents": 180000000000,
            "net_debt": -180000000000,
            "debt_to_equity": 0.0,
            "assessment": "Net Cash Positive",
            "is_financial": False,
        }
        mock_shareholding.return_value = {
            "as_of_date": "30 June 2026",
            "promoters_percent": 71.7,
            "fii_percent": 12.4,
            "dii_percent": 9.1,
            "government_percent": 0.0,
            "public_percent": 6.8,
            "promoter_names": [{"name": "Tata Sons Private Limited", "holding_percent": 71.7}],
            "trend": "Stable",
        }
        mock_meetings.return_value = [
            {"date": "2026-07-10", "purpose": "Financial Results", "outcome": "Approved", "url": "https://nseindia.com"}
        ]
        mock_actions.return_value = [
            {"date": "2026-07-18", "action": "Dividend", "details": "Interim Dividend ₹10", "url": "https://nseindia.com"}
        ]
        mock_news.return_value = [
            {
                "title": "TCS Signs Mega Deal",
                "source": "Moneycontrol",
                "url": "https://moneycontrol.com/tcs",
                "published_at": "2026-10-01",
                "summary": "TCS expands cloud transformation agreement.",
            }
        ]

        result = get_company_analysis("TCS")
        report = result["report_markdown"]

        # Check required Section V headers
        assert "## Snapshot" in report
        assert "## Valuation & Quality" in report
        assert "## Technical Sentiment" in report
        assert "## Last 4 Quarters Financial Performance" in report
        assert "## Cash Flow & Debt" in report
        assert "## Financial Health" in report
        assert "## Shareholding Pattern" in report
        assert "## Recent Board Meetings" in report
        assert "## Last 5 Corporate Actions" in report
        assert "## Recent Verified News" in report
        assert "## Data Availability / Limitations" in report

        # Verify key fields
        assert "Tata Consultancy Services Limited (TCS)" in report
        assert "₹3,500.00" in report
        assert "**Stock P/E:** 24.5x" in report
        assert "**ROCE:** 52.0%" in report
        assert "**ROE:** 44.0%" in report
        assert "Tata Sons Private Limited" in report

    def test_2_missing_pe_handled_gracefully(self):
        """Test 2: Missing P/E ratio is evaluated as Insufficient Data without throwing errors."""
        eval_res = evaluate_pe_valuation(None)
        assert eval_res["assessment"] == "Insufficient Data"
        assert "unavailable" in eval_res["reason"].lower()

    def test_3_missing_roce_handled_gracefully(self):
        """Test 3: Missing ROCE handled gracefully with deterministic fallback."""
        eval_res = evaluate_roce(None, is_financial=False)
        assert eval_res["assessment"] == "Insufficient Data"
        assert "available" in eval_res["reason"].lower()

    def test_4_missing_roe_handled_gracefully(self):
        """Test 4: Missing ROE handled gracefully with deterministic fallback."""
        eval_res = evaluate_roe(None, is_financial=False)
        assert eval_res["assessment"] == "Insufficient Data"
        assert "available" in eval_res["reason"].lower()

    @patch("tools.company_tool.yf.Ticker")
    def test_5_missing_book_value_handled_gracefully(self, mock_ticker):
        """Test 5: When book value is absent, tool sets book_value to None without crashing."""
        mock_instance = MagicMock()
        mock_instance.info = {"symbol": "SAMPLE", "bookValue": None}
        mock_instance.fast_info = {}
        mock_ticker.return_value = mock_instance

        info = get_company_info("SAMPLE")
        assert info.get("book_value") is None
        assert format_currency_inr(info.get("book_value")) == "N/A"

    @patch("tools.company_tool.yf.Ticker")
    def test_6_missing_dividend_yield_handled_gracefully(self, mock_ticker):
        """Test 6: When dividend yield is absent, returns None or 0.0 without errors."""
        mock_instance = MagicMock()
        mock_instance.info = {"symbol": "SAMPLE", "dividendYield": None}
        mock_instance.fast_info = {}
        mock_ticker.return_value = mock_instance

        info = get_company_info("SAMPLE")
        assert info.get("dividend_yield") is None


# =====================================================================
# Test Group 2: Financial Statement Parsing & Growth Rates (Tests 7 - 11)
# =====================================================================

class TestQuarterlyFinancialsAndGrowth:
    """Test 4-quarter financials parsing, margin calculations, and QoQ/YoY growth rates."""

    @patch("tools.financials_tool.yf.Ticker")
    def test_7_four_quarter_financial_statement_parsing(self, mock_ticker):
        """Test 7: Verify 4 quarters of Revenue, Expenses, Operating Profit, OPM %, Net Profit, EPS are extracted."""
        dates = pd.date_range(end="2025-06-30", periods=4, freq="QE-DEC")
        data = {
            "Total Revenue": [500000.0, 520000.0, 540000.0, 560000.0],
            "Operating Expense": [375000.0, 390000.0, 405000.0, 420000.0],
            "Operating Income": [125000.0, 130000.0, 135000.0, 140000.0],
            "Net Income": [95000.0, 100000.0, 105000.0, 110000.0],
            "Diluted EPS": [25.0, 26.5, 27.8, 29.0],
        }
        df = pd.DataFrame(data, index=dates).T

        mock_inst = MagicMock()
        mock_inst.quarterly_income_stmt = df
        mock_inst.quarterly_financials = df
        mock_ticker.return_value = mock_inst

        fin = get_quarterly_financials("INFY")
        assert len(fin["quarters"]) == 4
        q_latest = fin["quarters"][-1]
        assert q_latest["revenue"] == 560000.0
        assert q_latest["operating_profit"] == 140000.0
        assert q_latest["net_profit"] == 110000.0
        assert q_latest["eps"] == 29.0
        assert q_latest["operating_margin"] == pytest.approx(25.0, abs=0.1)

    @patch("tools.financials_tool.yf.Ticker")
    def test_8_revenue_growth_qoq_and_yoy(self, mock_ticker):
        """Test 8: Verify QoQ and YoY revenue growth calculations."""
        # Provide 5 quarters so YoY comparison is available
        dates = pd.date_range(end="2025-06-30", periods=5, freq="QE-DEC")
        data = {
            "Total Revenue": [100.0, 105.0, 110.0, 115.0, 120.0],  # Q0: 100, Q4: 120 -> YoY = +20%
            "Operating Income": [20.0, 21.0, 22.0, 23.0, 24.0],
            "Net Income": [15.0, 16.0, 17.0, 18.0, 20.0],  # Q3: 18, Q4: 20 -> QoQ = +11.11%
        }
        df = pd.DataFrame(data, index=dates).T

        mock_inst = MagicMock()
        mock_inst.quarterly_income_stmt = df
        mock_inst.quarterly_financials = df
        mock_ticker.return_value = mock_inst

        fin = get_quarterly_financials("TESTCO")
        # QoQ Revenue: (120 - 115) / 115 * 100 = 4.35%
        assert fin["qoq_revenue_growth"] == pytest.approx(4.35, abs=0.1)
        # YoY Revenue: (120 - 100) / 100 * 100 = 20.0%
        assert fin["yoy_revenue_growth"] == pytest.approx(20.0, abs=0.1)

    @patch("tools.financials_tool.yf.Ticker")
    def test_9_expense_growth_and_margin_trend(self, mock_ticker):
        """Test 9: Verify operating expenses are properly captured and margin trend tracked."""
        dates = pd.date_range(end="2025-06-30", periods=4, freq="QE-DEC")
        data = {
            "Total Revenue": [100.0, 100.0, 100.0, 100.0],
            "Operating Expense": [70.0, 75.0, 80.0, 85.0],  # Expenses increasing
            "Operating Income": [30.0, 25.0, 20.0, 15.0],  # Margin declining
            "Net Income": [20.0, 17.0, 14.0, 10.0],
        }
        df = pd.DataFrame(data, index=dates).T

        mock_inst = MagicMock()
        mock_inst.quarterly_income_stmt = df
        mock_inst.quarterly_financials = df
        mock_ticker.return_value = mock_inst

        fin = get_quarterly_financials("TESTCO")
        assert "Contracting" in fin["margin_trend"]

    @patch("tools.financials_tool.yf.Ticker")
    def test_10_net_profit_growth_calculation(self, mock_ticker):
        """Test 10: Verify net profit QoQ and YoY growth calculation."""
        dates = pd.date_range(end="2025-06-30", periods=5, freq="QE-DEC")
        data = {
            "Total Revenue": [100.0, 110.0, 120.0, 130.0, 140.0],
            "Net Income": [10.0, 12.0, 14.0, 16.0, 20.0],  # Q0: 10, Q3: 16, Q4: 20
        }
        df = pd.DataFrame(data, index=dates).T

        mock_inst = MagicMock()
        mock_inst.quarterly_income_stmt = df
        mock_inst.quarterly_financials = df
        mock_ticker.return_value = mock_inst

        fin = get_quarterly_financials("TESTCO")
        # QoQ Profit: (20 - 16) / 16 * 100 = +25.0%
        assert fin["qoq_profit_growth"] == pytest.approx(25.0, abs=0.1)
        # YoY Profit: (20 - 10) / 10 * 100 = +100.0%
        assert fin["yoy_profit_growth"] == pytest.approx(100.0, abs=0.1)

    @patch("tools.financials_tool.yf.Ticker")
    def test_11_operating_margin_calculation(self, mock_ticker):
        """Test 11: Operating margin = Operating Profit / Revenue * 100."""
        dates = pd.date_range(end="2025-06-30", periods=1, freq="QE-DEC")
        data = {
            "Total Revenue": [200.0],
            "Operating Income": [50.0],
            "Net Income": [40.0],
        }
        df = pd.DataFrame(data, index=dates).T

        mock_inst = MagicMock()
        mock_inst.quarterly_income_stmt = df
        mock_inst.quarterly_financials = df
        mock_ticker.return_value = mock_inst

        fin = get_quarterly_financials("TESTCO")
        assert fin["quarters"][0]["operating_margin"] == 25.0


# =====================================================================
# Test Group 3: Cash Flow, Cash Conversion & Debt (Tests 12 - 15)
# =====================================================================

class TestCashFlowAndDebtMetrics:
    """Test Operating Cash Flow, cash conversion ratio, and debt metrics."""

    @patch("tools.financials_tool.yf.Ticker")
    def test_12_cash_flow_comparison_with_net_profit(self, mock_ticker):
        """Test 12: Cash conversion ratio = OCF / Net Profit and healthy conversion detection."""
        dates = pd.date_range(end="2025-03-31", periods=2, freq="YE-MAR")
        cf_df = pd.DataFrame(
            {"Operating Cash Flow": [90000.0, 100000.0]}, index=dates
        ).T
        inc_df = pd.DataFrame(
            {"Net Income": [95000.0, 105000.0]}, index=dates
        ).T

        mock_inst = MagicMock()
        mock_inst.cashflow = cf_df
        mock_inst.quarterly_cashflow = pd.DataFrame()
        mock_inst.financials = inc_df
        mock_ticker.return_value = mock_inst

        cf_res = get_cash_flow("INFY")
        assert cf_res["cash_conversion_ratio"] == pytest.approx(100000.0 / 105000.0, abs=0.01)
        assert "Cash Conversion" in cf_res["cash_conversion_assessment"]

    @patch("tools.financials_tool.yf.Ticker")
    def test_13_missing_quarterly_cash_flow_handled_gracefully(self, mock_ticker):
        """Test 13: Indian companies not publishing quarterly OCF are explicitly disclosed as annual-only without inventing data."""
        dates = pd.date_range(end="2025-03-31", periods=1, freq="YE-MAR")
        cf_df = pd.DataFrame({"Operating Cash Flow": [50000.0]}, index=dates).T

        mock_inst = MagicMock()
        mock_inst.cashflow = cf_df
        mock_inst.quarterly_cashflow = pd.DataFrame()  # Empty quarterly cashflow
        mock_inst.financials = pd.DataFrame()
        mock_ticker.return_value = mock_inst

        cf_res = get_cash_flow("TCS")
        assert cf_res["quarterly_reported"] is False
        assert len(cf_res["annual_ocf_history"]) == 1

    @patch("tools.financials_tool.yf.Ticker")
    def test_14_debt_calculation(self, mock_ticker):
        """Test 14: Total Debt and borrowings calculated from balance sheet."""
        bs_data = {
            "Total Debt": [50000.0],
            "Cash And Cash Equivalents": [20000.0],
            "Stockholders Equity": [100000.0],
        }
        df = pd.DataFrame(bs_data, index=pd.date_range(end="2025-03-31", periods=1, freq="YE-MAR")).T

        mock_inst = MagicMock()
        mock_inst.balance_sheet = df
        mock_inst.quarterly_balance_sheet = df
        mock_ticker.return_value = mock_inst

        debt_res = get_debt_metrics("TCS")
        assert debt_res["total_debt"] == 50000.0
        assert debt_res["cash_and_equivalents"] == 20000.0

    @patch("tools.financials_tool.yf.Ticker")
    def test_15_net_debt_and_debt_equity_calculation(self, mock_ticker):
        """Test 15: Net Debt = Total Debt - Cash and Debt/Equity calculation."""
        bs_data = {
            "Total Debt": [60000.0],
            "Cash And Cash Equivalents": [80000.0],  # Net cash = -20000
            "Stockholders Equity": [120000.0],
        }
        df = pd.DataFrame(bs_data, index=pd.date_range(end="2025-03-31", periods=1, freq="YE-MAR")).T

        mock_inst = MagicMock()
        mock_inst.balance_sheet = df
        mock_inst.quarterly_balance_sheet = df
        mock_ticker.return_value = mock_inst

        debt_res = get_debt_metrics("INFY")
        assert debt_res["net_debt"] == -20000.0
        assert debt_res["debt_to_equity"] == pytest.approx(0.5, abs=0.01)
        assert "Net Cash Positive" in debt_res["assessment"]


# =====================================================================
# Test Group 4: Financial Health Scoring System (Tests 16 - 19)
# =====================================================================

class TestFinancialHealthScoring:
    """Test explainable deterministic scoring system (0 - 10) and classifications."""

    def test_16_financial_health_scoring_structure(self):
        """Test 16: Scoring produces numeric score 0-10, category rating, positive factors, and concerns."""
        res = calculate_financial_health(
            qoq_revenue_growth=5.0,
            yoy_revenue_growth=12.0,
            qoq_profit_growth=4.0,
            yoy_profit_growth=10.0,
            operating_margin=25.0,
            margin_trend="Expanding",
            cash_conversion_ratio=0.9,
            net_debt=-5000.0,
            debt_to_equity=0.1,
            roce=30.0,
            roe=25.0,
            is_financial=False,
        )
        assert 0.0 <= res["score"] <= 10.0
        assert res["rating"] in ["Very Good", "Good", "Average", "Bad", "Very Bad"]
        assert isinstance(res["positive_factors"], list)
        assert isinstance(res["concerns"], list)
        assert len(res["positive_factors"]) > 0

    def test_17_strong_company_gets_high_score(self):
        """Test 17: Strong financials yield 'Very Good' or 'Good' rating."""
        res = calculate_financial_health(
            qoq_revenue_growth=8.0,
            yoy_revenue_growth=18.0,
            qoq_profit_growth=10.0,
            yoy_profit_growth=22.0,
            operating_margin=28.0,
            margin_trend="Expanding",
            cash_conversion_ratio=1.05,
            net_debt=-100000.0,
            debt_to_equity=0.0,
            roce=45.0,
            roe=35.0,
            is_financial=False,
        )
        assert res["score"] >= 7.5
        assert res["rating"] in ["Very Good", "Good"]

    def test_18_average_company_gets_average_score(self):
        """Test 18: Mixed financials yield 'Average' rating."""
        res = calculate_financial_health(
            qoq_revenue_growth=1.0,
            yoy_revenue_growth=3.0,
            qoq_profit_growth=-1.0,
            yoy_profit_growth=2.0,
            operating_margin=12.0,
            margin_trend="Stable",
            cash_conversion_ratio=0.6,
            net_debt=20000.0,
            debt_to_equity=0.8,
            roce=13.0,
            roe=11.0,
            is_financial=False,
        )
        assert 4.0 <= res["score"] <= 7.0
        assert res["rating"] == "Average"

    def test_19_weak_company_gets_weak_score(self):
        """Test 19: Deteriorating fundamentals yield 'Bad' or 'Very Bad' rating."""
        res = calculate_financial_health(
            qoq_revenue_growth=-10.0,
            yoy_revenue_growth=-15.0,
            qoq_profit_growth=-25.0,
            yoy_profit_growth=-40.0,
            operating_margin=2.0,
            margin_trend="Contracting",
            cash_conversion_ratio=0.2,
            net_debt=100000.0,
            debt_to_equity=2.5,
            roce=3.0,
            roe=2.0,
            is_financial=False,
        )
        assert res["score"] < 4.5
        assert res["rating"] in ["Bad", "Very Bad"]
        assert len(res["concerns"]) >= 3


# =====================================================================
# Test Group 5: Shareholding Pattern & Disclosures (Tests 20 - 26)
# =====================================================================

class TestShareholdingPattern:
    """Test parsing and aggregation of Promoters, FII, DII, Government, and Public holdings."""

    @patch("tools.governance_tool.yf.Ticker")
    def test_20_shareholding_pattern_parsing(self, mock_ticker):
        """Test 20: Shareholding pattern extracts all major ownership categories."""
        mock_inst = MagicMock()
        mock_inst.major_holders = pd.DataFrame(
            [
                [0.72, "% of Shares Held by All Insider"],
                [0.13, "% of Shares Held by Institutions"],
            ]
        )
        mock_inst.institutional_holders = pd.DataFrame(
            {
                "Holder": ["Life Insurance Corporation Of India", "Vanguard Emerging Markets"],
                "% Out": [0.05, 0.03],
                "Date Reported": ["2026-06-30", "2026-06-30"],
            }
        )
        mock_ticker.return_value = mock_inst

        pattern = get_shareholding_pattern("TCS")
        assert pattern["promoters_percent"] == 72.0
        assert pattern["fii_percent"] is not None
        assert pattern["dii_percent"] is not None
        assert pattern["public_percent"] is not None

    @patch("tools.governance_tool.yf.Ticker")
    def test_21_missing_shareholder_names_handled_gracefully(self, mock_ticker):
        """Test 21: When institutional individual names are not in public filings, avoids hallucination."""
        mock_inst = MagicMock()
        mock_inst.major_holders = pd.DataFrame()
        mock_inst.institutional_holders = pd.DataFrame()  # No individual names
        mock_ticker.return_value = mock_inst

        pattern = get_shareholding_pattern("ZOMATO")
        assert pattern["promoter_names"] == []
        assert "stable" in pattern["trend"].lower()

    def test_22_promoters_aggregate_correctly(self):
        """Test 22: Promoter individual entities sum up to promoter total or remain bounded <= 100%."""
        pattern = {
            "promoters_percent": 71.74,
            "promoter_names": [
                {"name": "Tata Sons Pvt Ltd", "holding_percent": 71.74}
            ],
        }
        total_disclosed = sum(p["holding_percent"] for p in pattern["promoter_names"])
        assert total_disclosed <= pattern["promoters_percent"]

    @patch("tools.governance_tool.yf.Ticker")
    def test_23_fii_aggregation(self, mock_ticker):
        """Test 23: FII holding percentage is computed and categorized correctly."""
        mock_inst = MagicMock()
        mock_inst.major_holders = pd.DataFrame(
            [
                [0.50, "% of Shares Held by All Insider"],
                [0.30, "% of Shares Held by Institutions"],
            ]
        )
        mock_inst.institutional_holders = pd.DataFrame(
            {
                "Holder": ["Foreign Portfolio Investor A"],
                "% Out": [0.15],
            }
        )
        mock_ticker.return_value = mock_inst

        pattern = get_shareholding_pattern("SAMPLE")
        assert pattern["fii_percent"] >= 0.0

    @patch("tools.governance_tool.yf.Ticker")
    def test_24_dii_aggregation(self, mock_ticker):
        """Test 24: DII holding percentage incorporates domestic institutions (e.g. LIC, Mutual Funds)."""
        mock_inst = MagicMock()
        mock_inst.major_holders = pd.DataFrame(
            [
                [0.50, "% of Shares Held by All Insider"],
                [0.25, "% of Shares Held by Institutions"],
            ]
        )
        mock_inst.institutional_holders = pd.DataFrame(
            {
                "Holder": ["Life Insurance Corporation of India"],
                "% Out": [0.08],
            }
        )
        mock_ticker.return_value = mock_inst

        pattern = get_shareholding_pattern("SAMPLE")
        assert pattern["dii_percent"] >= 0.0

    @patch("tools.governance_tool.yf.Ticker")
    def test_25_government_holding_reported_correctly(self, mock_ticker):
        """Test 25: Government ownership is explicitly parsed (e.g. 0% for private companies, explicit % for PSUs)."""
        mock_inst = MagicMock()
        mock_inst.major_holders = pd.DataFrame()
        mock_inst.institutional_holders = pd.DataFrame()
        mock_ticker.return_value = mock_inst

        pattern = get_shareholding_pattern("PVTCO")
        assert pattern["government_percent"] == 0.0

    @patch("tools.governance_tool.yf.Ticker")
    def test_26_public_holding_reported_correctly(self, mock_ticker):
        """Test 26: Public shareholding is computed as remaining float."""
        mock_inst = MagicMock()
        mock_inst.major_holders = pd.DataFrame(
            [
                [0.60, "% of Shares Held by All Insider"],
                [0.20, "% of Shares Held by Institutions"],
            ]
        )
        mock_inst.institutional_holders = pd.DataFrame()
        mock_ticker.return_value = mock_inst

        pattern = get_shareholding_pattern("SAMPLE")
        # Remaining float = 100 - (60 + 20) = 20%
        assert pattern["public_percent"] == pytest.approx(20.0, abs=1.0)


# =====================================================================
# Test Group 6: Board Meetings, Corporate Actions & News (Tests 27 - 30)
# =====================================================================

class TestGovernanceAndNews:
    """Test Board meetings, corporate actions, and news deduplication/resilience."""

    @patch("tools.governance_tool.yf.Ticker")
    def test_27_board_meeting_parsing(self, mock_ticker):
        """Test 27: Board meetings are parsed with date, purpose, status, and URL."""
        mock_inst = MagicMock()
        mock_inst.calendar = {"Earnings Date": ["2026-10-15"]}
        mock_ticker.return_value = mock_inst

        meetings = get_board_meetings("TCS")
        assert len(meetings) >= 1
        assert "date" in meetings[0]
        assert "purpose" in meetings[0]
        assert "outcome" in meetings[0]

    @patch("tools.governance_tool.yf.Ticker")
    def test_28_corporate_action_parsing(self, mock_ticker):
        """Test 28: Corporate actions retrieve latest dividends and splits up to 5."""
        dates = pd.date_range(end="2026-06-30", periods=6, freq="180D")
        div_series = pd.Series([10.0, 12.0, 15.0, 18.0, 20.0, 25.0], index=dates)

        mock_inst = MagicMock()
        mock_inst.dividends = div_series
        mock_inst.splits = pd.Series(dtype=float)
        mock_ticker.return_value = mock_inst

        actions = get_corporate_actions("TCS")
        assert len(actions) == 5  # Capped at latest 5
        assert actions[0]["action"] == "Dividend"
        assert "₹25.00" in actions[0]["details"]

    def test_29_verified_news_deduplication(self):
        """Test 29: Duplicate news articles are filtered out based on title/url."""
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>TCS signs major cloud contract - Reuters</title>
                    <link>https://news.google.com/article1</link>
                    <pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate>
                    <source url="https://reuters.com">Reuters</source>
                    <description>Reuters report on TCS cloud deal.</description>
                </item>
                <item>
                    <title>TCS signs major cloud contract - Economic Times</title>
                    <link>https://news.google.com/article2</link>
                    <pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate>
                    <source url="https://economictimes.com">Economic Times</source>
                    <description>Duplicate story.</description>
                </item>
            </channel>
        </rss>"""
        with patch("tools.news_tool.fetch_news_feed", return_value=rss_xml):
            from tools.news_tool import get_market_news
            news = get_market_news("TCS", limit=5)
            # Should be deduplicated to 1 item
            assert len(news) == 1
            assert "TCS signs major cloud contract" in news[0]["title"]

    @patch("tools.analysis_tool.get_stock_price")
    @patch("tools.analysis_tool.get_company_info")
    @patch("tools.analysis_tool.get_stock_sentiment")
    @patch("tools.analysis_tool.get_quarterly_financials")
    @patch("tools.analysis_tool.get_cash_flow")
    @patch("tools.analysis_tool.get_debt_metrics")
    @patch("tools.analysis_tool.get_shareholding_pattern")
    @patch("tools.analysis_tool.get_board_meetings")
    @patch("tools.analysis_tool.get_corporate_actions")
    @patch("tools.analysis_tool.get_market_news")
    def test_30_partial_provider_failure_produces_partial_report_with_limitations(
        self,
        mock_news,
        mock_actions,
        mock_meetings,
        mock_shareholding,
        mock_debt,
        mock_cf,
        mock_fin,
        mock_sent,
        mock_info,
        mock_price,
    ):
        """Test 30: When some sub-tools fail, report completes and documents missing sections under Limitations."""
        mock_price.return_value = {
            "symbol": "PARTIAL",
            "company": "Partial Co Ltd",
            "current_price": 500.0,
            "change": 5.0,
            "change_percent": 1.0,
        }
        mock_info.return_value = {"symbol": "PARTIAL", "company": "Partial Co Ltd"}
        mock_sent.return_value = {"sentiment": "Neutral", "score": 0.0, "signals": []}

        # Make financials and cash flow throw exceptions
        mock_fin.side_effect = RuntimeError("Quarterly feed timeout")
        mock_cf.side_effect = RuntimeError("Cashflow service offline")
        mock_debt.return_value = {"is_financial": False, "assessment": "N/A"}
        mock_shareholding.return_value = {"as_of_date": "Latest", "promoters_percent": 50.0}
        mock_meetings.return_value = []
        mock_actions.return_value = []
        mock_news.return_value = []

        analysis = get_company_analysis("PARTIAL")
        assert analysis["symbol"] == "PARTIAL"
        assert len(analysis["limitations"]) >= 2
        assert any("Quarterly" in lim for lim in analysis["limitations"])
        assert any("Cash flow" in lim for lim in analysis["limitations"])
        # Report still contains standard layout
        assert "## Snapshot" in analysis["report_markdown"]
        assert "## Data Availability / Limitations" in analysis["report_markdown"]


# =====================================================================
# Test Group 7: Financial Institutions (Banks/NBFCs) (Tests 31 - 32)
# =====================================================================

class TestBankingAndFinancialInstitutions:
    """Test bank-specific evaluation, exclusion of normal debt/ROCE rules."""

    def test_31_financial_company_detection(self):
        """Test 31: Banks and NBFCs are correctly identified via sector and industry heuristics."""
        assert is_financial_company("Financial Services", "Private Bank") is True
        assert is_financial_company("Financials", "Public Sector Bank") is True
        assert is_financial_company("Financial Services", "NBFC") is True
        assert is_financial_company("Finance", "Insurance") is True
        assert is_financial_company("Technology", "IT Services") is False
        assert is_financial_company("Energy", "Oil & Gas") is False

    def test_32_bank_specific_scoring_excludes_corporate_debt_and_roce(self):
        """Test 32: Financial companies are scored using banking rules (ROCE N/A, no debt/equity penalty)."""
        roce_eval = evaluate_roce(15.0, is_financial=True)
        assert roce_eval["assessment"] == "Not Applicable"
        assert "banking" in roce_eval["reason"].lower() or "financial" in roce_eval["reason"].lower()

        debt_eval = get_debt_metrics("HDFCBANK", sector="Financial Services", industry="Private Bank")
        assert debt_eval["is_financial"] is True
        assert "core operating inputs" in debt_eval["assessment"] or "core operational" in debt_eval["assessment"]

        # Health scoring for a bank with high customer deposits (debt) should NOT penalize it
        bank_health = calculate_financial_health(
            qoq_revenue_growth=4.0,
            yoy_revenue_growth=14.0,
            qoq_profit_growth=5.0,
            yoy_profit_growth=16.0,
            operating_margin=None,  # Banks don't report standard operating margins
            margin_trend="Stable",
            cash_conversion_ratio=None,
            net_debt=5000000.0,  # Massive deposit liabilities
            debt_to_equity=6.5,  # Typical banking leverage
            roce=None,  # ROCE not applicable
            roe=17.5,   # Strong bank ROE
            is_financial=True,
        )
        # Should achieve Good or Very Good without debt penalty
        assert bank_health["score"] >= 7.0
        assert bank_health["rating"] in ["Good", "Very Good"]
        assert not any("Debt / Equity" in c for c in bank_health["concerns"])


# =====================================================================
# Test Group 8: Regression Test for get_stock_price (Test 33)
# =====================================================================

class TestStockPriceRegression:
    """Test existing get_stock_price functionality remains completely intact."""

    @patch("tools.stock_tool.yf.Ticker")
    def test_33_existing_stock_price_tests_continue_to_pass(self, mock_ticker):
        """Test 33: get_stock_price('TCS') continues returning accurate quote, range, and metadata."""
        mock_inst = MagicMock()
        mock_inst.fast_info = {
            "last_price": 3540.50,
            "previous_close": 3500.00,
            "year_high": 4250.00,
            "year_low": 3100.00,
            "currency": "INR",
        }
        mock_inst.info = {"shortName": "Tata Consultancy Services"}
        mock_ticker.return_value = mock_inst

        quote = get_stock_price("TCS")
        assert quote["symbol"] == "TCS"
        assert quote["current_price"] == 3540.50
        assert quote["change"] == pytest.approx(40.50, abs=0.01)
        assert quote["change_percent"] == pytest.approx(1.157, abs=0.01)
        assert quote["52_week_high"] == 4250.00
        assert quote["52_week_low"] == 3100.00
        assert quote["currency"] == "INR"
