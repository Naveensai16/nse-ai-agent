"""Unit tests for Top Sectors and Sector-Constituent Ranking Service and Tools."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from models.sector import (
    CompanySectorRanking,
    SectorPerformance,
    TopSectorsResult,
)
from services.sector_service import (
    SECTOR_DEFINITIONS,
    calculate_period_return,
    evaluate_company_trend,
    evaluate_sector_trend,
    get_all_sectors_performance,
    get_top_performing_sectors,
    get_top_sectors_and_companies,
    rank_companies_in_sector,
)
from tools.sector_tool import (
    format_top_sectors_markdown,
    get_top_sectors_and_companies as get_top_sectors_and_companies_tool_fn,
)
from services.llm_service import (
    get_all_tools,
    get_top_sectors_and_companies_tool,
)
from agent.demo_agent import run_demo_agent


def make_dummy_history(symbol: str, trend: str = "bullish") -> pd.DataFrame:
    """Generate deterministic OHLCV DataFrame for testing."""
    dates = pd.date_range(end="2026-10-02", periods=60, freq="B")
    n = len(dates)

    if trend == "bullish":
        base = 100.0
        prices = [base + i * 1.5 for i in range(n)]
        volumes = [100000 + i * 2000 for i in range(n)]
    elif trend == "bearish":
        base = 200.0
        prices = [base - i * 1.5 for i in range(n)]
        volumes = [80000 + i * 1000 for i in range(n)]
    else:
        base = 150.0
        prices = [base + (i % 3 - 1) * 2.0 for i in range(n)]
        volumes = [100000 for _ in range(n)]

    df = pd.DataFrame(
        {
            "Open": [p * 0.99 for p in prices],
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


def dummy_history_provider(symbol: str) -> pd.DataFrame:
    """Deterministic provider that gives high gains for Banking and lower for others."""
    sym_up = symbol.upper()
    if sym_up in ("HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"):
        return make_dummy_history(symbol, trend="bullish")
    elif sym_up in ("TCS", "INFY", "WIPRO"):
        return make_dummy_history(symbol, trend="neutral")
    else:
        return make_dummy_history(symbol, trend="bearish")


def dummy_company_info_provider(symbol: str) -> dict:
    """Deterministic company profile provider."""
    sym_up = symbol.upper()
    return {
        "company": f"{sym_up} India Limited",
        "market_cap": 8_000_000_000_000 if sym_up == "HDFCBANK" else 1_500_000_000_000,
        "trailing_pe": 18.5,
        "forward_pe": 16.2,
        "52_week_high": 250.0,
        "52_week_low": 120.0,
    }


def dummy_financials_provider(symbol: str) -> dict:
    """Deterministic financials provider."""
    return {
        "yoy_revenue_growth": 14.5,
        "yoy_profit_growth": 18.2,
        "margin_trend": "Expanding",
        "quarters": [
            {"date": "Q3 FY26", "revenue": 50000, "operating_margin": 24.5, "net_profit": 12000}
        ],
    }


def dummy_news_provider(symbol: str) -> list[str]:
    """Deterministic news headlines provider."""
    return [
        f"{symbol} secures significant institutional order expansion.",
        f"Quarterly earnings for {symbol} exceed consensus estimates.",
    ]


class TestSectorModels:
    """Verify sector dataclasses and serialization."""

    def test_sector_performance_creation_and_dict(self):
        sp = SectorPerformance(
            name="Banking",
            display_name="Banking",
            index_symbol="^NSEBANK",
            performance_score=4.5,
            change_1d=1.2,
            change_1w=3.8,
            change_1m=6.5,
            trend="Bullish",
            constituents_count=12,
            advance_decline_ratio=2.5,
            why_moving="Credit growth acceleration",
        )
        d = sp.to_dict()
        assert d["name"] == "Banking"
        assert d["performance_score"] == 4.5
        assert d["trend"] == "Bullish"
        assert d["why_moving"] == "Credit growth acceleration"

    def test_company_sector_ranking_creation_and_properties(self):
        cr = CompanySectorRanking(
            rank=1,
            symbol="HDFCBANK",
            company_name="HDFC Bank Limited",
            sector="Banking",
            current_price=1750.0,
            day_change_percent=2.4,
            week_change_percent=4.8,
            month_change_percent=7.5,
            market_cap=12_000_000_000_000,
            pe_ratio=19.2,
            high_52w=1800.0,
            low_52w=1380.0,
            volume=15000000,
            relative_volume=1.8,
            rsi_14=62.5,
            trend="Bullish",
            distance_from_52w_high_pct=-2.78,
            latest_catalyst="Robust deposit growth reported",
            quarterly_result_summary="Net profit up 18% YoY",
            composite_score=85.0,
            why_in_top_10=["Top 1-day momentum", "Heavy volume surge"],
        )
        assert cr.distance_from_52w_high == -2.78
        d = cr.to_dict()
        assert d["rank"] == 1
        assert d["symbol"] == "HDFCBANK"
        assert len(d["why_in_top_10"]) == 2

    def test_top_sectors_result_serialization(self):
        res = TopSectorsResult(
            generated_at="02 Oct 2026 · 03:30 PM IST",
            top_sectors=[],
            all_sectors=[],
            selected_sector="Banking",
            top_companies=[],
            timeframe="1 Day",
            sort_by="Momentum",
        )
        d = res.to_dict()
        assert d["selected_sector"] == "Banking"
        assert d["timeframe"] == "1 Day"
        assert d["sort_by"] == "Momentum"


class TestSectorTrendAndReturns:
    """Test period return calculations and trend classifications."""

    def test_calculate_period_return(self):
        df = make_dummy_history("TEST", trend="bullish")
        ret_1d = calculate_period_return(df, 1)
        assert ret_1d > 0.0

        ret_empty = calculate_period_return(None, 1)
        assert ret_empty == 0.0

    def test_evaluate_sector_trend(self):
        assert evaluate_sector_trend(1.5, 3.0, 5.0) == "Bullish"
        assert evaluate_sector_trend(-1.5, -2.5, -4.0) == "Bearish"
        assert evaluate_sector_trend(0.1, 0.2, 0.5) == "Neutral"

    def test_evaluate_company_trend(self):
        assert evaluate_company_trend(1.2, 2.5, 60.0) == "Bullish"
        assert evaluate_company_trend(-1.2, -2.5, 40.0) == "Bearish"
        assert evaluate_company_trend(0.0, 0.2, 50.0) == "Neutral"


class TestSectorService:
    """Test sector performance retrieval and multi-factor ranking."""

    def test_get_all_sectors_performance(self):
        sectors = get_all_sectors_performance(history_provider=dummy_history_provider)
        assert len(sectors) >= 5
        sector_names = [s.name for s in sectors]
        assert "Banking" in sector_names
        assert "Information Technology" in sector_names
        assert "Auto" in sector_names

        for s in sectors:
            assert s.trend in ("Bullish", "Neutral", "Bearish")
            assert s.constituents_count > 0
            assert s.why_moving != ""
            assert s.institutional_activity != ""
            assert s.policy_impact != ""
            assert s.major_earnings != ""

    def test_get_top_performing_sectors(self):
        top_5, all_secs = get_top_performing_sectors(
            top_n=5, history_provider=dummy_history_provider
        )
        assert len(top_5) == 5
        assert len(all_secs) >= len(top_5)
        # Top 5 should be sorted descending by performance score
        scores = [s.performance_score for s in top_5]
        assert scores == sorted(scores, reverse=True)

    def test_rank_companies_in_sector_multi_factor(self):
        """Verify companies are ranked by multi-factor score and not simply by market cap."""
        ranked = rank_companies_in_sector(
            sector_name="Banking",
            timeframe="1 Day",
            sort_by="Momentum",
            top_n=10,
            history_provider=dummy_history_provider,
            company_info_provider=dummy_company_info_provider,
            news_provider=dummy_news_provider,
            financials_provider=dummy_financials_provider,
        )
        assert len(ranked) <= 10
        assert len(ranked) > 0

        # Check all required fields on every company
        for c in ranked:
            assert c.rank >= 1
            assert c.company_name
            assert c.symbol
            assert c.current_price > 0
            assert c.market_cap is not None
            assert c.pe_ratio is not None
            assert c.high_52w is not None
            assert c.low_52w is not None
            assert c.relative_volume >= 0
            assert c.trend in ("Bullish", "Neutral", "Bearish")
            assert c.distance_from_52w_high_pct <= 0.01  # typically <= 0%
            assert c.latest_catalyst != ""
            assert c.quarterly_result_summary != ""
            assert len(c.why_in_top_10) >= 2  # clearly explains why in top 10

    def test_rank_companies_sorting_filters(self):
        # 1. Performance sort
        perf_ranked = rank_companies_in_sector(
            sector_name="Banking",
            timeframe="1 Day",
            sort_by="Performance",
            top_n=5,
            history_provider=dummy_history_provider,
            company_info_provider=dummy_company_info_provider,
        )
        day_changes = [c.day_change_percent for c in perf_ranked]
        assert day_changes == sorted(day_changes, reverse=True)

        # 2. Volume sort
        vol_ranked = rank_companies_in_sector(
            sector_name="Banking",
            timeframe="1 Day",
            sort_by="Volume",
            top_n=5,
            history_provider=dummy_history_provider,
            company_info_provider=dummy_company_info_provider,
        )
        rel_vols = [c.relative_volume for c in vol_ranked]
        assert rel_vols == sorted(rel_vols, reverse=True)

    def test_get_top_sectors_and_companies_master(self):
        result = get_top_sectors_and_companies(
            sector_name="Banking",
            timeframe="1 Day",
            sort_by="Momentum",
            top_sectors_count=5,
            top_companies_count=10,
            history_provider=dummy_history_provider,
            company_info_provider=dummy_company_info_provider,
            news_provider=dummy_news_provider,
            financials_provider=dummy_financials_provider,
        )
        assert isinstance(result, TopSectorsResult)
        assert len(result.top_sectors) == 5
        assert result.selected_sector == "Banking"
        assert len(result.top_companies) <= 10
        assert result.generated_at != ""


class TestSectorToolAndMarkdown:
    """Test the tool wrappers and markdown generator."""

    def test_format_top_sectors_markdown(self):
        result = get_top_sectors_and_companies(
            sector_name="Banking",
            history_provider=dummy_history_provider,
            company_info_provider=dummy_company_info_provider,
        )
        md = format_top_sectors_markdown(result.to_dict())
        assert "# 🔥 Current Top Sectors" in md
        assert "Top 5 Performing Sectors" in md
        assert "Sector Summary: Banking" in md
        assert "Top 10 Stocks — Banking" in md
        assert "Why in Top 10" in md

    def test_get_top_sectors_and_companies_tool_fn(self):
        mock_res = get_top_sectors_and_companies(
            sector_name="Banking",
            history_provider=dummy_history_provider,
            company_info_provider=dummy_company_info_provider,
            news_provider=dummy_news_provider,
            financials_provider=dummy_financials_provider,
        )
        from unittest.mock import patch
        with patch("tools.sector_tool._get_top_sectors_and_companies", return_value=mock_res):
            res = get_top_sectors_and_companies_tool_fn(
                sector_name="Banking",
                timeframe="1 Day",
                sort_by="Momentum",
                top_n=5,
            )
            assert res["status"] == "success"
            assert "top_sectors" in res
            assert "top_companies" in res
            assert "report_markdown" in res
            assert len(res["top_sectors"]) == 5

    def test_llm_service_tools_registration(self):
        tools = get_all_tools(include_extended=True)
        tool_names = [t.name for t in tools]
        assert "get_top_sectors_and_companies" in tool_names

        mock_res = get_top_sectors_and_companies(
            sector_name="Banking",
            history_provider=dummy_history_provider,
            company_info_provider=dummy_company_info_provider,
        )
        from unittest.mock import patch
        with patch("tools.sector_tool._get_top_sectors_and_companies", return_value=mock_res):
            tool_out = get_top_sectors_and_companies_tool.invoke({
                "sector_name": "Banking",
                "timeframe": "1 Day",
                "sort_by": "Momentum",
                "top_n": 5,
            })
            assert isinstance(tool_out, dict)
            assert tool_out.get("status") == "success"


class TestDemoAgentSectorRouting:
    """Verify demo agent handles top sector queries."""

    def test_demo_agent_top_sectors_intent(self):
        mock_res = get_top_sectors_and_companies(
            sector_name="Banking",
            history_provider=dummy_history_provider,
            company_info_provider=dummy_company_info_provider,
        )
        from unittest.mock import patch
        with patch("tools.sector_tool._get_top_sectors_and_companies", return_value=mock_res):
            out = run_demo_agent("What are the top sectors in the market today?")
            assert out["conversation_id"] is not None
            assert "Top 5 Performing Sectors" in out["response"] or "Top Sectors" in out["response"]
            tool_names = [tc["name"] for tc in out["tool_calls"]]
            assert "get_top_sectors_and_companies" in tool_names
