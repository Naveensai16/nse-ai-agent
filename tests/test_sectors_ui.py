"""Unit tests for Top Sectors & Companies Streamlit UI, view mode routing, and interactive dashboard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

import app
from models.sector import (
    CompanySectorRanking,
    SectorPerformance,
    TopSectorsResult,
)
from services.sector_service import get_top_sectors_and_companies


def _create_mock_sectors_result() -> TopSectorsResult:
    """Create a mock TopSectorsResult for Streamlit UI testing."""
    top_secs = [
        SectorPerformance(
            name="Banking",
            display_name="Banking",
            index_symbol="^NSEBANK",
            performance_score=3.85,
            change_1d=1.45,
            change_1w=4.20,
            change_1m=7.10,
            trend="Bullish",
            constituents_count=12,
            advance_decline_ratio=3.0,
            why_moving="Private & PSU banks are rallying on healthy loan growth.",
            important_news=["RBI liquidity stance supportive", "Asset quality near multi-year lows"],
            institutional_activity="Strong DII inflows into Tier-1 banks",
            policy_impact="Stable repo rates maintain margin stability",
            major_earnings="HDFC Bank and ICICI Bank report strong credit growth",
        ),
        SectorPerformance(
            name="Information Technology",
            display_name="Information Technology",
            index_symbol="^CNXIT",
            performance_score=2.90,
            change_1d=0.95,
            change_1w=3.10,
            change_1m=5.40,
            trend="Bullish",
            constituents_count=10,
            advance_decline_ratio=2.0,
            why_moving="IT stocks gain on international deal momentum.",
        ),
        SectorPerformance(
            name="Auto",
            display_name="Auto",
            index_symbol="^CNXAUTO",
            performance_score=2.10,
            change_1d=0.60,
            change_1w=2.40,
            change_1m=4.80,
            trend="Bullish",
            constituents_count=9,
            advance_decline_ratio=1.8,
            why_moving="Strong vehicle dispatch numbers.",
        ),
        SectorPerformance(
            name="Pharma & Healthcare",
            display_name="Pharma & Healthcare",
            index_symbol="^CNXPHARMA",
            performance_score=1.50,
            change_1d=0.30,
            change_1w=1.80,
            change_1m=3.20,
            trend="Neutral",
            constituents_count=9,
            advance_decline_ratio=1.2,
            why_moving="Steady domestic formulations and US generic stability.",
        ),
        SectorPerformance(
            name="Energy & Power",
            display_name="Energy & Power",
            index_symbol="^CNXENERGY",
            performance_score=1.20,
            change_1d=0.20,
            change_1w=1.50,
            change_1m=2.90,
            trend="Neutral",
            constituents_count=12,
            advance_decline_ratio=1.0,
            why_moving="Peak power demand reaches new highs.",
        ),
    ]

    top_companies = [
        CompanySectorRanking(
            rank=1,
            symbol="HDFCBANK",
            company_name="HDFC Bank Limited",
            sector="Banking",
            current_price=1740.0,
            day_change_percent=2.15,
            week_change_percent=4.80,
            month_change_percent=8.10,
            market_cap=13_000_000_000_000,
            pe_ratio=19.4,
            high_52w=1795.0,
            low_52w=1380.0,
            volume=14500000,
            relative_volume=1.85,
            rsi_14=63.2,
            trend="Bullish",
            distance_from_52w_high_pct=-3.06,
            latest_catalyst="Credit growth up 15% YoY with steady deposit accretion",
            quarterly_result_summary="Net profit rises 17.5% YoY; gross NPA down to 1.24%",
            composite_score=86.5,
            why_in_top_10=[
                "Strong 1-day momentum (+2.15%)",
                "Healthy volume expansion (1.85x average)",
                "Trading within 3.1% of 52-week high",
            ],
        ),
        CompanySectorRanking(
            rank=2,
            symbol="ICICIBANK",
            company_name="ICICI Bank Limited",
            sector="Banking",
            current_price=1280.0,
            day_change_percent=1.90,
            week_change_percent=4.20,
            month_change_percent=7.40,
            market_cap=9_000_000_000_000,
            pe_ratio=17.8,
            high_52w=1320.0,
            low_52w=980.0,
            volume=9800000,
            relative_volume=1.45,
            rsi_14=61.8,
            trend="Bullish",
            distance_from_52w_high_pct=-3.03,
            latest_catalyst="Consistent NIM above 4.3% with solid retail fee income",
            quarterly_result_summary="Net profit up 16.8% YoY",
            composite_score=82.0,
            why_in_top_10=[
                "Outperforming sector median by +0.7%",
                "Constructive RSI momentum at 61.8",
            ],
        ),
    ]

    return TopSectorsResult(
        generated_at="02 Oct 2026 · 03:30 PM IST",
        top_sectors=top_secs,
        all_sectors=top_secs,
        selected_sector="Banking",
        top_companies=top_companies,
        timeframe="1 Day",
        sort_by="Momentum",
    )


class TestSectorsSessionState:
    """Test session state initialization and defaults."""

    def test_init_session_state_initializes_sector_keys(self):
        with patch.object(app, "st") as mock_st:
            mock_session = {}
            mock_st.session_state = mock_session

            app.init_session_state()

            assert "selected_sector" in mock_session
            assert mock_session["selected_sector"] is None
            assert mock_session["sector_timeframe"] == "1 Day"
            assert mock_session["sector_sort_by"] == "Momentum"
            assert mock_session["sector_cache_token"] == 0
            assert mock_session["view_mode"] == "chat"


class TestSectorsSidebar:
    """Test sidebar button rendering and interaction."""

    def test_sidebar_contains_top_sectors_button(self):
        with patch.object(app, "st") as mock_st, \
             patch("app.list_conversations", return_value=[]):
            mock_st.session_state = {
                "current_conversation_id": None,
                "view_mode": "chat",
                "openai_api_key": "",
                "demo_mode": True,
            }
            mock_st.sidebar = MagicMock()
            mock_st.sidebar.text_input.return_value = ""
            # Simulate Top Sectors button being clicked
            def button_mock(label, **kwargs):
                if "Top Sectors" in label:
                    return True
                return False

            mock_st.sidebar.button.side_effect = button_mock

            app.render_sidebar()

            assert mock_st.session_state["view_mode"] == "sectors"
            mock_st.rerun.assert_called()


class TestWelcomeScreenCard:
    """Test welcome screen interactive cards."""

    def test_welcome_screen_renders_top_sectors_button(self):
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = {"view_mode": "chat"}
            mock_container = MagicMock()
            mock_st.container.return_value.__enter__.return_value = mock_container
            mock_st.columns.return_value = [MagicMock(), MagicMock()]

            def button_mock(label, **kwargs):
                if "Top Sectors" in label:
                    return True
                return False

            mock_st.button.side_effect = button_mock

            app.render_welcome_screen()

            assert mock_st.session_state["view_mode"] == "sectors"
            mock_st.rerun.assert_called()


class TestSectorsDashboard:
    """Test full dashboard rendering."""

    def test_render_top_sectors_dashboard_renders_cleanly(self):
        mock_result = _create_mock_sectors_result()

        with patch.object(app, "st") as mock_st, \
             patch("app._cached_get_top_sectors_and_companies", return_value=mock_result):
            mock_st.session_state = {
                "view_mode": "sectors",
                "selected_sector": "Banking",
                "sector_timeframe": "1 Day",
                "sector_sort_by": "Momentum",
                "sector_cache_token": 0,
            }
            mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
            mock_st.button.return_value = False
            mock_st.selectbox.side_effect = lambda label, options, **kw: options[0]
            mock_expander = MagicMock()
            mock_st.expander.return_value.__enter__.return_value = mock_expander

            app.render_top_sectors_dashboard()

            # Verify title and subheaders were called
            mock_st.title.assert_called_with("🔥 Top Sectors & Companies")
            assert any("Current Top 5 Sectors" in str(call) for call in mock_st.subheader.call_args_list)
            assert any("Top 10 Stocks" in str(call) for call in mock_st.subheader.call_args_list)

            # Verify dataframe was called to render the clean sortable table
            mock_st.dataframe.assert_called_once()
            args, kwargs = mock_st.dataframe.call_args
            table_rows = args[0]
            assert len(table_rows) == 2
            assert table_rows[0]["Company"] == "HDFC Bank Limited"
            assert table_rows[0]["Symbol"] == "HDFCBANK"
            assert "Why in Top 10" in table_rows[0]

    def test_dashboard_back_to_chat_button(self):
        mock_result = _create_mock_sectors_result()

        with patch.object(app, "st") as mock_st, \
             patch("app._cached_get_top_sectors_and_companies", return_value=mock_result):
            mock_st.session_state = {
                "view_mode": "sectors",
                "selected_sector": "Banking",
            }
            mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]

            def button_mock(label, **kwargs):
                if "Back to Chat" in label:
                    return True
                return False

            mock_st.button.side_effect = button_mock

            app.render_top_sectors_dashboard()

            assert mock_st.session_state["view_mode"] == "chat"
            mock_st.rerun.assert_called()


class TestAppViewRouting:
    """Test view_mode routing in render_app."""

    def test_render_app_routes_to_sectors_dashboard(self):
        with patch.object(app, "st") as mock_st, \
             patch("app.render_sidebar", return_value=None), \
             patch("app.render_top_sectors_dashboard") as mock_render_sectors:
            mock_st.session_state = {
                "view_mode": "sectors",
                "current_conversation_id": None,
                "openai_api_key": "",
                "demo_mode": True,
            }

            app.render_app()

            mock_render_sectors.assert_called_once()
