"""Unit tests for the 2-Day Trading Opportunities Streamlit UI, dedicated dashboard, and tool integrations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import app
from agent.demo_agent import run_demo_agent
from models.opportunity import (
    CatalystItem,
    MarketRegime,
    OpportunityScanResult,
    OpportunitySetup,
    RiskItem,
    ScoreBreakdown,
    TechnicalSignals,
)
from services.llm_service import get_all_tools, get_tool_by_name
from tools.opportunity_tool import get_short_term_opportunities


def _create_mock_scan_result() -> OpportunityScanResult:
    """Create a sample OpportunityScanResult for UI mocking."""
    regime = MarketRegime(
        regime="Bullish",
        nifty_change_1d=0.85,
        nifty_return_5d=2.1,
        nifty_above_20dma=True,
        nifty_above_50dma=True,
        nifty_current_value=25150.0,
        bank_nifty_change_1d=1.1,
        summary="NIFTY 50 is trading strongly above moving averages.",
    )

    signals = TechnicalSignals(
        current_price=2950.0,
        previous_close=2910.0,
        day_change_percent=1.37,
        symbol="RELIANCE",
        return_2d=2.1,
        return_5d=3.8,
        return_20d=7.2,
        dma_20=2880.0,
        dma_50=2810.0,
        distance_from_20dma=2.43,
        distance_from_50dma=4.98,
        volume=2500000.0,
        avg_volume_20d=1500000.0,
        volume_ratio=1.67,
        rsi_14=64.5,
        atr_14=42.0,
        high_20d=2960.0,
        low_20d=2780.0,
        is_breakout=True,
        is_above_20dma=True,
        is_above_50dma=True,
        is_dma_bullish_alignment=True,
        gap_percent=0.5,
        nearest_support=2880.0,
        nearest_resistance=3020.0,
        potential_upside_pct=2.37,
        potential_downside_pct=2.37,
        risk_reward_ratio=1.0,
    )

    breakdown = ScoreBreakdown(
        technical_momentum=22.0,
        volume_confirmation=12.0,
        market_sector_strength=14.0,
        positive_catalysts=16.0,
        financial_results=12.0,
        risk_adjustment=8.0,
        total_score=84.0,
    )

    candidate = OpportunitySetup(
        rank=1,
        symbol="RELIANCE",
        company_name="Reliance Industries Limited",
        sector="Energy & Conglomerate",
        current_price=2950.0,
        day_change_percent=1.37,
        volume_ratio=1.67,
        opportunity_score=84.0,
        confidence_label="High-Confidence Setup",
        score_breakdown=breakdown,
        technical_signals=signals,
        why_on_watchlist=[
            "Strong breakout above 20-day high with 1.67x volume surge",
            "Broader market in confirmed Bullish regime",
            "Robust quarterly profit acceleration",
        ],
        positive_catalysts=[
            CatalystItem(
                catalyst_type="positive",
                category="expansion",
                headline="Commissions new 5GW solar capacity unit in Jamnagar",
                summary="Major green energy capex operationalized ahead of schedule",
                source="Company Press Release",
                date="01 Oct 2026",
            )
        ],
        risks=[
            RiskItem(
                risk_type="overhead_resistance",
                description="Approaching psychological round number ₹3,000",
                severity="medium",
            )
        ],
        recent_news=[
            {
                "title": "Reliance expanding retail network with 200 new stores",
                "source": "Economic Times",
                "date": "02 Oct 2026",
                "summary": "Expansion underway targeting festive demand.",
                "url": "https://economictimes.indiatimes.com/example-link",
            }
        ],
        recent_filings=[],
    )

    result = OpportunityScanResult(
        generated_at="02 Oct 2026 · 03:30 PM IST",
        market_session="Closed",
        target_horizon="Next 1–2 NSE trading sessions (Mon, 05 Oct – Tue, 06 Oct 2026)",
        market_regime=regime,
        universe="NIFTY 200",
        total_scanned=100,
        candidates=[candidate],
        report_markdown="# 2-Day Trading Opportunities\nSample report content",
    )
    return result


# ==============================================================================
# UI Structure and State Transitions
# ==============================================================================


def test_init_session_state_includes_view_mode():
    """Verify init_session_state sets default view_mode to 'chat'."""
    fake_state = {}
    with patch("streamlit.session_state", fake_state):
        app.init_session_state()
        assert fake_state["view_mode"] == "chat"
        assert "opp_cache_token" in fake_state


def test_sidebar_opportunity_button_switches_mode():
    """Verify clicking 2-Day Trading Opportunities button sets view_mode to 'opportunities'."""
    fake_state = {
        "view_mode": "chat",
        "current_conversation_id": "test-cid",
        "openai_api_key": "",
        "demo_mode": True,
    }

    # Simulate button clicks: False for New Chat, True for Opportunities
    def fake_button(label, *args, **kwargs):
        if "2-Day Trading Opportunities" in label:
            return True
        return False

    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.sidebar.button", side_effect=fake_button), \
         patch("streamlit.sidebar.title"), \
         patch("streamlit.sidebar.caption"), \
         patch("streamlit.sidebar.markdown"), \
         patch("streamlit.sidebar.subheader"), \
         patch("streamlit.sidebar.text_input", return_value=""), \
         patch("streamlit.sidebar.checkbox", return_value=True), \
         patch("streamlit.sidebar.info"), \
         patch("app.list_conversations", return_value=[]), \
         patch("streamlit.rerun") as mock_rerun:

        app.render_sidebar()
        assert fake_state["view_mode"] == "opportunities"
        mock_rerun.assert_called_once()


def test_sidebar_new_chat_button_resets_to_chat():
    """Verify clicking + New Chat resets view_mode to 'chat' and clears current conversation."""
    fake_state = {
        "view_mode": "opportunities",
        "current_conversation_id": "test-cid",
        "openai_api_key": "",
        "demo_mode": True,
    }

    def fake_button(label, *args, **kwargs):
        if "New Chat" in label:
            return True
        return False

    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.sidebar.button", side_effect=fake_button), \
         patch("streamlit.sidebar.title"), \
         patch("streamlit.sidebar.caption"), \
         patch("streamlit.sidebar.markdown"), \
         patch("streamlit.sidebar.subheader"), \
         patch("streamlit.sidebar.text_input", return_value=""), \
         patch("streamlit.sidebar.checkbox", return_value=True), \
         patch("streamlit.sidebar.info"), \
         patch("app.list_conversations", return_value=[]), \
         patch("streamlit.rerun") as mock_rerun:

        app.render_sidebar()
        assert fake_state["view_mode"] == "chat"
        assert fake_state["current_conversation_id"] is None
        mock_rerun.assert_called_once()


def test_render_opportunities_dashboard_renders_all_sections():
    """Verify render_opportunities_dashboard renders metrics, regime, table, and cards without error."""
    mock_res = _create_mock_scan_result()
    fake_state = {"view_mode": "opportunities", "opp_cache_token": 0}

    def fake_columns(spec, *args, **kwargs):
        count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [MagicMock() for _ in range(count)]

    with patch("streamlit.session_state", fake_state), \
         patch("app._cached_scan_opportunities", return_value=mock_res), \
         patch("streamlit.title") as mock_title, \
         patch("streamlit.markdown"), \
         patch("streamlit.subheader"), \
         patch("streamlit.metric") as mock_metric, \
         patch("streamlit.dataframe") as mock_df, \
         patch("streamlit.expander"), \
         patch("streamlit.columns", side_effect=fake_columns), \
         patch("streamlit.selectbox", side_effect=["NIFTY 200", 5]), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.spinner"), \
         patch("streamlit.caption"), \
         patch("streamlit.write"), \
         patch("streamlit.warning"), \
         patch("streamlit.info"):

        app.render_opportunities_dashboard()
        mock_title.assert_called_once_with("2-Day Trading Opportunities")
        mock_df.assert_called_once()


# ==============================================================================
# Tool & Agent Integration
# ==============================================================================


def test_get_short_term_opportunities_tool_returns_dict():
    """Verify get_short_term_opportunities returns valid dictionary structure."""
    with patch("tools.opportunity_tool.scan_short_term_opportunities") as mock_scan:
        mock_scan.return_value = _create_mock_scan_result()
        res = get_short_term_opportunities(limit=5, universe="NIFTY 200")

        assert isinstance(res, dict)
        assert res["status"] == "success"
        assert "generated_at" in res
        assert "market_session" in res
        assert "target_horizon" in res
        assert "market_regime" in res
        assert len(res["candidates"]) == 1
        assert res["candidates"][0]["symbol"] == "RELIANCE"


def test_llm_service_registers_opportunity_tool():
    """Verify get_short_term_opportunities is registered in LLM service extended tools."""
    tool = get_tool_by_name("get_short_term_opportunities")
    assert tool is not None
    assert tool.name == "get_short_term_opportunities"

    all_extended_names = [t.name for t in get_all_tools(include_extended=True)]
    assert "get_short_term_opportunities" in all_extended_names


def test_demo_agent_handles_opportunity_prompt(tmp_path):
    """Verify demo agent dispatches opportunity queries to get_short_term_opportunities."""
    db_file = tmp_path / "test_opp.db"

    with patch("agent.demo_agent.get_short_term_opportunities") as mock_opp:
        mock_opp.return_value = {
            "report_markdown": "# 2-Day Trading Opportunities\nMocked setup list.",
            "candidates": [],
        }

        res = run_demo_agent("Show me 2-day trading opportunities", db_path=db_file)
        assert "tool_calls" in res
        call_names = [tc["name"] for tc in res["tool_calls"]]
        assert "get_short_term_opportunities" in call_names
        assert "2-Day Trading Opportunities" in res["response"]
