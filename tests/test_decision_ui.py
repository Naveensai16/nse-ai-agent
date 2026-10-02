"""Unit tests for Stock Decision Assistant Streamlit UI, view routing, and dashboard interactions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

import app
from models.decision import CatalystDetail, DecisionFactorSnapshot, DecisionResult, PeerMetric


def _create_mock_decision_result(intent: str = "new", purchase_price: float | None = None) -> DecisionResult:
    """Create deterministic mock DecisionResult for UI tests."""
    return DecisionResult(
        symbol="TATAPOWER",
        company_name="Tata Power Company Limited",
        current_price=410.50,
        intent=intent,
        horizon="1 Year",
        decision_indicator="BUY" if intent == "new" else "CONSIDER ADDING",
        analysis_confidence="High",
        generated_at="02 Oct 2026 · 03:30 PM IST",
        snapshot=DecisionFactorSnapshot(
            market="Bullish",
            sector="Strong",
            fundamentals="Improving",
            quarterly_results="Strong",
            valuation="Fair",
            technicals="Bullish",
            news="Positive",
            peer_position="Outperforming",
            risk="Moderate",
            market_observation="NIFTY holding above 20 DMA; low VIX.",
            sector_observation="Power sector witnessing peak demand growth.",
            fundamentals_observation="Consistent revenue and profit expansion.",
            quarterly_observation="Q3 earnings grew 18% YoY.",
            valuation_observation="P/E 28.5x below sector average.",
            peer_observation="Higher return ratios and green mix.",
            technical_observation="Above 20/50/200 DMAs with 62 RSI.",
            catalyst_observation="1 major positive catalyst identified.",
            risk_observation="Manageable leverage with strong cash flows.",
        ),
        why_decision=[
            "Robust 1-year earnings trajectory supported by renewable capacity addition.",
            "Valuation remains fair relative to utility sector median.",
            "Strong technical posture trading above all key moving averages.",
        ],
        reasons_supporting=[
            "Structural beneficiary of national clean energy transition.",
            "Strong multi-year earnings visibility with 18% PAT growth.",
        ],
        reasons_against=[
            "Valuation reflects premium growth expectations.",
        ],
        what_would_change_positive=[
            "Further contract wins in utility-scale solar/storage.",
            "Faster-than-expected debt reduction.",
        ],
        what_would_change_negative=[
            "Delays in project execution or tariff revisions.",
            "Sharp increase in interest rates.",
        ],
        market_environment="Bullish",
        market_explanation="NIFTY holding above 20 DMA; low VIX.",
        sector_name="Power & Utilities",
        sector_outlook="Strong",
        why_sector_moving="Surging industrial power demand and renewable capacity commissioning.",
        fundamentals_status="Improving",
        fundamentals_summary="Revenue grew 14% YoY; operating margins expanded to 22.5%.",
        roe=16.8,
        roce=18.2,
        debt_to_equity=1.45,
        quarterly_verdict="Strong",
        quarterly_revenue_yoy=14.0,
        quarterly_profit_yoy=18.0,
        quarterly_margin=22.5,
        quarterly_summary="Q3 PAT jumped 18% driven by solar EPC execution.",
        valuation_verdict="Fair",
        current_pe=28.5,
        sector_pe=32.0,
        price_to_book=3.8,
        valuation_summary="Trading at 28.5x P/E, comfortably below sector median of 32x.",
        peer_position="Outperforming",
        peer_metrics=[
            PeerMetric(
                symbol="NTPC",
                company_name="NTPC Limited",
                current_price=395.0,
                market_cap=3800000000000,
                pe_ratio=18.5,
                roe=13.5,
                roce=12.8,
                operating_margin=28.0,
                debt_to_equity=1.60,
                return_1y=42.0,
            )
        ],
        peer_summary="Higher ROE and stronger clean energy mix compared to peers.",
        technical_trend="Bullish",
        technical_state="Breaking out",
        rsi_14=62.4,
        dma_20=398.0,
        dma_50=385.0,
        dma_200=340.0,
        support_level=395.0,
        resistance_level=435.0,
        high_52w=445.0,
        low_52w=260.0,
        distance_52w_high_pct=-7.75,
        return_1d=1.75,
        return_1w=3.40,
        positive_catalysts=[
            CatalystDetail(
                headline="Commissioned 300 MW Solar Project in Gujarat",
                date="2026-09-28",
                source="Exchange Filing",
                explanation="Expands operational green portfolio by 8%.",
                potential_impact="Accretive to EBITDA from Q4.",
                catalyst_type="positive",
            )
        ],
        negative_catalysts=[],
        key_risks=[
            "Commodity price volatility affecting thermal fuel margins.",
            "Higher debt leverage due to accelerated renewable capex.",
        ],
        risk_level="Moderate",
        business_vs_stock_quality="Good Company + Fairly Valued Stock",
        purchase_price=purchase_price,
        quantity=100 if purchase_price else None,
        unrealized_gain_loss_pct=round(((410.50 - purchase_price) / purchase_price) * 100.0, 2) if purchase_price else None,
        existing_position_advice=f"Unrealized gain of +{round(((410.50 - purchase_price) / purchase_price) * 100.0, 2)}%" if purchase_price else "",
        composite_score=82.5,
        report_markdown="# Mock Decision Report",
    )


class TestDecisionSessionState:
    """Test session state initialization and defaults for Decision Assistant."""

    def test_init_session_state_initializes_decision_keys(self):
        with patch.object(app, "st") as mock_st:
            mock_session = {}
            mock_st.session_state = mock_session

            app.init_session_state()

            assert "decision_stock" in mock_session
            assert mock_session["decision_stock"] == "Tata Power"
            assert mock_session["decision_intent"] == "Thinking of Buying"
            assert mock_session["decision_horizon"] == "1 Year"
            assert mock_session["decision_purchase_price"] == 0.0
            assert mock_session["decision_quantity"] == 0
            assert mock_session["decision_cache_token"] == 0
            assert mock_session["view_mode"] == "chat"


class TestDecisionSidebar:
    """Test sidebar button rendering and interaction for Decision Assistant."""

    def test_sidebar_contains_decision_assistant_button(self):
        with patch.object(app, "st") as mock_st, patch("app.list_conversations", return_value=[]):
            mock_st.session_state = {
                "current_conversation_id": None,
                "view_mode": "chat",
                "openai_api_key": "",
                "demo_mode": True,
            }
            mock_st.sidebar = MagicMock()
            mock_st.sidebar.text_input.return_value = ""

            def button_mock(label, **kwargs):
                if "Stock Decision Assistant" in label:
                    return True
                return False

            mock_st.sidebar.button.side_effect = button_mock

            app.render_sidebar()

            assert mock_st.session_state["view_mode"] == "decision"
            mock_st.rerun.assert_called()


class TestWelcomeScreenCard:
    """Test welcome screen interactive cards for Decision Assistant."""

    def test_welcome_screen_renders_decision_button(self):
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = {"view_mode": "chat"}
            mock_container = MagicMock()
            mock_st.container.return_value.__enter__.return_value = mock_container
            mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

            def button_mock(label, **kwargs):
                if "Decision Assistant" in label:
                    return True
                return False

            mock_st.button.side_effect = button_mock

            app.render_welcome_screen()

            assert mock_st.session_state["view_mode"] == "decision"
            mock_st.rerun.assert_called()


class TestDecisionDashboard:
    """Test full dashboard rendering for Decision Assistant."""

    def test_decision_dashboard_renders_new_investment(self):
        mock_res = _create_mock_decision_result(intent="new")
        with patch.object(app, "st") as mock_st, patch(
            "app._cached_analyze_stock_decision", return_value=mock_res
        ):
            mock_st.session_state = {
                "decision_stock": "Tata Power",
                "decision_intent": "Thinking of Buying",
                "decision_horizon": "1 Year",
                "decision_purchase_price": 0.0,
                "decision_quantity": 0,
                "decision_cache_token": 0,
                "view_mode": "decision",
            }
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.tabs.return_value = [MagicMock() for _ in range(6)]
            mock_st.button.return_value = False
            mock_st.radio.return_value = "🛒 Thinking of Buying (New Investment)"
            mock_st.text_input.return_value = "Tata Power"
            mock_st.selectbox.return_value = "1 Year"

            app.render_decision_assistant_dashboard()

            # Verify banner and dataframe were invoked
            mock_st.success.assert_called()
            mock_st.dataframe.assert_called()

    def test_decision_dashboard_renders_existing_investment_with_price(self):
        mock_res = _create_mock_decision_result(intent="existing", purchase_price=350.0)
        with patch.object(app, "st") as mock_st, patch(
            "app._cached_analyze_stock_decision", return_value=mock_res
        ):
            mock_st.session_state = {
                "decision_stock": "Tata Power",
                "decision_intent": "Already Own",
                "decision_horizon": "1 Year",
                "decision_purchase_price": 350.0,
                "decision_quantity": 100,
                "decision_cache_token": 0,
                "view_mode": "decision",
            }
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.tabs.return_value = [MagicMock() for _ in range(6)]
            mock_st.button.return_value = False
            mock_st.radio.return_value = "💼 Already Own This Stock (Existing Investment)"
            mock_st.text_input.return_value = "Tata Power"
            mock_st.selectbox.return_value = "1 Year"
            mock_st.number_input.side_effect = [350.0, 100]

            app.render_decision_assistant_dashboard()

            # Verify success banner called for CONSIDER ADDING
            mock_st.success.assert_called()
            # Verify metric calls include unrealized return calculation
            metric_calls = mock_st.metric.call_args_list
            labels = [c[0][0] for c in metric_calls]
            assert "Purchase Price" in labels
            assert "Unrealized Return" in labels

    def test_decision_dashboard_back_to_chat(self):
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = {"view_mode": "decision"}
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.radio.return_value = "🛒 Thinking of Buying (New Investment)"
            mock_st.text_input.return_value = "Tata Power"
            mock_st.selectbox.return_value = "1 Year"

            def button_mock(label, **kwargs):
                if "Back to Chat" in label:
                    return True
                return False

            mock_st.button.side_effect = button_mock

            app.render_decision_assistant_dashboard()

            assert mock_st.session_state["view_mode"] == "chat"
            mock_st.rerun.assert_called()


class TestAppRouting:
    """Test top-level application routing for decision assistant."""

    def test_render_app_routes_to_decision_dashboard(self):
        with patch.object(app, "st") as mock_st, patch(
            "app.render_sidebar", return_value=None
        ), patch("app.render_decision_assistant_dashboard") as mock_render_decision:
            mock_st.session_state = {"view_mode": "decision"}

            app.render_app()

            mock_render_decision.assert_called_once()
