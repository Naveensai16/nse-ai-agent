"""Modern landing page / empty chat welcome screen component."""

from __future__ import annotations

import streamlit as st
from ui.theme import COLOR_ACCENT, COLOR_BG_CARD, COLOR_BORDER, COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY


def render_welcome_screen() -> None:
    """Render the high-end hero welcome screen with quick-action cards and sample query pills."""
    # Hero Title & Subtitle
    st.markdown(
        f"""
        <div style="text-align: center; padding: 24px 0 28px 0;">
            <div style="display: inline-flex; align-items: center; gap: 8px; padding: 4px 14px; border-radius: 999px; background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.22); color: #A5B4FC; font-size: 12px; font-weight: 600; margin-bottom: 12px;">
                <span>🇮🇳</span> National Stock Exchange of India &bull; Agentic Intelligence
            </div>
            <div style="font-size: 28px; font-weight: 700; color: {COLOR_TEXT_PRIMARY}; letter-spacing: -0.025em; line-height: 1.25;">
                What would you like to analyze today?
            </div>
            <div style="font-size: 14px; color: {COLOR_TEXT_SECONDARY}; max-width: 620px; margin: 8px auto 0 auto; line-height: 1.5;">
                Research NSE-listed equities, evaluate sector trends, screen multi-factor 2-day trading setups, or conduct comprehensive Buy / Hold / Exit decision analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 Main Feature Action Cards (Preserves compatibility with test_welcome_screen_renders_decision_button)
    cols = st.columns(3)
    col_feat1, col_feat2, col_feat3 = cols[0], cols[1], cols[2]

    with col_feat1:
        with st.container(border=True):
            st.markdown("#### 🧠 Stock Decision")
            st.caption(
                "Evaluate whether to BUY | HOLD | REDUCE | EXIT | WAIT for any Indian listed stock "
                "based on your investment horizon, valuation, fundamentals, technicals, news, and market regime."
            )
            if st.button("🧠 Open Decision Assistant", key="welcome_decision_btn", use_container_width=True, type="primary"):
                st.session_state["view_mode"] = "decision"
                st.rerun()

    with col_feat2:
        with st.container(border=True):
            st.markdown("#### 🔥 Top Sectors")
            st.caption(
                "Discover the current Top 5 performing NSE sectors and inspect multi-factor ranked "
                "top 10 companies with real-time momentum, earnings, and news catalysts."
            )
            if st.button("🔥 Open Top Sectors", key="welcome_top_sectors_btn", use_container_width=True):
                st.session_state["view_mode"] = "sectors"
                st.rerun()

    with col_feat3:
        with st.container(border=True):
            st.markdown("#### 📈 2-Day Trading")
            st.caption(
                "Screen high-liquidity NIFTY equities for favorable short-term setups over the next "
                "1–2 trading sessions using price action, volume confirmation, and verified corporate disclosures."
            )
            if st.button("📈 Open Opportunities", key="welcome_opp_btn", use_container_width=True):
                st.session_state["view_mode"] = "opportunities"
                st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Sample Query Quick-Pill Recommendations
    st.markdown(
        f"""
        <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {COLOR_TEXT_MUTED}; margin-bottom: 10px;">
            Suggested Prompts & Queries
        </div>
        """,
        unsafe_allow_html=True,
    )

    pills_col1, pills_col2 = st.columns(2)
    with pills_col1:
        st.markdown(
            f"""
            <div style="background: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;">
                <div style="font-size: 12px; font-weight: 600; color: #A5B4FC; margin-bottom: 4px;">🧠 Investment & Decision Questions</div>
                <div style="font-size: 12.5px; color: {COLOR_TEXT_SECONDARY}; line-height: 1.6;">
                    &bull; <em>"Should I buy India Cements?"</em><br>
                    &bull; <em>"I own Tata Power. Should I hold or sell?"</em><br>
                    &bull; <em>"I bought HDFC Bank at ₹1,650. Should I keep it?"</em><br>
                    &bull; <em>"Is Tata Motors good for a 2-year investment?"</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pills_col2:
        st.markdown(
            f"""
            <div style="background: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;">
                <div style="font-size: 12px; font-weight: 600; color: #38BDF8; margin-bottom: 4px;">📊 Market Research & Stock Comparisons</div>
                <div style="font-size: 12.5px; color: {COLOR_TEXT_SECONDARY}; line-height: 1.6;">
                    &bull; <em>"Tell me about TCS"</em><br>
                    &bull; <em>"Compare TCS with Infosys"</em><br>
                    &bull; <em>"Why is Reliance falling today?"</em><br>
                    &bull; <em>"What is the technical sentiment for SBIN?"</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
