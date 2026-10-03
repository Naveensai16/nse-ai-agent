"""Streamlit web application for the NSE Agentic AI Stock Analysis system.

Provides a clean ChatGPT-style user interface with persistent SQLite conversation management,
real-time tool calling visualization, friendly error presentation, responsive design,
and a dedicated 2-Day Trading Opportunities dashboard.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv

from agent.demo_agent import format_currency, run_demo_agent
from agent.runner import run_agent
from models.decision import DecisionResult
from models.opportunity import OpportunityScanResult
from models.sector import TopSectorsResult
from services.database_service import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    list_conversations,
)
from services.decision_service import analyze_stock_decision
from services.opportunity_service import scan_short_term_opportunities
from services.sector_service import get_top_sectors_and_companies
from utils.ui_helpers import (
    filter_chat_messages,
    format_conversation_title,
    format_error_message,
    format_tool_call_summary,
)
from services.symbol_resolver import (
    INDIAN_STOCK_MASTER,
    resolve_nse_symbol,
    search_stocks,
)
from ui.components import (
    render_compliance_notice,
    render_decision_card,
    render_metric_card,
    render_positive_negative_cards,
    render_stock_header,
    stock_autocomplete,
)
from ui.header import render_top_header
from ui.theme import get_global_css

# Load environment variables on startup
load_dotenv()


def init_session_state() -> None:
    """Initialize Streamlit session state variables for chat tracking and view routing."""
    if "current_conversation_id" not in st.session_state:
        st.session_state["current_conversation_id"] = None
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "chat"
    if "opp_cache_token" not in st.session_state:
        st.session_state["opp_cache_token"] = 0
    if "selected_sector" not in st.session_state:
        st.session_state["selected_sector"] = None
    if "sector_timeframe" not in st.session_state:
        st.session_state["sector_timeframe"] = "1 Day"
    if "sector_sort_by" not in st.session_state:
        st.session_state["sector_sort_by"] = "Momentum"
    if "sector_cache_token" not in st.session_state:
        st.session_state["sector_cache_token"] = 0
    if "decision_stock" not in st.session_state:
        st.session_state["decision_stock"] = "Tata Power"
    if "decision_intent" not in st.session_state:
        st.session_state["decision_intent"] = "Thinking of Buying"
    if "decision_horizon" not in st.session_state:
        st.session_state["decision_horizon"] = "1 Year"
    if "decision_purchase_price" not in st.session_state:
        st.session_state["decision_purchase_price"] = 0.0
    if "decision_quantity" not in st.session_state:
        st.session_state["decision_quantity"] = 0
    if "decision_cache_token" not in st.session_state:
        st.session_state["decision_cache_token"] = 0
    # Search and Decision Assistant lifecycle states
    if "decision_pending_input" not in st.session_state:
        st.session_state["decision_pending_input"] = None
    if "decision_selected_stock" not in st.session_state:
        st.session_state["decision_selected_stock"] = None
    if "decision_selected_symbol" not in st.session_state:
        st.session_state["decision_selected_symbol"] = None
    if "stock_search_text" not in st.session_state:
        st.session_state["stock_search_text"] = ""
    if "stock_suggestions" not in st.session_state:
        st.session_state["stock_suggestions"] = []
    if "selected_stock_name" not in st.session_state:
        st.session_state["selected_stock_name"] = None
    if "selected_symbol" not in st.session_state:
        st.session_state["selected_symbol"] = None
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "analysis_requested" not in st.session_state:
        st.session_state["analysis_requested"] = False


@st.cache_data(ttl=300, show_spinner=False)
def _cached_scan_opportunities(universe: str, limit: int, cache_token: int = 0) -> OpportunityScanResult:
    """Cached wrapper around scan_short_term_opportunities to accelerate page interactions."""
    return scan_short_term_opportunities(universe=universe, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_get_top_sectors_and_companies(
    sector_name: Optional[str],
    timeframe: str,
    sort_by: str,
    cache_token: int = 0,
) -> TopSectorsResult:
    """Cached wrapper around get_top_sectors_and_companies to accelerate sector dashboard interactions."""
    return get_top_sectors_and_companies(
        sector_name=sector_name,
        timeframe=timeframe,
        sort_by=sort_by,
        top_sectors_count=5,
        top_companies_count=10,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _cached_analyze_stock_decision(
    stock: str,
    intent: str,
    horizon: str,
    purchase_price: Optional[float] = None,
    quantity: Optional[int] = None,
    cache_token: int = 0,
) -> DecisionResult:
    """Cached wrapper around analyze_stock_decision to accelerate decision assistant interactions."""
    return analyze_stock_decision(
        stock=stock,
        intent=intent,
        horizon=horizon,
        purchase_price=purchase_price,
        quantity=quantity,
    )


def render_sidebar() -> Optional[str]:
    """Render the sidebar containing conversation history and controls.

    Returns:
        Optional[str]: Currently active conversation ID.
    """
    st.sidebar.title("📈 NSE AI Analyst")
    st.sidebar.caption("Intelligent Indian Stock Market Assistant")

    # "+ New Chat" action button
    if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state["current_conversation_id"] = None
        st.session_state["view_mode"] = "chat"
        st.rerun()

    # Dedicated button for 2-Day Trading Opportunities directly below New Chat
    is_opp_mode = st.session_state.get("view_mode") == "opportunities"
    opp_btn_type = "primary" if is_opp_mode else "secondary"
    if st.sidebar.button("📈 2-Day Trading Opportunities", use_container_width=True, type=opp_btn_type):
        st.session_state["view_mode"] = "opportunities"
        st.rerun()

    # Dedicated button for Top Sectors directly below 2-Day Trading Opportunities
    is_sectors_mode = st.session_state.get("view_mode") == "sectors"
    sec_btn_type = "primary" if is_sectors_mode else "secondary"
    if st.sidebar.button("🔥 Top Sectors", use_container_width=True, type=sec_btn_type):
        st.session_state["view_mode"] = "sectors"
        st.rerun()

    # Dedicated button for Stock Decision Assistant directly below Top Sectors
    is_decision_mode = st.session_state.get("view_mode") == "decision"
    dec_btn_type = "primary" if is_decision_mode else "secondary"
    if st.sidebar.button("🧠 Stock Decision Assistant", use_container_width=True, type=dec_btn_type):
        st.session_state["view_mode"] = "decision"
        st.rerun()

    # System Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ System Status")

    from services.ollama_service import check_ollama_health

    health = check_ollama_health()
    cfg_model = health.get("configured_model", "llama3.2:1b")
    if health.get("reachable") and health.get("model_available"):
        st.sidebar.success(f"🤖 **AI Engine**: Active (Ollama - {cfg_model})")
    elif health.get("reachable"):
        st.sidebar.warning(f"⚠️ **AI Engine**: Connected (Loading '{cfg_model}'...)")
    else:
        st.sidebar.info("🇮🇳 **NSE Assistant**: Active (Built-in Intelligence)")

    st.sidebar.caption(f"Model: **{cfg_model}** | Rate Limit: **30 req/min**")
    st.sidebar.caption("Exchange: **National Stock Exchange (NSE)**")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent Conversations")

    # Fetch recent conversations from SQLite
    try:
        conversations = list_conversations(limit=25)
    except Exception as e:
        st.sidebar.error("Failed to load chat history.")
        conversations = []

    current_id = st.session_state.get("current_conversation_id")

    if not conversations:
        st.sidebar.caption("No previous conversations yet.")
    else:
        for conv in conversations:
            cid = conv["conversation_id"]
            title = format_conversation_title(conv["title"], max_length=32)
            is_active = (cid == current_id) and (not is_opp_mode) and (not is_sectors_mode) and (not is_decision_mode)

            col_select, col_del = st.sidebar.columns([0.84, 0.16])
            with col_select:
                btn_label = f"👉 {title}" if is_active else f"💬 {title}"
                if st.button(
                    btn_label,
                    key=f"conv_{cid}",
                    use_container_width=True,
                    help=conv.get("title", ""),
                ):
                    st.session_state["current_conversation_id"] = cid
                    st.session_state["view_mode"] = "chat"
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_{cid}", help="Delete chat", use_container_width=True):
                    try:
                        delete_conversation(cid)
                        if st.session_state.get("current_conversation_id") == cid:
                            st.session_state["current_conversation_id"] = None
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error("Failed to delete chat.")

    return st.session_state.get("current_conversation_id")


def render_opportunities_dashboard() -> None:
    """Render dedicated dashboard for 2-day short-term trading setups on NSE."""
    render_top_header(
        title="2-Day Trading Opportunities",
        subtitle="Short-term momentum setups, volume breakouts & verified news catalysts",
    )

    header_col1, header_col2 = st.columns([0.82, 0.18])
    with header_col1:
        st.title("2-Day Trading Opportunities")
        st.markdown(
            "##### NSE stocks showing potentially favorable short-term setups based on price action, volume, "
            "market trend, corporate announcements, results and recent verified news."
        )
    with header_col2:
        st.write("")
        st.write("")
        if st.button("💬 Back to Chat", use_container_width=True):
            st.session_state["view_mode"] = "chat"
            st.rerun()

    # Top Controls Bar: Universe, Limit, Refresh
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([0.45, 0.25, 0.30])
    with ctrl_col1:
        universe_choice = st.selectbox(
            "Stock Universe",
            options=["NIFTY 200", "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100"],
            index=0,
            key="opp_universe_select",
            help="Select liquid universe of Indian equities to screen (default: NIFTY 200)",
        )
    with ctrl_col2:
        limit_choice = st.selectbox(
            "Top Candidates",
            options=[5, 10],
            index=0,
            key="opp_limit_select",
            help="Number of shortlisted candidates to display",
        )
    with ctrl_col3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.session_state["opp_cache_token"] = st.session_state.get("opp_cache_token", 0) + 1
            st.rerun()

    # Fetch scan results with caching
    cache_token = st.session_state.get("opp_cache_token", 0)
    with st.spinner("Analyzing liquid NSE equities, technical indicators & corporate catalysts..."):
        try:
            scan_res = _cached_scan_opportunities(universe_choice, limit_choice, cache_token)
        except Exception as exc:
            st.error(f"⚠️ Opportunity scanning encountered an issue: {exc}")
            return

    # Metadata Status Bar
    meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
    with meta_c1:
        gen_display = scan_res.generated_at.split(" ")[-1] if " " in scan_res.generated_at else scan_res.generated_at
        st.metric("Generated At", gen_display)
    with meta_c2:
        sess_icon = "🟢" if scan_res.market_session == "Open" else "🔴"
        st.metric("Market Session", f"{sess_icon} {scan_res.market_session}")
    with meta_c3:
        st.metric("Target Horizon", "1–2 Trading Sessions")
    with meta_c4:
        st.metric("Stocks Evaluated", f"{scan_res.total_scanned} ({scan_res.universe})")

    st.caption(f"🗓️ Target Horizon: **{scan_res.target_horizon}**")
    st.markdown("---")

    # Market Regime Banner
    regime = scan_res.market_regime
    reg_icon = "🟢" if regime.regime == "Bullish" else ("🔴" if regime.regime == "Bearish" else "🟡")

    st.subheader(f"{reg_icon} Market Regime: {regime.regime}")
    st.markdown(f"> **Market Context:** {regime.summary}")

    rg_c1, rg_c2, rg_c3, rg_c4 = st.columns(4)
    with rg_c1:
        n_val = f"{regime.nifty_current_value:,.2f}" if regime.nifty_current_value else "N/A"
        st.metric("NIFTY 50", n_val, f"{regime.nifty_change_1d:+.2f}%")
    with rg_c2:
        st.metric("NIFTY 5-Day Return", f"{regime.nifty_return_5d:+.2f}%")
    with rg_c3:
        dma_status = "Above 20 & 50 DMA" if (regime.nifty_above_20dma and regime.nifty_above_50dma) else ("Above 20 DMA" if regime.nifty_above_20dma else "Below Moving Averages")
        st.metric("Benchmark Trend", dma_status)
    with rg_c4:
        bn_str = f"{regime.bank_nifty_change_1d:+.2f}%" if regime.bank_nifty_change_1d is not None else "N/A"
        st.metric("BANK NIFTY (1d)", bn_str)

    st.markdown("---")

    # Watchlist Summary Table
    st.subheader("📋 Ranked Watchlist Summary")
    if not scan_res.candidates:
        st.info("No stocks in the selected universe met minimum momentum and volume thresholds.")
        return

    table_data = []
    for c in scan_res.candidates:
        table_data.append({
            "Rank": f"#{c.rank}",
            "Symbol": c.symbol,
            "Company": c.company_name,
            "Sector": c.sector,
            "Current Price": f"₹{c.signals.current_price:,.2f}",
            "Day Change": f"{c.signals.return_1d:+.2f}%" if c.signals.return_1d is not None else "N/A",
            "5d Return": f"{c.signals.return_5d:+.2f}%" if c.signals.return_5d is not None else "N/A",
            "Vol Ratio": f"{c.signals.volume_ratio:.2f}x" if c.signals.volume_ratio else "N/A",
            "RSI (14)": f"{c.signals.rsi_14:.1f}" if c.signals.rsi_14 else "N/A",
            "Score": f"{c.opportunity_score:.1f}/100",
            "Setup": c.confidence_label,
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Candidate Detail Cards
    st.subheader("🔍 Shortlisted Candidate Deep-Dives")

    for c in scan_res.candidates:
        badge_color = "🟢" if "High" in c.confidence_label else ("🟡" if "Moderate" in c.confidence_label else "⚪")
        card_header = f"#{c.rank}: {c.symbol} — {c.setup_title} ({badge_color} Score: {c.opportunity_score:.1f}/100 · {c.confidence_label})"

        with st.expander(card_header, expanded=(c.rank <= 3)):
            st.markdown(f"**{c.company_name}** | Sector: **{c.sector}** | NSE: `{c.symbol}`")

            # Quick Metrics Bar
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Current Price", f"₹{c.signals.current_price:,.2f}", f"{c.signals.return_1d:+.2f}%")
            with m2:
                st.metric("5-Day Return", f"{c.signals.return_5d:+.2f}%")
            with m3:
                st.metric("Volume vs 20d Avg", f"{c.signals.volume_ratio:.2f}x")
            with m4:
                st.metric("RSI (14)", f"{c.signals.rsi_14:.1f}")
            with m5:
                st.metric("Distance from 20 DMA", f"{c.signals.dist_from_20dma_pct:+.2f}%")

            # 1. Why on Watchlist
            st.markdown("#### 🎯 Why on the Watchlist?")
            for w in c.why_on_watchlist:
                st.markdown(f"- {w}")

            # 2. Technical Signals & Moving Averages
            st.markdown("#### 📊 Technical Signals & Moving Averages")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.write(f"- **20 DMA:** ₹{c.signals.dma_20:,.2f} ({'🟢 Above' if c.signals.is_above_20dma else '🔴 Below'})")
                st.write(f"- **50 DMA:** ₹{c.signals.dma_50:,.2f} ({'🟢 Above' if c.signals.is_above_50dma else '🔴 Below'})")
                st.write(f"- **Breakout Status:** {'⚡ Breakout above 20d High' if c.signals.is_breakout else 'Consolidation / Pullback'}")
            with t_col2:
                st.write(f"- **1-Day Return:** {c.signals.return_1d:+.2f}% | **2-Day Return:** {c.signals.return_2d:+.2f}%")
                st.write(f"- **5-Day Return:** {c.signals.return_5d:+.2f}% | **20-Day Return:** {c.signals.return_20d:+.2f}%")
                st.write(f"- **Daily ATR (14):** ₹{c.signals.atr_14:,.2f} ({(c.signals.atr_14 / c.signals.current_price * 100):.2f}% volatility)")

            # 3. Positive Catalysts & Fundamentals
            if c.positive_catalysts:
                st.markdown("#### 🚀 Positive Catalysts & Corporate Events")
                for cat in c.positive_catalysts:
                    cat_date = f" ({cat.date})" if cat.date else ""
                    st.markdown(f"- **{cat.category.replace('_', ' ').title()}:** {cat.headline}{cat_date} *(Source: {cat.source})*")

            # 4. Key Technical Levels & Risk/Reward
            st.markdown("#### 🛡️ Key Technical Levels & Risk/Reward")
            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                st.metric("Support Zone", f"₹{c.signals.support_level:,.2f}", f"{c.signals.downside_risk_pct:.2f}% risk")
            with l_col2:
                st.metric("Resistance / Target Zone", f"₹{c.signals.resistance_level:,.2f}", f"+{c.signals.upside_potential_pct:.2f}% upside")
            with l_col3:
                st.metric("Technical Risk/Reward", f"{c.signals.risk_reward_ratio:.2f} : 1")

            # 5. Risks & Invalidation Factors
            st.markdown("#### ⚠️ Risks & Invalidation Factors")
            for r in c.risk_factors:
                st.markdown(f"- {r.description}")

            # 6. Opening Gap Risk Warning
            st.warning(f"⚠️ **Opening Gap Warning**: {c.opening_gap_risk}")

            # 7. Recent News & Executive Summaries
            if c.recent_news:
                st.markdown("#### 📰 Recent News & Executive Summaries")
                for n in c.recent_news:
                    title = n.get("title", "Market Update")
                    src = n.get("source", "Financial News")
                    dt = n.get("published_date") or n.get("date") or ""
                    url = n.get("url") or n.get("link")
                    summary = n.get("summary") or n.get("description") or ""

                    header_line = f"**{title}**"
                    if dt:
                        header_line += f" *({dt})*"
                    st.markdown(header_line)
                    if summary:
                        st.caption(summary)
                    if url:
                        st.markdown(f"[🔗 Read full article on {src}]({url})")
                    st.write("")

    # Compliance Disclaimer Footer
    st.markdown("---")
    st.info(
        "🔒 **Strict Compliance & Risk Disclosure**: This short-term trading opportunities scanner identifies "
        "quantitative setups based strictly on verified price action, volume confirmation, moving averages, "
        "and public corporate disclosures. It is for research and educational purposes only. "
        "It does NOT guarantee profit or performance. Never trade without appropriate position sizing and risk management."
    )


def render_top_sectors_dashboard() -> None:
    """Render dedicated dashboard for Top Performing NSE Sectors and Top 10 Ranked Companies."""
    render_top_header(
        title="Top Performing Sectors",
        subtitle="Multi-factor sector ranking and industry leader analysis for NSE equities",
    )

    header_col1, header_col2 = st.columns([0.82, 0.18])
    with header_col1:
        st.title("🔥 Top Sectors & Companies")
        st.markdown(
            "##### Identify the current Top 5 performing sectors in the Indian stock market (NSE) "
            "and explore multi-factor ranked top 10 companies in each sector."
        )
    with header_col2:
        st.write("")
        st.write("")
        if st.button("💬 Back to Chat", use_container_width=True):
            st.session_state["view_mode"] = "chat"
            st.rerun()

    current_tf = st.session_state.get("sector_timeframe", "1 Day")
    current_sort = st.session_state.get("sector_sort_by", "Momentum")
    cache_token = st.session_state.get("sector_cache_token", 0)
    current_sector = st.session_state.get("selected_sector")

    with st.spinner("Analyzing NSE sector indices, constituent price action, volume, and catalysts..."):
        try:
            sec_res = _cached_get_top_sectors_and_companies(
                sector_name=current_sector,
                timeframe=current_tf,
                sort_by=current_sort,
                cache_token=cache_token,
            )
        except Exception as exc:
            st.error(f"⚠️ Failed to retrieve top sectors data: {exc}")
            return

    st.caption(f"🕒 **Last Updated:** {sec_res.generated_at} (NSE Market Data)")

    # 1. Current Top 5 Sectors
    st.subheader("Current Top 5 Sectors")

    top_sectors = sec_res.top_sectors
    top_sector_names = [s.name for s in top_sectors]

    # If current sector is not selected or not in top sectors, default to top 1
    if not current_sector or current_sector not in top_sector_names:
        current_sector = top_sector_names[0] if top_sector_names else "Banking"
        st.session_state["selected_sector"] = current_sector

    # Display Top 5 Sectors metrics row
    sec_cols = st.columns(len(top_sectors))
    for idx, (col, s) in enumerate(zip(sec_cols, top_sectors), 1):
        with col:
            t_icon = "🟢" if s.trend == "Bullish" else ("🔴" if s.trend == "Bearish" else "⚪")
            st.metric(
                label=f"#{idx} {s.display_name}",
                value=f"{s.performance_score:+.2f}%",
                delta=f"1D: {s.change_1d:+.2f}% | 1W: {s.change_1w:+.2f}%",
            )
            st.caption(f"Trend: **{t_icon} {s.trend}**")

    st.markdown("---")

    # 2. Sector Selection Dropdown and Filters
    f_col1, f_col2, f_col3, f_col4 = st.columns([0.35, 0.25, 0.25, 0.15])
    with f_col1:
        selected_idx = top_sector_names.index(current_sector) if current_sector in top_sector_names else 0
        new_sector = st.selectbox(
            "Select Sector",
            options=top_sector_names,
            index=selected_idx,
            key="sec_sector_dropdown",
            help="Select one of the current Top 5 performing NSE sectors",
        )
        if new_sector != current_sector:
            st.session_state["selected_sector"] = new_sector
            st.rerun()

    with f_col2:
        new_tf = st.selectbox(
            "Timeframe Filter",
            options=["1 Day", "1 Week", "1 Month"],
            index=["1 Day", "1 Week", "1 Month"].index(current_tf) if current_tf in ["1 Day", "1 Week", "1 Month"] else 0,
            key="sec_timeframe_select",
        )
        if new_tf != current_tf:
            st.session_state["sector_timeframe"] = new_tf
            st.rerun()

    with f_col3:
        new_sort = st.selectbox(
            "Sort By",
            options=["Momentum", "Performance", "Volume", "Market Cap", "RSI"],
            index=["Momentum", "Performance", "Volume", "Market Cap", "RSI"].index(current_sort) if current_sort in ["Momentum", "Performance", "Volume", "Market Cap", "RSI"] else 0,
            key="sec_sort_select",
        )
        if new_sort != current_sort:
            st.session_state["sector_sort_by"] = new_sort
            st.rerun()

    with f_col4:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state["sector_cache_token"] = cache_token + 1
            st.rerun()

    # 3. Sector Summary Card
    active_sector_perf = next((s for s in sec_res.all_sectors if s.name == current_sector), None)
    if active_sector_perf:
        with st.expander(f"📋 Sector Summary — {active_sector_perf.display_name}", expanded=True):
            st.markdown(f"**Why moving today:** {active_sector_perf.why_moving}")
            sum_c1, sum_c2 = st.columns(2)
            with sum_c1:
                st.markdown(f"**Institutional & FII/DII Activity:**\n{active_sector_perf.institutional_activity}")
                st.markdown(f"**Government & Policy Impact:**\n{active_sector_perf.policy_impact}")
            with sum_c2:
                st.markdown(f"**Major Earnings & Result Trends:**\n{active_sector_perf.major_earnings}")
                if active_sector_perf.important_news:
                    st.markdown("**Important News & Catalysts:**")
                    for news_item in active_sector_perf.important_news:
                        st.markdown(f"- {news_item}")

    st.markdown("---")

    # 4. Top 10 Stocks Table
    st.subheader(f"🔥 Top 10 Stocks — {current_sector}")
    top_stocks = sec_res.top_companies

    if not top_stocks:
        st.info(f"No company records found for {current_sector}.")
        return

    table_rows = []
    for c in top_stocks:
        trend_badge = "🟢 Bullish" if c.trend == "Bullish" else ("🔴 Bearish" if c.trend == "Bearish" else "⚪ Neutral")
        mcap_str = f"₹{c.market_cap / 10000000:,.0f} Cr" if c.market_cap else "N/A"
        pe_str = f"{c.pe_ratio:.1f}" if c.pe_ratio else "N/A"
        vol_str = f"{c.volume:,.0f}" if c.volume else "N/A"
        rel_vol_str = f"{c.relative_volume:.2f}x"
        rsi_str = f"{c.rsi_14:.1f}" if c.rsi_14 else "N/A"
        dist_str = f"{c.distance_from_52w_high_pct:+.1f}%"
        h52_str = f"₹{c.high_52w:,.1f}" if c.high_52w else "N/A"
        l52_str = f"₹{c.low_52w:,.1f}" if c.low_52w else "N/A"
        why_str = " • ".join(c.why_in_top_10)

        table_rows.append({
            "Rank": f"#{c.rank}",
            "Company": c.company_name,
            "Symbol": c.symbol,
            "Current Price": f"₹{c.current_price:,.2f}",
            "Day %": f"{c.day_change_percent:+.2f}%",
            "1-Week %": f"{c.week_change_percent:+.2f}%",
            "Market Cap": mcap_str,
            "P/E": pe_str,
            "52W High": h52_str,
            "52W Low": l52_str,
            "Volume": vol_str,
            "Rel Vol": rel_vol_str,
            "RSI": rsi_str,
            "Trend": trend_badge,
            "Dist 52W High": dist_str,
            "Why in Top 10": why_str,
        })

    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # 5. Expandable Company Deep-Dives
    st.markdown("#### 🔍 Top 10 Company Profiles & Catalysts")
    for c in top_stocks:
        trend_icon = "🟢" if c.trend == "Bullish" else ("🔴" if c.trend == "Bearish" else "⚪")
        header = f"#{c.rank} {c.company_name} ({c.symbol}) — ₹{c.current_price:,.2f} ({c.day_change_percent:+.2f}%) · {trend_icon} {c.trend}"

        with st.expander(header, expanded=(c.rank <= 3)):
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("Current Price", f"₹{c.current_price:,.2f}", f"{c.day_change_percent:+.2f}%")
            with m2:
                st.metric("1-Week Return", f"{c.week_change_percent:+.2f}%")
            with m3:
                st.metric("Relative Volume", f"{c.relative_volume:.2f}x")
            with m4:
                st.metric("RSI (14)", f"{c.rsi_14:.1f}" if c.rsi_14 else "N/A")
            with m5:
                st.metric("52W Range", f"₹{c.low_52w or 0:,.0f} – ₹{c.high_52w or 0:,.0f}")
            with m6:
                st.metric("From 52W High", f"{c.distance_from_52w_high_pct:+.1f}%")

            cd_c1, cd_c2 = st.columns(2)
            with cd_c1:
                st.markdown("##### 🎯 Why in Top 10?")
                for reason in c.why_in_top_10:
                    st.markdown(f"- {reason}")
                st.markdown(f"**Latest Catalyst / News:**\n> {c.latest_catalyst}")

            with cd_c2:
                st.markdown("##### 📊 Financials & Fundamentals")
                st.markdown(f"- **P/E Ratio:** {f'{c.pe_ratio:.2f}' if c.pe_ratio else 'N/A'}")
                st.markdown(f"- **Market Cap:** {f'₹{c.market_cap / 10000000:,.0f} Cr' if c.market_cap else 'N/A'}")
                st.markdown(f"**Quarterly Results Summary:**\n> {c.quarterly_result_summary}")

    # Compliance Disclaimer Footer
    st.markdown("---")
    st.info(
        "🔒 **Educational & Research Notice**: Sector rankings and constituent company scores are computed "
        "deterministically using a multi-factor combination of price momentum, relative volume, technical indicators, "
        "earnings growth, and publicly available news. They do not constitute financial advice or investment recommendations."
    )


def render_decision_assistant_dashboard() -> None:
    """Render dedicated dashboard for Stock Decision Assistant (Buy / Hold / Sell / Reduce / Exit / Wait analysis)."""
    render_top_header(
        title="Stock Decision Assistant",
        subtitle="Deterministic multi-factor Buy | Hold | Reduce | Exit | Wait analysis for NSE equities",
    )

    header_col1, header_col2 = st.columns([0.82, 0.18])
    with header_col1:
        st.title("🧠 Stock Decision Assistant")
        st.markdown(
            "##### Comprehensive Buy | Hold | Reduce | Exit | Wait analysis for Indian equities "
            "based on investment horizon, multi-factor fundamentals, valuation, technicals, risks, and market conditions."
        )
    with header_col2:
        st.write("")
        st.write("")
        if st.button("💬 Back to Chat", use_container_width=True):
            st.session_state["view_mode"] = "chat"
            st.rerun()
            return

    # 1. Mode Selector
    mode_options = ["🛒 Thinking of Buying (New Investment)", "💼 Already Own This Stock (Existing Investment)"]
    current_intent_raw = st.session_state.get("decision_intent", "Thinking of Buying")
    default_mode_idx = 1 if "Already Own" in current_intent_raw or current_intent_raw == "existing" else 0
    selected_mode = st.radio(
        "Select Investment Context",
        options=mode_options,
        index=default_mode_idx,
        horizontal=True,
        key="decision_mode_radio",
    )
    is_existing = "Already Own" in selected_mode
    intent_arg = "existing" if is_existing else "new"
    st.session_state["decision_intent"] = selected_mode

    # 2. Input Controls
    horizon_options = ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "3+ Years"]
    current_horizon = st.session_state.get("decision_horizon", "1 Year")
    h_idx = horizon_options.index(current_horizon) if current_horizon in horizon_options else 3

    if not is_existing:
        in_col1, in_col2, in_col3, in_col4 = st.columns([0.45, 0.25, 0.15, 0.15])
        with in_col1:
            stock_input = stock_autocomplete(
                "Stock Name or Symbol",
                value=st.session_state.get("stock_search_text", st.session_state.get("decision_stock", "")),
                key="decision_stock_input",
                placeholder="e.g. Tata Motors, RELIANCE, TCS, HDFC Bank",
                help="Type company name or NSE ticker symbol to view live suggestions",
            )
        with in_col2:
            horizon_input = st.selectbox(
                "Intended Horizon",
                options=horizon_options,
                index=h_idx,
                key="decision_horizon_select",
                help="Select your planned investment duration",
            )
        with in_col3:
            st.write("")
            st.write("")
            analyze_clicked = st.button("🚀 Analyze", use_container_width=True, type="primary")
        with in_col4:
            st.write("")
            st.write("")
            refresh_btn = st.button("🔄 Refresh", use_container_width=True)
            if refresh_btn:
                st.session_state["decision_cache_token"] = st.session_state.get("decision_cache_token", 0) + 1
        purchase_price_val = None
        quantity_val = None
    else:
        in_col1, in_col2, in_col3, in_col4, in_col5 = st.columns([0.30, 0.20, 0.20, 0.15, 0.15])
        with in_col1:
            stock_input = stock_autocomplete(
                "Stock Name or Symbol",
                value=st.session_state.get("stock_search_text", st.session_state.get("decision_stock", "")),
                key="decision_stock_input_ex",
                placeholder="e.g. Tata Motors, HDFC Bank, PNC Infratech",
                help="Type company name or NSE ticker symbol to view live suggestions",
            )
        with in_col2:
            horizon_input = st.selectbox(
                "Holding Horizon",
                options=horizon_options,
                index=h_idx,
                key="decision_horizon_select_ex",
            )
        with in_col3:
            purchase_price_raw = st.number_input(
                "Purchase Price (₹)",
                min_value=0.0,
                value=float(st.session_state.get("decision_purchase_price", 0.0)),
                step=1.0,
                key="decision_price_input",
                help="Optional: your average purchase price",
            )
        with in_col4:
            quantity_raw = st.number_input(
                "Quantity Owned",
                min_value=0,
                value=int(st.session_state.get("decision_quantity", 0)),
                step=1,
                key="decision_qty_input",
                help="Optional: number of shares owned",
            )
        with in_col5:
            st.write("")
            st.write("")
            analyze_clicked = st.button("🚀 Analyze", use_container_width=True, type="primary")
            refresh_btn = st.button("🔄 Refresh", use_container_width=True)
            if refresh_btn:
                st.session_state["decision_cache_token"] = st.session_state.get("decision_cache_token", 0) + 1

        purchase_price_val = float(purchase_price_raw) if purchase_price_raw > 0 else None
        quantity_val = int(quantity_raw) if quantity_raw > 0 else None

    # Update session state with current inputs
    st.session_state["decision_horizon"] = horizon_input
    if is_existing:
        st.session_state["decision_purchase_price"] = purchase_price_val or 0.0
        st.session_state["decision_quantity"] = quantity_val or 0

    # If user clicked refresh on an existing analysis, re-run analysis
    if refresh_btn and (st.session_state.get("selected_symbol") or st.session_state.get("analysis_result")):
        analyze_clicked = True

    if analyze_clicked:
        raw_query = (
            st.session_state.get("selected_symbol")
            or st.session_state.get("decision_selected_symbol")
            or stock_input
            or st.session_state.get("decision_stock")
            or ""
        ).strip()
        if not raw_query:
            st.warning("Please enter a stock name or NSE symbol above.")
            return

        selected_sym = st.session_state.get("selected_symbol")

        # Check for ambiguity if user typed a broad search without picking a suggestion
        suggestions = st.session_state.get("stock_suggestions") or []
        if not suggestions:
            suggestions = search_stocks(raw_query, limit=15)

        is_exact_symbol = raw_query.upper() in INDIAN_STOCK_MASTER
        is_exact_company = any(raw_query.lower() == s.get("company_name", "").lower() for s in suggestions)

        if not selected_sym and not is_exact_symbol and not is_exact_company and len(suggestions) > 1:
            st.warning(f'Multiple stocks match "{raw_query}". Please select the stock you want to analyze.')
            st.session_state["analysis_requested"] = False
            return

        if selected_sym and selected_sym in INDIAN_STOCK_MASTER:
            target_symbol = selected_sym
            company_display_name = (
                st.session_state.get("selected_stock_name")
                or st.session_state.get("decision_selected_stock")
                or INDIAN_STOCK_MASTER[selected_sym]["company_name"]
            )
            st.session_state["selected_symbol"] = target_symbol
            st.session_state["decision_selected_symbol"] = target_symbol
            st.session_state["selected_stock_name"] = company_display_name
            st.session_state["decision_selected_stock"] = company_display_name
        elif is_exact_symbol:
            target_symbol = raw_query.upper()
            company_display_name = INDIAN_STOCK_MASTER[target_symbol]["company_name"]
            st.session_state["selected_symbol"] = target_symbol
            st.session_state["decision_selected_symbol"] = target_symbol
            st.session_state["selected_stock_name"] = company_display_name
            st.session_state["decision_selected_stock"] = company_display_name
        else:
            resolved = resolve_nse_symbol(raw_query, allow_online_lookup=False)
            if resolved.get("is_ambiguous"):
                st.warning(f'Multiple stocks match "{raw_query}". Please select the stock you want to analyze.')
                st.session_state["analysis_requested"] = False
                return
            if not resolved.get("symbol"):
                st.error(f"Could not resolve '{raw_query}' to an NSE ticker symbol. Please check the company name or symbol.")
                st.session_state["analysis_requested"] = False
                return
            target_symbol = resolved["symbol"]
            company_display_name = resolved.get("company_name", target_symbol)
            st.session_state["selected_symbol"] = target_symbol
            st.session_state["decision_selected_symbol"] = target_symbol
            st.session_state["selected_stock_name"] = company_display_name
            st.session_state["decision_selected_stock"] = company_display_name

        st.session_state["analysis_requested"] = True
        st.session_state["decision_stock"] = target_symbol
        cache_token = st.session_state.get("decision_cache_token", 0)

        with st.spinner(f"Analyzing {company_display_name} across fundamentals, valuation, sector, technicals, news & risks..."):
            try:
                decision_res = _cached_analyze_stock_decision(
                    stock=target_symbol,
                    intent=intent_arg,
                    horizon=horizon_input,
                    purchase_price=purchase_price_val,
                    quantity=quantity_val,
                    cache_token=cache_token,
                )
                st.session_state["analysis_result"] = decision_res
            except Exception as exc:
                st.error(f"⚠️ Decision analysis encountered an issue: {exc}")
                st.session_state["analysis_result"] = None
                return

    decision_res = st.session_state.get("analysis_result")
    if not decision_res:
        st.info("👆 Enter a stock name or NSE symbol above and click **🚀 Analyze** to generate the complete investment decision report.")
        return

    # Check for resolution failure
    if not decision_res.resolved_symbol:
        st.error(f"⚠️ Could not resolve NSE symbol. Please check the company name or symbol.")
        return

    # Top Status Bar
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Company / NSE", f"{decision_res.resolved_symbol}", decision_res.company_name)
    with m_col2:
        price_val = f"₹{decision_res.current_price:,.2f}" if decision_res.current_price else "N/A"
        day_chg = f"{decision_res.day_change_percent:+.2f}%" if decision_res.day_change_percent is not None else ""
        st.metric("Current Price", price_val, day_chg)
    with m_col3:
        conf_icon = "🟢" if decision_res.analysis_confidence == "High" else ("🟡" if decision_res.analysis_confidence == "Medium" else "⚪")
        st.metric("Confidence", f"{conf_icon} {decision_res.analysis_confidence}")
    with m_col4:
        st.metric("Last Updated", decision_res.last_updated.split(" ")[-1] if " " in decision_res.last_updated else decision_res.last_updated)

    # Existing Position Card if applicable
    if is_existing and purchase_price_val and purchase_price_val > 0 and decision_res.current_price:
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        pnl_pct = ((decision_res.current_price - purchase_price_val) / purchase_price_val) * 100
        pnl_color = "🟢" if pnl_pct >= 0 else "🔴"
        with p_c1:
            st.metric("Purchase Price", f"₹{purchase_price_val:,.2f}")
        with p_c2:
            st.metric("Unrealized Return", f"{pnl_color} {pnl_pct:+.2f}%")
        with p_c3:
            if quantity_val and quantity_val > 0:
                cur_val = decision_res.current_price * quantity_val
                st.metric("Current Value", f"₹{cur_val:,.2f}")
            else:
                st.metric("Sector", decision_res.sector)
        with p_c4:
            if quantity_val and quantity_val > 0:
                tot_pnl = (decision_res.current_price - purchase_price_val) * quantity_val
                st.metric("Total P&L", f"₹{tot_pnl:+,.2f}")
            else:
                st.metric("Horizon", decision_res.investment_horizon)

        st.caption(
            "💡 *Disciplined Investing Note: Your past entry price does NOT determine future market direction. "
            "The recommendation is strictly forward-looking based on risk-reward for your remaining horizon.*"
        )

    # 3. Big Decision Banner
    st.markdown("---")
    indicator = decision_res.decision_indicator
    if indicator in ["BUY", "CONSIDER ADDING"]:
        banner_type = st.success
        icon = "🟢"
    elif indicator in ["HOLD", "WAIT"]:
        banner_type = st.info
        icon = "🟡"
    else:  # REDUCE, EXIT, AVOID FOR NOW
        banner_type = st.error
        icon = "🔴"

    banner_type(
        f"### {icon} Decision Recommendation: **{indicator}**  \n"
        f"**Target Horizon:** {decision_res.investment_horizon} | **Profile:** {decision_res.intent_type.title()} Investment  \n"
        f"**Business Quality vs Stock Quality:** {decision_res.business_vs_stock_quality}"
    )

    # Key Decision Bullet Points
    st.markdown("#### 🎯 Core Decision Thesis")
    for reason in decision_res.key_reasons:
        st.markdown(f"- {reason}")

    st.markdown("---")

    # 4. Factor Snapshot Table
    st.subheader("📋 Decision Factor Snapshot")
    snap = decision_res.snapshot
    snap_data = [
        {"Factor": "Market Condition", "Status": snap.market_condition, "Observation": snap.market_summary},
        {"Factor": "Sector Outlook", "Status": snap.sector_outlook, "Observation": f"{decision_res.sector}: {snap.sector_summary}"},
        {"Factor": "Fundamentals", "Status": snap.fundamentals, "Observation": snap.fundamentals_summary},
        {"Factor": "Quarterly Results", "Status": snap.quarterly_results, "Observation": snap.quarterly_summary},
        {"Factor": "Valuation", "Status": snap.valuation, "Observation": snap.valuation_summary},
        {"Factor": "Peer Position", "Status": snap.peer_position, "Observation": snap.peer_summary},
        {"Factor": "Technical Setup", "Status": f"{snap.technicals} ({snap.technical_state})", "Observation": snap.technical_summary},
        {"Factor": "News & Catalysts", "Status": f"🟢 {snap.positive_catalyst_count} / 🔴 {snap.negative_catalyst_count}", "Observation": snap.catalyst_summary},
        {"Factor": "Risk Level", "Status": snap.risk_level, "Observation": snap.risk_summary},
    ]
    st.dataframe(snap_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 5. Tabbed Detailed Analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏛️ Market & Sector",
        "📊 Fundamentals & Earnings",
        "💰 Valuation & Peers",
        "📈 Technical Setup",
        "📰 News & Catalysts",
        "⚠️ Risks & Thesis Invalidation",
    ])

    with tab1:
        st.markdown("#### 🏛️ Market & Sector Context")
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.markdown(f"**NIFTY 50 Benchmark Trend:** `{decision_res.market_trend}`")
            st.markdown(f"**NIFTY Level:** ₹{decision_res.nifty_value:,.2f} ({decision_res.nifty_change_1d:+.2f}%)")
            st.markdown(f"**India VIX:** {decision_res.india_vix:.2f} ({'Elevated Volatility' if decision_res.india_vix > 18 else 'Normal Range'})")
            st.markdown(f"**FII / DII Institutional Flow:** {decision_res.fii_dii_summary}")
        with m_c2:
            st.markdown(f"**Sector:** {decision_res.sector}")
            st.markdown(f"**Sector Outlook:** `{decision_res.sector_outlook}`")
            st.markdown(f"**Sector Performance:** 1D {decision_res.sector_change_1d:+.2f}% | 1W {decision_res.sector_change_1w:+.2f}%")
            st.markdown(f"**Sector Drivers:** {decision_res.sector_why}")

    with tab2:
        st.markdown("#### 📊 Fundamentals & Quarterly Performance")
        f_c1, f_c2, f_c3, f_c4 = st.columns(4)
        with f_c1:
            st.metric("Operating Margin", f"{decision_res.operating_margin:.1f}%" if decision_res.operating_margin else "N/A")
        with f_c2:
            st.metric("Return on Equity (ROE)", f"{decision_res.roe:.1f}%" if decision_res.roe else "N/A")
        with f_c3:
            st.metric("ROCE", f"{decision_res.roce:.1f}%" if decision_res.roce else "N/A")
        with f_c4:
            st.metric("Debt-to-Equity", f"{decision_res.debt_to_equity:.2f}" if decision_res.debt_to_equity else "N/A")

        st.markdown(f"**Fundamental Trend:** `{decision_res.fundamental_trend}`")
        st.markdown(f"**Quarterly Results Quality:** `{decision_res.quarterly_results_quality}`")
        st.markdown(f"> **Quarterly Earnings Summary:** {decision_res.quarterly_results_summary}")
        st.markdown(f"> **4-Quarter Trend & Growth:** {decision_res.fundamental_trend_details}")

    with tab3:
        st.markdown("#### 💰 Valuation & Sector Peer Benchmarking")
        v_c1, v_c2, v_c3, v_c4 = st.columns(4)
        with v_c1:
            pe_str = f"{decision_res.pe_ratio:.2f}" if decision_res.pe_ratio else "N/A"
            st.metric("Stock P/E", pe_str)
        with v_c2:
            sec_pe_str = f"{decision_res.sector_pe:.2f}" if decision_res.sector_pe is not None else "N/A"
            st.metric("Sector Average P/E", sec_pe_str)
        with v_c3:
            pb_str = f"{decision_res.pb_ratio:.2f}" if decision_res.pb_ratio else "N/A"
            st.metric("Price-to-Book (P/B)", pb_str)
        with v_c4:
            div_str = f"{decision_res.dividend_yield:.2f}%" if decision_res.dividend_yield else "N/A"
            st.metric("Dividend Yield", div_str)

        st.markdown(f"**Valuation Classification:** `{decision_res.valuation_assessment}`")
        st.markdown(f"> {decision_res.valuation_details}")

        st.markdown("##### 🏢 Peer Benchmarking Table")
        if decision_res.peers:
            peer_table = []
            for p in decision_res.peers:
                peer_table.append({
                    "Symbol": p.symbol,
                    "Company": p.company_name,
                    "Price": f"₹{p.price:,.2f}",
                    "Market Cap": f"₹{p.market_cap / 10000000:,.0f} Cr" if p.market_cap else "N/A",
                    "P/E": f"{p.pe_ratio:.1f}" if p.pe_ratio else "N/A",
                    "ROE": f"{p.roe:.1f}%" if p.roe else "N/A",
                    "ROCE": f"{p.roce:.1f}%" if p.roce else "N/A",
                    "Op Margin": f"{p.operating_margin:.1f}%" if p.operating_margin else "N/A",
                    "D/E": f"{p.debt_to_equity:.2f}" if p.debt_to_equity is not None else "N/A",
                    "1Y Return": f"{p.return_1y:+.1f}%" if p.return_1y is not None else "N/A",
                })
            st.dataframe(peer_table, use_container_width=True, hide_index=True)
            st.markdown(f"**Peer Position:** `{decision_res.peer_comparison_label}` — {decision_res.peer_details}")

    with tab4:
        st.markdown("#### 📈 Technical Setup & Key Levels")
        t_c1, t_c2, t_c3, t_c4 = st.columns(4)
        with t_c1:
            st.metric("RSI (14)", f"{decision_res.rsi_14:.1f}" if decision_res.rsi_14 else "N/A")
        with t_c2:
            st.metric("Technical Trend", decision_res.technical_trend)
        with t_c3:
            st.metric("Support Level", f"₹{decision_res.support_level:,.2f}" if decision_res.support_level else "N/A")
        with t_c4:
            st.metric("Resistance Level", f"₹{decision_res.resistance_level:,.2f}" if decision_res.resistance_level else "N/A")

        dma_c1, dma_c2, dma_c3, dma_c4 = st.columns(4)
        with dma_c1:
            st.write(f"- **20 DMA:** ₹{decision_res.dma_20:,.2f}" if decision_res.dma_20 else "- **20 DMA:** N/A")
        with dma_c2:
            st.write(f"- **50 DMA:** ₹{decision_res.dma_50:,.2f}" if decision_res.dma_50 else "- **50 DMA:** N/A")
        with dma_c3:
            st.write(f"- **200 DMA:** ₹{decision_res.dma_200:,.2f}" if decision_res.dma_200 else "- **200 DMA:** N/A")
        with dma_c4:
            st.write(f"- **From 52W High:** {decision_res.distance_from_52w_high_pct:+.1f}%")

        st.markdown(f"**Chart State:** `{decision_res.technical_state}`")
        st.markdown(f"> {decision_res.technical_details}")

    with tab5:
        st.markdown("#### 📰 News & Verified Catalysts")
        cat_c1, cat_c2 = st.columns(2)
        with cat_c1:
            st.markdown("##### 🟢 Positive Catalysts")
            if decision_res.positive_catalysts:
                for cat in decision_res.positive_catalysts:
                    st.markdown(f"**{cat.headline}**")
                    if cat.source or cat.date:
                        st.caption(f"Source: {cat.source} | Date: {cat.date}")
                    st.write(f"{cat.explanation}")
                    if cat.potential_impact:
                        st.write(f"*Potential Impact:* {cat.potential_impact}")
                    if cat.url:
                        st.markdown(f"[🔗 Read full article]({cat.url})")
                    st.write("---")
            else:
                st.info("No prominent positive catalysts identified in recent disclosures.")

        with cat_c2:
            st.markdown("##### 🔴 Negative Catalysts & Concerns")
            if decision_res.negative_catalysts:
                for cat in decision_res.negative_catalysts:
                    st.markdown(f"**{cat.headline}**")
                    if cat.source or cat.date:
                        st.caption(f"Source: {cat.source} | Date: {cat.date}")
                    st.write(f"{cat.explanation}")
                    if cat.potential_impact:
                        st.write(f"*Potential Impact:* {cat.potential_impact}")
                    if cat.url:
                        st.markdown(f"[🔗 Read full article]({cat.url})")
                    st.write("---")
            else:
                st.info("No prominent negative catalysts or headwind articles identified.")

    with tab6:
        st.markdown("#### ⚠️ Key Company Risks & Thesis Invalidation")
        st.markdown(f"**Risk Level:** `{decision_res.risk_level}`")
        st.markdown("##### Specific Risk Factors:")
        for r in decision_res.risk_factors:
            st.markdown(f"- ⚠️ {r}")

        st.markdown("---")
        st.markdown("#### 🔄 What Would Change This View?")
        render_positive_negative_cards(
            positive_items=decision_res.what_makes_view_more_positive,
            negative_items=decision_res.what_makes_view_more_negative,
        )

    # 6. Expandable Markdown Report
    st.markdown("---")
    with st.expander("📄 View & Copy Full Markdown Report", expanded=False):
        st.code(decision_res.report_markdown, language="markdown")

    # 7. Regulatory Disclaimer
    st.info(
        "🔒 **Educational & Research Notice**: This analysis is generated deterministically using quantitative multi-factor models "
        "combining market data, financial statements, valuation metrics, technical indicators, and verified news. "
        "It does not constitute SEBI-registered investment advice or a recommendation to buy or sell securities. "
        "Always consult a qualified financial advisor and conduct your own due diligence before making investment decisions."
    )


def render_welcome_screen() -> None:
    """Render welcome card, quick feature buttons, and sample prompts for brand new conversations."""
    # Top Hero Header
    st.markdown(
        """
        <div style="text-align: center; padding: 18px 0 22px 0;">
            <div style="display: inline-flex; align-items: center; gap: 8px; padding: 4px 14px; border-radius: 999px; background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.22); color: #A5B4FC; font-size: 12px; font-weight: 600; margin-bottom: 12px;">
                <span>🇮🇳</span> National Stock Exchange of India &bull; Agentic Intelligence
            </div>
            <div style="font-size: 26px; font-weight: 700; color: #F3F6FA; letter-spacing: -0.025em; line-height: 1.25;">
                What would you like to analyze today?
            </div>
            <div style="font-size: 13.5px; color: #A3ADBD; max-width: 620px; margin: 8px auto 0 auto; line-height: 1.5;">
                Research NSE equities, evaluate sector trends, screen multi-factor 2-day trading setups, or conduct comprehensive Buy / Hold / Exit decision analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    col_feat1 = cols[0]
    col_feat2 = cols[1] if len(cols) > 1 else cols[0]
    col_feat3 = cols[2] if len(cols) > 2 else cols[0]
    with col_feat1:
        with st.container(border=True):
            st.markdown("#### 🧠 Stock Decision")
            st.caption(
                "Analyze whether to BUY | HOLD | REDUCE | EXIT | WAIT for any Indian listed stock "
                "based on horizon, valuation, fundamentals, technicals, news, and market conditions."
            )
            if st.button("🧠 Open Decision Assistant", key="welcome_decision_btn", use_container_width=True, type="primary"):
                st.session_state["view_mode"] = "decision"
                st.rerun()

    with col_feat2:
        with st.container(border=True):
            st.markdown("#### 🔥 Top Sectors")
            st.caption("Discover the top 5 performing NSE sectors and top 10 ranked companies with multi-factor scoring.")
            if st.button("🔥 Open Top Sectors", key="welcome_top_sectors_btn", use_container_width=True):
                st.session_state["view_mode"] = "sectors"
                st.rerun()

    with col_feat3:
        with st.container(border=True):
            st.markdown("#### 📈 2-Day Trading")
            st.caption("Scan liquid NSE stocks for high-probability setups over the next 1–2 trading sessions.")
            if st.button("📈 Open Opportunities", key="welcome_opp_btn", use_container_width=True):
                st.session_state["view_mode"] = "opportunities"
                st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Clean Prompts Grid via pure HTML/CSS Flexbox container
    st.markdown(
        """
        <div style="display: flex; gap: 14px; flex-wrap: wrap; width: 100%;">
            <div style="flex: 1; min-width: 280px; background: #151C26; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 14px 16px;">
                <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #A5B4FC; margin-bottom: 8px;">
                    🧠 Investment & Decision Questions
                </div>
                <div style="font-size: 13px; color: #A3ADBD; line-height: 1.7;">
                    &bull; <em>"Should I buy India Cements?"</em><br>
                    &bull; <em>"I own Tata Power. Should I hold or sell?"</em><br>
                    &bull; <em>"I bought HDFC Bank at ₹1,650. Should I keep it?"</em><br>
                    &bull; <em>"Can I hold Tata Motors for 1 year?"</em>
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: #151C26; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 14px 16px;">
                <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #38BDF8; margin-bottom: 8px;">
                    📊 Market Research & Comparisons
                </div>
                <div style="font-size: 13px; color: #A3ADBD; line-height: 1.7;">
                    &bull; <em>"Tell me about TCS"</em><br>
                    &bull; <em>"Compare TCS with Infosys"</em><br>
                    &bull; <em>"What is the technical sentiment for RELIANCE?"</em><br>
                    &bull; <em>"What is the latest news about Tata Motors?"</em>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_messages(conversation_id: Optional[str]) -> None:
    """Load and render persisted chat messages from the active conversation.

    Args:
        conversation_id: UUID of active conversation, if any.
    """
    if not conversation_id:
        render_welcome_screen()
        return

    try:
        raw_messages = get_messages(conversation_id)
        displayable = filter_chat_messages(raw_messages)
    except Exception as e:
        st.error(f"Error reading conversation messages: {format_error_message(e)}")
        displayable = []

    if not displayable:
        render_welcome_screen()
        return

    for msg in displayable:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

    if displayable and displayable[-1]["role"] == "user":
        with st.chat_message("assistant"):
            st.info("ℹ️ No response recorded for the previous message. Please re-send your query below.")


def process_user_input(prompt: str, current_id: Optional[str]) -> None:
    """Handle incoming user prompt, invoke LangGraph or Demo agent, and render responses.

    Args:
        prompt: User question or instruction.
        current_id: Existing conversation ID or None.
    """
    # Create conversation in database if not yet existing
    if not current_id:
        try:
            new_title = format_conversation_title(prompt, max_length=40)
            new_conv = create_conversation(title=new_title)
            current_id = new_conv["conversation_id"]
            st.session_state["current_conversation_id"] = current_id
        except Exception as e:
            st.error(f"Failed to create conversation: {format_error_message(e)}")
            return

    # Immediately render user query in chat UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show loading state and invoke agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing NSE market data..."):
            try:
                from services.market_assistant_service import run_market_assistant

                result = run_market_assistant(
                    query=prompt,
                    conversation_id=current_id,
                )

                assistant_response = result.get("response", "")
                st.markdown(assistant_response)

                # Show tool activities in a clean collapsible expander (not as chat messages)
                tool_calls = result.get("tool_calls", [])
                if tool_calls:
                    summaries = [format_tool_call_summary(tc) for tc in tool_calls]
                    with st.expander(
                        f"🛠️ Executed {len(summaries)} market data tool{'s' if len(summaries) > 1 else ''}",
                        expanded=False,
                    ):
                        for s in summaries:
                            st.write(f"- {s}")

            except Exception as e:
                friendly_error = format_error_message(e)
                error_msg = f"⚠️ {friendly_error}"
                # Persist the error to SQLite so the conversation turn is properly completed and visible
                try:
                    add_message(current_id, role="assistant", content=error_msg)
                except Exception:
                    pass
                st.error(friendly_error)

    # Rerun to sync full persisted state with UI
    st.rerun()


def render_app() -> None:
    """Main application layout and execution entry point."""
    st.set_page_config(
        page_title="NSE AI Stock Analyst",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Apply modern financial dark theme CSS
    is_dashboard = st.session_state.get("view_mode") in ("opportunities", "sectors", "decision")
    st.markdown(get_global_css(is_dashboard=is_dashboard), unsafe_allow_html=True)

    active_cid = render_sidebar()

    # Dedicated 2-Day Trading Opportunities Dashboard View
    if st.session_state.get("view_mode") == "opportunities":
        render_opportunities_dashboard()
        return

    # Dedicated Top Sectors & Companies Dashboard View
    if st.session_state.get("view_mode") == "sectors":
        render_top_sectors_dashboard()
        return

    # Dedicated Stock Decision Assistant Dashboard View
    if st.session_state.get("view_mode") == "decision":
        render_decision_assistant_dashboard()
        return

    # Render top header bar with live status pill
    render_top_header(
        title="🇮🇳 NSE AI Market Assistant",
        subtitle="Intelligent Real-time Indian Equity Research & Market Analysis",
    )

    # Active conversation header
    if active_cid:
        active_conv = get_conversation(active_cid)
        if active_conv:
            st.subheader(active_conv.get("title", "Conversation"))
    else:
        st.subheader("New Stock Analysis")

    # Render previous conversation turns
    render_chat_messages(active_cid)

    # Chat input box at page bottom
    user_prompt = st.chat_input("Ask about any NSE stock, market, index, news or comparison...")
    if user_prompt:
        process_user_input(user_prompt.strip(), active_cid)


if __name__ == "__main__":
    render_app()
