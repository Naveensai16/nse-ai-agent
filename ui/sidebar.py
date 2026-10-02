"""Sidebar navigation, chat history, and configuration component."""

from __future__ import annotations

import os
from typing import Optional
import streamlit as st

from services.database_service import delete_conversation, list_conversations
from utils.ui_helpers import format_conversation_title


def render_sidebar() -> Optional[str]:
    """Render the modern dark fintech sidebar.

    Returns:
        Optional[str]: Currently active conversation ID, or None.
    """
    # Brand Header
    st.sidebar.title("📈 NSE AI Analyst")
    st.sidebar.caption("Intelligent Indian Stock Market Assistant")

    # 1. Primary Navigation Actions
    if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state["current_conversation_id"] = None
        st.session_state["view_mode"] = "chat"
        st.rerun()

    is_opp_mode = st.session_state.get("view_mode") == "opportunities"
    opp_btn_type = "primary" if is_opp_mode else "secondary"
    if st.sidebar.button("📈 2-Day Trading Opportunities", use_container_width=True, type=opp_btn_type):
        st.session_state["view_mode"] = "opportunities"
        st.rerun()

    is_sectors_mode = st.session_state.get("view_mode") == "sectors"
    sec_btn_type = "primary" if is_sectors_mode else "secondary"
    if st.sidebar.button("🔥 Top Sectors", use_container_width=True, type=sec_btn_type):
        st.session_state["view_mode"] = "sectors"
        st.rerun()

    is_decision_mode = st.session_state.get("view_mode") == "decision"
    dec_btn_type = "primary" if is_decision_mode else "secondary"
    if st.sidebar.button("🧠 Stock Decision Assistant", use_container_width=True, type=dec_btn_type):
        st.session_state["view_mode"] = "decision"
        st.rerun()

    # 2. Configuration & API Key Expander / Drawer
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings & API Key")

    env_key = os.getenv("OPENAI_API_KEY", "")
    session_key = st.session_state.get("openai_api_key", "")
    effective_key = (session_key or env_key).strip()

    user_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=session_key or env_key,
        type="password",
        placeholder="sk-proj-...",
        help="Enter your OpenAI API key to activate AI agent. Never logged.",
    )
    if user_key and user_key.strip() != session_key:
        st.session_state["openai_api_key"] = user_key.strip()
        os.environ["OPENAI_API_KEY"] = user_key.strip()
        effective_key = user_key.strip()
        st.session_state["demo_mode"] = False
        st.rerun()

    demo_checked = st.sidebar.checkbox(
        "🧪 Live Demo Mode (No API key)",
        value=st.session_state.get("demo_mode", not bool(effective_key)),
        help="Compare stocks and view real-time NSE data directly without an OpenAI API key.",
    )
    st.session_state["demo_mode"] = demo_checked

    if not effective_key and not demo_checked:
        st.sidebar.warning("⚠️ OpenAI API key missing. Enter key above or check Demo Mode.")
    elif demo_checked and not effective_key:
        st.sidebar.info("💡 **Live Demo Mode Active**: Real-time NSE data without API key.")
    elif effective_key:
        st.sidebar.success("✅ **AI Agent Active**: Full GPT reasoning enabled.")

    llm_provider = os.getenv("LLM_PROVIDER", "openai").upper()
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    st.sidebar.caption(f"Provider: **{llm_provider}** | Model: **{llm_model}**")
    st.sidebar.caption("Exchange: **National Stock Exchange (NSE)**")

    # 3. Recent Chat History
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent Conversations")

    try:
        conversations = list_conversations(limit=25)
    except Exception:
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
                    except Exception:
                        st.sidebar.error("Failed to delete chat.")

    return st.session_state.get("current_conversation_id")
