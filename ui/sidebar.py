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

    # 2. System Status & Engine
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
