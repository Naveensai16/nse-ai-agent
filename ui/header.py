"""Top header bar component with live market status pill and title."""

from __future__ import annotations

from datetime import datetime
import pytz
import streamlit as st
from ui.theme import COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY


def get_current_ist_time_str() -> str:
    """Get current time formatted in IST."""
    try:
        ist = pytz.timezone("Asia/Kolkata")
        return datetime.now(ist).strftime("%H:%M IST")
    except Exception:
        return datetime.now().strftime("%H:%M")


def render_top_header(
    title: str = "NSE Intelligence",
    subtitle: str = "Indian Equity Market Research & Agentic Intelligence",
    show_live_pill: bool = True,
) -> None:
    """Render a modern top application header with live market status pill."""
    ist_time = get_current_ist_time_str()

    live_pill_html = ""
    if show_live_pill:
        live_pill_html = f"""
        <div class="market-status-pill">
            <span class="market-status-dot"></span>
            <span>Market Data Live &bull; NSE &bull; {ist_time}</span>
        </div>
        """

    header_html = f"""
    <div class="top-app-header">
        <div class="top-header-left">
            <div class="top-header-title">{title}</div>
            <div class="top-header-sub">{subtitle}</div>
        </div>
        <div class="top-header-right">
            {live_pill_html}
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
