"""Top header bar component with live market status pill and title."""

from __future__ import annotations

from datetime import datetime
import pytz
import streamlit as st


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
    """Render a modern top application header with live market status pill.

    Constructs clean, single-line unindented HTML to prevent Markdown parser
    from incorrectly interpreting 4+ spaces of indentation as code blocks.
    """
    ist_time = get_current_ist_time_str()

    live_pill_html = ""
    if show_live_pill:
        live_pill_html = (
            '<div class="market-status-pill">'
            '<span class="market-status-dot"></span>'
            f"<span>Market Data Live &bull; NSE &bull; {ist_time}</span>"
            "</div>"
        )

    header_html = (
        '<div class="top-app-header">'
        '<div class="top-header-left">'
        f'<div class="top-header-title">{title}</div>'
        f'<div class="top-header-sub">{subtitle}</div>'
        "</div>"
        f'<div class="top-header-right">{live_pill_html}</div>'
        "</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)
