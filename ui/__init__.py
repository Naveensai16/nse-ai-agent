"""UI package for the modern, premium NSE AI Agent frontend."""

from __future__ import annotations

from ui.components import (
    render_compliance_notice,
    render_decision_card,
    render_metric_card,
    render_positive_negative_cards,
    render_stock_header,
    stock_autocomplete,
)
from ui.header import render_top_header
from ui.sidebar import render_sidebar
from ui.theme import get_global_css
from ui.welcome import render_welcome_screen

__all__ = [
    "get_global_css",
    "render_compliance_notice",
    "render_decision_card",
    "render_metric_card",
    "render_positive_negative_cards",
    "render_sidebar",
    "render_stock_header",
    "render_top_header",
    "render_welcome_screen",
    "stock_autocomplete",
]
