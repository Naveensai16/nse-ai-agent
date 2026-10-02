"""Reusable modern UI component renderers for financial metrics, headers, cards, and reports."""

from __future__ import annotations

from typing import Any, Optional
import streamlit as st

from ui.theme import (
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
)


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "neu",
) -> None:
    """Render a clean financial metric card.

    Args:
        label: Metric title (e.g. 'Market Cap', 'P/E', 'ROE').
        value: Primary metric number or string.
        delta: Optional secondary change or subtitle (e.g. '+1.45%').
        delta_type: 'pos' (green), 'neg' (red), or 'neu' (muted).
    """
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-delta {delta_type}">{delta}</div>'

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stock_header(
    company_name: str,
    symbol: str,
    current_price: Optional[float] = None,
    day_change_pct: Optional[float] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    bse_code: Optional[str] = None,
    isin: Optional[str] = None,
    exchange: str = "NSE",
) -> None:
    """Render a header banner with company identity, live price and badges."""
    price_html = ""
    if current_price is not None:
        chg_str = f"{day_change_pct:+.2f}%" if day_change_pct is not None else ""
        chg_cls = "pos" if (day_change_pct or 0) >= 0 else "neg"
        chg_color = COLOR_POSITIVE if (day_change_pct or 0) >= 0 else COLOR_NEGATIVE
        price_html = f"""
        <div class="company-header-price-block">
            <div class="company-header-price">₹{current_price:,.2f}</div>
            <div class="company-header-change" style="color: {chg_color};">{chg_str}</div>
        </div>
        """

    meta_parts = [f"<strong>{symbol}</strong> • {exchange}"]
    if sector:
        meta_parts.append(sector)
    if industry:
        meta_parts.append(industry)
    if bse_code:
        meta_parts.append(f"BSE: {bse_code}")
    if isin:
        meta_parts.append(f"ISIN: {isin}")

    meta_str = " &nbsp;|&nbsp; ".join(meta_parts)

    html = f"""
    <div class="company-header-card">
        <div>
            <div class="company-header-title">{company_name}</div>
            <div class="company-header-meta">{meta_str}</div>
        </div>
        {price_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_decision_card(
    indicator: str,
    horizon: str,
    confidence: str,
    business_vs_stock: str,
    key_reasons: list[str],
) -> None:
    """Render the central Stock Decision recommendation banner and core thesis."""
    ind_upper = indicator.upper()
    if any(k in ind_upper for k in ("BUY", "CONSIDER ADDING")):
        cls = "buy"
        badge_bg = "rgba(34, 197, 94, 0.15)"
        badge_border = "rgba(34, 197, 94, 0.35)"
        badge_color = "#4ADE80"
        icon = "✦"
    elif any(k in ind_upper for k in ("HOLD", "WAIT")):
        cls = "hold"
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_border = "rgba(245, 158, 11, 0.35)"
        badge_color = "#FBBF24"
        icon = "●"
    else:
        cls = "exit"
        badge_bg = "rgba(239, 68, 68, 0.15)"
        badge_border = "rgba(239, 68, 68, 0.35)"
        badge_color = "#F87171"
        icon = "✕"

    reasons_li = "".join(f"<li style='margin-bottom: 6px; color: {COLOR_TEXT_PRIMARY}; font-size: 13.5px;'>{r}</li>" for r in key_reasons)

    html = f"""
    <div class="decision-banner-box {cls}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; color: {COLOR_TEXT_MUTED};">
                Investment Recommendation
            </div>
            <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color}; font-size: 12px; font-weight: 700;">
                <span>{icon}</span> {indicator}
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-bottom: 14px; font-size: 12.5px; color: {COLOR_TEXT_SECONDARY};">
            <div>Horizon: <strong style="color: {COLOR_TEXT_PRIMARY};">{horizon}</strong></div>
            <div>Confidence: <strong style="color: {badge_color};">{confidence}</strong></div>
        </div>
        <div style="font-size: 13px; color: {COLOR_TEXT_SECONDARY}; margin-bottom: 14px; line-height: 1.5;">
            {business_vs_stock}
        </div>
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: {COLOR_TEXT_MUTED}; margin-bottom: 8px;">
            Core Thesis
        </div>
        <ul style="padding-left: 20px; margin: 0;">
            {reasons_li}
        </ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_positive_negative_cards(
    positive_items: list[str],
    negative_items: list[str],
) -> None:
    """Render the clean two-column 'What Would Change This View' component."""
    col1, col2 = st.columns(2)

    with col1:
        pos_list = "".join(f"<li style='margin-bottom: 6px; font-size: 13px;'>{item}</li>" for item in positive_items)
        html_pos = f"""
        <div class="view-change-card positive">
            <div style="display: flex; align-items: center; gap: 6px; color: #4ADE80; font-size: 13px; font-weight: 700; margin-bottom: 10px;">
                <span>↑</span> View Becomes More Positive If:
            </div>
            <ul style="padding-left: 18px; margin: 0; color: {COLOR_TEXT_PRIMARY};">
                {pos_list}
            </ul>
        </div>
        """
        st.markdown(html_pos, unsafe_allow_html=True)

    with col2:
        neg_list = "".join(f"<li style='margin-bottom: 6px; font-size: 13px;'>{item}</li>" for item in negative_items)
        html_neg = f"""
        <div class="view-change-card negative">
            <div style="display: flex; align-items: center; gap: 6px; color: #F87171; font-size: 13px; font-weight: 700; margin-bottom: 10px;">
                <span>↓</span> View Becomes More Negative If:
            </div>
            <ul style="padding-left: 18px; margin: 0; color: {COLOR_TEXT_PRIMARY};">
                {neg_list}
            </ul>
        </div>
        """
        st.markdown(html_neg, unsafe_allow_html=True)


def render_compliance_notice(custom_text: Optional[str] = None) -> None:
    """Render a compact, professional disclaimer box without intrusive alerts."""
    default_text = (
        "Investment Research Notice: Analysis is generated using deterministic quantitative multi-factor models. "
        "It does not constitute SEBI-registered investment advice. Please consult a licensed financial advisor."
    )
    text = custom_text or default_text
    with st.expander("ⓘ Regulatory Disclosure & Investment Notice", expanded=False):
        st.caption(text)
