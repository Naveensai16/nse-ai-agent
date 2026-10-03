"""Reusable modern UI component renderers for financial metrics, headers, cards, and reports."""

from __future__ import annotations

import textwrap
from typing import Any, Optional
import streamlit as st

from services.symbol_resolver import INDIAN_STOCK_MASTER, search_stocks
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


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_search_stocks(query: str, limit: int = 15) -> list[dict[str, Any]]:
    """Cached wrapper around search_stocks to ensure instantaneous autocomplete response."""
    return search_stocks(query=query, limit=limit)


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
    delta_html = f'<div class="metric-delta {delta_type}">{delta}</div>' if delta else ""
    html = f"""
<div class="metric-card">
<div class="metric-label">{label}</div>
<div class="metric-value">{value}</div>
{delta_html}
</div>
"""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


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
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


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
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


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
        st.markdown(textwrap.dedent(html_pos).strip(), unsafe_allow_html=True)

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
        st.markdown(textwrap.dedent(html_neg).strip(), unsafe_allow_html=True)


def render_compliance_notice(custom_text: Optional[str] = None) -> None:
    """Render a compact, professional disclaimer box without intrusive alerts."""
    default_text = (
        "Investment Research Notice: Analysis is generated using deterministic quantitative multi-factor models. "
        "It does not constitute SEBI-registered investment advice. Please consult a licensed financial advisor."
    )
    text = custom_text or default_text
    with st.expander("ⓘ Regulatory Disclosure & Investment Notice", expanded=False):
        st.caption(text)


def stock_autocomplete(
    label: str = "Stock Name or Symbol",
    value: str = "",
    key: str = "decision_stock_input",
    placeholder: str = "e.g. Tata Motors, RELIANCE, TCS, HDFC Bank",
    help: Optional[str] = None,
) -> str:
    """Render a searchable autocomplete stock input with dynamic suggestions dropdown.

    Provides both options to the user:
    1. Continue typing manually
    2. Select one of the suggestions from the dropdown

    Maintains separate session state values:
    - stock_search_text: Current typed text
    - stock_suggestions: List of matching stocks
    - selected_stock_name: Formatted company name
    - selected_symbol: Canonical NSE symbol
    - analysis_result: Analyzed report result (None until Analyze clicked)
    - analysis_requested: True only when Analyze button clicked

    Returns:
        str: Selected or typed company name / NSE ticker symbol.
    """
    # 0. Apply pending input BEFORE the widget is instantiated on this rerun
    pending_val = None
    if f"{key}_pending" in st.session_state:
        pending_val = st.session_state.pop(f"{key}_pending")
    elif "decision_pending_input" in st.session_state and key == "decision_stock_input":
        pending_val = st.session_state.pop("decision_pending_input")

    if pending_val is not None:
        st.session_state[key] = pending_val
        current_value = pending_val
    elif key not in st.session_state and value:
        st.session_state[key] = value
        current_value = value
    else:
        current_value = st.session_state.get(key, value)

    # 1. Main input box
    typed_input = st.text_input(
        label,
        value=current_value,
        key=key,
        placeholder=placeholder,
        help=help,
    )

    clean_text = typed_input.strip() if typed_input else ""
    st.session_state["stock_search_text"] = clean_text

    # If user cleared the input, reset selection states
    if not clean_text:
        st.session_state["selected_symbol"] = None
        st.session_state["decision_selected_symbol"] = None
        st.session_state["selected_stock_name"] = None
        st.session_state["decision_selected_stock"] = None
        st.session_state["stock_suggestions"] = []
        st.session_state["analysis_result"] = None
        st.session_state["analysis_requested"] = False
        return ""

    # Check if a symbol was already selected and clean_text still matches it
    current_sym = (
        st.session_state.get("selected_symbol")
        or st.session_state.get("decision_selected_symbol")
        or st.session_state.get(f"{key}_canonical")
    )
    if current_sym and current_sym in INDIAN_STOCK_MASTER:
        master_data = INDIAN_STOCK_MASTER[current_sym]
        disp = f"{master_data.get('company_name', current_sym)} ({current_sym})"
        # If user changed the text so it no longer matches the selected stock
        if (
            clean_text.lower() not in (disp.lower(), current_sym.lower(), master_data.get("company_name", "").lower())
            and not clean_text.upper().endswith(f"({current_sym})")
        ):
            # User is typing a different query: clear previous selection!
            st.session_state["selected_symbol"] = None
            st.session_state["decision_selected_symbol"] = None
            st.session_state["selected_stock_name"] = None
            st.session_state["decision_selected_stock"] = None
            st.session_state.pop(f"{key}_canonical", None)
            st.session_state["analysis_result"] = None
            st.session_state["analysis_requested"] = False
            current_sym = None
        else:
            sector_txt = f" | Sector: **{master_data['sector']}**" if master_data.get("sector") else ""
            st.caption(f"✓ Selected: **{master_data.get('company_name', current_sym)}** (`{current_sym}`){sector_txt}")
            st.session_state["selected_symbol"] = current_sym
            st.session_state["decision_selected_symbol"] = current_sym
            st.session_state["selected_stock_name"] = master_data.get("company_name", current_sym)
            st.session_state["decision_selected_stock"] = master_data.get("company_name", current_sym)
            return current_sym

    # Check if user typed an exact symbol directly (e.g. "TATAMOTORS", "RELIANCE")
    if clean_text.upper() in INDIAN_STOCK_MASTER:
        sym = clean_text.upper()
        master_data = INDIAN_STOCK_MASTER[sym]
        sector_txt = f" | Sector: **{master_data['sector']}**" if master_data.get("sector") else ""
        st.caption(f"✓ Exact Symbol: **{master_data.get('company_name', sym)}** (`{sym}`){sector_txt}")
        st.session_state["selected_symbol"] = sym
        st.session_state["decision_selected_symbol"] = sym
        st.session_state["selected_stock_name"] = master_data.get("company_name", sym)
        st.session_state["decision_selected_stock"] = master_data.get("company_name", sym)
        st.session_state[f"{key}_canonical"] = sym
        return sym

    # Search for autocomplete suggestions
    matches = _cached_search_stocks(clean_text, limit=15)
    st.session_state["stock_suggestions"] = matches

    if matches:
        # Check if the user already selected or typed an exact match to a single suggestion
        is_exact_match = (
            len(matches) == 1
            and (
                matches[0]["display_text"].lower() == clean_text.lower()
                or matches[0]["symbol"].lower() == clean_text.lower()
                or matches[0]["company_name"].lower() == clean_text.lower()
                or clean_text.upper().endswith(f"({matches[0]['symbol']})")
            )
        )

        if is_exact_match:
            # Display subtle confirmation pill
            m = matches[0]
            sector_txt = f" | Sector: **{m['sector']}**" if m.get("sector") else ""
            st.caption(f"✓ Selected: **{m['company_name']}** (`{m['symbol']}`){sector_txt}")
            st.session_state["selected_symbol"] = m["symbol"]
            st.session_state["decision_selected_symbol"] = m["symbol"]
            st.session_state["selected_stock_name"] = m["company_name"]
            st.session_state["decision_selected_stock"] = m["company_name"]
            st.session_state[f"{key}_canonical"] = m["symbol"]
            return m["symbol"]

        # Build options dictionary
        suggestion_map = {m["display_text"]: m for m in matches}
        placeholder_label = f"▾ {len(matches)} matching stock{'s' if len(matches) > 1 else ''} found — Click to select (or continue typing):"
        options = [placeholder_label] + list(suggestion_map.keys())

        # Render suggestions selectbox dropdown
        chosen = st.selectbox(
            "Suggested Stocks",
            options=options,
            index=0,
            key=f"{key}_suggestions_dropdown",
            label_visibility="collapsed",
            help="Select a matching NSE stock to populate the input",
        )

        # If a valid suggestion was selected by the user
        if chosen in suggestion_map:
            selected_record = suggestion_map[chosen]
            # CRITICAL: Do NOT set st.session_state[key] after instantiation!
            # Instead, set pending input to be applied BEFORE instantiation on next rerun:
            st.session_state[f"{key}_pending"] = selected_record["display_text"]
            st.session_state["decision_pending_input"] = selected_record["display_text"]
            st.session_state[f"{key}_canonical"] = selected_record["symbol"]
            st.session_state["selected_symbol"] = selected_record["symbol"]
            st.session_state["decision_selected_symbol"] = selected_record["symbol"]
            st.session_state["selected_stock_name"] = selected_record["company_name"]
            st.session_state["decision_selected_stock"] = selected_record["company_name"]
            st.session_state["stock_search_text"] = selected_record["display_text"]
            st.session_state["analysis_requested"] = False
            st.session_state["analysis_result"] = None
            st.session_state.pop(f"{key}_suggestions_dropdown", None)
            st.rerun()

    return typed_input
