"""Theme configuration, design tokens, and global CSS for the NSE AI Agent application."""

from __future__ import annotations

# Color Palette Constants
COLOR_BG_MAIN = "#0B0F14"
COLOR_BG_SECONDARY = "#111720"
COLOR_BG_CARD = "#151C26"
COLOR_BG_ELEVATED = "#19222E"
COLOR_BORDER = "rgba(255, 255, 255, 0.07)"
COLOR_BORDER_SUBTLE = "rgba(255, 255, 255, 0.04)"
COLOR_BORDER_FOCUS = "rgba(99, 102, 241, 0.4)"

COLOR_TEXT_PRIMARY = "#F3F6FA"
COLOR_TEXT_SECONDARY = "#A3ADBD"
COLOR_TEXT_MUTED = "#6F7A8A"

COLOR_ACCENT = "#6366F1"
COLOR_ACCENT_HOVER = "#4F46E5"
COLOR_ACCENT_BG = "rgba(99, 102, 241, 0.12)"
COLOR_ACCENT_BORDER = "rgba(99, 102, 241, 0.28)"

COLOR_POSITIVE = "#22C55E"
COLOR_POSITIVE_BG = "rgba(34, 197, 94, 0.08)"
COLOR_POSITIVE_BORDER = "rgba(34, 197, 94, 0.22)"

COLOR_NEGATIVE = "#EF4444"
COLOR_NEGATIVE_BG = "rgba(239, 68, 68, 0.08)"
COLOR_NEGATIVE_BORDER = "rgba(239, 68, 68, 0.22)"

COLOR_WARNING = "#F59E0B"
COLOR_WARNING_BG = "rgba(245, 158, 11, 0.08)"
COLOR_WARNING_BORDER = "rgba(245, 158, 11, 0.22)"

COLOR_INFO = "#38BDF8"
COLOR_INFO_BG = "rgba(56, 189, 248, 0.08)"
COLOR_INFO_BORDER = "rgba(56, 189, 248, 0.22)"


def get_global_css(is_dashboard: bool = False) -> str:
    """Generate injected CSS to transform Streamlit into a modern fintech dark UI."""
    max_w = "1360px" if is_dashboard else "1160px"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global reset & typography */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        color: {COLOR_TEXT_PRIMARY};
        letter-spacing: -0.011em;
    }}

    /* Main background */
    .stApp {{
        background-color: {COLOR_BG_MAIN} !important;
    }}

    /* Hide Streamlit top header bar clutter while keeping sidebar toggle */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        border-bottom: none !important;
    }}
    #MainMenu, footer, .stDeployButton {{
        visibility: hidden !important;
        display: none !important;
    }}

    /* Centered content container */
    .main .block-container {{
        max-width: {max_w} !important;
        padding-top: 1.25rem !important;
        padding-bottom: 5.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }}

    /* Custom thin dark scrollbars */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLOR_BG_MAIN};
    }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(255, 255, 255, 0.12);
        border-radius: 999px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(255, 255, 255, 0.22);
    }}

    /* -------------------------------------------------------------
       Sidebar Styling (270px width, sleek dark, refined controls)
       ------------------------------------------------------------- */
    section[data-testid="stSidebar"] {{
        width: 275px !important;
        min-width: 275px !important;
        max-width: 275px !important;
        background-color: {COLOR_BG_SECONDARY} !important;
        border-right: 1px solid {COLOR_BORDER} !important;
    }}

    section[data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    /* Sidebar buttons (Navigation items) */
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100% !important;
        height: 44px !important;
        border-radius: 10px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 0 14px !important;
        margin-bottom: 4px !important;
        transition: all 150ms ease !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
        color: {COLOR_TEXT_SECONDARY} !important;
    }}

    section[data-testid="stSidebar"] .stButton > button:hover {{
        background-color: rgba(255, 255, 255, 0.04) !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }}

    /* Active navigation button */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {COLOR_ACCENT_BG} !important;
        border: 1px solid {COLOR_ACCENT_BORDER} !important;
        color: #C7D2FE !important;
        font-weight: 600 !important;
    }}

    /* Sidebar section headers */
    .sidebar-section-label {{
        font-size: 10.5px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: {COLOR_TEXT_MUTED} !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.5rem !important;
        padding-left: 0.5rem !important;
    }}

    /* Sidebar Brand Header */
    .sidebar-brand-box {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 6px 16px 6px;
        border-bottom: 1px solid {COLOR_BORDER};
        margin-bottom: 14px;
    }}
    .sidebar-brand-icon {{
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }}
    .sidebar-brand-text {{
        display: flex;
        flex-direction: column;
    }}
    .sidebar-brand-title {{
        font-size: 15px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        line-height: 1.2;
    }}
    .sidebar-brand-sub {{
        font-size: 11px;
        color: {COLOR_TEXT_MUTED};
        font-weight: 400;
    }}

    /* -------------------------------------------------------------
       Top Global Header Bar
       ------------------------------------------------------------- */
    .top-app-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    .top-header-left {{
        display: flex;
        flex-direction: column;
        gap: 2px;
    }}
    .top-header-title {{
        font-size: 20px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        letter-spacing: -0.02em;
    }}
    .top-header-sub {{
        font-size: 12.5px;
        color: {COLOR_TEXT_MUTED};
    }}
    .top-header-right {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .market-status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #4ADE80;
    }}
    .market-status-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #22C55E;
        box-shadow: 0 0 8px #22C55E;
    }}

    /* -------------------------------------------------------------
       Cards & Metric Grids
       ------------------------------------------------------------- */
    .metric-card {{
        background: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        transition: border-color 150ms ease, transform 150ms ease;
    }}
    .metric-card:hover {{
        border-color: rgba(255, 255, 255, 0.14);
    }}
    .metric-label {{
        font-size: 11px;
        font-weight: 600;
        color: {COLOR_TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .metric-value {{
        font-size: 20px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        font-feature-settings: "tnum";
    }}
    .metric-delta {{
        font-size: 11.5px;
        font-weight: 500;
    }}
    .metric-delta.pos {{
        color: {COLOR_POSITIVE};
    }}
    .metric-delta.neg {{
        color: {COLOR_NEGATIVE};
    }}
    .metric-delta.neu {{
        color: {COLOR_TEXT_MUTED};
    }}

    /* -------------------------------------------------------------
       Streamlit Widgets Overrides
       ------------------------------------------------------------- */
    /* Selectboxes, Text Inputs & Number Inputs */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input {{
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 10px !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        font-size: 13.5px !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div:hover,
    .stTextInput input:hover,
    .stNumberInput input:hover {{
        border-color: rgba(255, 255, 255, 0.16) !important;
    }}
    .stTextInput input:focus,
    .stNumberInput input:focus {{
        border-color: {COLOR_ACCENT} !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        transition: all 150ms ease !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLOR_ACCENT}, #4F46E5) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #4F46E5, #4338CA) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: rgba(255, 255, 255, 0.18) !important;
        background: {COLOR_BG_ELEVATED} !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        overflow: hidden !important;
    }}
    div[data-testid="stExpander"] > details > summary {{
        padding: 12px 16px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        border-radius: 12px !important;
    }}
    div[data-testid="stExpander"] > details > summary:hover {{
        background-color: rgba(255, 255, 255, 0.02) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 1px solid {COLOR_BORDER} !important;
        gap: 8px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        color: {COLOR_TEXT_SECONDARY} !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        padding: 8px 14px !important;
        border: none !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: #A5B4FC !important;
        font-weight: 600 !important;
        border-bottom: 2px solid {COLOR_ACCENT} !important;
    }}

    /* Chat Messages */
    div[data-testid="stChatMessage"] {{
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12) !important;
    }}
    div[data-testid="stChatMessage"][data-testid*="user"] {{
        background-color: {COLOR_BG_ELEVATED} !important;
        border-color: rgba(99, 102, 241, 0.15) !important;
    }}

    /* Floating Chat Input at Bottom */
    div[data-testid="stChatInput"] {{
        border-radius: 14px !important;
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        padding: 4px 6px !important;
        transition: border-color 150ms ease, box-shadow 150ms ease !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: {COLOR_ACCENT} !important;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.18) !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: {COLOR_TEXT_PRIMARY} !important;
        font-size: 14px !important;
    }}

    /* Dataframes */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}

    /* Stock Header Banner */
    .company-header-card {{
        background: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }}
    .company-header-title {{
        font-size: 22px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    .company-header-meta {{
        font-size: 13px;
        color: {COLOR_TEXT_MUTED};
        margin-top: 4px;
        display: flex;
        gap: 12px;
        align-items: center;
    }}
    .company-header-price-block {{
        text-align: right;
    }}
    .company-header-price {{
        font-size: 26px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        font-feature-settings: "tnum";
    }}
    .company-header-change {{
        font-size: 13.5px;
        font-weight: 600;
    }}

    /* Decision Card */
    .decision-banner-box {{
        background: {COLOR_BG_CARD};
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        border-left: 4px solid {COLOR_ACCENT};
        border-top: 1px solid {COLOR_BORDER};
        border-right: 1px solid {COLOR_BORDER};
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    .decision-banner-box.buy {{
        border-left-color: {COLOR_POSITIVE};
        background: linear-gradient(180deg, rgba(34, 197, 94, 0.05) 0%, {COLOR_BG_CARD} 100%);
    }}
    .decision-banner-box.hold {{
        border-left-color: {COLOR_WARNING};
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.05) 0%, {COLOR_BG_CARD} 100%);
    }}
    .decision-banner-box.exit {{
        border-left-color: {COLOR_NEGATIVE};
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.05) 0%, {COLOR_BG_CARD} 100%);
    }}

    /* Two-column change cards (Positive / Negative If) */
    .view-change-card {{
        background: {COLOR_BG_CARD};
        border-radius: 12px;
        padding: 16px 18px;
        height: 100%;
    }}
    .view-change-card.positive {{
        border: 1px solid {COLOR_POSITIVE_BORDER};
        background: {COLOR_POSITIVE_BG};
    }}
    .view-change-card.negative {{
        border: 1px solid {COLOR_NEGATIVE_BORDER};
        background: {COLOR_NEGATIVE_BG};
    }}

    /* Disclaimer box */
    .compliance-pill-box {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid {COLOR_BORDER};
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 11.5px;
        color: {COLOR_TEXT_MUTED};
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 16px;
    }}
    </style>
    """
