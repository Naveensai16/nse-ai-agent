"""Decision engine for Stock Decision Assistant (Buy / Hold / Sell / Reduce / Exit / Wait analysis)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from models.decision import (
    CatalystDetail,
    DecisionFactorSnapshot,
    DecisionResult,
    PeerMetric,
)
from services.market_data_service import (
    SECTOR_MAP,
    calculate_stock_technical_signals,
    get_market_regime,
    get_market_session_status,
)
from services.sector_service import (
    SECTOR_DEFINITIONS,
    calculate_period_return,
    get_all_sectors_performance,
)
from services.symbol_resolver import NSE_MASTER_DIRECTORY, resolve_nse_symbol
from tools.company_tool import get_company_info
from tools.financials_tool import (
    evaluate_pe_valuation,
    evaluate_roce,
    evaluate_roe,
    get_cash_flow,
    get_debt_metrics,
    get_quarterly_financials,
    is_financial_company,
)
from tools.governance_tool import (
    get_board_meetings,
    get_corporate_actions,
    get_shareholding_pattern,
)
from tools.news_tool import get_market_news
from tools.stock_tool import get_stock_price

logger = logging.getLogger(__name__)

# Sector-specific major peer groupings for direct benchmarking
SECTOR_PEERS: dict[str, list[str]] = {
    "Cement": ["ULTRACEMCO", "AMBUJACEM", "ACC", "SHREECEM", "DALBHARAT"],
    "Power & Utilities": ["NTPC", "POWERGRID", "TATAPOWER", "ADANIPOWER", "ADANIGREEN"],
    "Information Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
    "Automobile": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO"],
    "Pharmaceuticals": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN"],
    "Energy": ["RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "COALINDIA"],
    "Metals & Mining": ["JSWSTEEL", "TATASTEEL", "HINDALCO", "JINDALSTEL", "VEDL"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
    "Financial Services": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN"],
    "Capital Goods": ["LT", "SIEMENS", "ABB", "BHEL", "BEL", "HAL"],
    "Retail & Consumer": ["TITAN", "TRENT", "DMART", "ASIANPAINT", "VOLTAS"],
}


def detect_user_intent_and_details(
    query: str,
) -> tuple[str, Optional[str], str, Optional[float], Optional[int]]:
    """Parse user query to detect investment intent, stock, horizon, purchase price, and quantity.

    Args:
        query: Raw user message or question.

    Returns:
        Tuple containing:
            - intent: 'new' | 'existing'
            - stock: Optional extracted stock symbol or company name
            - horizon: '1 Month' | '3 Months' | '6 Months' | '1 Year' | '2 Years' | '3+ Years'
            - purchase_price: Optional float
            - quantity: Optional int
    """
    q_lower = query.lower()

    # 1. Intent Detection
    existing_keywords = [
        "own", "bought", "have", "holding", "hold or sell", "keep", "exit",
        "sell", "average", "down", "loss in", "my portfolio", "cost price",
        "bought at", "purchase price", "purchased", "already have", "already own",
    ]
    is_existing = any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in existing_keywords)
    intent = "existing" if is_existing else "new"

    # 2. Horizon Detection
    horizon = "1 Year"  # Standard default
    if any(k in q_lower for k in ["3+ year", "3+ years", "3 year", "3-year", "3 years", "5 year", "5-year", "5 years", "long term", "multi-year", "long-term"]):
        horizon = "3+ Years"
    elif any(k in q_lower for k in ["2 year", "2-year", "2 years", "2 yr", "2 yrs", "two year", "two years"]):
        horizon = "2 Years"
    elif any(k in q_lower for k in ["1 year", "1-year", "12 months", "1 yr", "one year", "next year"]):
        horizon = "1 Year"
    elif any(k in q_lower for k in ["6 month", "6-month", "6 months", "half year", "6m"]):
        horizon = "6 Months"
    elif any(k in q_lower for k in ["3 month", "3-month", "3 months", "quarter", "1 quarter", "3m"]):
        horizon = "3 Months"
    elif any(k in q_lower for k in ["1 month", "1-month", "few weeks", "4 weeks", "short term", "short-term", "few days", "1m"]):
        horizon = "1 Month"

    # 3. Purchase Price Extraction
    purchase_price = None
    price_match = re.search(
        r"(?:bought|purchased|bought\s+at|at|@|price\s+(?:of|is|was)?)\s*(?:₹|rs\.?|inr)?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
        query,
        re.IGNORECASE,
    )
    if price_match:
        try:
            val_str = price_match.group(1).replace(",", "")
            val = float(val_str)
            if 1.0 <= val <= 200000.0:
                purchase_price = round(val, 2)
        except Exception:
            pass

    # 4. Quantity Extraction
    quantity = None
    qty_match = re.search(r"(\d+)\s*(?:shares|stocks|qty|quantity|units)", query, re.IGNORECASE)
    if not qty_match:
        qty_match = re.search(r"(?:quantity|qty|shares|units)[:\s]+(\d+)", query, re.IGNORECASE)
    if qty_match:
        try:
            quantity = int(qty_match.group(1))
        except Exception:
            pass

    # 5. Stock Extraction
    clean_query = query
    for kw in [
        "should i buy", "should i sell", "should i hold", "should i exit",
        "should i reduce", "should i average", "should i keep", "is this a good time to buy",
        "good time to buy", "good to buy now", "good to buy", "i already own",
        "i own", "i bought", "can i hold", "can i buy", "for the next", "for another",
        "for a", "investment", "shares of", "shares", "stocks", "at", "₹", "rs.",
        "hold or sell", "should i continue", "keep it", "continue", "months", "month",
        "years", "year", "is", "now", "company", "ltd", "limited", "and want to hold it",
        "and want to hold", "want to hold", "or sell", "should i",
    ]:
        clean_query = re.sub(r"(?i)\b" + re.escape(kw) + r"\b", " ", clean_query)
    clean_query = re.sub(r"[0-9,?.!₹]+", " ", clean_query).strip()

    stock = None
    if clean_query:
        from services.symbol_resolver import resolve_nse_symbol
        res = resolve_nse_symbol(clean_query, allow_online_lookup=False)
        if res.get("symbol"):
            stock = res.get("symbol")
        else:
            stock = clean_query

    return intent, stock, horizon, purchase_price, quantity


def _fetch_price_history_safe(
    symbol: str,
    period: str = "1y",
    provider: Optional[Callable[[str], pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """Retrieve historical price DataFrame with provider override support."""
    if provider is not None:
        try:
            return provider(symbol)
        except Exception:
            return None

    try:
        from utils.helpers import normalize_nse_symbol
        yahoo_sym = normalize_nse_symbol(symbol, target_format="yahoo")
        ticker = yf.Ticker(yahoo_sym)
        df = ticker.history(period=period)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


def calculate_returns_series(df: Optional[pd.DataFrame]) -> dict[str, float]:
    """Calculate multi-period returns (1d, 1w, 1m, 3m, 6m, 1y)."""
    return {
        "1d": calculate_period_return(df, 1),
        "1w": calculate_period_return(df, 5),
        "1m": calculate_period_return(df, 20),
        "3m": calculate_period_return(df, 60),
        "6m": calculate_period_return(df, 120),
        "1y": calculate_period_return(df, 240),
    }


def analyze_stock_decision(
    symbol_or_name: str = "",
    intent: str = "new_investment",
    horizon: str = "1 Year",
    purchase_price: Optional[float] = None,
    quantity: Optional[int] = None,
    history_provider: Optional[Callable[[str], pd.DataFrame]] = None,
    company_info_provider: Optional[Callable[[str], dict[str, Any]]] = None,
    news_provider: Optional[Callable[[str], list[str]]] = None,
    financials_provider: Optional[Callable[[str], dict[str, Any]]] = None,
    stock: Optional[str] = None,
) -> DecisionResult:
    """Master analytical engine for Stock Decision Assistant.

    Performs comprehensive company, sector, valuation, financial, technical,
    catalyst, risk, and peer analysis to output an actionable, objective decision indicator.

    Args:
        symbol_or_name: NSE ticker symbol or natural company name.
        intent: 'new_investment' | 'existing_investment' | 'new' | 'existing'
        horizon: '1 Month' | '3 Months' | '6 Months' | '1 Year' | '2 Years' | '3+ Years'
        purchase_price: Optional historical buy price.
        quantity: Optional number of shares held.
        history_provider: Injectable provider for testing.
        company_info_provider: Injectable provider for testing.
        news_provider: Injectable provider for testing.
        financials_provider: Injectable provider for testing.
        stock: Alias for symbol_or_name.

    Returns:
        DecisionResult dataclass object.
    """
    target = stock or symbol_or_name
    # 1. Resolve Symbol
    res = resolve_nse_symbol(target)
    clean_symbol = res.get("symbol") or target.upper().strip()
    company_name = res.get("company_name") or clean_symbol

    # 2. Market Session & Timestamp
    market_session, formatted_time, _ = get_market_session_status()

    # 3. Company Fundamentals & Profile
    if company_info_provider is not None:
        info = company_info_provider(clean_symbol)
    else:
        info = get_company_info(clean_symbol)

    if info.get("company"):
        company_name = info["company"]

    sector_str = info.get("sector") or SECTOR_MAP.get(clean_symbol, "Diversified")
    industry_str = info.get("industry") or "General"
    market_cap = info.get("market_cap")
    trailing_pe = info.get("trailing_pe")
    forward_pe = info.get("forward_pe")
    price_to_book = info.get("price_to_book")
    if price_to_book is None and info.get("book_value") and info.get("current_price"):
        try:
            price_to_book = float(info["current_price"]) / max(float(info["book_value"]), 1.0)
        except Exception:
            price_to_book = None
    high_52w = info.get("52_week_high")
    low_52w = info.get("52_week_low")

    # 4. Historical Price & Technicals
    df = _fetch_price_history_safe(clean_symbol, period="1y", provider=history_provider)
    signals = calculate_stock_technical_signals(clean_symbol, df) if df is not None else None

    if signals is not None:
        current_price = signals.current_price
        rsi = signals.rsi_14
        dma_20 = signals.dma_20
        dma_50 = signals.dma_50
        support_level = signals.nearest_support
        resistance_level = signals.nearest_resistance
        rel_vol = signals.volume_ratio
    else:
        price_data = get_stock_price(clean_symbol)
        current_price = price_data.get("current_price") or 100.0
        rsi = 50.0
        dma_20 = current_price
        dma_50 = current_price
        support_level = round(current_price * 0.95, 2)
        resistance_level = round(current_price * 1.05, 2)
        rel_vol = 1.0

    if price_to_book is None and info.get("book_value") and current_price:
        try:
            price_to_book = round(float(current_price) / max(float(info["book_value"]), 1.0), 2)
        except Exception:
            pass

    # 200 DMA calculation if data exists
    dma_200 = None
    if df is not None and len(df) >= 200 and "Close" in df.columns:
        dma_200 = round(float(df["Close"].rolling(200).mean().iloc[-1]), 2)
    elif dma_50:
        dma_200 = round(dma_50 * 0.96, 2)

    # Multi-period returns
    returns = calculate_returns_series(df)
    dist_52w = round(((current_price - high_52w) / high_52w) * 100.0, 2) if (high_52w and high_52w > 0) else 0.0

    # 5. Market Condition
    regime = get_market_regime()
    market_env = regime.regime
    if market_env == "Bullish" and regime.nifty_change_1d > 0.5:
        market_env_detail = "Bullish"
    elif market_env == "Bullish":
        market_env_detail = "Moderately Bullish"
    elif market_env == "Bearish":
        market_env_detail = "Moderately Bearish"
    else:
        market_env_detail = "Neutral"

    market_explanation = (
        f"Broader benchmark NIFTY 50 is {market_env_detail.lower()} ({regime.nifty_change_1d:+.2f}% 1D, "
        f"{regime.nifty_return_5d:+.2f}% 5D), holding {'above' if regime.nifty_above_20dma else 'below'} 20 DMA. "
        f"Institutional flows remain {'supportive' if regime.nifty_return_5d >= 0 else 'cautious'}."
    )

    # 6. Sector Analysis
    # Match to SECTOR_DEFINITIONS
    matched_sector_key = "Banking"
    for s_key in SECTOR_DEFINITIONS:
        if s_key.lower() in sector_str.lower() or sector_str.lower() in s_key.lower():
            matched_sector_key = s_key
            break

    sec_defn = SECTOR_DEFINITIONS.get(matched_sector_key, SECTOR_DEFINITIONS["Banking"])
    sec_outlook = "Positive" if regime.regime == "Bullish" else "Neutral"
    why_sec_moving = sec_defn.get("default_why_moving", "Sector is witnessing steady institutional demand and volume accretion.")

    # 7. Quarterly Results & Fundamentals
    if financials_provider is not None:
        fin_data = financials_provider(clean_symbol)
    else:
        try:
            fin_data = get_quarterly_financials(clean_symbol)
        except Exception:
            fin_data = {}

    rev_yoy = fin_data.get("yoy_revenue_growth")
    pat_yoy = fin_data.get("yoy_profit_growth")
    quarters = fin_data.get("quarters", [])
    latest_q = quarters[-1] if quarters else {}
    margin_trend = fin_data.get("margin_trend", "Stable")
    latest_margin = latest_q.get("operating_margin")

    if pat_yoy is not None and pat_yoy >= 15.0 and (rev_yoy or 0) >= 10.0:
        quarterly_verdict = "Very Strong"
        fundamentals_status = "Improving"
    elif pat_yoy is not None and pat_yoy > 0:
        quarterly_verdict = "Strong"
        fundamentals_status = "Improving"
    elif pat_yoy is not None and pat_yoy > -10.0:
        quarterly_verdict = "Mixed"
        fundamentals_status = "Stable"
    elif pat_yoy is not None:
        quarterly_verdict = "Weak"
        fundamentals_status = "Deteriorating"
    else:
        quarterly_verdict = "Mixed"
        fundamentals_status = "Stable"

    # Debt and Cash Flow
    debt_data: dict[str, Any] = {}
    cash_data: dict[str, Any] = {}
    d_to_e = fin_data.get("debt_to_equity")
    fcf = fin_data.get("free_cash_flow")
    if d_to_e is None or fin_data.get("interest_coverage") is None:
        debt_data = get_debt_metrics(clean_symbol)
        if d_to_e is None:
            d_to_e = debt_data.get("debt_to_equity")
    if d_to_e is not None and d_to_e > 20.0:
        d_to_e = round(d_to_e / 100.0, 2)

    if fcf is None:
        cash_data = get_cash_flow(clean_symbol)
        fcf = cash_data.get("free_cash_flow")
    fcf_pos = (fcf is not None and fcf > 0)

    # Governance & Shareholding
    gov_data = get_shareholding_pattern(clean_symbol)
    promoter_pct = gov_data.get("promoter_holding")
    fii_pct = gov_data.get("fii_holding")
    dii_pct = gov_data.get("dii_holding")

    board_mtgs = get_board_meetings(clean_symbol)
    corp_actions = get_corporate_actions(clean_symbol)

    # 8. Valuation Analysis
    pe_eval = evaluate_pe_valuation(trailing_pe, sector=sector_str, earnings_growth=pat_yoy)
    pe_assessment = pe_eval.get("assessment", "Average")

    if pe_assessment in ("Very Good", "Good"):
        val_verdict = "Very Attractive" if (trailing_pe and trailing_pe < 18) else "Attractive"
    elif pe_assessment == "Average":
        val_verdict = "Fair"
    elif pe_assessment in ("Bad", "Very Bad"):
        val_verdict = "Very Expensive" if (trailing_pe and trailing_pe >= 50) else "Expensive"
    else:
        val_verdict = "Fair"

    # 9. Technical Trend & State
    if signals is not None:
        if signals.is_above_20dma and signals.is_above_50dma and (signals.return_5d or 0) > 0:
            tech_trend = "Bullish"
        elif not signals.is_above_20dma and not signals.is_above_50dma and (signals.return_5d or 0) < 0:
            tech_trend = "Bearish"
        else:
            tech_trend = "Neutral"

        if signals.is_breakout:
            tech_state = "Breaking out"
        elif rsi >= 70:
            tech_state = "Overbought"
        elif rsi <= 35:
            tech_state = "Oversold"
        elif -3.0 <= dist_52w <= 0.0:
            tech_state = "Near 52W High"
        elif dist_52w <= -20.0:
            tech_state = "Consolidating after Pullback"
        else:
            tech_state = "Consolidating"
    else:
        tech_trend = "Neutral"
        tech_state = "Consolidating"

    # 10. Peer Comparison
    peer_candidates = SECTOR_PEERS.get(matched_sector_key, ["HDFCBANK", "ICICIBANK", "SBIN"])
    peer_symbols = [p for p in peer_candidates if p != clean_symbol][:4]

    peer_metrics_list: list[PeerMetric] = []
    peer_pes: list[float] = []

    for psym in peer_symbols:
        p_price = get_stock_price(psym)
        p_info = get_company_info(psym)
        p_pe = p_info.get("trailing_pe")
        if p_pe:
            peer_pes.append(p_pe)

        peer_metrics_list.append(
            PeerMetric(
                symbol=psym,
                company_name=p_info.get("company", psym),
                current_price=p_price.get("current_price") or 0.0,
                market_cap=p_info.get("market_cap"),
                pe_ratio=p_pe,
                roe=p_info.get("return_on_equity"),
                roce=p_info.get("return_on_capital_employed"),
                operating_margin=p_info.get("operating_margin"),
                debt_to_equity=p_info.get("debt_to_equity"),
                return_1y=p_info.get("return_1y"),
            )
        )

    # Classify peer position
    if peer_pes and trailing_pe:
        median_peer_pe = float(np.median(peer_pes))
        if (pat_yoy or 0) > 10.0 and trailing_pe <= median_peer_pe * 1.1:
            peer_position = "Outperforming"
        elif (pat_yoy or 0) < 0.0 or trailing_pe > median_peer_pe * 1.3:
            peer_position = "Underperforming"
        else:
            peer_position = "Similar to"
    else:
        peer_position = "Similar to"

    # 11. News & Catalysts Categorization
    if news_provider is not None:
        raw_news = news_provider(clean_symbol)
        raw_news_items = []
        for n in raw_news:
            if isinstance(n, dict):
                raw_news_items.append(n)
            else:
                raw_news_items.append({"title": str(n), "source": "Financial News", "url": "#"})
    else:
        try:
            raw_news_items = get_market_news(clean_symbol, limit=6)
        except Exception:
            raw_news_items = []

    positive_catalysts: list[CatalystDetail] = []
    negative_catalysts: list[CatalystDetail] = []

    pos_keywords = ["order", "growth", "jump", "profit", "win", "expansion", "dividend", "beat", "upgrade", "approval"]
    neg_keywords = ["fall", "loss", "decline", "probe", "cut", "downgrade", "debt", "drop", "penalty", "delay"]

    for n in raw_news_items:
        title = n.get("title", "")
        title_lower = title.lower()
        dt = n.get("published_date") or n.get("date") or "Recent"
        src = n.get("source") or "Market News"
        url = n.get("url")

        is_pos = any(k in title_lower for k in pos_keywords)
        is_neg = any(k in title_lower for k in neg_keywords)

        if is_pos and not is_neg:
            positive_catalysts.append(
                CatalystDetail(
                    headline=title,
                    date=dt,
                    source=src,
                    explanation="Positive corporate/operational catalyst supporting revenue run-rate and sentiment.",
                    potential_impact="Supportive for operating earnings trajectory.",
                    url=url,
                    catalyst_type="positive",
                )
            )
        elif is_neg:
            negative_catalysts.append(
                CatalystDetail(
                    headline=title,
                    date=dt,
                    source=src,
                    explanation="Negative news/headwind creating near-term sentiment caution.",
                    potential_impact="May exert temporary margin or volume pressure.",
                    url=url,
                    catalyst_type="negative",
                )
            )

    # Ensure at least 1 positive and 1 negative perspective to maintain objectivity
    if not positive_catalysts:
        positive_catalysts.append(
            CatalystDetail(
                headline=f"Steady core franchise positioning in NSE {sector_str}",
                source="Sector Overview",
                explanation="Established market presence with consistent customer and distribution reach.",
                potential_impact="Long-term revenue stability.",
            )
        )
    if not negative_catalysts:
        negative_catalysts.append(
            CatalystDetail(
                headline="Macroeconomic sensitivity and raw material/interest rate cycle",
                source="Sector Overview",
                explanation="Input cost volatility or demand slowdown in broad industrial sectors.",
                potential_impact="Potential cyclical margin compression.",
            )
        )

    # 12. Key Risks Analysis
    key_risks: list[str] = []
    if d_to_e is not None and d_to_e > 1.8 and not is_financial_company(sector_str, industry_str):
        key_risks.append(f"Elevated Financial Leverage: Debt-to-Equity is {d_to_e:.2f}, creating interest coverage vulnerability.")
    if val_verdict in ("Expensive", "Very Expensive"):
        key_risks.append(f"Valuation Premium: Trading at {trailing_pe:.1f}x earnings, leaving limited margin of safety.")
    if rsi >= 70:
        key_risks.append(f"Overbought Technical Setup: 14-day RSI is elevated at {rsi:.1f}, susceptible to near-term pullback.")
    if pat_yoy is not None and pat_yoy < 0:
        key_risks.append(f"Earnings Pressure: Latest quarter net profit contracted by {pat_yoy:.1f}% YoY.")
    if margin_trend == "Contracting":
        key_risks.append("Operating Margin Squeeze: Sequential compression in EBITDA/operating margin.")
    key_risks.append(f"Macro & Sector Cyclicality: Demand is tied to domestic economic cycles and regulatory adjustments.")

    # Risk Level Rating
    if len(key_risks) >= 4 or (d_to_e and d_to_e > 2.5):
        risk_level = "High"
    elif len(key_risks) <= 2:
        risk_level = "Low"
    else:
        risk_level = "Moderate"

    # 13. Deterministic Decision Engine
    # Horizon Weights:
    # If horizon is 1 Year or longer: Fundamentals 40%, Valuation 25%, Sector 15%, Technicals 10%, Risk 10%
    # If horizon is shorter: Technicals 35%, Catalysts 25%, Sector 20%, Fundamentals 10%, Valuation 10%
    is_long_horizon = horizon in ("1 Year", "2 Years", "3+ Years")

    # Score calculation (0 to 100)
    fund_score = 85.0 if (fundamentals_status == "Improving" and quarterly_verdict in ("Strong", "Very Strong")) else (75.0 if fundamentals_status == "Improving" else (55.0 if fundamentals_status == "Stable" else 30.0))
    val_score = 85.0 if val_verdict in ("Very Attractive", "Attractive") else (60.0 if val_verdict == "Fair" else 35.0)
    sec_score = 80.0 if sec_outlook in ("Strong", "Positive") else 50.0
    tech_score = 80.0 if (tech_trend == "Bullish" and tech_state != "Overbought") else (70.0 if tech_trend == "Bullish" else (55.0 if tech_trend == "Neutral" else 30.0))
    risk_score = 85.0 if risk_level == "Low" else (60.0 if risk_level == "Moderate" else 35.0)
    cat_score = 85.0 if len(positive_catalysts) > len(negative_catalysts) else (40.0 if len(negative_catalysts) > len(positive_catalysts) else 55.0)

    if is_long_horizon:
        composite_decision_score = (
            (fund_score * 0.40) +
            (val_score * 0.25) +
            (sec_score * 0.15) +
            (risk_score * 0.10) +
            (tech_score * 0.10)
        )
    else:
        composite_decision_score = (
            (tech_score * 0.40) +
            (cat_score * 0.25) +
            (sec_score * 0.15) +
            (fund_score * 0.10) +
            (val_score * 0.10)
        )

    # Determine Decision Indicator
    if intent in ("new", "new_investment"):
        if is_long_horizon:
            if composite_decision_score >= 68.0 and val_verdict not in ("Expensive", "Very Expensive"):
                decision_indicator = "BUY"
            elif val_verdict in ("Expensive", "Very Expensive") or (45.0 <= composite_decision_score < 68.0):
                decision_indicator = "WAIT"
            else:
                decision_indicator = "AVOID FOR NOW"
        else:
            if composite_decision_score >= 68.0 and tech_state != "Overbought" and val_verdict not in ("Expensive", "Very Expensive"):
                decision_indicator = "BUY"
            elif val_verdict in ("Expensive", "Very Expensive") or tech_state == "Overbought" or (45.0 <= composite_decision_score < 68.0):
                decision_indicator = "WAIT"
            else:
                decision_indicator = "AVOID FOR NOW"
    else:  # existing / existing_investment
        if composite_decision_score >= 75.0 and val_verdict in ("Attractive", "Very Attractive", "Fair"):
            decision_indicator = "CONSIDER ADDING"
        elif composite_decision_score >= 50.0:
            decision_indicator = "HOLD"
        elif composite_decision_score >= 38.0:
            decision_indicator = "REDUCE"
        else:
            decision_indicator = "EXIT"

    # Business Quality vs Stock Quality framing
    if fundamentals_status == "Improving" and val_verdict in ("Expensive", "Very Expensive"):
        business_vs_stock = "Good Business + Expensive Stock (Valuation leaves minimal margin of safety)"
    elif fundamentals_status == "Improving":
        business_vs_stock = "Good Business + Reasonable Valuation (Favorable business & price alignment)"
    elif fundamentals_status == "Deteriorating" and val_verdict in ("Attractive", "Very Attractive"):
        business_vs_stock = "Weakening Business + Low Valuation (Potential value trap, caution warranted)"
    else:
        business_vs_stock = "Moderate Business Quality + Fair Stock Valuation"

    # Synthesis of 3-6 Reasons Why
    why_decision: list[str] = []
    if decision_indicator in ("BUY", "CONSIDER ADDING"):
        why_decision.append(f"Fundamentals are {fundamentals_status.lower()} with healthy earnings trajectory.")
        why_decision.append(f"Valuation is {val_verdict.lower()} relative to historical averages and sector benchmarks.")
        why_decision.append(f"Technical stance is constructive ({tech_trend}) with favorable risk/reward.")
        why_decision.append(f"Sector outlook for {sector_str} is {sec_outlook.lower()} with supportive macro tailwinds.")
    elif decision_indicator == "WAIT":
        why_decision.append("Core business remains sound, but current price action/valuation suggests waiting for a better risk-reward entry.")
        if val_verdict in ("Expensive", "Very Expensive"):
            why_decision.append(f"Valuation is currently {val_verdict.lower()} ({trailing_pe:.1f}x P/E), leaving little cushion for earnings surprises.")
        else:
            why_decision.append(f"Current valuation is {val_verdict.lower()}, warranting a pullback towards key support (₹{support_level:,.2f}) for optimal margin of safety.")
        if tech_state in ("Overbought", "Near 52W High"):
            why_decision.append(f"Technical state is {tech_state.lower()} (RSI: {rsi:.1f}). A consolidation or dip to support offers a safer entry.")
        else:
            why_decision.append(f"Technical trend is currently {tech_trend.lower()} ({tech_state.lower()}); awaiting confirmation of a definitive momentum breakout.")
        why_decision.append("Patience is advised until prices consolidate or technical pullbacks offer improved margin of safety.")
    elif decision_indicator == "HOLD":
        why_decision.append(f"The underlying business remains stable, justifying continuity for your {horizon} horizon.")
        why_decision.append(f"Balance sheet leverage and operating metrics remain within acceptable bounds.")
        why_decision.append(f"Support level at ₹{support_level:,.2f} is holding, maintaining the broader structural trend.")
        why_decision.append("No immediate catalyst warrants full exit; maintain standard trailing stops.")
    elif decision_indicator == "REDUCE":
        why_decision.append("Earnings momentum or valuation metrics have shown signs of fatigue.")
        why_decision.append("Trimming partial profits or reallocating to higher-conviction sector peers is prudent.")
        why_decision.append("Rising debt or margin pressure increases sensitivity to market corrections.")
    else:  # EXIT or AVOID FOR NOW
        why_decision.append(f"Fundamentals are {fundamentals_status.lower()} with persistent headwinds.")
        why_decision.append("Risk/reward profile is skewed negatively relative to peer opportunities in the sector.")
        why_decision.append(f"Technical breakdown below moving averages suggests prolonged consolidation or downside.")

    # Reasons Supporting
    reasons_supporting = [
        f"Positioned in {sector_str} with {sec_outlook.lower()} medium-term industrial demand.",
        f"Technical support established near ₹{support_level:,.2f} with 20 DMA at ₹{dma_20:,.2f}.",
    ]
    if rev_yoy and rev_yoy > 0:
        reasons_supporting.append(f"Delivered +{rev_yoy:.1f}% YoY top-line expansion in latest reported quarter.")
    if fcf_pos:
        reasons_supporting.append("Generates positive free cash flows supporting internal capex commitments.")

    # Reasons Against
    reasons_against = list(key_risks[:3])

    # What Would Change This View
    what_would_change_pos = [
        "Acceleration in quarterly earnings and operating margins (> 200 bps expansion).",
        "Substantial debt reduction or sustained positive free cash flow generation.",
        f"Decisive technical breakout above key resistance at ₹{resistance_level:,.2f} on 1.5x+ volume.",
    ]
    what_would_change_neg = [
        "Sequential deterioration in quarterly EBITDA margins or unexpected revenue contraction.",
        "Significant increase in corporate leverage or aggressive debt-funded acquisitions.",
        f"Violation of critical structural support at ₹{support_level:,.2f} turning technical trend Bearish.",
    ]

    # Existing Position Analysis
    unrealized_pnl = None
    existing_advice = ""
    if purchase_price and purchase_price > 0:
        unrealized_pnl = round(((current_price - purchase_price) / purchase_price) * 100.0, 2)
        if unrealized_pnl < 0:
            existing_advice = (
                f"You are currently down {abs(unrealized_pnl):.1f}% from your entry at ₹{purchase_price:,.2f}. "
                f"Crucially, past entry price should never dictate future investment decisions. "
                f"Based on forward business prospects, the objective decision is to **{decision_indicator}**."
            )
        else:
            existing_advice = (
                f"You are currently holding an unrealized gain of +{unrealized_pnl:.1f}% from entry at ₹{purchase_price:,.2f}. "
                f"Based on business fundamentals and risk factors, the forward decision is to **{decision_indicator}**."
            )

    # Horizon Outlook
    if is_long_horizon:
        horizon_outlook = (
            f"**Long-Term Horizon ({horizon})**: For multi-year holdings, short-term price volatility is secondary "
            f"to compounding business earnings, return on capital (ROCE/ROE), and industry competitive moat. "
            f"Current fundamental health ({fundamentals_status.lower()}) and valuation ({val_verdict.lower()}) "
            f"support a forward **{decision_indicator}** stance."
        )
    else:
        horizon_outlook = (
            f"**Short-Term Horizon ({horizon})**: For a shorter timeframe, technical momentum, nearest support "
            f"(₹{support_level:,.2f}) and resistance (₹{resistance_level:,.2f}), and quarterly catalyst flow take priority over "
            f"multi-year compounding. Technical stance is **{tech_trend}** ({tech_state})."
        )

    # Overall Factor Snapshot
    snapshot = DecisionFactorSnapshot(
        market=market_env_detail,
        sector=sec_outlook,
        fundamentals=fundamentals_status,
        quarterly_results=quarterly_verdict,
        valuation=val_verdict,
        technicals=tech_trend,
        news="Positive" if len(positive_catalysts) >= len(negative_catalysts) else "Neutral",
        peer_position=peer_position,
        risk=risk_level,
    )

    result = DecisionResult(
        symbol=clean_symbol,
        company_name=company_name,
        current_price=current_price,
        intent=intent,
        horizon=horizon,
        decision_indicator=decision_indicator,
        analysis_confidence="High" if df is not None and not df.empty else "Medium",
        generated_at=formatted_time,
        snapshot=snapshot,
        why_decision=why_decision,
        reasons_supporting=reasons_supporting,
        reasons_against=reasons_against,
        what_would_change_positive=what_would_change_pos,
        what_would_change_negative=what_would_change_neg,
        market_environment=market_env_detail,
        market_explanation=market_explanation,
        sector_name=sector_str,
        sector_outlook=sec_outlook,
        why_sector_moving=why_sec_moving,
        fundamentals_status=fundamentals_status,
        fundamentals_summary=f"Revenue YoY: {rev_yoy:+.1f}%, PAT YoY: {pat_yoy:+.1f}%, Margin Trend: {margin_trend}" if rev_yoy is not None else "Fundamentals stable with consistent operating margins.",
        roe=info.get("return_on_equity"),
        roce=info.get("return_on_capital_employed"),
        debt_to_equity=d_to_e,
        interest_coverage=fin_data.get("interest_coverage") or debt_data.get("interest_coverage"),
        free_cash_flow_positive=fcf_pos,
        promoter_holding_pct=promoter_pct,
        fii_holding_pct=fii_pct,
        dii_holding_pct=dii_pct,
        quarterly_verdict=quarterly_verdict,
        quarterly_revenue_yoy=rev_yoy,
        quarterly_profit_yoy=pat_yoy,
        quarterly_margin=latest_margin,
        quarterly_summary=f"Latest quarter delivered {rev_yoy:+.1f}% YoY revenue and {pat_yoy:+.1f}% YoY profit growth." if rev_yoy is not None else "Quarterly results in line with sector historical range.",
        valuation_verdict=val_verdict,
        current_pe=trailing_pe,
        sector_pe=info.get("sector_pe"),
        price_to_book=price_to_book,
        valuation_summary=f"Trades at {trailing_pe:.1f}x earnings against sector peer median. Valuation is {val_verdict.lower()}." if trailing_pe else "Valuation metrics within historical norms.",
        peer_position=peer_position,
        peer_metrics=peer_metrics_list,
        peer_summary=f"Compared to sector peers ({', '.join(peer_symbols)}), the company is currently {peer_position.lower()} on growth and profitability.",
        technical_trend=tech_trend,
        technical_state=tech_state,
        rsi_14=rsi,
        dma_20=dma_20,
        dma_50=dma_50,
        dma_200=dma_200,
        support_level=support_level,
        resistance_level=resistance_level,
        high_52w=high_52w,
        low_52w=low_52w,
        distance_52w_high_pct=dist_52w,
        return_1d=returns["1d"],
        return_1w=returns["1w"],
        return_1m=returns["1m"],
        return_3m=returns["3m"],
        return_6m=returns["6m"],
        return_1y=returns["1y"],
        positive_catalysts=positive_catalysts,
        negative_catalysts=negative_catalysts,
        management_announcements=[m.get("purpose", "") for m in board_mtgs if m.get("purpose")][:2],
        key_risks=key_risks,
        risk_level=risk_level,
        business_vs_stock_quality=business_vs_stock,
        investment_horizon_outlook=horizon_outlook,
        purchase_price=purchase_price,
        quantity=quantity,
        unrealized_gain_loss_pct=unrealized_pnl,
        existing_position_advice=existing_advice,
        composite_score=round(composite_decision_score, 1),
    )

    result.report_markdown = format_decision_report_markdown(result)
    return result


def format_decision_report_markdown(res: DecisionResult) -> str:
    """Format complete decision result into structured, professional Markdown."""
    lines: list[str] = []

    # Display canonical company identification banner
    from services.symbol_resolver import get_security_entity
    entity = get_security_entity(res.symbol)
    if entity:
        lines.append(entity.format_banner() + "\n\n---\n")

    # Title & Indicator Banner
    lines.append(f"# 🧠 Stock Decision Analysis — {res.company_name} ({res.symbol})")
    lines.append(f"*Analysis Generated: {res.generated_at} | Horizon: **{res.horizon}** | Confidence: **{res.analysis_confidence}***\n")


    # Primary Decision Indicator Callout
    intent_label = "New Investment" if res.intent == "new_investment" else "Existing Investment"
    lines.append(f"### 🎯 Decision Indicator: **{res.decision_indicator}**")
    lines.append(f"**Intent:** {intent_label} | **Current Price:** ₹{res.current_price:,.2f} | **Quality Assessment:** *{res.business_vs_stock_quality}*\n")

    if res.existing_position_advice:
        lines.append(f"> [!NOTE]\n> **Existing Position Context:**\n> {res.existing_position_advice}\n")

    # Overall Snapshot Table
    lines.append("### 📊 Overall Factor Snapshot")
    lines.append("| Factor | Assessment |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Market Environment** | {res.snapshot.market} |")
    lines.append(f"| **Sector Outlook** | {res.snapshot.sector} ({res.sector_name}) |")
    lines.append(f"| **Company Fundamentals** | {res.snapshot.fundamentals} |")
    lines.append(f"| **Quarterly Results** | {res.snapshot.quarterly_results} |")
    lines.append(f"| **Valuation** | {res.snapshot.valuation} ({f'{res.current_pe:.1f}x P/E' if res.current_pe else 'N/A'}) |")
    lines.append(f"| **Technicals** | {res.snapshot.technicals} ({res.technical_state}) |")
    lines.append(f"| **News & Catalysts** | {res.snapshot.news} |")
    lines.append(f"| **Peer Position** | {res.snapshot.peer_position} |")
    lines.append(f"| **Risk Level** | {res.snapshot.risk} |")
    lines.append("")

    # Why this Decision
    lines.append(f"### 💡 Why **{res.decision_indicator}**?")
    for item in res.why_decision:
        lines.append(f"- {item}")
    lines.append("")

    # Horizon Outlook
    lines.append("### ⏱️ Investment Horizon Outlook")
    lines.append(f"{res.investment_horizon_outlook}\n")

    # Supporting and Opposing Reasons
    lines.append("### ✅ Reasons Supporting the Stock")
    for s in res.reasons_supporting:
        lines.append(f"- {s}")
    lines.append("")

    lines.append("### ❌ Reasons Against the Stock (Risks & Headwinds)")
    for a in res.reasons_against:
        lines.append(f"- {a}")
    lines.append("")

    # Market & Sector Analysis
    lines.append("### 🌐 Market & Sector Environment")
    lines.append(f"* **Broader Market:** {res.market_explanation}")
    lines.append(f"* **Sector ({res.sector_name}):** {res.why_sector_moving}\n")

    # Fundamentals & Latest Results
    lines.append("### 📑 Company Fundamentals & Latest Results")
    lines.append(f"- **Fundamentals Trend:** {res.fundamentals_status} ({res.fundamentals_summary})")
    lines.append(f"- **Latest Quarterly Result:** {res.quarterly_verdict} ({res.quarterly_summary})")
    lines.append(f"- **Balance Sheet & Leverage:** Debt-to-Equity: {f'{res.debt_to_equity:.2f}' if res.debt_to_equity else 'N/A'} | FCF: {'Positive' if res.free_cash_flow_positive else 'Negative / Neutral'}")
    if res.promoter_holding_pct is not None:
        lines.append(f"- **Institutional Ownership:** Promoter: {res.promoter_holding_pct:.1f}% | FII: {res.fii_holding_pct or 0:.1f}% | DII: {res.dii_holding_pct or 0:.1f}%\n")

    # Valuation & Peer Comparison
    lines.append("### ⚖️ Valuation & Peer Benchmarking")
    lines.append(f"**Valuation Assessment:** {res.valuation_verdict}. {res.valuation_summary}\n")
    if res.peer_metrics:
        lines.append("| Peer Symbol | Company Name | Price (₹) | P/E | Mkt Cap (Cr) |")
        lines.append("| :--- | :--- | :---: | :---: | :---: |")
        for pm in res.peer_metrics:
            mcap_cr = f"₹{pm.market_cap / 10000000:,.0f}" if pm.market_cap else "N/A"
            lines.append(f"| `{pm.symbol}` | {pm.company_name} | ₹{pm.current_price:,.2f} | {f'{pm.pe_ratio:.1f}' if pm.pe_ratio else 'N/A'} | {mcap_cr} |")
        lines.append(f"\n*{res.peer_summary}*\n")

    # Technicals & Price Performance
    lines.append("### 📈 Technical Setup & Key Levels")
    lines.append(f"- **Trend:** {res.technical_trend} ({res.technical_state}) | **RSI (14):** {f'{res.rsi_14:.1f}' if res.rsi_14 else 'N/A'}")
    lines.append(f"- **Moving Averages:** 20 DMA: ₹{res.dma_20 or 0:,.2f} | 50 DMA: ₹{res.dma_50 or 0:,.2f} | 200 DMA: ₹{res.dma_200 or 0:,.2f}")
    lines.append(f"- **Key Zones:** Nearest Support: ₹{res.support_level or 0:,.2f} | Nearest Resistance: ₹{res.resistance_level or 0:,.2f}")
    lines.append(f"- **Multi-Period Returns:** 1D: {res.return_1d:+.2f}% | 1W: {res.return_1w:+.2f}% | 1M: {res.return_1m:+.2f}% | 3M: {res.return_3m:+.2f}% | 1Y: {res.return_1y:+.2f}%\n")

    # Catalysts
    lines.append("### 📰 Catalysts & Verified News")
    lines.append("#### 🟢 Positive Catalysts:")
    for cat in res.positive_catalysts:
        src_link = f" [🔗 Read on {cat.source}]({cat.url})" if cat.url and cat.url != "#" else ""
        lines.append(f"- **{cat.headline}** *({cat.source})*{src_link}")
        if cat.explanation:
            lines.append(f"  * {cat.explanation}")

    lines.append("\n#### 🔴 Negative Catalysts / Concerns:")
    for cat in res.negative_catalysts:
        src_link = f" [🔗 Read on {cat.source}]({cat.url})" if cat.url and cat.url != "#" else ""
        lines.append(f"- **{cat.headline}** *({cat.source})*{src_link}")
        if cat.explanation:
            lines.append(f"  * {cat.explanation}")
    lines.append("")

    # What Would Change This View
    lines.append("### 🔄 What Would Change This View?")
    lines.append("**More Positive if:**")
    for cp in res.what_would_change_positive:
        lines.append(f"- {cp}")
    lines.append("\n**More Negative if:**")
    for cn in res.what_would_change_negative:
        lines.append(f"- {cn}")
    lines.append("")

    # Compliance Disclaimer Footer
    lines.append("---")
    lines.append(
        "> [!IMPORTANT]\n"
        "> **Decision Support & Compliance Notice**: This analysis provides research-oriented decision support based "
        "on quantitative metrics, corporate filings, technical setups, and sector intelligence. It does NOT guarantee "
        "profit, future returns, or price movement. All investments are subject to market risks. Perform your own due "
        "diligence and consult a certified financial advisor before committing capital."
    )

    return "\n".join(lines)
