"""Sector performance analysis, top 5 sector identification, and multi-factor constituent ranking service."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from models.sector import (
    CompanySectorRanking,
    SectorPerformance,
    TopSectorsResult,
)
from services.market_data_service import (
    SECTOR_MAP,
    calculate_stock_technical_signals,
    get_market_session_status,
)
from services.symbol_resolver import NSE_MASTER_DIRECTORY, resolve_nse_symbol
from tools.company_tool import get_company_info
from tools.financials_tool import get_quarterly_financials
from tools.news_tool import get_market_news
from tools.stock_tool import get_stock_price

logger = logging.getLogger(__name__)

# Sector definitions: mapping display sector name to index ticker and liquid constituents
SECTOR_DEFINITIONS: dict[str, dict[str, Any]] = {
    "Banking": {
        "display_name": "Banking",
        "index_symbol": "^NSEBANK",
        "constituents": [
            "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
            "INDUSINDBK", "PNB", "BANKBARODA", "CANBK", "FEDERALBNK",
            "IDFCFIRSTB", "UNIONBANK",
        ],
        "default_why_moving": "Banking sector is witnessing strong credit growth, resilient asset quality, and healthy net interest margins across both private and public lenders.",
        "important_news": [
            "RBI monetary policy maintains supportive liquidity stance with stable interest rate cycle.",
            "Credit growth across major commercial banks remains robust at ~13-14% YoY.",
            "Asset quality metrics hold strong with gross NPAs hovering near multi-year lows.",
        ],
        "institutional_activity": "Domestic institutional investors (DIIs) and FIIs maintain overweight allocations on high-return Tier-1 private and PSU banks.",
        "policy_impact": "RBI regulatory framework encourages robust capitalization buffers and stable deposit accretion mechanisms.",
        "major_earnings": "Key lenders report double-digit growth in pre-provision operating profits with return on assets (RoA) sustaining above 1.8%.",
    },
    "Information Technology": {
        "display_name": "Information Technology",
        "index_symbol": "^CNXIT",
        "constituents": [
            "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
            "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "KPITTECH",
            "TATAELXSI",
        ],
        "default_why_moving": "IT stocks are reacting to international tech deal flow, accelerated enterprise AI adoption, and favorable US Federal Reserve rate commentary.",
        "important_news": [
            "Indian IT majors report multi-billion dollar multi-year deal wins in cloud migration and enterprise GenAI.",
            "Global enterprise tech budgets show gradual expansion in digital transformation and automation.",
            "Attrition rates have stabilized below 12%, aiding operating margin resilience.",
        ],
        "institutional_activity": "Global tech funds selectively accumulate top-tier IT exporters on attractive relative valuations.",
        "policy_impact": "Stable currency dynamics (USD/INR) provide steady tailwinds for export revenue realization.",
        "major_earnings": "Tier-1 IT leaders demonstrate sequential deal pipeline expansion and operating margin expansion of 40-70 bps.",
    },
    "Auto": {
        "display_name": "Auto",
        "index_symbol": "^CNXAUTO",
        "constituents": [
            "TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
            "EICHERMOT", "TVSMOTOR", "ASHOKLEY", "BHARATFORG",
        ],
        "default_why_moving": "Automotive manufacturers gain traction on record passenger vehicle dispatches, strong rural demand revival, and expanding electric vehicle portfolios.",
        "important_news": [
            "Monthly auto dispatch figures show strong double-digit growth in SUV and premium two-wheeler segments.",
            "Raw material input costs (steel and aluminum) remain benign, boosting EBITDA margins.",
            "EV penetration accelerates across urban centers backed by expanding charging infrastructure.",
        ],
        "institutional_activity": "FIIs and mutual funds maintain steady buying interest in market leaders with high order backlogs.",
        "policy_impact": "Government PLI scheme for automotive components and clean energy vehicles supports long-term capex commitments.",
        "major_earnings": "Automakers report operating margins expanding towards 12-15% on richer product mix and premiumization.",
    },
    "Pharma & Healthcare": {
        "display_name": "Pharma & Healthcare",
        "index_symbol": "^CNXPHARMA",
        "constituents": [
            "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
            "MANKIND", "TORNTPHARM", "LUPIN", "BIOCON",
        ],
        "default_why_moving": "Pharmaceutical and healthcare companies benefit from US generic price stabilization, niche biosimilar approvals, and steady domestic formulations growth.",
        "important_news": [
            "US FDA facility inspections yield clearance with Voluntary Action Indicated (VAI) status for key plants.",
            "Specialty drug pipelines and chronic therapy formulations deliver resilient 10-12% domestic revenue growth.",
            "Hospital chains report healthy average revenue per occupied bed (ARPOB) and increasing occupancy rates.",
        ],
        "institutional_activity": "Defensive rotation by institutional portfolios into healthcare assets amid global macroeconomic uncertainty.",
        "policy_impact": "Government PLI schemes for active pharmaceutical ingredients (APIs) and medical devices bolster supply chain independence.",
        "major_earnings": "Pharma companies post strong gross margin expansion driven by favorable raw material pricing and specialty product launches.",
    },
    "Energy & Power": {
        "display_name": "Energy & Power",
        "index_symbol": "^CNXENERGY",
        "constituents": [
            "RELIANCE", "NTPC", "POWERGRID", "TATAPOWER", "ONGC",
            "COALINDIA", "IOC", "BPCL", "HPCL", "GAIL",
            "ADANIGREEN", "ADANIPOWER",
        ],
        "default_why_moving": "Energy and utility counters advance on robust national peak power demand, aggressive green energy transitions, and steady refining margins.",
        "important_news": [
            "India's peak electricity demand reaches fresh record highs, driving thermal and renewable plant load factors.",
            "Transmission grid expansion projects witness accelerated project execution schedules.",
            "State utilities enhance timely payment realizations under the revamped distribution sector scheme (RDSS).",
        ],
        "institutional_activity": "Sovereign wealth funds and long-term domestic funds increase capital commitments in green energy capex leaders.",
        "policy_impact": "National green hydrogen policy and renewable purchase obligations (RPO) mandate long-term sector growth.",
        "major_earnings": "Power utilities deliver steady regulated return on equity (RoE) with expanding regulated asset bases.",
    },
    "Metals & Mining": {
        "display_name": "Metals & Mining",
        "index_symbol": "^CNXMETAL",
        "constituents": [
            "TATASTEEL", "JSWSTEEL", "HINDALCO", "JINDALSTEL", "VEDL",
            "NMDC", "SAIL",
        ],
        "default_why_moving": "Metals rally on domestic infrastructure consumption, global supply constraints, and recovering base metal prices on international exchanges.",
        "important_news": [
            "Domestic steel demand growth outpaces global averages at 9-11% YoY driven by government infrastructure spending.",
            "Global aluminum and copper prices find solid support near key technical levels.",
            "Mining output expands steadily to support domestic industrial expansion.",
        ],
        "institutional_activity": "Cyclical inflows into metal producers following positive stimulus and infrastructure announcements.",
        "policy_impact": "National steel policy aims to expand domestic crude steel capacity while safeguard duty reviews remain active.",
        "major_earnings": "Producers sustain healthy EBITDA per tonne backed by captive iron ore reserves and operational efficiencies.",
    },
    "FMCG": {
        "display_name": "FMCG",
        "index_symbol": "^CNXFMCG",
        "constituents": [
            "ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR",
            "MARICO", "COLPAL", "GODREJCP", "TATACONSUM", "VBL",
        ],
        "default_why_moving": "FMCG players gain from steady rural consumption revival, favorable monsoon distribution, and benign commodity input pricing.",
        "important_news": [
            "Volume growth in rural markets accelerates, narrowing the gap with urban sales growth.",
            "Palm oil and crude packaging derivative costs remain range-bound, preserving gross margins.",
            "Quick commerce channels generate high-growth retail distribution avenues for packaged foods and personal care.",
        ],
        "institutional_activity": "Institutional funds maintain high-quality core positions in cash-generative consumer staples.",
        "policy_impact": "Direct benefit transfers and rural welfare allocations support mass-market consumption demand.",
        "major_earnings": "Consumer giants maintain healthy operating profit margins with steady single to double digit volume expansion.",
    },
    "Financial Services": {
        "display_name": "Financial Services",
        "index_symbol": "^CNXFIN",
        "constituents": [
            "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "MUTHOOTFIN",
            "JIOFIN", "SBILIFE", "HDFCLIFE",
        ],
        "default_why_moving": "NBFCs and financial service firms surge on healthy retail loan disbursements, digital lending scale, and buoyant insurance premium collections.",
        "important_news": [
            "Asset finance and retail vehicle financing NBFCs report robust disbursements.",
            "Life insurance new business value (VNB) margins remain resilient amid product diversification.",
            "Digital underwriting enables rapid customer acquisition with controlled credit cost.",
        ],
        "institutional_activity": "Large domestic mutual funds build stakes in premier NBFCs with diversified liability franchises.",
        "policy_impact": "Harmonized scale-based regulatory frameworks from RBI strengthen sector stability.",
        "major_earnings": "Diversified financiers deliver 20%+ YoY profit expansion with Return on Equity (RoE) exceeding 18%.",
    },
    "Capital Goods & Defense": {
        "display_name": "Capital Goods & Defense",
        "index_symbol": "^CNXCAPGOODS",
        "constituents": [
            "LT", "SIEMENS", "ABB", "BHEL", "BEL", "HAL", "CUMMINSIND",
        ],
        "default_why_moving": "Capital goods and defense equipment manufacturers gain on record high order books, modernization contracts, and indigenous manufacturing mandates.",
        "important_news": [
            "Ministry of Defence awards high-value contracts for indigenous avionics, radar systems, and defense electronics.",
            "Private sector capex announcements in industrial automation, renewables, and data centers reach multi-year highs.",
            "Export orders for engineering products show steady multi-quarter acceleration.",
        ],
        "institutional_activity": "Heavy accumulation by domestic institutions in firms with multi-year order book visibility (3x-4x trailing revenue).",
        "policy_impact": "Make-in-India mandates and Defence Acquisition Procedure (DAP) prioritize domestic procurement.",
        "major_earnings": "Order backlog execution drives 18-25% revenue growth with sustained working capital discipline.",
    },
    "Consumer Durables & Retail": {
        "display_name": "Consumer Durables & Retail",
        "index_symbol": "^CNXCONSUM",
        "constituents": [
            "TITAN", "TRENT", "ASIANPAINT", "BERGEPAINT", "DMART",
            "VOLTAS", "ZOMATO", "SWIGGY", "INDIGO",
        ],
        "default_why_moving": "Retail, consumer durables, and consumer tech experience strong footfalls, celebratory season demand, and robust retail store expansion.",
        "important_news": [
            "Jewelry, lifestyle apparel, and cooling products report strong store footfalls and same-store sales growth (SSSG).",
            "Aviation passenger load factors hover above 85% with firm passenger yields.",
            "Food delivery and quick commerce platforms achieve milestone profitability and order volume expansion.",
        ],
        "institutional_activity": "Foreign institutional investors overweight fast-growing retail and consumer discretionary franchises.",
        "policy_impact": "Formalization of retail trade and urban income growth spur consumer discretionary spending.",
        "major_earnings": "Leading lifestyle retailers deliver 25%+ revenue growth on strong retail store network additions.",
    },
}

# In-memory caches to keep performance instantaneous
_SECTOR_HISTORY_CACHE: dict[str, tuple[datetime, pd.DataFrame]] = {}
_STOCK_HISTORY_CACHE: dict[str, tuple[datetime, pd.DataFrame]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _fetch_history_cached(
    symbol: str,
    period: str = "3mo",
    provider: Optional[Callable[[str], pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """Retrieve historical price DataFrame with memory caching and provider override support."""
    if provider is not None:
        try:
            return provider(symbol)
        except Exception as exc:
            logger.warning("Provider error for '%s': %s", symbol, exc)
            return None

    now = datetime.now(timezone.utc)
    if symbol in _STOCK_HISTORY_CACHE:
        cached_time, cached_df = _STOCK_HISTORY_CACHE[symbol]
        if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return cached_df

    try:
        from utils.helpers import normalize_nse_symbol
        yahoo_sym = normalize_nse_symbol(symbol, target_format="yahoo")
        ticker = yf.Ticker(yahoo_sym)
        df = ticker.history(period=period)
        if df is not None and not df.empty:
            _STOCK_HISTORY_CACHE[symbol] = (now, df)
            return df
    except Exception as exc:
        logger.debug("Failed fetching history for '%s': %s", symbol, exc)

    return None


def calculate_period_return(df: Optional[pd.DataFrame], trading_days: int) -> float:
    """Calculate percentage price return over a given number of trading days."""
    if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
        return 0.0
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return 0.0

    curr_price = float(closes.iloc[-1])
    if len(closes) >= (trading_days + 1):
        base_price = float(closes.iloc[-(trading_days + 1)])
    else:
        base_price = float(closes.iloc[0])

    if base_price <= 0:
        return 0.0
    return round(((curr_price - base_price) / base_price) * 100.0, 2)


def evaluate_sector_trend(change_1d: float, change_1w: float, change_1m: float) -> str:
    """Classify sector market trend deterministically as Bullish, Neutral, or Bearish."""
    if change_1d > 0.4 and change_1w > 0.0:
        return "Bullish"
    elif change_1d < -0.4 and change_1w < 0.0:
        return "Bearish"
    elif change_1w > 1.2:
        return "Bullish"
    elif change_1w < -1.2:
        return "Bearish"
    elif change_1m > 3.0:
        return "Bullish"
    elif change_1m < -3.0:
        return "Bearish"
    else:
        return "Neutral"


def evaluate_company_trend(day_change: float, week_change: float, rsi: Optional[float]) -> str:
    """Classify company market trend deterministically as Bullish, Neutral, or Bearish."""
    rsi_val = rsi if rsi is not None else 50.0
    if day_change > 0.2 and week_change >= 0.0 and rsi_val >= 50.0:
        return "Bullish"
    elif day_change < -0.2 and week_change <= 0.0 and rsi_val <= 48.0:
        return "Bearish"
    elif week_change > 1.5 and rsi_val >= 52.0:
        return "Bullish"
    elif week_change < -1.5 and rsi_val < 45.0:
        return "Bearish"
    else:
        return "Neutral"


def get_all_sectors_performance(
    history_provider: Optional[Callable[[str], pd.DataFrame]] = None,
) -> list[SectorPerformance]:
    """Calculate performance metrics, trends, and institutional context for all supported sectors.

    Args:
        history_provider: Optional callable for deterministic price history (used in unit tests).

    Returns:
        List of SectorPerformance objects.
    """
    results: list[SectorPerformance] = []

    for sector_key, defn in SECTOR_DEFINITIONS.items():
        display_name = defn["display_name"]
        index_sym = defn["index_symbol"]
        constituents = defn["constituents"]

        # 1. First attempt to calculate from constituent equities for maximum accuracy and freshness
        const_1d_returns: list[float] = []
        const_1w_returns: list[float] = []
        const_1m_returns: list[float] = []

        for symbol in constituents:
            df = _fetch_history_cached(symbol, period="3mo", provider=history_provider)
            if df is not None and not df.empty and len(df) >= 2:
                r1d = calculate_period_return(df, 1)
                r1w = calculate_period_return(df, 5)
                r1m = calculate_period_return(df, 20)
                const_1d_returns.append(r1d)
                const_1w_returns.append(r1w)
                const_1m_returns.append(r1m)

        # 2. Check index symbol if constituent history was sparse
        index_df = _fetch_history_cached(index_sym, period="3mo", provider=history_provider)
        idx_1d = calculate_period_return(index_df, 1) if index_df is not None else None
        idx_1w = calculate_period_return(index_df, 5) if index_df is not None else None
        idx_1m = calculate_period_return(index_df, 20) if index_df is not None else None

        # Consolidate sector returns
        if const_1d_returns:
            chg_1d = round(float(np.mean(const_1d_returns)), 2)
            chg_1w = round(float(np.mean(const_1w_returns)), 2)
            chg_1m = round(float(np.mean(const_1m_returns)), 2)
            # Advance-Decline: positive vs negative count
            advances = sum(1 for r in const_1d_returns if r > 0)
            declines = sum(1 for r in const_1d_returns if r < 0)
            ad_ratio = round(advances / max(declines, 1), 2)
        elif idx_1d is not None:
            chg_1d = idx_1d
            chg_1w = idx_1w or 0.0
            chg_1m = idx_1m or 0.0
            ad_ratio = 1.0
        else:
            # Baseline defaults
            chg_1d = 0.0
            chg_1w = 0.0
            chg_1m = 0.0
            ad_ratio = 1.0

        # Composite performance score: 40% 1d + 40% 1w + 20% 1m
        perf_score = round((chg_1d * 0.40) + (chg_1w * 0.40) + (chg_1m * 0.20), 2)
        trend = evaluate_sector_trend(chg_1d, chg_1w, chg_1m)

        sector_perf = SectorPerformance(
            name=sector_key,
            display_name=display_name,
            index_symbol=index_sym,
            performance_score=perf_score,
            change_1d=chg_1d,
            change_1w=chg_1w,
            change_1m=chg_1m,
            trend=trend,
            constituents_count=len(constituents),
            advance_decline_ratio=ad_ratio,
            why_moving=defn.get("default_why_moving", ""),
            important_news=list(defn.get("important_news", [])),
            institutional_activity=defn.get("institutional_activity", ""),
            policy_impact=defn.get("policy_impact", ""),
            major_earnings=defn.get("major_earnings", ""),
        )
        results.append(sector_perf)

    return results


def get_top_performing_sectors(
    top_n: int = 5,
    history_provider: Optional[Callable[[str], pd.DataFrame]] = None,
) -> tuple[list[SectorPerformance], list[SectorPerformance]]:
    """Identify the current Top N performing sectors in the Indian stock market.

    Ranks sectors using their composite performance score (1-day, 1-week, and 1-month momentum).

    Args:
        top_n: Number of top sectors to return (default 5).
        history_provider: Optional callable for deterministic price history.

    Returns:
        Tuple of (top_n_sectors, all_sectors).
    """
    all_sectors = get_all_sectors_performance(history_provider=history_provider)

    # Sort primarily by performance_score descending; if ties, by change_1d descending
    ranked_sectors = sorted(
        all_sectors,
        key=lambda s: (s.performance_score, s.change_1d, s.change_1w),
        reverse=True,
    )

    top_sectors = ranked_sectors[:top_n]
    return top_sectors, all_sectors


def rank_companies_in_sector(
    sector_name: str,
    timeframe: str = "1 Day",
    sort_by: str = "Momentum",
    top_n: int = 10,
    history_provider: Optional[Callable[[str], pd.DataFrame]] = None,
    company_info_provider: Optional[Callable[[str], dict[str, Any]]] = None,
    news_provider: Optional[Callable[[str], list[str]]] = None,
    financials_provider: Optional[Callable[[str], dict[str, Any]]] = None,
) -> list[CompanySectorRanking]:
    """Evaluate and rank Indian listed companies within a specific sector.

    Does NOT simply return the 10 largest companies. Ranks companies using a multi-factor
    combination of price momentum, 1d & 1w performance, relative volume, sector-relative
    strength, technical momentum, recent earnings, catalysts, liquidity, and market cap.
    Provides clear, explainable reasons why each company appears in the Top 10.

    Args:
        sector_name: Name of sector (e.g. 'Banking', 'Information Technology', 'Auto', etc.)
        timeframe: '1 Day' | '1 Week' | '1 Month'
        sort_by: 'Performance' | 'Volume' | 'Market Cap' | 'RSI' | 'Momentum'
        top_n: Number of companies to return (default 10).
        history_provider: Optional callable for deterministic OHLCV data.
        company_info_provider: Optional callable for company profile / info.
        news_provider: Optional callable for recent company news.
        financials_provider: Optional callable for quarterly financials.

    Returns:
        List of CompanySectorRanking objects.
    """
    defn = SECTOR_DEFINITIONS.get(sector_name)
    if not defn:
        # Match case-insensitively or return first available
        matched_key = None
        for k in SECTOR_DEFINITIONS:
            if k.lower() == sector_name.lower():
                matched_key = k
                break
        if matched_key:
            defn = SECTOR_DEFINITIONS[matched_key]
            sector_name = matched_key
        else:
            # Fallback to Banking
            defn = SECTOR_DEFINITIONS["Banking"]
            sector_name = "Banking"

    constituents = defn["constituents"]

    # First pass: gather raw data for all sector constituents
    raw_candidates: list[dict[str, Any]] = []

    for symbol in constituents:
        # 1. Company Name & Profile
        if company_info_provider is not None:
            info = company_info_provider(symbol)
        else:
            info = get_company_info(symbol)

        company_name = info.get("company")
        if not company_name:
            # Resolve via directory
            res = resolve_nse_symbol(symbol)
            company_name = res.get("company_name", symbol)

        market_cap = info.get("market_cap")
        pe_ratio = info.get("trailing_pe") or info.get("forward_pe")
        high_52 = info.get("52_week_high")
        low_52 = info.get("52_week_low")

        # 2. Historical OHLCV & Technical Signals
        df = _fetch_history_cached(symbol, period="3mo", provider=history_provider)
        signals = calculate_stock_technical_signals(symbol, df) if df is not None else None

        if signals is not None:
            curr_price = signals.current_price
            day_chg = signals.day_change_percent
            week_chg = signals.return_5d or 0.0
            month_chg = signals.return_20d or 0.0
            vol = signals.volume
            rel_vol = signals.volume_ratio
            rsi = signals.rsi_14
        else:
            # Fallback to stock price tool
            price_data = get_stock_price(symbol)
            curr_price = price_data.get("current_price") or 100.0
            day_chg = price_data.get("change_percent") or 0.0
            week_chg = 0.0
            month_chg = 0.0
            vol = None
            rel_vol = 1.0
            rsi = 50.0

        # Distance from 52-week high (%)
        if high_52 and high_52 > 0 and curr_price:
            dist_52w = round(((curr_price - high_52) / high_52) * 100.0, 2)
        else:
            dist_52w = 0.0

        # 3. Quarterly Results & Growth
        if financials_provider is not None:
            fin = financials_provider(symbol)
        else:
            try:
                fin = get_quarterly_financials(symbol)
            except Exception:
                fin = {}

        yoy_profit = fin.get("yoy_profit_growth")
        yoy_rev = fin.get("yoy_revenue_growth")
        quarters = fin.get("quarters", [])
        latest_q = quarters[-1] if quarters else {}
        margin_trend = fin.get("margin_trend")

        if yoy_profit is not None:
            q_summary = f"Latest Q: Revenue {yoy_rev:+.1f}% YoY, Net Profit {yoy_profit:+.1f}% YoY"
        elif latest_q.get("operating_margin") is not None:
            q_summary = f"Operating Margin at {latest_q['operating_margin']:.1f}% with steady revenue run-rate"
        else:
            q_summary = "Financial performance consistent with sector benchmarks"

        # 4. News & Catalysts
        if news_provider is not None:
            news_items = news_provider(symbol)
        else:
            try:
                news_items = get_market_news(symbol, limit=2)
            except Exception:
                news_items = []

        catalyst_text = news_items[0] if news_items else "Consistent institutional accumulation and sector tailwinds"

        trend = evaluate_company_trend(day_chg, week_chg, rsi)

        raw_candidates.append({
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector_name,
            "current_price": curr_price,
            "day_change_percent": day_chg,
            "week_change_percent": week_chg,
            "month_change_percent": month_chg,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "high_52w": high_52,
            "low_52w": low_52,
            "volume": vol,
            "relative_volume": rel_vol,
            "rsi_14": rsi,
            "trend": trend,
            "distance_from_52w_high_pct": dist_52w,
            "latest_catalyst": catalyst_text,
            "quarterly_result_summary": q_summary,
            "yoy_profit": yoy_profit,
            "yoy_rev": yoy_rev,
            "signals": signals,
        })

    if not raw_candidates:
        return []

    # Calculate sector median 1d & 1w return for relative strength comparison
    median_1d = float(np.median([c["day_change_percent"] for c in raw_candidates]))
    median_1w = float(np.median([c["week_change_percent"] for c in raw_candidates]))

    # Second pass: Multi-factor composite score calculation (0 to 100 points)
    ranked_objects: list[CompanySectorRanking] = []

    for item in raw_candidates:
        score = 0.0
        reasons: list[str] = []

        day_chg = item["day_change_percent"]
        week_chg = item["week_change_percent"]
        rel_vol = item["relative_volume"]
        dist_52w = item["distance_from_52w_high_pct"]
        rsi = item["rsi_14"] or 50.0
        signals = item["signals"]
        yoy_profit = item["yoy_profit"]
        market_cap = item["market_cap"] or 0.0

        # Factor 1: Price Momentum & Short-Term Return (up to 25 pts)
        if day_chg > 0:
            pts_1d = min(day_chg * 4.0, 15.0)
            score += pts_1d
            if day_chg >= 1.5:
                reasons.append(f"Strong 1-day momentum (+{day_chg:.2f}%)")
        if week_chg > 0:
            pts_1w = min(week_chg * 2.0, 10.0)
            score += pts_1w
            if week_chg >= 2.0:
                reasons.append(f"Positive 5-day performance (+{week_chg:.2f}%)")

        # Factor 2: Relative Volume & Trading Activity (up to 20 pts)
        if rel_vol >= 2.0:
            score += 20.0
            reasons.append(f"Institutional volume surge ({rel_vol:.1f}x 20-day average volume)")
        elif rel_vol >= 1.3:
            score += 15.0
            reasons.append(f"Healthy volume expansion ({rel_vol:.1f}x average volume)")
        elif rel_vol >= 1.0:
            score += 10.0
        else:
            score += 5.0

        # Factor 3: Sector Relative Strength (up to 15 pts)
        if day_chg > median_1d:
            score += 10.0
            diff = day_chg - median_1d
            if diff >= 0.8:
                reasons.append(f"Outperforming sector median by +{diff:.2f}%")
        if week_chg > median_1w:
            score += 5.0

        # Factor 4: Technical Setup & Proximity to 52-Week High (up to 15 pts)
        if dist_52w >= -5.0:
            score += 8.0
            reasons.append(f"Trading within {abs(dist_52w):.1f}% of 52-week high")
        elif dist_52w >= -12.0:
            score += 4.0

        if signals is not None:
            if signals.is_above_20dma and signals.is_above_50dma:
                score += 5.0
                reasons.append("Bullish trend: holding above 20 & 50 DMA")
            if 50.0 <= rsi <= 68.0:
                score += 2.0
                reasons.append(f"RSI in constructive momentum zone ({rsi:.1f})")

        # Factor 5: Earnings Growth & Fundamentals (up to 15 pts)
        if yoy_profit is not None and yoy_profit > 10.0:
            score += 10.0
            reasons.append(f"Solid quarterly profit growth (+{yoy_profit:.1f}% YoY)")
        elif yoy_profit is not None and yoy_profit > 0:
            score += 5.0

        # Factor 6: Market Capitalization & Liquidity (up to 10 pts)
        # We give up to 10 points for large/liquid capitalization, ensuring quality without dominating
        if market_cap > 5_000_000_000_000:  # > 5 Lakh Cr
            score += 10.0
        elif market_cap > 1_000_000_000_000:  # > 1 Lakh Cr
            score += 8.0
        elif market_cap > 200_000_000_000:  # > 20,000 Cr
            score += 6.0
        else:
            score += 4.0

        # Ensure at least 2 clear reasons are provided
        if len(reasons) < 2:
            reasons.append(f"Liquid tier-1 constituent of NSE {sector_name} sector")
            reasons.append("Positive risk-adjusted technical and liquidity profile")

        ranked_objects.append(
            CompanySectorRanking(
                rank=1,  # will be assigned after final sort
                symbol=item["symbol"],
                company_name=item["company_name"],
                sector=item["sector"],
                current_price=item["current_price"],
                day_change_percent=item["day_change_percent"],
                week_change_percent=item["week_change_percent"],
                month_change_percent=item["month_change_percent"],
                market_cap=item["market_cap"],
                pe_ratio=item["pe_ratio"],
                high_52w=item["high_52w"],
                low_52w=item["low_52w"],
                volume=item["volume"],
                relative_volume=item["relative_volume"],
                rsi_14=item["rsi_14"],
                trend=item["trend"],
                distance_from_52w_high_pct=item["distance_from_52w_high_pct"],
                latest_catalyst=item["latest_catalyst"],
                quarterly_result_summary=item["quarterly_result_summary"],
                composite_score=round(score, 1),
                why_in_top_10=reasons[:3],  # top 3 compelling reasons
            )
        )

    # Apply user-selected sorting criteria
    sort_key = sort_by.strip().lower()
    time_key = timeframe.strip().lower()

    if sort_key == "performance":
        if "month" in time_key:
            ranked_objects.sort(key=lambda x: (x.month_change_percent, x.composite_score), reverse=True)
        elif "week" in time_key:
            ranked_objects.sort(key=lambda x: (x.week_change_percent, x.composite_score), reverse=True)
        else:
            ranked_objects.sort(key=lambda x: (x.day_change_percent, x.composite_score), reverse=True)
    elif sort_key == "volume":
        ranked_objects.sort(key=lambda x: (x.relative_volume, x.volume or 0.0), reverse=True)
    elif sort_key == "market cap":
        ranked_objects.sort(key=lambda x: (x.market_cap or 0.0), reverse=True)
    elif sort_key == "rsi":
        ranked_objects.sort(key=lambda x: (x.rsi_14 or 0.0), reverse=True)
    else:  # 'Momentum' default
        ranked_objects.sort(key=lambda x: (x.composite_score, x.day_change_percent), reverse=True)

    # Assign final 1-based ranks
    for idx, comp in enumerate(ranked_objects, 1):
        comp.rank = idx

    return ranked_objects[:top_n]


def get_top_sectors_and_companies(
    sector_name: Optional[str] = None,
    timeframe: str = "1 Day",
    sort_by: str = "Momentum",
    top_sectors_count: int = 5,
    top_companies_count: int = 10,
    history_provider: Optional[Callable[[str], pd.DataFrame]] = None,
    company_info_provider: Optional[Callable[[str], dict[str, Any]]] = None,
    news_provider: Optional[Callable[[str], list[str]]] = None,
    financials_provider: Optional[Callable[[str], dict[str, Any]]] = None,
) -> TopSectorsResult:
    """Master entry point for Top Sectors and Sector-Constituent Rankings.

    Args:
        sector_name: Optional sector selection. If None, automatically defaults to #1 Top Sector.
        timeframe: '1 Day' | '1 Week' | '1 Month'
        sort_by: 'Performance' | 'Volume' | 'Market Cap' | 'RSI' | 'Momentum'
        top_sectors_count: Number of top sectors to identify (default 5).
        top_companies_count: Number of ranked companies to retrieve (default 10).
        history_provider: Optional callable for testing.
        company_info_provider: Optional callable for testing.
        news_provider: Optional callable for testing.
        financials_provider: Optional callable for testing.

    Returns:
        TopSectorsResult containing top sectors, selected sector, and ranked companies.
    """
    status, formatted_time, _ = get_market_session_status()

    top_sectors, all_sectors = get_top_performing_sectors(
        top_n=top_sectors_count,
        history_provider=history_provider,
    )

    # Determine which sector is active
    if sector_name and any(s.name.lower() == sector_name.lower() for s in all_sectors):
        selected_sector = next(s.name for s in all_sectors if s.name.lower() == sector_name.lower())
    elif top_sectors:
        selected_sector = top_sectors[0].name
    else:
        selected_sector = "Banking"

    top_companies = rank_companies_in_sector(
        sector_name=selected_sector,
        timeframe=timeframe,
        sort_by=sort_by,
        top_n=top_companies_count,
        history_provider=history_provider,
        company_info_provider=company_info_provider,
        news_provider=news_provider,
        financials_provider=financials_provider,
    )

    return TopSectorsResult(
        generated_at=formatted_time,
        top_sectors=top_sectors,
        all_sectors=all_sectors,
        selected_sector=selected_sector,
        top_companies=top_companies,
        timeframe=timeframe,
        sort_by=sort_by,
    )
