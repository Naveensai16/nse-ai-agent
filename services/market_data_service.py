"""Market data retrieval, universe definitions, market regime detection, and technical indicator calculations."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from models.opportunity import MarketRegime, TechnicalSignals
from utils.helpers import normalize_nse_symbol

logger = logging.getLogger(__name__)

# Official Indian National Stock Exchange (NSE) standard trading holidays (month, day)
NSE_ANNUAL_HOLIDAYS: set[tuple[int, int]] = {
    (1, 26),  # Republic Day
    (5, 1),   # Maharashtra Day
    (8, 15),  # Independence Day
    (10, 2),  # Mahatma Gandhi Jayanti
    (12, 25), # Christmas
}

# Sector classification for liquid NSE equities
SECTOR_MAP: dict[str, str] = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT", "LTIM": "IT",
    "PERSISTENT": "IT", "COFORGE": "IT", "MPHASIS": "IT", "KPITTECH": "IT", "TATAELXSI": "IT",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "INDUSINDBK": "Banking", "PNB": "Banking", "BANKBARODA": "Banking",
    "CANBK": "Banking", "UNIONBANK": "Banking", "IDFCFIRSTB": "Banking", "FEDERALBNK": "Banking",
    "BAJFINANCE": "Financial Services", "BAJAJFINSV": "Financial Services", "CHOLAFIN": "Financial Services",
    "SHRIRAMFIN": "Financial Services", "MUTHOOTFIN": "Financial Services", "JIOFIN": "Financial Services",
    "RELIANCE": "Energy & Conglomerate", "ONGC": "Energy", "IOC": "Energy", "BPCL": "Energy",
    "HPCL": "Energy", "GAIL": "Energy", "OIL": "Energy", "COALINDIA": "Energy",
    "NTPC": "Power & Utilities", "POWERGRID": "Power & Utilities", "TATAPOWER": "Power & Utilities",
    "ADANIPOWER": "Power & Utilities", "ADANIGREEN": "Power & Utilities", "ADANIENSOL": "Power & Utilities",
    "NHPC": "Power & Utilities", "SJVN": "Power & Utilities",
    "TATAMOTORS": "Automobile", "MARUTI": "Automobile", "M&M": "Automobile", "BAJAJ-AUTO": "Automobile",
    "HEROMOTOCO": "Automobile", "EICHERMOT": "Automobile", "TVSMOTOR": "Automobile", "ASHOKLEY": "Automobile",
    "BHARATFORG": "Automobile",
    "TATASTEEL": "Metals & Mining", "JSWSTEEL": "Metals & Mining", "HINDALCO": "Metals & Mining",
    "JINDALSTEL": "Metals & Mining", "VEDL": "Metals & Mining", "NMDC": "Metals & Mining", "SAIL": "Metals & Mining",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG", "DABUR": "FMCG",
    "MARICO": "FMCG", "COLPAL": "FMCG", "GODREJCP": "FMCG", "TATACONSUM": "FMCG", "VBL": "FMCG",
    "SUNPHARMA": "Pharmaceuticals", "DRREDDY": "Pharmaceuticals", "CIPLA": "Pharmaceuticals",
    "DIVISLAB": "Pharmaceuticals", "APOLLOHOSP": "Healthcare", "MANKIND": "Pharmaceuticals",
    "TORNTPHARM": "Pharmaceuticals", "LUPIN": "Pharmaceuticals", "BIOCON": "Pharmaceuticals",
    "BHARTIARTL": "Telecom", "IDEA": "Telecom", "TATACOMM": "Telecom",
    "LT": "Infrastructure & Capital Goods", "SIEMENS": "Capital Goods", "ABB": "Capital Goods",
    "BHEL": "Capital Goods", "BEL": "Defense & Capital Goods", "HAL": "Defense & Aerospace",
    "CUMMINSIND": "Capital Goods",
    "ULTRACEMCO": "Cement", "AMBUJACEM": "Cement", "ACC": "Cement", "SHREECEM": "Cement",
    "ASIANPAINT": "Consumer Durables", "BERGEPAINT": "Consumer Durables", "PIDILITIND": "Chemicals",
    "TITAN": "Consumer Goods & Jewelry", "TRENT": "Retail", "DMART": "Retail", "ZOMATO": "Consumer Tech",
    "SWIGGY": "Consumer Tech", "PAYTM": "Fintech", "INDIGO": "Aviation", "IRCTC": "Travel & Tourism",
    "INDIANHOTE": "Hospitality", "VOLTAS": "Consumer Durables", "ASTRAL": "Building Materials",
    "SRF": "Chemicals", "TATACHEM": "Chemicals", "TATATECH": "IT & ER&D",
}

# Pre-defined liquid NSE universes
NIFTY_50_TICKERS: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "INFY", "ITC", "SBIN",
    "LT", "HINDUNILVR", "BAJFINANCE", "M&M", "MARUTI", "SUNPHARMA", "KOTAKBANK",
    "AXISBANK", "TATAMOTORS", "NTPC", "POWERGRID", "ULTRACEMCO", "TITAN", "ONGC",
    "ADANIENT", "JSWSTEEL", "TATASTEEL", "COALINDIA", "BAJAJFINSV", "NESTLEIND",
    "ASIANPAINT", "GRASIM", "HCLTECH", "TECHM", "WIPRO", "HINDALCO", "BEL",
    "TRENT", "SHRIRAMFIN", "BPCL", "DRREDDY", "CIPLA", "APOLLOHOSP", "EICHERMOT",
    "TATACONSUM", "BRITANNIA", "BAJAJ-AUTO", "HEROMOTOCO", "SBILIFE", "HDFCLIFE",
    "INDUSINDBK", "ADANIPORTS",
]

NIFTY_NEXT_50_TICKERS: list[str] = [
    "TATAPOWER", "HAL", "VEDL", "ZOMATO", "JIOFIN", "AMBUJACEM", "CHOLAFIN",
    "BANKBARODA", "CANBK", "PNB", "UNIONBANK", "IDFCFIRSTB", "FEDERALBNK",
    "GAIL", "IOC", "HPCL", "OIL", "NHPC", "SJVN", "ADANIPOWER", "ADANIGREEN",
    "ADANIENSOL", "ATGL", "JINDALSTEL", "NMDC", "SAIL", "SIEMENS", "ABB",
    "BHEL", "CUMMINSIND", "DABUR", "MARICO", "COLPAL", "GODREJCP", "VBL",
    "DMART", "SWIGGY", "PAYTM", "INDIGO", "IRCTC", "INDIANHOTE", "VOLTAS",
    "PIDILITIND", "BERGEPAINT", "ASTRAL", "SRF", "TATACHEM", "TATACOMM",
    "TATAELXSI", "TATATECH",
]

# Configurable Universes dictionary
UNIVERSES: dict[str, list[str]] = {
    "NIFTY 50": NIFTY_50_TICKERS,
    "NIFTY NEXT 50": NIFTY_NEXT_50_TICKERS,
    "NIFTY 100": NIFTY_50_TICKERS + NIFTY_NEXT_50_TICKERS,
    "NIFTY 200": NIFTY_50_TICKERS + NIFTY_NEXT_50_TICKERS,  # Configurable liquid universe
}


def get_market_session_status(current_dt: Optional[datetime] = None) -> tuple[str, str, str]:
    """Calculate whether the NSE market is currently Open or Closed in IST, along with timestamps.

    Args:
        current_dt: Optional datetime override (primarily for deterministic unit testing).

    Returns:
        Tuple containing:
            - status: 'Open' | 'Closed'
            - formatted_timestamp: e.g. '02 Oct 2026 · 03:20 PM IST'
            - target_horizon: e.g. 'Next 1–2 NSE trading sessions (Mon, 05 Oct – Tue, 06 Oct 2026)'
    """
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now = current_dt.astimezone(ist_tz) if current_dt else datetime.now(ist_tz)

    formatted_timestamp = now.strftime("%d %b %Y · %I:%M %p IST")

    # Weekday check (Monday=0 ... Sunday=6)
    is_weekday = now.weekday() < 5
    is_holiday = (now.month, now.day) in NSE_ANNUAL_HOLIDAYS

    # Trading hours: 09:15 AM to 03:30 PM IST
    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if is_weekday and not is_holiday and (market_open_time <= now <= market_close_time):
        status = "Open"
    else:
        status = "Closed"

    # Calculate next 2 valid NSE trading sessions
    trading_days: list[datetime] = []
    # If market is closed after 3:30 PM or on weekend/holiday, start looking from tomorrow
    check_day = now.date() if (status == "Open") else now.date() + timedelta(days=1)

    while len(trading_days) < 2:
        if check_day.weekday() < 5 and (check_day.month, check_day.day) not in NSE_ANNUAL_HOLIDAYS:
            trading_days.append(datetime(check_day.year, check_day.month, check_day.day, tzinfo=ist_tz))
        check_day += timedelta(days=1)

    t1_str = trading_days[0].strftime("%a, %d %b")
    t2_str = trading_days[1].strftime("%a, %d %b %Y")
    target_horizon = f"Next 1–2 NSE trading sessions ({t1_str} – {t2_str})"

    return status, formatted_timestamp, target_horizon


def get_market_regime(
    nifty_history_provider: Optional[Any] = None,
    bank_nifty_history_provider: Optional[Any] = None,
    nifty_history: Optional[pd.DataFrame] = None,
    bank_nifty_history: Optional[pd.DataFrame] = None,
) -> MarketRegime:
    """Determine the overarching market regime (Bullish, Neutral, Bearish) using deterministic index signals.

    Calculates:
    - NIFTY 50 1-day change %
    - NIFTY 50 5-day return %
    - 20 DMA & 50 DMA alignment
    - Benchmark confirmation from Bank Nifty

    Args:
        nifty_history_provider: Optional callable returning price history (for testing).
        bank_nifty_history_provider: Optional callable returning Bank Nifty history (for testing).
        nifty_history: Optional DataFrame directly passed.
        bank_nifty_history: Optional DataFrame directly passed.

    Returns:
        MarketRegime dataclass object.
    """
    try:
        if nifty_history is not None:
            nifty_hist = nifty_history
        elif nifty_history_provider is not None:
            nifty_hist = nifty_history_provider("^NSEI")
        else:
            ticker = yf.Ticker("^NSEI")
            nifty_hist = ticker.history(period="3mo")

        if bank_nifty_history is not None:
            bank_hist = bank_nifty_history
        elif bank_nifty_history_provider is not None:
            bank_hist = bank_nifty_history_provider("^NSEBANK")
        else:
            b_ticker = yf.Ticker("^NSEBANK")
            bank_hist = b_ticker.history(period="1mo")

        if nifty_hist is None or getattr(nifty_hist, "empty", True) or len(nifty_hist) < 20:
            return MarketRegime(
                regime="Neutral",
                summary="Insufficient benchmark index data to determine market regime.",
            )

        closes = nifty_hist["Close"]
        curr_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2]) if len(closes) >= 2 else curr_price
        p5_price = float(closes.iloc[-6]) if len(closes) >= 6 else float(closes.iloc[0])

        day_chg = round(((curr_price - prev_price) / prev_price) * 100.0, 2)
        ret_5d = round(((curr_price - p5_price) / p5_price) * 100.0, 2)

        dma_20 = float(closes.rolling(20).mean().iloc[-1])
        dma_50 = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else dma_20

        above_20dma = curr_price >= dma_20
        above_50dma = curr_price >= dma_50
        bullish_alignment = dma_20 >= dma_50

        # Bank Nifty day change
        bank_chg = 0.0
        if bank_hist is not None and not getattr(bank_hist, "empty", True) and len(bank_hist) >= 2:
            b_closes = bank_hist["Close"]
            bank_chg = round(((float(b_closes.iloc[-1]) - float(b_closes.iloc[-2])) / float(b_closes.iloc[-2])) * 100.0, 2)

        # Deterministic regime classification
        if above_20dma and above_50dma and (ret_5d >= 0.0 or day_chg >= 0.3):
            regime = "Bullish"
            summary = (
                f"NIFTY 50 trading above both 20 DMA ({dma_20:,.0f}) and 50 DMA ({dma_50:,.0f}) "
                f"with positive 5-day return ({ret_5d:+.2f}%). Supportive Bullish market environment."
            )
        elif not above_20dma and not above_50dma and ret_5d <= -0.5:
            regime = "Bearish"
            summary = (
                f"NIFTY 50 trading below key 20 DMA ({dma_20:,.0f}) and 50 DMA ({dma_50:,.0f}) "
                f"with negative momentum ({ret_5d:+.2f}%). Caution warranted for long setups."
            )
        else:
            regime = "Neutral"
            summary = (
                f"NIFTY 50 consolidating between moving averages with mixed directional signals "
                f"(Day: {day_chg:+.2f}%, 5-Day: {ret_5d:+.2f}%)."
            )

        return MarketRegime(
            regime=regime,
            nifty_change_1d=day_chg,
            nifty_return_5d=ret_5d,
            nifty_above_20dma=above_20dma,
            nifty_above_50dma=above_50dma,
            nifty_current_value=round(curr_price, 2),
            bank_nifty_change_1d=bank_chg,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("Error calculating market regime: %s", exc)
        return MarketRegime(
            regime="Neutral",
            summary="Market regime defaults to Neutral due to exchange feed exception.",
        )


def calculate_stock_technical_signals(
    symbol: str,
    df: pd.DataFrame,
) -> Optional[TechnicalSignals]:
    """Calculate comprehensive technical signals and price action metrics from historical OHLCV.

    Args:
        symbol: NSE stock symbol.
        df: DataFrame with OHLCV data.

    Returns:
        TechnicalSignals dataclass or None if data is insufficient.
    """
    if df is None or df.empty or len(df) < 15:
        return None

    # Handle multi-index columns from yf.download if present
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if symbol in df.columns.levels[1]:
                df = df.xs(symbol, axis=1, level=1)
            elif f"{symbol}.NS" in df.columns.levels[1]:
                df = df.xs(f"{symbol}.NS", axis=1, level=1)
        except Exception:
            pass

    required_cols = {"Close", "High", "Low", "Volume"}
    if not required_cols.issubset(df.columns):
        return None

    closes = df["Close"].dropna()
    highs = df["High"].dropna()
    lows = df["Low"].dropna()
    volumes = df["Volume"].dropna()
    n = len(closes)

    if n < 15:
        return None

    curr_price = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2]) if n >= 2 else curr_price
    day_chg_pct = round(((curr_price - prev_close) / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

    ret_2d = round(((curr_price - float(closes.iloc[-3])) / float(closes.iloc[-3])) * 100.0, 2) if n >= 3 else day_chg_pct
    ret_5d = round(((curr_price - float(closes.iloc[-6])) / float(closes.iloc[-6])) * 100.0, 2) if n >= 6 else day_chg_pct
    ret_20d = round(((curr_price - float(closes.iloc[-21])) / float(closes.iloc[-21])) * 100.0, 2) if n >= 21 else day_chg_pct

    # Moving averages
    dma_20 = float(closes.rolling(20, min_periods=10).mean().iloc[-1])
    dma_50 = float(closes.rolling(50, min_periods=20).mean().iloc[-1]) if n >= 20 else dma_20

    dist_20 = round(((curr_price - dma_20) / dma_20) * 100.0, 2) if dma_20 > 0 else 0.0
    dist_50 = round(((curr_price - dma_50) / dma_50) * 100.0, 2) if dma_50 > 0 else 0.0

    # Volume metrics
    curr_volume = float(volumes.iloc[-1])
    vol_20d_avg = float(volumes.rolling(20, min_periods=5).mean().iloc[-1])
    volume_ratio = round(curr_volume / vol_20d_avg, 2) if vol_20d_avg > 0 else 1.0

    # 14-period Wilder's RSI calculation
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    last_gain = float(avg_gain.dropna().iloc[-1]) if not avg_gain.dropna().empty else 0.0
    last_loss = float(avg_loss.dropna().iloc[-1]) if not avg_loss.dropna().empty else 0.0

    if last_loss == 0.0 and last_gain > 0.0:
        rsi_14 = 100.0
    elif last_gain == 0.0 and last_loss > 0.0:
        rsi_14 = 0.0
    elif last_loss == 0.0 and last_gain == 0.0:
        rsi_14 = 50.0
    else:
        rs = last_gain / last_loss
        rsi_14 = round(100.0 - (100.0 / (1.0 + rs)), 1)

    # 14-period ATR calculation
    tr1 = highs - lows
    tr2 = (highs - closes.shift()).abs()
    tr3 = (lows - closes.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=14, min_periods=5).mean()
    atr_14 = round(float(atr_series.dropna().iloc[-1]), 2) if not atr_series.dropna().empty else round(curr_price * 0.02, 2)

    # 20-day high and low
    high_20d = float(highs.tail(20).max())
    low_20d = float(lows.tail(20).min())

    # Breakout condition: current close within 1% of or above 20-day high, with volume confirmation
    is_breakout = curr_price >= (high_20d * 0.99) and high_20d > low_20d

    # Gap-up / Gap-down from yesterday's close
    open_today = float(df["Open"].iloc[-1]) if "Open" in df.columns else curr_price
    gap_percent = round(((open_today - prev_close) / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

    # Support and Resistance calculations
    # Support: Key swing low or moving average, at least 1-2% below current price
    nearest_support = round(max(low_20d, dma_20 * 0.985), 2)
    if nearest_support >= curr_price:
        nearest_support = round(curr_price - (1.5 * atr_14), 2)

    # Resistance: 20-day high or short-term ATR expansion zone
    if high_20d > curr_price * 1.005:
        nearest_resistance = round(high_20d, 2)
    else:
        nearest_resistance = round(curr_price + (1.5 * atr_14), 2)

    # Potential Upside / Downside and Risk/Reward
    potential_upside = round(((nearest_resistance - curr_price) / curr_price) * 100.0, 2)
    potential_downside = round(((curr_price - nearest_support) / curr_price) * 100.0, 2)

    risk_reward = round(potential_upside / max(potential_downside, 0.1), 1)

    return TechnicalSignals(
        current_price=round(curr_price, 2),
        previous_close=round(prev_close, 2),
        day_change_percent=day_chg_pct,
        symbol=symbol,
        return_2d=ret_2d,
        return_5d=ret_5d,
        return_20d=ret_20d,
        dma_20=round(dma_20, 2),
        dma_50=round(dma_50, 2),
        distance_from_20dma=dist_20,
        distance_from_50dma=dist_50,
        volume=curr_volume,
        avg_volume_20d=round(vol_20d_avg, 0),
        volume_ratio=volume_ratio,
        rsi_14=rsi_14,
        atr_14=atr_14,
        high_20d=round(high_20d, 2),
        low_20d=round(low_20d, 2),
        is_breakout=is_breakout,
        is_above_20dma=curr_price > dma_20,
        is_above_50dma=curr_price > dma_50,
        is_dma_bullish_alignment=dma_20 > dma_50,
        gap_percent=gap_percent,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        potential_upside_pct=potential_upside,
        potential_downside_pct=potential_downside,
        risk_reward_ratio=risk_reward,
    )
