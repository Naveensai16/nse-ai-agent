"""Stock price retrieval tool using yfinance for NSE equities."""

import logging
import math
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from utils.helpers import normalize_nse_symbol

logger = logging.getLogger(__name__)


def _clean_number(val: Any) -> Optional[float]:
    """Convert a value to a sanitized float, returning None for NaN, Inf, or invalid values."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 2)
    except (ValueError, TypeError):
        return None


def _clean_str(val: Any) -> Optional[str]:
    """Convert a value to a non-empty string, returning None if empty or invalid."""
    if val is None:
        return None
    try:
        s = str(val).strip()
        return s if s else None
    except Exception:
        return None


def _safe_get_property(source: Any, *keys: str) -> Any:
    """Safely retrieve a value from a dictionary or object using multiple candidate keys."""
    if source is None:
        return None
    for key in keys:
        # 1. Try dict-like .get()
        if hasattr(source, "get"):
            try:
                val = source.get(key)
                if val is not None:
                    return val
            except Exception:
                pass
        # 2. Try subscripting source[key]
        try:
            val = source[key]
            if val is not None:
                return val
        except Exception:
            pass
        # 3. Try attribute getattr(source, key)
        try:
            val = getattr(source, key, None)
            if val is not None:
                return val
        except Exception:
            pass
    return None


def get_stock_price(symbol: str) -> dict:
    """Fetch real-time / latest stock price information for an NSE stock symbol.

    Uses yfinance to query price metrics via fast_info, info, and history fallbacks.
    Does not manufacture missing market data; missing or unavailable fields default to None.

    Args:
        symbol: Raw NSE symbol, e.g. 'TCS', 'reliance', 'INFY.NS', 'HDFCBANK-EQ'.

    Returns:
        Dictionary containing:
            - symbol: Clean NSE symbol (e.g. 'TCS') or None if invalid
            - yahoo_symbol: Yahoo symbol (e.g. 'TCS.NS') or None if invalid
            - company: Company name or None
            - current_price: Latest price or None
            - previous_close: Previous close price or None
            - change: Absolute price change (current - previous) or None
            - change_percent: Percentage price change or None
            - 52_week_high: 52-week high price or None
            - 52_week_low: 52-week low price or None
            - currency: Trading currency (e.g. 'INR') or None
    """
    result: dict = {
        "symbol": None,
        "yahoo_symbol": None,
        "company": None,
        "current_price": None,
        "previous_close": None,
        "change": None,
        "change_percent": None,
        "52_week_high": None,
        "52_week_low": None,
        "currency": None,
    }

    # 1. Validate and normalize symbol
    try:
        from services.symbol_resolver import resolve_nse_symbol
        res = resolve_nse_symbol(symbol)
        if res.get("symbol"):
            symbol = res["symbol"]
    except Exception:
        pass

    try:
        clean_symbol = normalize_nse_symbol(symbol, target_format="clean")
        yahoo_symbol = normalize_nse_symbol(symbol, target_format="yahoo")
    except (ValueError, TypeError) as exc:
        logger.warning("Symbol normalization failed for '%s': %s", symbol, exc)
        return result

    result["symbol"] = clean_symbol
    result["yahoo_symbol"] = yahoo_symbol

    # 2. Instantiate Ticker safely
    try:
        ticker = yf.Ticker(yahoo_symbol)
    except Exception as exc:
        logger.warning("Failed to initialize yfinance.Ticker for '%s': %s", yahoo_symbol, exc)
        return result

    # 3. Retrieve fast_info safely
    fast_info = None
    try:
        fast_info = ticker.fast_info
    except Exception as exc:
        logger.debug("fast_info unavailable for '%s': %s", yahoo_symbol, exc)

    # 4. Retrieve info safely
    info = None
    try:
        info_data = ticker.info
        if isinstance(info_data, dict):
            info = info_data
    except Exception as exc:
        logger.debug("info unavailable for '%s': %s", yahoo_symbol, exc)

    # 4b. Fallback to migrated ticker if primary ticker has no quote (e.g. corporate restructuring)
    YAHOO_MIGRATED_TICKERS = {"TATAMOTORS.NS": "TMCV.NS"}
    if yahoo_symbol in YAHOO_MIGRATED_TICKERS:
        has_valid_data = False
        if fast_info:
            try:
                has_valid_data = fast_info.last_price is not None
            except Exception:
                pass
        if not has_valid_data and isinstance(info, dict):
            has_valid_data = bool(info.get("shortName") or info.get("longName") or info.get("currentPrice"))

        if not has_valid_data:
            migrated_sym = YAHOO_MIGRATED_TICKERS[yahoo_symbol]
            try:
                m_ticker = yf.Ticker(migrated_sym)
                m_fast = getattr(m_ticker, "fast_info", None)
                m_info = getattr(m_ticker, "info", None)
                ticker = m_ticker
                fast_info = m_fast
                info = m_info if isinstance(m_info, dict) else None
            except Exception:
                pass

    # Extract company name (primarily from info)
    company = None
    if info:
        company = _clean_str(info.get("longName") or info.get("shortName"))
    result["company"] = company

    # Extract current price: fast_info -> info fallback
    current_price = None
    if fast_info:
        current_price = _clean_number(
            _safe_get_property(fast_info, "last_price", "lastPrice", "regular_market_price")
        )
    if current_price is None and info:
        current_price = _clean_number(
            info.get("currentPrice") or info.get("regularMarketPrice")
        )

    # Extract previous close: fast_info -> info fallback
    previous_close = None
    if fast_info:
        previous_close = _clean_number(
            _safe_get_property(
                fast_info,
                "previous_close",
                "previousClose",
                "regular_market_previous_close",
            )
        )
    if previous_close is None and info:
        previous_close = _clean_number(
            info.get("previousClose") or info.get("regularMarketPreviousClose")
        )

    # Extract 52-week high & low: fast_info -> info fallback
    high_52 = None
    low_52 = None
    if fast_info:
        high_52 = _clean_number(
            _safe_get_property(
                fast_info,
                "year_high",
                "yearHigh",
                "fifty_two_week_high",
                "fiftyTwoWeekHigh",
            )
        )
        low_52 = _clean_number(
            _safe_get_property(
                fast_info,
                "year_low",
                "yearLow",
                "fifty_two_week_low",
                "fiftyTwoWeekLow",
            )
        )
    if high_52 is None and info:
        high_52 = _clean_number(info.get("fiftyTwoWeekHigh"))
    if low_52 is None and info:
        low_52 = _clean_number(info.get("fiftyTwoWeekLow"))

    # Extract currency: fast_info -> info fallback
    currency = None
    if fast_info:
        currency = _clean_str(_safe_get_property(fast_info, "currency"))
    if currency is None and info:
        currency = _clean_str(info.get("currency"))

    # 5. History fallback if prices are still missing
    if current_price is None or previous_close is None:
        try:
            hist = ticker.history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                valid_closes = hist["Close"].dropna()
                if len(valid_closes) >= 1 and current_price is None:
                    current_price = _clean_number(valid_closes.iloc[-1])
                if len(valid_closes) >= 2 and previous_close is None:
                    previous_close = _clean_number(valid_closes.iloc[-2])
        except Exception as exc:
            logger.debug("history fallback failed for '%s': %s", yahoo_symbol, exc)

    # Assign resolved fields
    result["current_price"] = current_price
    result["previous_close"] = previous_close
    result["52_week_high"] = high_52
    result["52_week_low"] = low_52
    result["currency"] = currency

    # 6. Calculate change and change_percent only when both prices exist
    if current_price is not None and previous_close is not None:
        change = round(current_price - previous_close, 2)
        result["change"] = change
        if previous_close != 0:
            result["change_percent"] = round((change / previous_close) * 100, 2)
        else:
            result["change_percent"] = None
    else:
        result["change"] = None
        result["change_percent"] = None

    return result
