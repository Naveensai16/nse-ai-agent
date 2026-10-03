"""Stock price retrieval tool using yfinance for NSE equities."""

import logging
import math
import threading
import time
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from data.stock_master import INDIAN_STOCK_MASTER
from utils.helpers import normalize_nse_symbol

logger = logging.getLogger(__name__)

# Fast in-memory thread-safe TTL price cache (avoids repeated network trips)
_PRICE_CACHE: dict[str, tuple[float, dict]] = {}
_PRICE_CACHE_LOCK = threading.Lock()
_PRICE_CACHE_TTL = 90.0  # 90 seconds TTL


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

    # Check in-memory cache first
    now = time.time()
    sym_upper = (symbol or "").upper().replace(".NS", "").replace("-EQ", "").strip()
    with _PRICE_CACHE_LOCK:
        if sym_upper in _PRICE_CACHE:
            ts, cached_res = _PRICE_CACHE[sym_upper]
            if now - ts < _PRICE_CACHE_TTL:
                return cached_res.copy()

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

    # Check cache again under clean_symbol
    with _PRICE_CACHE_LOCK:
        if clean_symbol in _PRICE_CACHE:
            ts, cached_res = _PRICE_CACHE[clean_symbol]
            if now - ts < _PRICE_CACHE_TTL:
                return cached_res.copy()

    result["symbol"] = clean_symbol
    result["yahoo_symbol"] = yahoo_symbol

    # Instant company name from verified local master (0ms latency)
    company = INDIAN_STOCK_MASTER.get(clean_symbol, {}).get("company_name")
    result["company"] = company

    # 2. Instantiate Ticker safely
    try:
        ticker = yf.Ticker(yahoo_symbol)
    except Exception as exc:
        logger.warning("Failed to initialize yfinance.Ticker for '%s': %s", yahoo_symbol, exc)
        return result

    # 3. Retrieve fast_info (takes ~0.05s, lightweight)
    fast_info = None
    try:
        fast_info = ticker.fast_info
    except Exception as exc:
        logger.debug("fast_info unavailable for '%s': %s", yahoo_symbol, exc)

    # 4. Extract metrics from fast_info
    current_price = None
    previous_close = None
    high_52 = None
    low_52 = None
    currency = "INR"

    if fast_info:
        current_price = _clean_number(
            _safe_get_property(fast_info, "last_price", "lastPrice", "regular_market_price")
        )
        previous_close = _clean_number(
            _safe_get_property(
                fast_info,
                "previous_close",
                "previousClose",
                "regular_market_previous_close",
            )
        )
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
        currency = _clean_str(_safe_get_property(fast_info, "currency")) or "INR"

    # 5. Slow fallback ONLY if fast_info yielded no price
    if current_price is None:
        info = None
        try:
            info_data = ticker.info
            if isinstance(info_data, dict):
                info = info_data
        except Exception:
            pass

        if info:
            if not company:
                company = _clean_str(info.get("longName") or info.get("shortName"))
                result["company"] = company
            current_price = _clean_number(info.get("currentPrice") or info.get("regularMarketPrice"))
            if previous_close is None:
                previous_close = _clean_number(info.get("previousClose") or info.get("regularMarketPreviousClose"))
            if high_52 is None:
                high_52 = _clean_number(info.get("fiftyTwoWeekHigh"))
            if low_52 is None:
                low_52 = _clean_number(info.get("fiftyTwoWeekLow"))

    # 6. History fallback if prices are still missing
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
    result["company"] = company or clean_symbol
    result["current_price"] = current_price
    result["previous_close"] = previous_close
    result["52_week_high"] = high_52
    result["52_week_low"] = low_52
    result["currency"] = currency

    # 7. Calculate change and change_percent
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

    # Store in memory cache
    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE[clean_symbol] = (time.time(), result.copy())
        if sym_upper != clean_symbol:
            _PRICE_CACHE[sym_upper] = (time.time(), result.copy())

    return result
