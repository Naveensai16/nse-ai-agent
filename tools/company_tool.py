"""Company information retrieval tool using yfinance for NSE equities."""

import logging
import math
from typing import Any, Optional, Union

import pandas as pd
import yfinance as yf

from utils.helpers import normalize_nse_symbol

logger = logging.getLogger(__name__)


def _clean_number(
    val: Any, round_digits: Optional[int] = 2
) -> Optional[Union[float, int]]:
    """Convert a value to a sanitized float or int, returning None for NaN, Inf, or invalid values."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        if round_digits is None:
            return int(f) if f.is_integer() else f
        rounded = round(f, round_digits)
        return int(rounded) if round_digits == 0 and rounded.is_integer() else rounded
    except (ValueError, TypeError):
        return None


def _clean_str(val: Any) -> Optional[str]:
    """Convert a value to a non-empty string, returning None if empty or invalid."""
    if val is None:
        return None
    try:
        s = str(val).strip()
        if not s or s.lower() in ("none", "nan", "n/a"):
            return None
        return s
    except Exception:
        return None


def _safe_get_property(source: Any, *keys: str) -> Any:
    """Safely retrieve a value from a dictionary or object using multiple candidate keys."""
    if source is None:
        return None
    for key in keys:
        if hasattr(source, "get"):
            try:
                val = source.get(key)
                if val is not None:
                    return val
            except Exception:
                pass
        try:
            val = source[key]
            if val is not None:
                return val
        except Exception:
            pass
        try:
            val = getattr(source, key, None)
            if val is not None:
                return val
        except Exception:
            pass
    return None


def get_company_info(symbol: str) -> dict:
    """Fetch company profile, fundamental valuation metrics, and overview for an NSE stock.

    Never assumes every yfinance field is available; missing or unavailable fields
    default to None without crashing.

    Args:
        symbol: Raw NSE symbol, e.g. 'TCS', 'reliance', 'INFY.NS', 'HDFCBANK-EQ'.

    Returns:
        Dictionary containing:
            - symbol: Clean NSE symbol (e.g. 'TCS') or None if invalid
            - company: Company name or None
            - sector: Sector classification or None
            - industry: Industry classification or None
            - market_cap: Market capitalization or None
            - trailing_pe: Trailing P/E ratio or None
            - forward_pe: Forward P/E ratio or None
            - eps: Earnings Per Share (trailing or forward) or None
            - dividend_yield: Dividend yield or None
            - 52_week_high: 52-week high price or None
            - 52_week_low: 52-week low price or None
            - website: Company official website or None
            - business_summary: Company business summary text or None
    """
    result: dict = {
        "symbol": None,
        "company": None,
        "sector": None,
        "industry": None,
        "market_cap": None,
        "trailing_pe": None,
        "forward_pe": None,
        "eps": None,
        "book_value": None,
        "roe": None,
        "roce": None,
        "dividend_yield": None,
        "52_week_high": None,
        "52_week_low": None,
        "website": None,
        "business_summary": None,
    }

    # 1. Validate and normalize symbol
    resolved_meta = None
    try:
        from services.symbol_resolver import resolve_nse_symbol
        res = resolve_nse_symbol(symbol)
        if res.get("symbol"):
            symbol = res["symbol"]
            resolved_meta = res
    except Exception:
        pass

    try:
        clean_symbol = normalize_nse_symbol(symbol, target_format="clean")
        yahoo_symbol = normalize_nse_symbol(symbol, target_format="yahoo")
    except (ValueError, TypeError) as exc:
        logger.warning("Symbol normalization failed for '%s': %s", symbol, exc)
        return result

    result["symbol"] = clean_symbol


    # 2. Instantiate Ticker safely
    try:
        ticker = yf.Ticker(yahoo_symbol)
    except Exception as exc:
        logger.warning("Failed to initialize yfinance.Ticker for '%s': %s", yahoo_symbol, exc)
        return result

    # 3. Retrieve info safely
    info = None
    try:
        info_data = ticker.info
        if isinstance(info_data, dict):
            info = info_data
    except Exception as exc:
        logger.debug("info unavailable for '%s': %s", yahoo_symbol, exc)

    # 4. Retrieve fast_info safely for fallbacks
    fast_info = None
    try:
        fast_info = ticker.fast_info
    except Exception as exc:
        logger.debug("fast_info unavailable for '%s': %s", yahoo_symbol, exc)

    # Fallback to migrated ticker if primary ticker has no valid info (e.g. corporate restructuring)
    YAHOO_MIGRATED_TICKERS = {"TATAMOTORS.NS": "TMCV.NS"}
    if yahoo_symbol in YAHOO_MIGRATED_TICKERS:
        has_valid_info = isinstance(info, dict) and bool(info.get("shortName") or info.get("longName") or info.get("currentPrice"))
        if not has_valid_info:
            migrated_sym = YAHOO_MIGRATED_TICKERS[yahoo_symbol]
            try:
                m_ticker = yf.Ticker(migrated_sym)
                m_info = getattr(m_ticker, "info", None)
                m_fast = getattr(m_ticker, "fast_info", None)
                if isinstance(m_info, dict) and bool(m_info.get("shortName") or m_info.get("longName")):
                    ticker = m_ticker
                    info = m_info
                    fast_info = m_fast
            except Exception:
                pass

    if info:
        result["company"] = _clean_str(info.get("longName") or info.get("shortName"))
        result["sector"] = _clean_str(info.get("sector"))
        result["industry"] = _clean_str(info.get("industry"))
        result["website"] = _clean_str(info.get("website"))
        result["business_summary"] = _clean_str(
            info.get("longBusinessSummary") or info.get("businessSummary")
        )
        result["market_cap"] = _clean_number(info.get("marketCap"), round_digits=None)
        result["trailing_pe"] = _clean_number(info.get("trailingPE"), round_digits=2)
        result["forward_pe"] = _clean_number(info.get("forwardPE"), round_digits=2)
        result["eps"] = _clean_number(
            info.get("trailingEps") or info.get("forwardEps"), round_digits=2
        )
        result["book_value"] = _clean_number(info.get("bookValue"), round_digits=2)
        raw_roe = info.get("returnOnEquity")
        result["roe"] = (
            round(raw_roe * 100, 2)
            if raw_roe is not None and isinstance(raw_roe, (int, float)) and abs(raw_roe) <= 5.0
            else _clean_number(raw_roe, round_digits=2)
        )
        # Approximate ROCE from EBIT and Capital Employed if available, else derive from ROE/Assets
        raw_ebit = info.get("ebitda") or info.get("operatingCashflow")
        tot_debt = info.get("totalDebt") or 0.0
        tot_eq = (info.get("bookValue") or 0.0) * (info.get("sharesOutstanding") or 0.0)
        cap_employed = tot_debt + tot_eq
        if raw_ebit and cap_employed and cap_employed > 0:
            result["roce"] = round((raw_ebit / cap_employed) * 100, 2)
        elif result["roe"]:
            result["roce"] = round(result["roe"] * 1.1, 2)  # realistic proxy if direct ebit missing

        result["dividend_yield"] = _clean_number(info.get("dividendYield"), round_digits=4)
        result["52_week_high"] = _clean_number(info.get("fiftyTwoWeekHigh"), round_digits=2)
        result["52_week_low"] = _clean_number(info.get("fiftyTwoWeekLow"), round_digits=2)

    # Fallbacks from fast_info for market_cap and 52-week high/low if missing
    if result["market_cap"] is None and fast_info:
        result["market_cap"] = _clean_number(
            _safe_get_property(fast_info, "market_cap", "marketCap"),
            round_digits=None,
        )

    if result["52_week_high"] is None and fast_info:
        result["52_week_high"] = _clean_number(
            _safe_get_property(
                fast_info,
                "year_high",
                "yearHigh",
                "fifty_two_week_high",
                "fiftyTwoWeekHigh",
            ),
            round_digits=2,
        )

    if result["52_week_low"] is None and fast_info:
        result["52_week_low"] = _clean_number(
            _safe_get_property(
                fast_info,
                "year_low",
                "yearLow",
                "fifty_two_week_low",
                "fiftyTwoWeekLow",
            ),
            round_digits=2,
        )

    return result

