"""Market index and historical price retrieval tools using yfinance for Indian equities."""

import logging
import math
import re
from typing import Any, Optional, Union

import pandas as pd
import yfinance as yf

from utils.helpers import is_valid_nse_symbol, normalize_nse_symbol

logger = logging.getLogger(__name__)

# Centralized mapping of Indian market indices to canonical name and Yahoo Finance ticker.
# Do not duplicate or scatter this mapping across the application.
INDEX_MAPPING: dict[str, dict[str, str]] = {
    "NIFTY": {"name": "NIFTY 50", "ticker": "^NSEI"},
    "NIFTY50": {"name": "NIFTY 50", "ticker": "^NSEI"},
    "NIFTY 50": {"name": "NIFTY 50", "ticker": "^NSEI"},
    "^NSEI": {"name": "NIFTY 50", "ticker": "^NSEI"},
    "BANKNIFTY": {"name": "BANK NIFTY", "ticker": "^NSEBANK"},
    "BANK NIFTY": {"name": "BANK NIFTY", "ticker": "^NSEBANK"},
    "NIFTYBANK": {"name": "BANK NIFTY", "ticker": "^NSEBANK"},
    "NIFTY BANK": {"name": "BANK NIFTY", "ticker": "^NSEBANK"},
    "^NSEBANK": {"name": "BANK NIFTY", "ticker": "^NSEBANK"},
    "SENSEX": {"name": "SENSEX", "ticker": "^BSESN"},
    "BSE SENSEX": {"name": "SENSEX", "ticker": "^BSESN"},
    "BSESENSEX": {"name": "SENSEX", "ticker": "^BSESN"},
    "^BSESN": {"name": "SENSEX", "ticker": "^BSESN"},
    "NIFTYIT": {"name": "NIFTY IT", "ticker": "^CNXIT"},
    "NIFTY IT": {"name": "NIFTY IT", "ticker": "^CNXIT"},
    "^CNXIT": {"name": "NIFTY IT", "ticker": "^CNXIT"},
}

SUPPORTED_PERIODS = ("5d", "1mo", "3mo", "6mo", "1y")


class PriceHistory(list):
    """Reusable list container representing historical OHLCV price bars.

    Supports list operations, attribute access, and dict-like access for flexible use across
    analysis tools and agents.
    """

    def __init__(
        self,
        data: Optional[list[dict[str, Any]]] = None,
        symbol: Optional[str] = None,
        yahoo_symbol: Optional[str] = None,
        period: str = "1mo",
    ) -> None:
        super().__init__(data or [])
        self.symbol = symbol
        self.yahoo_symbol = yahoo_symbol
        self.period = period

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item in ("data", "history", "records", "candles"):
                return list(self)
            if item == "symbol":
                return self.symbol
            if item == "yahoo_symbol":
                return self.yahoo_symbol
            if item == "period":
                return self.period
            if item == "count":
                return len(self)
            raise KeyError(f"Key '{item}' not found on PriceHistory")
        return super().__getitem__(item)

    def to_dict(self) -> dict[str, Any]:
        """Convert container to standard dictionary format."""
        return {
            "symbol": self.symbol,
            "yahoo_symbol": self.yahoo_symbol,
            "period": self.period,
            "data": list(self),
            "count": len(self),
        }


def _clean_number(val: Any, round_digits: int = 2) -> Optional[float]:
    """Convert value to sanitized float, returning None for NaN, Inf, or invalid input."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, round_digits)
    except (ValueError, TypeError):
        return None


def _clean_int(val: Any) -> Optional[int]:
    """Convert value to sanitized integer, returning None for invalid values."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


def _safe_get_property(source: Any, *keys: str) -> Any:
    """Safely retrieve value from a dictionary or object using multiple candidate keys."""
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


def resolve_index(index_name: str) -> Optional[dict[str, str]]:
    """Resolve an index name or alias to its canonical representation and ticker.

    Args:
        index_name: Raw index name (e.g., 'NIFTY', 'NIFTY 50', 'BANKNIFTY', 'SENSEX').

    Returns:
        Dict with 'name' and 'ticker' if recognized, otherwise None.
    """
    if not isinstance(index_name, str):
        return None

    cleaned = re.sub(r"\s+", " ", index_name.strip()).upper()
    if cleaned in INDEX_MAPPING:
        return INDEX_MAPPING[cleaned]

    no_spaces = cleaned.replace(" ", "")
    if no_spaces in INDEX_MAPPING:
        return INDEX_MAPPING[no_spaces]

    return None


def get_market_index(index_name: str) -> dict[str, Any]:
    """Fetch current market index quote and daily change.

    Supported index aliases:
        - NIFTY, NIFTY50, NIFTY 50
        - BANKNIFTY, BANK NIFTY, NIFTY BANK
        - SENSEX, BSE SENSEX
        - NIFTY IT, NIFTYIT

    Args:
        index_name: Name or alias of the market index.

    Returns:
        Dictionary containing:
            - index: Canonical index name (or None if unsupported)
            - ticker: Yahoo Finance ticker (or None if unsupported)
            - current_value: Current index level or None
            - previous_close: Previous day close or None
            - change: Absolute point change or None
            - change_percent: Percentage change or None
    """
    result: dict[str, Any] = {
        "index": None,
        "ticker": None,
        "current_value": None,
        "previous_close": None,
        "change": None,
        "change_percent": None,
    }

    resolved = resolve_index(index_name)
    if not resolved:
        logger.warning("Unsupported index requested: '%s'", index_name)
        return result

    result["index"] = resolved["name"]
    result["ticker"] = resolved["ticker"]

    try:
        ticker = yf.Ticker(resolved["ticker"])
    except Exception as exc:
        logger.warning(
            "Failed to initialize yfinance.Ticker for index '%s': %s",
            resolved["ticker"],
            exc,
        )
        return result

    # 1. Retrieve fast_info
    fast_info = None
    try:
        fast_info = ticker.fast_info
    except Exception as exc:
        logger.debug("fast_info unavailable for index '%s': %s", resolved["ticker"], exc)

    # 2. Retrieve info
    info = None
    try:
        info_data = ticker.info
        if isinstance(info_data, dict):
            info = info_data
    except Exception as exc:
        logger.debug("info unavailable for index '%s': %s", resolved["ticker"], exc)

    # Extract current value
    current_val = None
    if fast_info:
        current_val = _clean_number(
            _safe_get_property(
                fast_info, "last_price", "lastPrice", "regular_market_price"
            )
        )
    if current_val is None and info:
        current_val = _clean_number(
            info.get("regularMarketPrice") or info.get("currentPrice")
        )

    # Extract previous close
    prev_close = None
    if fast_info:
        prev_close = _clean_number(
            _safe_get_property(
                fast_info,
                "previous_close",
                "previousClose",
                "regular_market_previous_close",
            )
        )
    if prev_close is None and info:
        prev_close = _clean_number(
            info.get("regularMarketPreviousClose") or info.get("previousClose")
        )

    # Fallback to history(period="5d") if values are missing
    if current_val is None or prev_close is None:
        try:
            hist = ticker.history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                valid_closes = hist["Close"].dropna()
                if len(valid_closes) >= 1 and current_val is None:
                    current_val = _clean_number(valid_closes.iloc[-1])
                if len(valid_closes) >= 2 and prev_close is None:
                    prev_close = _clean_number(valid_closes.iloc[-2])
        except Exception as exc:
            logger.debug(
                "history fallback failed for index '%s': %s", resolved["ticker"], exc
            )

    result["current_value"] = current_val
    result["previous_close"] = prev_close

    # Calculate change and change_percent
    if current_val is not None and prev_close is not None:
        change = round(current_val - prev_close, 2)
        result["change"] = change
        if prev_close != 0:
            result["change_percent"] = round((change / prev_close) * 100, 2)
        else:
            result["change_percent"] = None
    else:
        result["change"] = None
        result["change_percent"] = None

    return result


def get_price_history(
    symbol: str, period: str = "1mo"
) -> PriceHistory:
    """Fetch reusable historical OHLCV price series for an NSE equity or index.

    Supported periods: '5d', '1mo', '3mo', '6mo', '1y'

    Args:
        symbol: Stock symbol (e.g., 'TCS', 'reliance', 'INFY.NS') or index (e.g., 'NIFTY 50', 'SENSEX').
        period: Time window string. Must be one of SUPPORTED_PERIODS. Defaults to '1mo'.

    Returns:
        PriceHistory list containing dicts with keys:
            - date: 'YYYY-MM-DD'
            - open: float or None
            - high: float or None
            - low: float or None
            - close: float or None
            - volume: int or None

    Raises:
        ValueError: If period is not in SUPPORTED_PERIODS.
    """
    if period not in SUPPORTED_PERIODS:
        raise ValueError(
            f"Unsupported period '{period}'. Supported periods: {SUPPORTED_PERIODS}"
        )

    # 1. Resolve ticker symbol (Index or Equity)
    resolved_index = resolve_index(symbol) if isinstance(symbol, str) else None
    if isinstance(symbol, str) and not resolved_index and not symbol.startswith("^"):
        try:
            from services.symbol_resolver import resolve_nse_symbol
            res = resolve_nse_symbol(symbol)
            if res.get("symbol"):
                symbol = res["symbol"]
        except Exception:
            pass

    if resolved_index:
        clean_symbol = resolved_index["name"]
        yahoo_symbol = resolved_index["ticker"]
    elif isinstance(symbol, str) and symbol.startswith("^"):
        clean_symbol = symbol
        yahoo_symbol = symbol
    elif isinstance(symbol, str) and is_valid_nse_symbol(symbol):
        try:
            clean_symbol = normalize_nse_symbol(symbol, target_format="clean")
            yahoo_symbol = normalize_nse_symbol(symbol, target_format="yahoo")
        except (ValueError, TypeError):
            return PriceHistory([], symbol=None, yahoo_symbol=None, period=period)
    else:
        logger.warning("Unsupported or invalid symbol for historical data: '%s'", symbol)
        return PriceHistory([], symbol=None, yahoo_symbol=None, period=period)

    # 2. Query historical data from yfinance
    try:
        ticker = yf.Ticker(yahoo_symbol)
        hist = ticker.history(period=period)
    except Exception as exc:
        logger.warning(
            "Failed to fetch price history for '%s': %s", yahoo_symbol, exc
        )
        return PriceHistory(
            [], symbol=clean_symbol, yahoo_symbol=yahoo_symbol, period=period
        )

    if hist is None or hist.empty:
        logger.debug("Empty history returned for '%s' (%s)", yahoo_symbol, period)
        return PriceHistory(
            [], symbol=clean_symbol, yahoo_symbol=yahoo_symbol, period=period
        )

    # 3. Format historical records
    records: list[dict[str, Any]] = []
    for idx, row in hist.iterrows():
        if hasattr(idx, "strftime"):
            date_str = idx.strftime("%Y-%m-%d")
        else:
            date_str = str(idx)[:10]

        records.append(
            {
                "date": date_str,
                "open": _clean_number(row.get("Open")),
                "high": _clean_number(row.get("High")),
                "low": _clean_number(row.get("Low")),
                "close": _clean_number(row.get("Close")),
                "volume": _clean_int(row.get("Volume")),
            }
        )

    return PriceHistory(
        records, symbol=clean_symbol, yahoo_symbol=yahoo_symbol, period=period
    )


# Liquid constituent universe representing NSE core large-caps for gainers/losers calculations
NIFTY_CORE_SYMBOLS = (
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "BHARTIARTL",
    "ITC",
    "SBIN",
    "HINDUNILVR",
    "LT",
    "BAJFINANCE",
    "HCLTECH",
    "MARUTI",
    "SUNPHARMA",
    "WIPRO",
    "KOTAKBANK",
    "TITAN",
    "ONGC",
    "NTPC",
    "AXISBANK",
)

# In-memory movers cache
_MOVERS_CACHE: dict[str, tuple[float, tuple[list[dict[str, Any]], list[dict[str, Any]]]]] = {}


class GainersLosersProvider:
    """Isolated provider for discovering top gaining and losing NSE equities.

    Queries verified market data for the NSE benchmark universe without fabricating data.
    """

    def __init__(self, symbols: tuple[str, ...] = NIFTY_CORE_SYMBOLS) -> None:
        self.symbols = symbols

    def fetch_market_movers(
        self, limit: int = 5
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fetch quotes across constituent universe concurrently and rank top gainers and losers.

        Returns:
            Tuple of (gainers_list, losers_list).
        """
        import concurrent.futures
        import time

        cache_key = f"movers_{limit}"
        now = time.time()
        if cache_key in _MOVERS_CACHE:
            ts, cached_data = _MOVERS_CACHE[cache_key]
            if now - ts < 120.0:  # 2 minute cache
                return cached_data

        def _fetch_single_mover(symbol: str) -> Optional[dict[str, Any]]:
            try:
                yahoo_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yahoo_symbol)
                fast_info = getattr(ticker, "fast_info", None)
                if not fast_info:
                    return None

                price = _clean_number(
                    _safe_get_property(
                        fast_info, "last_price", "lastPrice", "regular_market_price"
                    )
                )
                prev_close = _clean_number(
                    _safe_get_property(
                        fast_info,
                        "previous_close",
                        "previousClose",
                        "regular_market_previous_close",
                    )
                )

                if price is not None and prev_close is not None and prev_close > 0:
                    change = round(price - prev_close, 2)
                    change_percent = round((change / prev_close) * 100, 2)
                    return {
                        "symbol": symbol,
                        "yahoo_symbol": yahoo_symbol,
                        "price": price,
                        "previous_close": prev_close,
                        "change": change,
                        "change_percent": change_percent,
                    }
            except Exception as exc:
                logger.debug("Failed to fetch quote for '%s' in mover scan: %s", symbol, exc)
            return None

        movers: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {executor.submit(_fetch_single_mover, sym): sym for sym in self.symbols}
            for fut in concurrent.futures.as_completed(future_to_sym):
                res = fut.result()
                if res:
                    movers.append(res)

        if not movers:
            return [], []

        # Sort for gainers (highest positive change first)
        gainers = sorted(
            [m for m in movers if m["change_percent"] is not None],
            key=lambda x: x["change_percent"],
            reverse=True,
        )

        # Sort for losers (lowest / most negative change first)
        losers = sorted(
            [m for m in movers if m["change_percent"] is not None],
            key=lambda x: x["change_percent"],
        )

        result = (gainers[:limit], losers[:limit])
        _MOVERS_CACHE[cache_key] = (now, result)
        return result


_DEFAULT_MOVER_PROVIDER = GainersLosersProvider()


def get_top_gainers(
    limit: int = 5, provider: Optional[GainersLosersProvider] = None
) -> list[dict[str, Any]]:
    """Retrieve top gaining NSE stocks ranked by percentage change.

    Args:
        limit: Number of gainers to return. Defaults to 5.
        provider: Optional custom or mocked GainersLosersProvider.

    Returns:
        List of stock mover dictionaries.
    """
    prov = provider or _DEFAULT_MOVER_PROVIDER
    try:
        gainers, _ = prov.fetch_market_movers(limit=limit)
        return gainers
    except Exception as exc:
        logger.warning("Error fetching top gainers: %s", exc)
        return []


def get_top_losers(
    limit: int = 5, provider: Optional[GainersLosersProvider] = None
) -> list[dict[str, Any]]:
    """Retrieve top losing NSE stocks ranked by percentage change.

    Args:
        limit: Number of losers to return. Defaults to 5.
        provider: Optional custom or mocked GainersLosersProvider.

    Returns:
        List of stock mover dictionaries.
    """
    prov = provider or _DEFAULT_MOVER_PROVIDER
    try:
        _, losers = prov.fetch_market_movers(limit=limit)
        return losers
    except Exception as exc:
        logger.warning("Error fetching top losers: %s", exc)
        return []

