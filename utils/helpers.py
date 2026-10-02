"""Utility functions and helpers for the NSE Agentic AI application."""

import re
from typing import Literal

# Type definition for supported symbol formatting targets
SymbolFormat = Literal["clean", "yahoo", "tradingview"]

# Regex pattern for a valid base NSE ticker:
# - Uppercase alphanumeric characters
# - Can include '&' (e.g., M&M, L&TFH)
# - Can include hyphens (e.g., BAJAJ-AUTO, MCDOWELL-N)
# Length typically ranges between 1 and 20 characters
_BASE_SYMBOL_REGEX = re.compile(r"^[A-Z0-9&]+(?:-[A-Z0-9]+)?$")


def normalize_nse_symbol(symbol: str, target_format: SymbolFormat = "clean") -> str:
    """Normalize and format an Indian National Stock Exchange (NSE) ticker symbol.

    Handles common user inputs, exchange prefixes, market suffixes,
    and equity series notations.

    Examples:
        >>> normalize_nse_symbol("reliance")
        'RELIANCE'
        >>> normalize_nse_symbol("NSE:INFY")
        'INFY'
        >>> normalize_nse_symbol("  tcs.ns  ")
        'TCS'
        >>> normalize_nse_symbol("HDFCBANK-EQ")
        'HDFCBANK'
        >>> normalize_nse_symbol("m&m", target_format="yahoo")
        'M&M.NS'
        >>> normalize_nse_symbol("infy", target_format="tradingview")
        'NSE:INFY'

    Args:
        symbol: The raw input symbol or ticker string.
        target_format: The desired output format:
            - "clean": Raw NSE ticker (e.g., 'RELIANCE', 'M&M')
            - "yahoo": Yahoo Finance ticker format (e.g., 'RELIANCE.NS')
            - "tradingview": TradingView ticker format (e.g., 'NSE:RELIANCE')

    Returns:
        The normalized symbol string formatted according to target_format.

    Raises:
        TypeError: If symbol is not a string.
        ValueError: If symbol is empty, invalid, or target_format is unsupported.
    """
    if not isinstance(symbol, str):
        raise TypeError(f"Symbol must be a string, got {type(symbol).__name__}")

    valid_formats = ("clean", "yahoo", "tradingview")
    if target_format not in valid_formats:
        raise ValueError(
            f"Invalid target_format '{target_format}'. Expected one of: {valid_formats}"
        )

    # 1. Clean whitespace and uppercase
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Symbol cannot be empty or whitespace only.")

    # 2. Strip known exchange prefixes (e.g., 'NSE:', 'NSE_')
    if cleaned.startswith("NSE:"):
        cleaned = cleaned[4:].strip()
    elif cleaned.startswith("NSE_"):
        cleaned = cleaned[4:].strip()

    # 3. Strip Yahoo/BSE/NSE suffixes (e.g., '.NS', '.BO')
    if cleaned.endswith(".NS"):
        cleaned = cleaned[:-3].strip()
    elif cleaned.endswith(".BO"):
        # Note: If user provided BSE suffix for an NSE app, strip it to extract the ticker
        cleaned = cleaned[:-3].strip()

    # 4. Strip equity series tags (e.g., '-EQ', '.EQ', '-BE')
    for suffix in ("-EQ", ".EQ", "-BE", ".BE"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break

    # 5. Validate the extracted base ticker
    if not cleaned or not _BASE_SYMBOL_REGEX.match(cleaned):
        raise ValueError(f"'{symbol}' is not a valid NSE ticker symbol.")

    # 6. Apply requested output format
    if target_format == "clean":
        return cleaned
    elif target_format == "yahoo":
        return f"{cleaned}.NS"
    elif target_format == "tradingview":
        return f"NSE:{cleaned}"

    return cleaned


def is_valid_nse_symbol(symbol: str) -> bool:
    """Check whether a given string can be parsed into a valid NSE symbol.

    Args:
        symbol: The symbol string to test.

    Returns:
        True if symbol normalizes successfully, False otherwise.
    """
    if not isinstance(symbol, str):
        return False
    try:
        normalize_nse_symbol(symbol)
        return True
    except (ValueError, TypeError):
        return False
