"""Conversation context management and multi-turn coreference resolution service.

Extracts referenced entities from persisted conversation turns in SQLite, enabling follow-up questions
(e.g., 'What is its 52 week high?', 'Which has better PE?') without requiring an OpenAI API key.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

from data.stock_master import INDIAN_STOCK_MASTER
from services.database_service import get_messages

logger = logging.getLogger(__name__)

# Pronouns and coreference indicator phrases
COREFERENCE_PATTERNS = (
    r"\b(it|its|this|that|the stock|the share|the company|of it|for it|this company|this stock)\b",
    r"\b(which is better|which has better|which one|between both|between them|compare them|which of them)\b",
)


def get_conversation_context(
    conversation_id: Optional[str],
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Inspect previous messages in the conversation to extract referenced symbols and entities.

    Args:
        conversation_id: Unique UUID of active conversation, if any.
        db_path: Optional SQLite database file path.

    Returns:
        Dictionary with:
            - 'last_symbol': Most recently focused single NSE stock symbol or None
            - 'last_symbols': List of symbols discussed in the most recent turn
            - 'last_company_name': Company name of last focused stock
            - 'turns_count': Total persisted message count
    """
    context: dict[str, Any] = {
        "last_symbol": None,
        "last_symbols": [],
        "last_company_name": None,
        "turns_count": 0,
    }

    if not conversation_id:
        return context

    try:
        messages = get_messages(conversation_id, db_path=db_path)
    except Exception as exc:
        logger.debug("Could not read conversation messages for context: %s", exc)
        return context

    if not messages:
        return context

    context["turns_count"] = len(messages)

    # Scan messages in reverse chronological order to find the latest stock entities
    found_symbols: list[str] = []

    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue

        # Ignore assistant fallback or disambiguation messages (they contain examples like "TCS", not focused stocks)
        if role == "assistant":
            if (
                "couldn't determine exactly what market information" in content
                or "Ambiguous Group: Multiple Companies Listed" in content
                or "You can ask things like" in content
            ):
                continue
            # Also ignore market-wide scan responses (tables containing 10+ stocks)
            if any(
                phrase in content
                for phrase in (
                    "NSE Stocks Near Their 52-Week",
                    "Top NSE Gainers Today",
                    "Top NSE Losers Today",
                    "High-Volume & Active NSE Equities",
                    "Sector Equities (NSE)",
                    "Short-Term Trading Opportunities",
                )
            ):
                continue

            # Check for single-stock response headers: e.g. "### 📈 Tata Power (TATAPOWER)" or "### 🎯 Tata Power (`TATAPOWER`)"
            hdr_match = re.search(r"###\s+[^\n]+?[\(`]([A-Z0-9\-]{2,12})[\)`]", content)
            if hdr_match:
                sym = hdr_match.group(1).upper()
                if sym in INDIAN_STOCK_MASTER:
                    found_symbols = [sym]
                    break

            # Corporate card format: e.g. "**NSE:** `TATAPOWER`"
            card_match = re.search(r"\*\*NSE:\*\*\s+`?([A-Z0-9\-]{2,12})`?", content)
            if card_match:
                sym = card_match.group(1).upper()
                if sym in INDIAN_STOCK_MASTER:
                    found_symbols = [sym]
                    break

        elif role == "user":
            from agent.demo_agent import extract_symbols_from_query

            user_syms = extract_symbols_from_query(content)
            if user_syms:
                found_symbols = user_syms
                break

    if found_symbols:
        context["last_symbols"] = found_symbols
        context["last_symbol"] = found_symbols[0]
        rec = INDIAN_STOCK_MASTER.get(found_symbols[0], {})
        context["last_company_name"] = rec.get("company_name", found_symbols[0])

    return context


def resolve_coreferences(
    query: str,
    current_symbols: list[str],
    context: dict[str, Any],
) -> tuple[list[str], Optional[str]]:
    """Resolve pronouns like 'its' or comparison follow-ups using conversation context.

    Examples:
    1. Turn 1: 'Tell me about Tata Power'
       Turn 2: 'What is its 52 week high?' -> resolves 'its' to 'TATAPOWER'
    2. Turn 1: 'Compare TCS and Infosys'
       Turn 2: 'Which has better PE?' -> resolves context to ['TCS', 'INFY']

    Args:
        query: User input text.
        current_symbols: Symbols already extracted from the current query.
        context: Context dictionary from get_conversation_context.

    Returns:
        Tuple of (resolved_symbols_list, coreference_note_if_applied).
    """
    if current_symbols:
        # If user explicitly specified symbols in the current turn, respect their choice
        return current_symbols, None

    q_lower = query.lower()

    # CRITICAL PROTECTION: Market-wide, plural, screener, or sector queries must NEVER
    # be hijacked by previously focused single stock context!
    # Examples: "52 Week High Stocks", "52 weeks low", "top gainers", "banking stocks"
    is_market_wide = any(
        kw in q_lower
        for kw in (
            "52 week high",
            "52-week high",
            "52w high",
            "52 weeks high",
            "52 week highs",
            "52 weeks highs",
            "52 week low",
            "52-week low",
            "52w low",
            "52 weeks low",
            "52 week lows",
            "52 weeks lows",
            "yearly high",
            "yearly low",
            "year high",
            "year low",
            "all time high",
            "all-time high",
            "all time low",
            "all-time low",
            "gainer",
            "gainers",
            "loser",
            "losers",
            "most active",
            "volume spike",
            "volume surge",
            "volume shocker",
            "volume shockers",
            "market overview",
            "how is market",
            "market today",
            "nifty",
            "sensex",
            "bank nifty",
            "sector",
            "sectors",
            "industry",
            "opportunities",
            "trading opportunities",
            "short-term",
            "short term",
            "2-day",
            "trading setup",
            "screener",
        )
    )

    is_plural_request = any(
        w in q_lower
        for w in (
            "stocks",
            "shares",
            "companies",
            "equities",
            "which stocks",
            "top stocks",
            "best stocks",
            "all stocks",
            "list stocks",
        )
    )

    # Check for explicit pronouns or follow-up indicators
    has_pronoun = any(re.search(pat, q_lower) for pat in COREFERENCE_PATTERNS)

    # If it is a market-wide or plural query and lacks an explicit pronoun ('its', 'this stock'),
    # DO NOT resolve to a single stock!
    if (is_market_wide or is_plural_request) and not has_pronoun:
        return current_symbols, None

    is_attribute_query = any(
        kw in q_lower
        for kw in (
            "pe",
            "p/e",
            "price",
            "results",
            "quarterly",
            "dividend",
            "sentiment",
            "rsi",
            "technicals",
            "market cap",
            "promoter",
            "should i buy",
            "hold or sell",
            "better",
            "cheaper",
        )
    )

    last_symbols = context.get("last_symbols") or []
    last_single = context.get("last_symbol")

    # If the user asks a comparative follow-up (e.g. "Which has better PE?", "Which is cheaper?")
    # and previous turn had 2+ symbols:
    if len(last_symbols) >= 2 and any(
        w in q_lower
        for w in (
            "which",
            "better",
            "cheaper",
            "between both",
            "compare",
            "winner",
            "higher",
            "lower",
        )
    ):
        note = f"Resolved follow-up context to previously compared companies: {', '.join(last_symbols)}"
        return list(last_symbols), note

    # If the user asks a single-stock follow-up (e.g. "What is its 52 week high?", "price of it", "what about PE?")
    # and previous turn had a focused stock:
    if (has_pronoun or is_attribute_query) and last_single:
        comp_name = context.get("last_company_name", last_single)
        note = f"Resolved follow-up context ('its') to {comp_name} ({last_single})"
        return [last_single], note

    return current_symbols, None
