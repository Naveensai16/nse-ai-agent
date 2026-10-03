"""OpenAI natural-language understanding, structured classification, and grounded response synthesis.

Provides server-side OpenAI integration with strict token/rate limits, timeout handling,
scope-aware structured intent classification, entity validation, and grounded response synthesis.
Never hardcodes or exposes API credentials to users or frontend.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from data.stock_master import CONGLOMERATE_GROUPS, INDIAN_STOCK_MASTER
from services.intent_service import MarketIntent, QueryScope, detect_intent, normalize_query
from agent.demo_agent import extract_symbols_from_query
from services.symbol_resolver import check_conglomerate_ambiguity, resolve_nse_symbol

logger = logging.getLogger(__name__)

# Server-side environment variable configuration
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "12.0"))
MAX_PROMPT_LENGTH = int(os.getenv("OPENAI_MAX_PROMPT_LENGTH", "1000"))
MAX_CLASSIFICATION_TOKENS = 400
MAX_RESPONSE_TOKENS = 900
RATE_LIMIT_REQUESTS_PER_MINUTE = 30


# ==============================================================================
# PER-SESSION IN-MEMORY RATE LIMITER
# ==============================================================================


class SessionRateLimiter:
    """Thread-safe sliding-window rate limiter to protect server-side OpenAI token consumption."""

    def __init__(self, max_per_minute: int = RATE_LIMIT_REQUESTS_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, session_id: str) -> bool:
        """Check if session is allowed to make a request under the rate limit."""
        now = time.time()
        window_start = now - 60.0

        timestamps = self._requests.get(session_id, [])
        # Prune timestamps older than 60 seconds
        valid_timestamps = [t for t in timestamps if t > window_start]

        if len(valid_timestamps) >= self.max_per_minute:
            self._requests[session_id] = valid_timestamps
            return False

        valid_timestamps.append(now)
        self._requests[session_id] = valid_timestamps
        return True


rate_limiter = SessionRateLimiter()


def get_server_openai_api_key() -> str:
    """Retrieve OpenAI API key strictly server-side without exposing to client.

    Checks:
    1. os.environ['OPENAI_API_KEY']
    2. streamlit.secrets['OPENAI_API_KEY'] (if running in Streamlit)
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st

            key = st.secrets.get("OPENAI_API_KEY", "").strip()
        except Exception:
            pass
    return key


def mask_sensitive_error(error: Exception) -> str:
    """Sanitize error messages to ensure API keys are never exposed in exceptions or logs."""
    msg = str(error)
    key = get_server_openai_api_key()
    if key and key in msg:
        msg = msg.replace(key, "sk-******")
    msg = re.sub(r"sk-[a-zA-Z0-9_\-]{15,}", "sk-******", msg)
    return msg


# ==============================================================================
# STRUCTURED INTENT AND SCOPE CLASSIFICATION
# ==============================================================================


@dataclass
class StructuredClassification:
    """Structured result of natural language intent and scope classification."""

    intent: str
    scope: str
    symbols: list[str] = field(default_factory=list)
    company_names: list[str] = field(default_factory=list)
    sector: Optional[str] = None
    index: Optional[str] = None
    timeframe: str = "TODAY"
    include_catalysts: bool = False
    needs_clarification: bool = False
    clarification_group: Optional[str] = None
    confidence: float = 1.0
    source: str = "openai"  # 'openai' or 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CLASSIFICATION_SYSTEM_PROMPT = """You are a precision Natural-Language Understanding classifier for an Indian National Stock Exchange (NSE) market assistant.
Given a user query and optional previous conversation context, output a strict JSON object with these exact keys:

{
  "intent": "<INTENT_ENUM>",
  "scope": "<SCOPE_ENUM>",
  "symbols": ["<NSE_TICKER>", ...],
  "company_names": ["<COMPANY_NAME>", ...],
  "sector": "<SECTOR_NAME_OR_NULL>",
  "index": "<INDEX_NAME_OR_NULL>",
  "timeframe": "<TIMEFRAME>",
  "include_catalysts": <true|false>,
  "needs_clarification": <true|false>,
  "clarification_group": "<GROUP_NAME_OR_NULL>"
}

Supported INTENT_ENUM values:
- 52_WEEK_HIGH_STOCKS (market-wide scan for stocks near yearly high)
- 52_WEEK_LOW_STOCKS (market-wide scan for stocks near yearly low)
- TOP_GAINERS (top gaining stocks across NSE today)
- TOP_LOSERS (top falling stocks across NSE today)
- VOLUME_SPIKE / MOST_ACTIVE (unusual volume or high turnover stocks)
- MARKET_OVERVIEW (index status: Nifty 50, Sensex, Bank Nifty, overall market)
- SHORT_TERM_OPPORTUNITIES (2-day swing trading setups)
- STOCKS_BY_SECTOR (stocks in IT, Banking, Auto, Pharma, FMCG, Metals, Realty, etc.)
- SECTOR_PERFORMANCE (sector rankings, leading/lagging sectors)
- STOCK_PRICE (current quote/price of a single specific stock)
- STOCK_OVERVIEW (general details, overview, market cap of a stock)
- STOCK_FUNDAMENTALS (PE, valuation, dividend, debt, financials of a stock)
- STOCK_TECHNICALS (RSI, 52-week range, moving averages of a stock)
- STOCK_52_WEEK_HIGH_LOW (52-week high and low of a specific single stock)
- STOCK_NEWS (recent news/catalysts for a specific stock)
- STOCK_RESULTS (quarterly results, earnings for a specific stock)
- STOCK_DECISION (buy/sell/hold decision or advisory question for a stock)
- STOCK_COMPARISON (comparing 2 or more specific stocks, e.g. TCS vs Infosys)
- AMBIGUOUS_STOCK (mentioning a group/conglomerate name like 'Tata', 'Adani', 'Bajaj', 'Birla')
- FALLBACK (unclear, off-topic, or greeting)

Supported SCOPE_ENUM values:
- MARKET_WIDE (e.g. "52 Week High Stocks", "top gainers", "volume shockers")
- SINGLE_STOCK (e.g. "What is ITC 52 week high?", "Tata Motors price")
- MULTI_STOCK (e.g. "Compare TCS and Infosys")
- SECTOR (e.g. "Banking stocks", "IT sector performance")
- INDEX (e.g. "How is Nifty doing?", "Sensex status")
- AMBIGUOUS (e.g. "Tata", "Adani")
- GENERAL (e.g. "market today", general inquiry)

CRITICAL RULES:
1. Distinguish MARKET_WIDE vs SINGLE_STOCK carefully:
   - "52 Week High Stocks", "52 weeks high", "yearly high shares", "which stocks touched 52w high?" are MARKET_WIDE (symbols: []).
   - "What is ITC's 52 week high?", "Tata Power 52 week high" are SINGLE_STOCK (symbols: ["ITC"] or ["TATAPOWER"]).
2. If the user mentions a conglomerate like "Tata", "Adani", "Bajaj", "HDFC" without specifying a single listed company:
   - Set scope: "AMBIGUOUS", intent: "AMBIGUOUS_STOCK", needs_clarification: true, clarification_group: "Tata".
3. For follow-up queries like "What is its PE?" or "price of it":
   - Use the provided conversation context to resolve the referenced stock symbol.
4. Output ONLY the valid JSON object. No conversational prefix or markdown tags."""


def classify_query(
    query: str,
    symbols: Optional[list[str]] = None,
    context: Optional[dict[str, Any]] = None,
    session_id: str = "default",
) -> StructuredClassification:
    """Classify user query using OpenAI structured output when configured, falling back deterministically.

    Args:
        query: Raw user query string.
        symbols: Pre-resolved or extracted symbols (if already known).
        context: Optional conversation context dictionary.
        session_id: Unique session identifier for rate limiting.

    Returns:
        StructuredClassification object with verified intent, scope, and entities.
    """
    from services.conversation_context import resolve_coreferences

    context = context or {}
    clean_query = query.strip()[:MAX_PROMPT_LENGTH]

    # Resolve symbols incorporating context if not already pre-provided
    if symbols is not None and len(symbols) > 0:
        resolved_symbols = list(symbols)
    else:
        direct_symbols = extract_symbols_from_query(clean_query)
        resolved_symbols, _ = resolve_coreferences(clean_query, direct_symbols, context)

    api_key = get_server_openai_api_key()

    # If OpenAI API key is available and rate limit allows, attempt OpenAI classification
    if api_key and rate_limiter.is_allowed(session_id):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=DEFAULT_TIMEOUT, max_retries=1)

            ctx_hint = ""
            last_sym = context.get("last_symbol")
            last_syms = context.get("last_symbols", [])
            if last_sym:
                ctx_hint = f"\nPrevious conversation focused stock: {last_sym} (All recently discussed: {', '.join(last_syms)})"

            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                temperature=0.0,
                max_tokens=MAX_CLASSIFICATION_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"User Query: {clean_query}{ctx_hint}"},
                ],
            )

            raw_json = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_json)

            # Validate and canonicalize parsed output
            intent_str = parsed.get("intent", "").upper().strip()
            scope_str = parsed.get("scope", "").upper().strip()
            extracted_syms = parsed.get("symbols", [])
            comp_names = parsed.get("company_names", [])

            # Canonicalize symbols against INDIAN_STOCK_MASTER
            valid_syms: list[str] = []
            for s in extracted_syms:
                res = resolve_nse_symbol(str(s))
                if res and res.symbol in INDIAN_STOCK_MASTER and res.symbol not in valid_syms:
                    valid_syms.append(res.symbol)

            # Also check company names if symbols empty
            if not valid_syms and comp_names:
                for c in comp_names:
                    res = resolve_nse_symbol(str(c))
                    if res and res.symbol in INDIAN_STOCK_MASTER and res.symbol not in valid_syms:
                        valid_syms.append(res.symbol)

            # If model didn't return symbols but context/rules resolved them, merge
            if not valid_syms and resolved_symbols and scope_str != QueryScope.MARKET_WIDE.value:
                valid_syms = list(resolved_symbols)

            # Safety enforcement: Market-wide queries must never have single-stock symbols
            if scope_str == QueryScope.MARKET_WIDE.value:
                valid_syms = []

            # Check conglomerate ambiguity override
            clean_q = normalize_query(clean_query).lower()
            if not valid_syms:
                for w in clean_q.split():
                    if w in CONGLOMERATE_GROUPS:
                        ambig_res = check_conglomerate_ambiguity(w, w)
                        if ambig_res and ambig_res.get("candidates"):
                            return StructuredClassification(
                                intent=MarketIntent.AMBIGUOUS_STOCK.value,
                                scope=QueryScope.AMBIGUOUS.value,
                                needs_clarification=True,
                                clarification_group=w.title(),
                                source="openai_validated",
                            )

            # Map intent string to enum safely
            matched_intent = MarketIntent.FALLBACK.value
            for mi in MarketIntent:
                if mi.value == intent_str:
                    matched_intent = mi.value
                    break

            # If valid intent was recognized
            if matched_intent != MarketIntent.FALLBACK.value:
                return StructuredClassification(
                    intent=matched_intent,
                    scope=scope_str or (QueryScope.SINGLE_STOCK.value if valid_syms else QueryScope.GENERAL.value),
                    symbols=valid_syms,
                    company_names=comp_names,
                    sector=parsed.get("sector"),
                    index=parsed.get("index"),
                    timeframe=parsed.get("timeframe", "TODAY"),
                    include_catalysts=bool(parsed.get("include_catalysts", False)),
                    needs_clarification=bool(parsed.get("needs_clarification", False)),
                    clarification_group=parsed.get("clarification_group"),
                    source="openai",
                )

        except Exception as exc:
            logger.warning("OpenAI classification failed or timed out: %s. Using deterministic fallback.", mask_sensitive_error(exc))

    # ==========================================================================
    # DETERMINISTIC CLASSIFICATION (High-accuracy fallback & local engine)
    # ==========================================================================
    clean_q = normalize_query(clean_query)

    # Conglomerate ambiguity check
    has_conglomerate = False
    ambig_group = None
    for w in clean_q.lower().split():
        if w in CONGLOMERATE_GROUPS and not resolved_symbols:
            has_conglomerate = True
            ambig_group = w.title()
            break

    det_intent, det_params = detect_intent(
        query=clean_query,
        symbols=resolved_symbols,
        has_ambiguous_conglomerate=has_conglomerate,
        context=context,
    )

    scope = det_params.get("scope")
    if not scope:
        if resolved_symbols and len(resolved_symbols) == 1:
            scope = QueryScope.SINGLE_STOCK.value
        elif resolved_symbols and len(resolved_symbols) > 1:
            scope = QueryScope.MULTI_STOCK.value
        elif has_conglomerate and not resolved_symbols:
            scope = QueryScope.AMBIGUOUS.value
        else:
            scope = QueryScope.GENERAL.value

    return StructuredClassification(
        intent=det_intent.value if hasattr(det_intent, "value") else str(det_intent),
        scope=scope,
        symbols=resolved_symbols,
        sector=det_params.get("sector"),
        index=det_params.get("index"),
        include_catalysts=det_params.get("include_catalysts", False),
        needs_clarification=has_conglomerate,
        clarification_group=ambig_group,
        source="deterministic",
    )


# ==============================================================================
# GROUNDED RESPONSE SYNTHESIS (No Hallucinations)
# ==============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are an expert National Stock Exchange (NSE) Market Analyst.
Your job is to format and explain the factual real-time stock market data provided to you.

STRICT CONSTRAINTS (ZERO TOLERANCE FOR HALLUCINATION):
1. NEVER invent, extrapolate, or guess stock prices, percentage changes, 52-week ranges, P/E ratios, market caps, or dates.
2. Rely EXCLUSIVELY on the provided grounded factual data block.
3. If an explanation or catalyst is requested ("why?"), provide only catalysts found in the provided news/results context. If no news is available, explicitly state: "No recent company-specific catalyst was found from available NSE disclosures."
4. Format all prices in Indian Rupees (₹) and percentage changes with explicit signs (+/-).
5. Produce clean, professional GitHub-flavored markdown with clear headers, bold metrics, and structured bullet points or tables.
"""


def synthesize_grounded_response(
    query: str,
    grounded_data_summary: str,
    deterministic_markdown: str,
    session_id: str = "default",
) -> str:
    """Use OpenAI to synthesize a polished, natural-language explanation strictly grounded in fetched facts.

    If OpenAI is not configured, rate-limited, or fails, returns the deterministic markdown directly.
    """
    api_key = get_server_openai_api_key()
    if not api_key or not rate_limiter.is_allowed(session_id):
        return deterministic_markdown

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=DEFAULT_TIMEOUT, max_retries=1)

        user_content = (
            f"User Question: {query}\n\n"
            f"Grounded Market Data & Disclosures:\n"
            f"\"\"\"\n{grounded_data_summary}\n\"\"\"\n\n"
            f"Draft Reference Output:\n"
            f"\"\"\"\n{deterministic_markdown}\n\"\"\"\n\n"
            f"Synthesize the final professional response following all constraints."
        )

        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0.2,
            max_tokens=MAX_RESPONSE_TOKENS,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )

        synthesized = response.choices[0].message.content
        if synthesized and len(synthesized.strip()) > 30:
            return synthesized.strip()

    except Exception as exc:
        logger.warning("OpenAI grounded synthesis failed: %s. Using deterministic output.", mask_sensitive_error(exc))

    return deterministic_markdown
