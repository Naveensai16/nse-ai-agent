"""Ollama natural-language understanding, structured classification, and grounded response synthesis.

Provides private server-side Ollama integration with strict concurrency/rate limits, timeout handling,
scope-aware structured intent classification, entity validation, and grounded response synthesis.
Never hardcodes or requires AI credentials from users or frontend.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import requests

from data.stock_master import CONGLOMERATE_GROUPS, INDIAN_STOCK_MASTER
from services.intent_service import MarketIntent, QueryScope, detect_intent, normalize_query
from services.symbol_resolver import check_conglomerate_ambiguity, resolve_nse_symbol

logger = logging.getLogger(__name__)

# Server-side environment configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30.0"))
MAX_PROMPT_LENGTH = int(os.getenv("OLLAMA_MAX_PROMPT_LENGTH", "1500"))
MAX_CLASSIFICATION_TOKENS = 80
MAX_RESPONSE_TOKENS = 250
RATE_LIMIT_REQUESTS_PER_MINUTE = 30
MAX_CONCURRENCY = int(os.getenv("OLLAMA_MAX_CONCURRENCY", "4"))

# Concurrency semaphore to prevent overwhelming local CPU/GPU
_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENCY)

# In-memory query classification cache to eliminate redundant CPU latency
_classification_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


# ==============================================================================
# PER-SESSION IN-MEMORY RATE LIMITER
# ==============================================================================


class SessionRateLimiter:
    """Thread-safe sliding-window rate limiter to protect server-side Ollama compute."""

    def __init__(self, max_per_minute: int = RATE_LIMIT_REQUESTS_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, session_id: str) -> bool:
        """Check if session is allowed to make a request under the rate limit."""
        with self._lock:
            now = time.time()
            window_start = now - 60.0

            timestamps = self._requests.get(session_id, [])
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= self.max_per_minute:
                self._requests[session_id] = valid_timestamps
                return False

            valid_timestamps.append(now)
            self._requests[session_id] = valid_timestamps
            return True


rate_limiter = SessionRateLimiter()


# ==============================================================================
# OLLAMA HEALTH & MODEL AVAILABILITY CHECK (CACHED)
# ==============================================================================

_health_cache: dict[str, Any] = {
    "reachable": False,
    "model_available": False,
    "available_models": [],
    "checked_at": 0.0,
    "error": None,
}
_health_lock = threading.Lock()


def check_ollama_health(ttl_seconds: float = 30.0, force_refresh: bool = False) -> dict[str, Any]:
    """Check if the local/server Ollama instance is reachable and the model is loaded.

    Caches health status to avoid pinging Ollama on every Streamlit rerun.
    """
    global _health_cache
    now = time.time()

    with _health_lock:
        if not force_refresh and (now - _health_cache["checked_at"]) < ttl_seconds:
            return dict(_health_cache)

        tags_url = f"{OLLAMA_BASE_URL}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                # Check if configured model is available (exact or prefix match e.g. 'llama3.2:1b' vs 'llama3.2:1b')
                target_base = OLLAMA_MODEL.split(":")[0]
                model_found = any(
                    m == OLLAMA_MODEL or m.startswith(f"{OLLAMA_MODEL}:") or m.startswith(f"{target_base}:")
                    for m in models
                )
                _health_cache = {
                    "reachable": True,
                    "model_available": model_found,
                    "available_models": models,
                    "configured_model": OLLAMA_MODEL,
                    "base_url": OLLAMA_BASE_URL,
                    "checked_at": now,
                    "error": None,
                }
            else:
                _health_cache = {
                    "reachable": False,
                    "model_available": False,
                    "available_models": [],
                    "configured_model": OLLAMA_MODEL,
                    "base_url": OLLAMA_BASE_URL,
                    "checked_at": now,
                    "error": f"Ollama HTTP {resp.status_code}",
                }
        except Exception as exc:
            _health_cache = {
                "reachable": False,
                "model_available": False,
                "available_models": [],
                "configured_model": OLLAMA_MODEL,
                "base_url": OLLAMA_BASE_URL,
                "checked_at": now,
                "error": str(exc),
            }

        return dict(_health_cache)


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
    source: str = "ollama"  # 'ollama' or 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CLASSIFICATION_SYSTEM_PROMPT = """You are a precision Natural-Language Understanding classifier for an Indian National Stock Exchange (NSE) market assistant.
Output ONLY a strict JSON object with these exact keys:

{
  "intent": "<INTENT_ENUM>",
  "scope": "<SCOPE_ENUM>",
  "symbols": ["<NSE_TICKER>"],
  "company_names": ["<COMPANY_NAME>"],
  "sector": "<SECTOR_NAME_OR_NULL>",
  "index": "<INDEX_NAME_OR_NULL>",
  "timeframe": "<TIMEFRAME>",
  "include_catalysts": <true|false>,
  "needs_clarification": <true|false>,
  "clarification_group": "<GROUP_NAME_OR_NULL>"
}

Supported INTENT_ENUM values:
- 52_WEEK_HIGH_STOCKS (all/market-wide scan for stocks near yearly high)
- 52_WEEK_LOW_STOCKS (all/market-wide scan for stocks near yearly low)
- TOP_GAINERS (top gaining stocks across NSE today)
- TOP_LOSERS (top falling stocks across NSE today)
- VOLUME_SPIKE (high volume or turnover stocks, most action, buzzing shares)
- MARKET_OVERVIEW (index status: Nifty 50, Sensex, Bank Nifty, overall market)
- SHORT_TERM_OPPORTUNITIES (2-day swing trading setups)
- STOCKS_BY_SECTOR (stocks in IT, Banking, Auto, Pharma, FMCG, Metals, Realty, etc.)
- SECTOR_PERFORMANCE (sector rankings, leading/lagging sectors)
- STOCK_PRICE (current quote/price of a single specific stock)
- STOCK_OVERVIEW (general details, overview, market cap of a stock)
- STOCK_FUNDAMENTALS (PE, valuation, dividend, debt, financials of a stock)
- STOCK_TECHNICALS (RSI, moving averages, technical sentiment of a stock)
- STOCK_52_WEEK_HIGH_LOW (52-week high and low of a specific single stock)
- STOCK_NEWS (recent news/catalysts for a specific stock)
- STOCK_RESULTS (quarterly results, earnings for a specific stock)
- STOCK_DECISION (buy/sell/hold decision or advisory question for a stock)
- STOCK_COMPARISON (comparing 2 or more specific stocks, e.g. TCS vs Infosys)
- AMBIGUOUS_STOCK (only when user provides an ambiguous group name like 'Tata' alone)
- FALLBACK (unclear, off-topic, or greeting)

Supported SCOPE_ENUM values:
- MARKET_WIDE (e.g. "52 Week High Stocks", "top gainers", "volume shockers")
- SINGLE_STOCK (e.g. "What is ITC 52 week high?", "Tata Motors price", "Tell me about YES Bank")
- MULTI_STOCK (e.g. "Compare TCS and Infosys")
- SECTOR (e.g. "Banking stocks", "IT sector performance")
- INDEX (e.g. "How is Nifty doing?", "Sensex status")
- AMBIGUOUS (only for group name like "Tata", "Adani", "Bajaj" without specific company)
- GENERAL (e.g. "market today", general inquiry)

Few-Shot Classification Examples:
Query: "52 Week High Stocks" -> {"intent": "52_WEEK_HIGH_STOCKS", "scope": "MARKET_WIDE", "symbols": [], "company_names": [], "needs_clarification": false, "clarification_group": null}
Query: "What is ITC 52 week high?" -> {"intent": "STOCK_52_WEEK_HIGH_LOW", "scope": "SINGLE_STOCK", "symbols": ["ITC"], "company_names": ["ITC"], "needs_clarification": false, "clarification_group": null}
Query: "Tell me about YES Bank" -> {"intent": "STOCK_OVERVIEW", "scope": "SINGLE_STOCK", "symbols": ["YESBANK"], "company_names": ["YES Bank"], "needs_clarification": false, "clarification_group": null}
Query: "Tata" -> {"intent": "AMBIGUOUS_STOCK", "scope": "AMBIGUOUS", "symbols": [], "company_names": [], "needs_clarification": true, "clarification_group": "Tata"}
Query: "What is its PE?" with previous stock TATAPOWER -> {"intent": "STOCK_FUNDAMENTALS", "scope": "SINGLE_STOCK", "symbols": ["TATAPOWER"], "company_names": ["Tata Power"], "needs_clarification": false, "clarification_group": null}
Output ONLY the valid JSON object without markdown fences."""


def _call_ollama_chat(
    messages: list[dict[str, str]],
    model: str = OLLAMA_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 500,
    format_json: bool = False,
    timeout: float = OLLAMA_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Execute raw HTTP request to Ollama /api/chat with timeout and concurrency limits."""
    acquired = _concurrency_semaphore.acquire(timeout=2.0)
    if not acquired:
        logger.warning("Ollama concurrency limit reached. Skipping LLM call.")
        return None

    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 1024,
                "num_thread": 8,
            },
        }
        if format_json:
            payload["format"] = "json"

        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            msg = data.get("message", {})
            return msg.get("content", "")
        else:
            logger.warning("Ollama returned HTTP %d: %s", resp.status_code, resp.text[:200])
            return None
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out after %.1fs", timeout)
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to Ollama server at %s", OLLAMA_BASE_URL)
        return None
    except Exception as exc:
        logger.warning("Ollama request failed: %s", exc)
        return None
    finally:
        _concurrency_semaphore.release()


def _classify_deterministically(
    clean_query: str,
    resolved_symbols: list[str],
    context: Optional[dict[str, Any]] = None,
) -> StructuredClassification:
    """Instant deterministic classification fallback and local engine."""
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


def classify_query(
    query: str,
    symbols: Optional[list[str]] = None,
    context: Optional[dict[str, Any]] = None,
    session_id: str = "default",
    use_llm: bool = True,
) -> StructuredClassification:
    """Classify user query using Ollama structured JSON output, falling back deterministically.

    Args:
        query: Raw user query string.
        symbols: Pre-resolved or extracted symbols (if already known).
        context: Optional conversation context dictionary.
        session_id: Unique session identifier for rate limiting.
        use_llm: If False, skips Ollama network call and uses instant deterministic classification.

    Returns:
        StructuredClassification object with verified intent, scope, and entities.
    """
    from agent.demo_agent import extract_symbols_from_query
    from services.conversation_context import resolve_coreferences

    context = context or {}
    clean_query = query.strip()[:MAX_PROMPT_LENGTH]

    # Resolve symbols incorporating context if not already pre-provided
    if symbols is not None and len(symbols) > 0:
        resolved_symbols = list(symbols)
    else:
        direct_symbols = extract_symbols_from_query(clean_query)
        resolved_symbols, _ = resolve_coreferences(clean_query, direct_symbols, context)

    # In-memory cache check to eliminate CPU duplicate inference
    cache_key = f"{clean_query.lower()}|{sorted(resolved_symbols)}"
    with _cache_lock:
        if cache_key in _classification_cache:
            return _classification_cache[cache_key]

    # If LLM disabled, jump directly to high-precision deterministic classification
    if not use_llm:
        sc = _classify_deterministically(clean_query, resolved_symbols, context)
        with _cache_lock:
            _classification_cache[cache_key] = sc
        return sc

    # Check health and rate limit
    health = check_ollama_health()
    if health["reachable"] and health["model_available"] and rate_limiter.is_allowed(session_id):
        try:
            ctx_hint = ""
            last_sym = context.get("last_symbol")
            last_syms = context.get("last_symbols", [])
            if last_sym:
                ctx_hint = f"\nPrevious conversation focused stock: {last_sym} (All recently discussed: {', '.join(last_syms)})"

            messages = [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: {clean_query}{ctx_hint}"},
            ]

            content = _call_ollama_chat(
                messages=messages,
                model=OLLAMA_MODEL,
                temperature=0.0,
                max_tokens=MAX_CLASSIFICATION_TOKENS,
                format_json=True,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )

            if content:
                # Attempt to parse JSON
                try:
                    parsed = json.loads(content)
                except Exception:
                    # Retry once with explicit json correction if malformed
                    retry_messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": "Your previous output was not valid JSON. Provide ONLY the valid JSON object."},
                    ]
                    retry_content = _call_ollama_chat(
                        messages=retry_messages,
                        model=OLLAMA_MODEL,
                        temperature=0.0,
                        max_tokens=MAX_CLASSIFICATION_TOKENS,
                        format_json=True,
                        timeout=10.0,
                    )
                    parsed = json.loads(retry_content) if retry_content else {}

                intent_str = parsed.get("intent", "").upper().strip()
                scope_str = parsed.get("scope", "").upper().strip()
                extracted_syms = parsed.get("symbols", [])
                comp_names = parsed.get("company_names", [])

                # Canonicalize symbols against INDIAN_STOCK_MASTER
                valid_syms: list[str] = []
                for s in extracted_syms:
                    res = resolve_nse_symbol(str(s))
                    sym_val = res.get("symbol") if isinstance(res, dict) else getattr(res, "symbol", None)
                    if sym_val and sym_val in INDIAN_STOCK_MASTER and sym_val not in valid_syms:
                        valid_syms.append(sym_val)

                # Also check company names if symbols empty
                if not valid_syms and comp_names:
                    for c in comp_names:
                        res = resolve_nse_symbol(str(c))
                        sym_val = res.get("symbol") if isinstance(res, dict) else getattr(res, "symbol", None)
                        if sym_val and sym_val in INDIAN_STOCK_MASTER and sym_val not in valid_syms:
                            valid_syms.append(sym_val)

                # Merge with context/rule resolved symbols if appropriate
                if not valid_syms and resolved_symbols and scope_str != QueryScope.MARKET_WIDE.value:
                    valid_syms = list(resolved_symbols)

                # Safety enforcement: Market-wide queries must never have single-stock symbols
                clean_q = normalize_query(clean_query).lower()
                is_market_52w_high = ("52" in clean_q or "yearly" in clean_q or "annual" in clean_q) and "high" in clean_q and not valid_syms
                is_market_52w_low = ("52" in clean_q or "yearly" in clean_q or "annual" in clean_q) and "low" in clean_q and not valid_syms

                if scope_str == QueryScope.MARKET_WIDE.value or is_market_52w_high or is_market_52w_low:
                    valid_syms = []
                    scope_str = QueryScope.MARKET_WIDE.value
                    if is_market_52w_high:
                        intent_str = MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS.value
                    elif is_market_52w_low:
                        intent_str = MarketIntent.FIFTY_TWO_WEEK_LOW_STOCKS.value

                # Conglomerate ambiguity override check
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
                                    source="ollama_validated",
                                )

                # Single stock refinement
                if valid_syms:
                    scope_str = QueryScope.SINGLE_STOCK.value if len(valid_syms) == 1 else QueryScope.MULTI_STOCK.value
                    if ("52" in clean_q or "yearly" in clean_q or "annual" in clean_q) and ("high" in clean_q or "low" in clean_q):
                        intent_str = MarketIntent.STOCK_52_WEEK_HIGH_LOW.value
                    elif any(k in clean_q for k in ("pe", "p/e", "valuation", "fundamental")):
                        intent_str = MarketIntent.STOCK_FUNDAMENTALS.value
                    elif any(k in clean_q for k in ("news", "headline")):
                        intent_str = MarketIntent.STOCK_NEWS.value
                    elif any(k in clean_q for k in ("rsi", "technical", "sentiment")):
                        intent_str = MarketIntent.STOCK_TECHNICALS.value
                    elif any(k in clean_q for k in ("result", "quarter", "earning")):
                        intent_str = MarketIntent.STOCK_RESULTS.value
                    elif any(k in clean_q for k in ("buy", "sell", "hold", "exit", "target", "invest")):
                        intent_str = MarketIntent.STOCK_DECISION.value

                # Map intent string to enum safely
                matched_intent = MarketIntent.FALLBACK.value
                for mi in MarketIntent:
                    if mi.value == intent_str:
                        matched_intent = mi.value
                        break

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
                        needs_clarification=False,
                        clarification_group=None,
                        source="ollama",
                    )
                    with _cache_lock:
                        _classification_cache[cache_key] = sc
                    return sc

        except Exception as exc:
            logger.warning("Ollama classification failed or timed out: %s. Using deterministic fallback.", exc)

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

    sc = StructuredClassification(
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
    with _cache_lock:
        _classification_cache[cache_key] = sc
    return sc


# ==============================================================================
# GROUNDED RESPONSE SYNTHESIS (Zero Hallucination)
# ==============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a factual Indian stock market analyst.
You are provided with verified real-time National Stock Exchange (NSE) market data and disclosures.
Summarize and explain the provided factual data concisely in 1 to 2 clear paragraphs.
Never invent numbers, dates, or prices not in the supplied text.
If any metric is missing, note that it is currently unavailable."""


def synthesize_grounded_response(
    query: str,
    grounded_data_summary: str,
    deterministic_markdown: str,
    session_id: str = "default",
) -> str:
    """Use Ollama to synthesize a polished, natural-language explanation strictly grounded in fetched facts.

    If Ollama is not configured, unreachable, or fails, returns the deterministic markdown directly.
    """
    health = check_ollama_health()
    if not (health["reachable"] and health["model_available"]) or not rate_limiter.is_allowed(session_id):
        return deterministic_markdown

    try:
        trimmed_summary = grounded_data_summary[:800]
        user_content = (
            f"User Question: {query}\n\n"
            f"Verified Market Data:\n{trimmed_summary}\n\n"
            f"Provide a concise, grounded factual explanation."
        )

        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        synthesized = _call_ollama_chat(
            messages=messages,
            model=OLLAMA_MODEL,
            temperature=0.1,
            max_tokens=150,
            format_json=False,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )

        if synthesized and len(synthesized.strip()) > 25:
            cleaned_synth = synthesized.strip()
            # If deterministic markdown has structured tables or cards, combine them
            if "|" in deterministic_markdown or "###" in deterministic_markdown:
                return f"{cleaned_synth}\n\n---\n\n{deterministic_markdown}"
            return cleaned_synth

    except Exception as exc:
        logger.warning("Ollama grounded synthesis failed: %s. Using deterministic output.", exc)

    return deterministic_markdown
