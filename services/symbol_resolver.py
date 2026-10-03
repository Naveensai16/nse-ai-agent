"""Authoritative symbol and company entity resolution service for Indian equities (NSE / BSE).

Converts natural-language company names, colloquial aliases, legal entity variants,
and NSE/BSE ticker symbols into canonical SecurityEntity objects and ticker symbols.

Enforces:
1. Priority 1: Exact NSE Symbol (case-insensitive, strip prefixes/suffixes)
2. Priority 2: Exact Official Company Name
3. Priority 3: Exact Known Alias (e.g. 'YES Bank' -> YESBANK, 'SBI' -> SBIN)
4. Priority 4: Normalized Exact Match (suffix stripping, concatenations)
5. Priority 5: Token Matching with strict Generic Word Protection
6. Priority 6: Fuzzy String Similarity with distinctive token protection
7. Priority 7: Dynamic Online Search Fallback

Strict Generic Word Suppression:
- Generic words (Bank, Power, Steel, Cement, Motors, India, Finance, etc.)
  must NEVER match alone.
- Conglomerates ('Tata', 'Adani', 'Birla') return is_ambiguous=True with candidate options.
- NO MATCH is always better than WRONG MATCH.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Optional

import yfinance as yf

from data.stock_master import (
    CONGLOMERATE_GROUPS,
    GENERIC_FINANCIAL_WORDS,
    INDIAN_STOCK_MASTER,
)
from models.security import SecurityEntity

logger = logging.getLogger(__name__)

# Common corporate and legal suffixes to strip during normalization
LEGAL_ENTITY_SUFFIXES: tuple[str, ...] = (
    "private limited",
    "pvt ltd",
    "pvt limited",
    "company limited",
    "co limited",
    "co ltd",
    "comp ltd",
    "limited",
    "ltd",
    "company",
    "corporation",
    "corp",
    "co",
    "share",
    "shares",
    "stocks",
    "stock",
    "enterprise",
    "enterprises",
)

# Exported master directory for backward compatibility with existing tests
NSE_MASTER_DIRECTORY: dict[str, tuple[str, list[str]]] = {
    sym: (rec["company_name"], rec.get("aliases", []))
    for sym, rec in INDIAN_STOCK_MASTER.items()
}

# Lookup indices
_EXACT_SYMBOL_MAP: dict[str, str] = {}
_OFFICIAL_NAME_MAP: dict[str, str] = {}
_ALIAS_LOOKUP_MAP: dict[str, str] = {}
_CONCATENATED_MAP: dict[str, str] = {}


def _clean_text(s: str) -> str:
    """Normalize string by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not s:
        return ""
    # Retain alphanumeric characters, '&' and whitespace
    subbed = re.sub(r"[^\w\s&]", " ", s.lower())
    return " ".join(subbed.split())


def _strip_entity_suffixes(cleaned_phrase: str) -> str:
    """Remove legal and entity suffixes (e.g. 'company limited', 'ltd') from end of phrase."""
    if not cleaned_phrase:
        return ""

    phrase = cleaned_phrase.strip()
    modified = True
    while modified:
        modified = False
        for suffix in LEGAL_ENTITY_SUFFIXES:
            if phrase == suffix:
                continue
            if phrase.endswith(" " + suffix):
                phrase = phrase[: -len(suffix)].strip()
                modified = True
                break
    return phrase


def _build_indices() -> None:
    """Build high-speed lookup tables from INDIAN_STOCK_MASTER."""
    global _EXACT_SYMBOL_MAP, _OFFICIAL_NAME_MAP, _ALIAS_LOOKUP_MAP, _CONCATENATED_MAP
    if _EXACT_SYMBOL_MAP:
        return

    sym_map: dict[str, str] = {}
    name_map: dict[str, str] = {}
    alias_map: dict[str, str] = {}
    concat_map: dict[str, str] = {}

    for sym, data in INDIAN_STOCK_MASTER.items():
        sym_clean = sym.upper().strip()
        sym_map[sym_clean] = sym_clean

        official_name = data.get("company_name", "")
        clean_off = _clean_text(official_name)
        if clean_off:
            name_map[clean_off] = sym_clean
            stripped_off = _strip_entity_suffixes(clean_off)
            if stripped_off and stripped_off not in GENERIC_FINANCIAL_WORDS:
                alias_map[stripped_off] = sym_clean
            # Concatenated form of official name
            no_space_off = clean_off.replace(" ", "")
            if len(no_space_off) >= 3:
                concat_map[no_space_off] = sym_clean

        # Register aliases
        for alias in data.get("aliases", []):
            clean_al = _clean_text(alias)
            if not clean_al or len(clean_al) < 2:
                continue
            # Do NOT register generic words as aliases!
            if clean_al in GENERIC_FINANCIAL_WORDS:
                continue
            alias_map[clean_al] = sym_clean
            
            stripped_al = _strip_entity_suffixes(clean_al)
            if stripped_al and stripped_al not in GENERIC_FINANCIAL_WORDS:
                alias_map[stripped_al] = sym_clean

            no_spaces = clean_al.replace(" ", "")
            if len(no_spaces) >= 3 and no_spaces not in GENERIC_FINANCIAL_WORDS:
                concat_map[no_spaces] = sym_clean

    _EXACT_SYMBOL_MAP = sym_map
    _OFFICIAL_NAME_MAP = name_map
    _ALIAS_LOOKUP_MAP = alias_map
    _CONCATENATED_MAP = concat_map


_build_indices()


def is_generic_query(cleaned_query: str) -> bool:
    """Check if query consists solely of generic financial/corporate words and is NOT a known company."""
    if not cleaned_query:
        return False
    clean = cleaned_query.strip().lower()
    # If the cleaned query is a registered official company name or registered alias, it represents a real listed entity!
    if clean in _OFFICIAL_NAME_MAP or clean in _ALIAS_LOOKUP_MAP:
        return False
    words = clean.split()
    if not words:
        return False
    return all(w in GENERIC_FINANCIAL_WORDS for w in words)


def check_conglomerate_ambiguity(
    raw_query: str,
    cleaned_query: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Check if the query is an ambiguous conglomerate group name."""
    if not raw_query:
        return None
    if cleaned_query is None:
        cleaned_query = _clean_text(raw_query)
    stripped = _strip_entity_suffixes(cleaned_query)
    candidates_group = CONGLOMERATE_GROUPS.get(cleaned_query) or CONGLOMERATE_GROUPS.get(stripped)
    if candidates_group:
        cand_list = candidates_group["candidates"]
        names_str = ", ".join(f"{c['name']} ({c['symbol']})" for c in cand_list[:4])
        return {
            "input": raw_query,
            "symbol": None,
            "yahoo_symbol": None,
            "company_name": None,
            "exchange": "NSE",
            "confidence": 0.0,
            "match_type": "ambiguous_conglomerate",
            "is_ambiguous": True,
            "candidates": cand_list,
            "error": (
                f"'{raw_query}' is a conglomerate group with multiple listed entities. "
                f"Please specify which company you mean (e.g., {names_str})."
            ),
        }
    return None


def _strip_carrier_phrases(text: str) -> str:
    """Strip conversational prefixes, suffixes, and intent phrasing from query."""
    t = text.strip()
    
    prefix_patterns = [
        r"^(?:tell\s+me\s+about|tell\s+about|can\s+you\s+tell\s+me\s+about)\s+",
        r"^(?:what\s+is\s+the\s+target\s+(?:for|of)|what\s+is\s+target\s+(?:for|of))\s+",
        r"^(?:what\s+is\s+the\s+price\s+of|what\s+is\s+price\s+of|share\s+price\s+of|stock\s+price\s+of)\s+",
        r"^(?:what\s+about|how\s+about|how\s+is)\s+",
        r"^(?:should\s+i\s+(?:buy|hold|sell|exit|average|keep)|can\s+i\s+(?:buy|hold|sell|exit|average|keep))\s+",
        r"^(?:is\s+this\s+a\s+good\s+time\s+to\s+buy|is\s+it\s+good\s+to\s+buy|is)\s+",
        r"^(?:i\s+already\s+own|i\s+own|i\s+bought|i\s+purchased|i\s+have)\s+",
        r"^(?:give\s+me\s+(?:details|information|info|news|analysis)\s+(?:of|about|on))\s+",
        r"^(?:analyze|analysis\s+of|details\s+of)\s+",
    ]
    for pat in prefix_patterns:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()

    suffix_patterns = [
        r"\s+(?:stock|stocks|share|shares)\s+for\s+\d+\s*(?:year|years|month|months|days|day|session|sessions)[\?\.]*$",
        r"\s+for\s+(?:another\s+)?\d+\s*(?:year|years|month|months|days|day|session|sessions)[\?\.]*$",
        r"\s+for\s+the\s+(?:next\s+)?\d+\s*(?:year|years|month|months|days|day|session|sessions)[\?\.]*$",
        r"\s+for\s+the\s+(?:long|short)\s+term[\?\.]*$",
        r"\s+for\s+(?:long|short)\s+term[\?\.]*$",
        r"\s+for\s+(?:swing|day)\s+trading[\?\.]*$",
        r"\s+(?:a\s+)?good\s+buy[\?\.]*$",
        r"\s+good\s+to\s+buy(?:\s+now)?[\?\.]*$",
        r"\s+worth\s+buying[\?\.]*$",
        r"\s+target(?:\s+price)?[\?\.]*$",
        r"\s+outlook[\?\.]*$",
        r"[\?\.]+$",
    ]
    for pat in suffix_patterns:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()

    return t



def resolve_nse_symbol(
    query: str,
    allow_online_lookup: bool = True,
) -> dict[str, Any]:
    """Resolve natural company names, aliases, and tickers to an authoritative NSE security.

    Args:
        query: Company name or ticker string (e.g. 'Tata Power', 'YES Bank', 'SBIN').
        allow_online_lookup: Whether to query Yahoo Finance Search as a fallback.

    Returns:
        Dictionary containing canonical security information.
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "input": query,
            "symbol": None,
            "yahoo_symbol": None,
            "company_name": None,
            "exchange": "NSE",
            "confidence": 0.0,
            "is_ambiguous": False,
            "candidates": [],
            "error": "Query cannot be empty or non-string.",
        }

    raw_query = query.strip()
    q_clean = _clean_text(raw_query)

    # -------------------------------------------------------------
    # Step 0: Conglomerate Group & Pure Generic Word Checks
    # -------------------------------------------------------------
    conglomerate_res = check_conglomerate_ambiguity(raw_query, q_clean)
    if conglomerate_res:
        return conglomerate_res

    # Check if user query is purely generic words (e.g. 'Bank', 'Power', 'Cement')
    if is_generic_query(q_clean):
        return {
            "input": raw_query,
            "symbol": None,
            "yahoo_symbol": None,
            "company_name": None,
            "exchange": "NSE",
            "confidence": 0.0,
            "match_type": "generic_word_suppressed",
            "is_ambiguous": True,
            "candidates": [],
            "error": (
                f"'{raw_query}' is a generic sector or corporate term, not a specific listed company. "
                "Please provide a specific company name (e.g., 'YES Bank', 'HDFC Bank', 'State Bank of India')."
            ),
        }

    # -------------------------------------------------------------
    # Priority 1: Exact NSE Symbol
    # -------------------------------------------------------------
    sym_candidate = raw_query.upper().strip()
    for prefix in ("NSE:", "NSE_"):
        if sym_candidate.startswith(prefix):
            sym_candidate = sym_candidate[len(prefix) :].strip()
    for suffix in (".NS", ".BO", "-EQ", ".EQ", "-BE"):
        if sym_candidate.endswith(suffix):
            sym_candidate = sym_candidate[: -len(suffix)].strip()

    if sym_candidate in _EXACT_SYMBOL_MAP:
        resolved_sym = _EXACT_SYMBOL_MAP[sym_candidate]
        master_data = INDIAN_STOCK_MASTER.get(resolved_sym, {})
        return {
            "input": raw_query,
            "symbol": resolved_sym,
            "yahoo_symbol": f"{resolved_sym}.NS",
            "company_name": master_data.get("company_name", resolved_sym),
            "isin": master_data.get("isin"),
            "sector": master_data.get("sector"),
            "industry": master_data.get("industry"),
            "bse_code": master_data.get("bse_code"),
            "exchange": "NSE",
            "confidence": 1.0,
            "match_type": "exact_symbol",
            "is_ambiguous": False,
            "candidates": [],
        }

    # -------------------------------------------------------------
    # Priority 2: Exact Official Company Name
    # -------------------------------------------------------------
    if q_clean in _OFFICIAL_NAME_MAP:
        resolved_sym = _OFFICIAL_NAME_MAP[q_clean]
        master_data = INDIAN_STOCK_MASTER.get(resolved_sym, {})
        return {
            "input": raw_query,
            "symbol": resolved_sym,
            "yahoo_symbol": f"{resolved_sym}.NS",
            "company_name": master_data.get("company_name", resolved_sym),
            "isin": master_data.get("isin"),
            "sector": master_data.get("sector"),
            "industry": master_data.get("industry"),
            "bse_code": master_data.get("bse_code"),
            "exchange": "NSE",
            "confidence": 1.0,
            "match_type": "exact_company_name",
            "is_ambiguous": False,
            "candidates": [],
        }

    # -------------------------------------------------------------
    # Priority 3: Exact Known Alias
    # -------------------------------------------------------------
    if q_clean in _ALIAS_LOOKUP_MAP:
        resolved_sym = _ALIAS_LOOKUP_MAP[q_clean]
        master_data = INDIAN_STOCK_MASTER.get(resolved_sym, {})
        return {
            "input": raw_query,
            "symbol": resolved_sym,
            "yahoo_symbol": f"{resolved_sym}.NS",
            "company_name": master_data.get("company_name", resolved_sym),
            "isin": master_data.get("isin"),
            "sector": master_data.get("sector"),
            "industry": master_data.get("industry"),
            "bse_code": master_data.get("bse_code"),
            "exchange": "NSE",
            "confidence": 1.0,
            "match_type": "exact_alias",
            "is_ambiguous": False,
            "candidates": [],
        }

    # -------------------------------------------------------------
    # Priority 4: Normalized Exact Match (Suffix Stripping & Concatenation)
    # -------------------------------------------------------------
    stripped_q = _strip_entity_suffixes(q_clean)
    if stripped_q and stripped_q in _ALIAS_LOOKUP_MAP:
        resolved_sym = _ALIAS_LOOKUP_MAP[stripped_q]
        master_data = INDIAN_STOCK_MASTER.get(resolved_sym, {})
        return {
            "input": raw_query,
            "symbol": resolved_sym,
            "yahoo_symbol": f"{resolved_sym}.NS",
            "company_name": master_data.get("company_name", resolved_sym),
            "isin": master_data.get("isin"),
            "sector": master_data.get("sector"),
            "industry": master_data.get("industry"),
            "bse_code": master_data.get("bse_code"),
            "exchange": "NSE",
            "confidence": 1.0,
            "match_type": "suffix_stripped_alias",
            "is_ambiguous": False,
            "candidates": [],
        }

    no_spaces_q = q_clean.replace(" ", "")
    if no_spaces_q in _CONCATENATED_MAP:
        resolved_sym = _CONCATENATED_MAP[no_spaces_q]
        master_data = INDIAN_STOCK_MASTER.get(resolved_sym, {})
        return {
            "input": raw_query,
            "symbol": resolved_sym,
            "yahoo_symbol": f"{resolved_sym}.NS",
            "company_name": master_data.get("company_name", resolved_sym),
            "isin": master_data.get("isin"),
            "sector": master_data.get("sector"),
            "industry": master_data.get("industry"),
            "bse_code": master_data.get("bse_code"),
            "exchange": "NSE",
            "confidence": 0.95,
            "match_type": "concatenated_alias",
            "is_ambiguous": False,
            "candidates": [],
        }

    if stripped_q:
        no_spaces_stripped = stripped_q.replace(" ", "")
        if no_spaces_stripped in _CONCATENATED_MAP:
            resolved_sym = _CONCATENATED_MAP[no_spaces_stripped]
            master_data = INDIAN_STOCK_MASTER.get(resolved_sym, {})
            return {
                "input": raw_query,
                "symbol": resolved_sym,
                "yahoo_symbol": f"{resolved_sym}.NS",
                "company_name": master_data.get("company_name", resolved_sym),
                "isin": master_data.get("isin"),
                "sector": master_data.get("sector"),
                "industry": master_data.get("industry"),
                "bse_code": master_data.get("bse_code"),
                "exchange": "NSE",
                "confidence": 0.95,
                "match_type": "concatenated_stripped_alias",
                "is_ambiguous": False,
                "candidates": [],
            }

    # -------------------------------------------------------------
    # Priority 4b: Conversational Carrier Stripping & Substring Entity Matching
    # -------------------------------------------------------------
    stripped_carrier = _strip_carrier_phrases(raw_query)
    if stripped_carrier and stripped_carrier.lower() != raw_query.lower():
        sub_res = resolve_nse_symbol(stripped_carrier, allow_online_lookup=allow_online_lookup)
        if sub_res.get("symbol") or sub_res.get("is_ambiguous"):
            sub_res["input"] = raw_query
            return sub_res

    # Scan query for longest known multi-word alias (length >= 2 words)
    # This directly finds "yes bank", "state bank of india", "tata power", etc. in queries like
    # "Tell me about YES Bank" or "What is target for yes bank?"
    matched_sub_sym: Optional[str] = None
    max_alias_words = 0
    max_alias_len = 0

    for al_clean, sym in _ALIAS_LOOKUP_MAP.items():
        al_words = al_clean.split()
        if len(al_words) < 2:
            continue
        pattern = r"\b" + re.escape(al_clean) + r"\b"
        if re.search(pattern, q_clean):
            if len(al_words) > max_alias_words or (len(al_words) == max_alias_words and len(al_clean) > max_alias_len):
                matched_sub_sym = sym
                max_alias_words = len(al_words)
                max_alias_len = len(al_clean)

    if matched_sub_sym:
        master_data = INDIAN_STOCK_MASTER.get(matched_sub_sym, {})
        return {
            "input": raw_query,
            "symbol": matched_sub_sym,
            "yahoo_symbol": f"{matched_sub_sym}.NS",
            "company_name": master_data.get("company_name", matched_sub_sym),
            "isin": master_data.get("isin"),
            "sector": master_data.get("sector"),
            "industry": master_data.get("industry"),
            "bse_code": master_data.get("bse_code"),
            "exchange": "NSE",
            "confidence": 0.95,
            "match_type": "substring_alias_match",
            "is_ambiguous": False,
            "candidates": [],
        }

    # Also scan for exact standalone ticker in multi-word sentence (e.g. 'TCS', 'INFY', 'SBI')
    for sym_code in _EXACT_SYMBOL_MAP.keys():
        if len(sym_code) < 3 or sym_code.lower() in GENERIC_FINANCIAL_WORDS:
            continue
        pattern = r"\b" + re.escape(sym_code.lower()) + r"\b"
        if re.search(pattern, q_clean):
            master_data = INDIAN_STOCK_MASTER.get(sym_code, {})
            return {
                "input": raw_query,
                "symbol": sym_code,
                "yahoo_symbol": f"{sym_code}.NS",
                "company_name": master_data.get("company_name", sym_code),
                "isin": master_data.get("isin"),
                "sector": master_data.get("sector"),
                "industry": master_data.get("industry"),
                "bse_code": master_data.get("bse_code"),
                "exchange": "NSE",
                "confidence": 0.95,
                "match_type": "substring_symbol_match",
                "is_ambiguous": False,
                "candidates": [],
            }

    # -------------------------------------------------------------
    # Priority 5: Token / Substring Matching with Generic Word Protection
    # -------------------------------------------------------------
    query_words = (stripped_q or q_clean).split()
    query_tokens = set(query_words)
    distinctive_tokens = {w for w in query_tokens if w not in GENERIC_FINANCIAL_WORDS}

    # If query has at least one distinctive token and multiple words:
    if distinctive_tokens and len(query_words) >= 2:
        best_sym: Optional[str] = None
        best_overlap_count = 0
        best_key_len = 999

        for key, sym in _ALIAS_LOOKUP_MAP.items():
            key_words = key.split()
            key_tokens = set(key_words)
            # Require all distinctive tokens of query to be present in key!
            if not distinctive_tokens.issubset(key_tokens):
                continue

            common = query_tokens.intersection(key_tokens)
            # If all query tokens are matched or distinctive tokens strictly match key
            if len(common) > best_overlap_count or (len(common) == best_overlap_count and len(key_words) < best_key_len):
                best_sym = sym
                best_overlap_count = len(common)
                best_key_len = len(key_words)

        if best_sym:
            master_data = INDIAN_STOCK_MASTER.get(best_sym, {})
            return {
                "input": raw_query,
                "symbol": best_sym,
                "yahoo_symbol": f"{best_sym}.NS",
                "company_name": master_data.get("company_name", best_sym),
                "isin": master_data.get("isin"),
                "sector": master_data.get("sector"),
                "industry": master_data.get("industry"),
                "bse_code": master_data.get("bse_code"),
                "exchange": "NSE",
                "confidence": 0.90,
                "match_type": "token_match",
                "is_ambiguous": False,
                "candidates": [],
            }

    # -------------------------------------------------------------
    # Priority 6: Fuzzy String Similarity with Distinctive Protection
    # -------------------------------------------------------------
    # NEVER allow fuzzy matching if the query is just a generic word or too short
    target_clean = stripped_q or q_clean
    if len(target_clean) >= 4 and not is_generic_query(target_clean):
        # We only consider candidate keys that share distinctive character patterns
        candidate_keys = [
            k for k in _ALIAS_LOOKUP_MAP.keys()
            if not is_generic_query(k) and abs(len(k) - len(target_clean)) <= 4
        ]
        close_matches = difflib.get_close_matches(target_clean, candidate_keys, n=1, cutoff=0.78)
        if close_matches:
            match_key = close_matches[0]
            sim = difflib.SequenceMatcher(None, target_clean, match_key).ratio()
            # Double check that we don't accidentally match across distinct entity types
            # (e.g. check first char or major word overlap)
            resolved_sym = _ALIAS_LOOKUP_MAP[match_key]
            master_data = INDIAN_STOCK_MASTER.get(resolved_sym, {})
            return {
                "input": raw_query,
                "symbol": resolved_sym,
                "yahoo_symbol": f"{resolved_sym}.NS",
                "company_name": master_data.get("company_name", resolved_sym),
                "isin": master_data.get("isin"),
                "sector": master_data.get("sector"),
                "industry": master_data.get("industry"),
                "bse_code": master_data.get("bse_code"),
                "exchange": "NSE",
                "confidence": round(sim, 2),
                "match_type": "fuzzy_alias",
                "is_ambiguous": False,
                "candidates": [],
            }

    # -------------------------------------------------------------
    # Priority 7: Dynamic Online Search Fallback
    # -------------------------------------------------------------
    if allow_online_lookup and len(target_clean) >= 3 and not is_generic_query(target_clean):
        search_terms = [target_clean, raw_query]
        for term in search_terms:
            try:
                search_results = yf.Search(term)
                quotes = getattr(search_results, "quotes", []) or []
                for q in quotes:
                    sym = q.get("symbol", "")
                    exch = q.get("exchange", "")
                    if sym.endswith(".NS") or exch in ("NSI", "NSE"):
                        clean_sym = sym[:-3] if sym.endswith(".NS") else sym
                        clean_sym = clean_sym.split(".")[0].upper()
                        # Verify the result is not generic
                        short_name = q.get("shortname") or q.get("longname") or clean_sym
                        return {
                            "input": raw_query,
                            "symbol": clean_sym,
                            "yahoo_symbol": f"{clean_sym}.NS",
                            "company_name": short_name,
                            "isin": None,
                            "sector": None,
                            "industry": None,
                            "bse_code": None,
                            "exchange": "NSE",
                            "confidence": 0.85,
                            "match_type": "online_search",
                            "is_ambiguous": False,
                            "candidates": [],
                        }
            except Exception as exc:
                logger.debug("Online search fallback error for '%s': %s", term, exc)

    return {
        "input": raw_query,
        "symbol": None,
        "yahoo_symbol": None,
        "company_name": None,
        "exchange": "NSE",
        "confidence": 0.0,
        "match_type": "no_match",
        "is_ambiguous": False,
        "candidates": [],
        "error": f"Could not resolve '{raw_query}' to an NSE ticker symbol.",
    }


def get_security_entity(symbol_or_query: str) -> Optional[SecurityEntity]:
    """Retrieve or resolve a canonical SecurityEntity instance."""
    if not symbol_or_query:
        return None
    res = resolve_nse_symbol(symbol_or_query, allow_online_lookup=False)
    if not res.get("symbol") and not res.get("is_ambiguous"):
        return None
    return SecurityEntity(
        symbol=res.get("symbol"),
        company_name=res.get("company_name") or (symbol_or_query if res.get("is_ambiguous") else None),
        yahoo_symbol=res.get("yahoo_symbol"),
        isin=res.get("isin"),
        sector=res.get("sector"),
        industry=res.get("industry"),
        bse_code=res.get("bse_code"),
        exchange=res.get("exchange", "NSE"),
        confidence=res.get("confidence", 1.0),
        match_type=res.get("match_type", "exact_symbol"),
        is_ambiguous=res.get("is_ambiguous", False),
        candidates=res.get("candidates", []),
        error=res.get("error"),
    )


def search_stocks(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search or autocomplete Indian listed equities by ticker, name, or alias.

    Args:
        query: User input query string.
        limit: Maximum results to return (default 10).

    Returns:
        List of matching company dictionary summaries.
    """
    if not query or not query.strip():
        return []

    clean_q = _clean_text(query)
    q_upper = query.upper().strip()
    results: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()

    def _add_result(sym: str, score: int) -> None:
        if sym in seen_symbols:
            return
        data = INDIAN_STOCK_MASTER.get(sym, {})
        seen_symbols.add(sym)
        results.append({
            "symbol": sym,
            "company_name": data.get("company_name", sym),
            "sector": data.get("sector"),
            "industry": data.get("industry"),
            "isin": data.get("isin"),
            "bse_code": data.get("bse_code"),
            "display": f"{data.get('company_name', sym)} ({sym})",
            "_score": score,
        })

    # 0. Check for parenthesized symbol or exact display match e.g. "Tata Power Company Limited (TATAPOWER)"
    paren_match = re.search(r"\(([A-Z0-9_\-\.\&]+)\)\s*$", query.strip(), re.IGNORECASE)
    if paren_match:
        extracted_sym = paren_match.group(1).upper()
        if extracted_sym in INDIAN_STOCK_MASTER:
            _add_result(extracted_sym, score=100)
    for sym, data in INDIAN_STOCK_MASTER.items():
        disp = f"{data.get('company_name', sym)} ({sym})"
        if disp.strip().lower() == query.strip().lower():
            _add_result(sym, score=100)

    # 1. Exact match (symbol or company name)
    if q_upper in INDIAN_STOCK_MASTER:
        _add_result(q_upper, score=100)
    for sym, data in INDIAN_STOCK_MASTER.items():
        if _clean_text(data.get("company_name", "")) == clean_q:
            _add_result(sym, score=100)

    # 2. Priority 1: Company name starts with entered text
    for sym, data in INDIAN_STOCK_MASTER.items():
        name_clean = _clean_text(data.get("company_name", ""))
        if name_clean.startswith(clean_q):
            _add_result(sym, score=90)

    # 3. Priority 2: Symbol starts with entered text
    for sym in INDIAN_STOCK_MASTER.keys():
        if sym.startswith(q_upper):
            _add_result(sym, score=80)

    # 4. Priority 3: Company name contains entered text
    for sym, data in INDIAN_STOCK_MASTER.items():
        name_clean = _clean_text(data.get("company_name", ""))
        if clean_q in name_clean:
            _add_result(sym, score=70)

    # 5. Priority 4: Alias starts with or contains entered text
    for sym, data in INDIAN_STOCK_MASTER.items():
        for al in data.get("aliases", []):
            al_clean = _clean_text(al)
            if al_clean.startswith(clean_q):
                _add_result(sym, score=60)
                break
            elif clean_q in al_clean:
                _add_result(sym, score=50)
                break

    # 6. Priority 5: Symbol contains entered text
    for sym in INDIAN_STOCK_MASTER.keys():
        if clean_q in sym.lower():
            _add_result(sym, score=40)

    # Sort results by score descending, then shorter company name, then alphabetically
    results.sort(key=lambda x: (-x["_score"], len(x["company_name"]), x["company_name"]))
    for r in results:
        r.pop("_score", None)
        r["display_text"] = r["display"]

    return results[:limit]
