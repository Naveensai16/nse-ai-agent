"""Fallback demo / offline rule-based agent for the NSE stock analysis application.

Enables instant evaluation and testing of real live market data, company fundamentals,
sentiment analysis, and comparative stock intelligence for arbitrary N companies without
requiring an external OpenAI API key.
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import yfinance as yf

from data.stock_master import (
    CONGLOMERATE_GROUPS,
    GENERIC_FINANCIAL_WORDS,
    INDIAN_STOCK_MASTER,
)
from services.database_service import add_message, create_conversation, get_conversation, initialize_database
from services.symbol_resolver import (
    _ALIAS_LOOKUP_MAP,
    NSE_MASTER_DIRECTORY,
    _clean_text,
    check_conglomerate_ambiguity,
    get_security_entity,
    is_generic_query,
    resolve_nse_symbol,
)
from services.decision_service import detect_user_intent_and_details
from tools.analysis_tool import get_company_analysis

from tools.company_tool import get_company_info
from tools.decision_tool import get_stock_decision
from tools.market_tool import get_market_index, get_top_gainers, get_top_losers
from tools.news_tool import get_market_news
from tools.opportunity_tool import get_short_term_opportunities
from tools.sector_tool import get_top_sectors_and_companies
from tools.sentiment_tool import get_stock_sentiment
from tools.stock_tool import get_stock_price
from utils.helpers import normalize_nse_symbol

logger = logging.getLogger(__name__)

# Common stock name / alias to NSE ticker symbol mapping
STOCK_ALIASES: dict[str, str] = {
    "INDIA CEMENTS": "INDIACEM",
    "INDIA CEMENT": "INDIACEM",
    "INDIACEM": "INDIACEM",
    "PNC INFRATECH": "PNCINFRA",
    "PNC INFRA": "PNCINFRA",
    "PNCINFRA": "PNCINFRA",
    "SBI": "SBIN",
    "STATE BANK OF INDIA": "SBIN",
    "SBIN": "SBIN",
    "SBI BANK": "SBIN",
    "STATE BANK": "SBIN",
    "HDFC": "HDFCBANK",
    "HDFC BANK": "HDFCBANK",
    "HDFCBANK": "HDFCBANK",
    "ICICI": "ICICIBANK",
    "ICICI BANK": "ICICIBANK",
    "ICICIBANK": "ICICIBANK",
    "KOTAK": "KOTAKBANK",
    "KOTAK BANK": "KOTAKBANK",
    "KOTAKBANK": "KOTAKBANK",
    "AXIS": "AXISBANK",
    "AXIS BANK": "AXISBANK",
    "AXISBANK": "AXISBANK",
    "INDUSIND": "INDUSINDBK",
    "INDUSIND BANK": "INDUSINDBK",
    "PNB": "PNB",
    "PUNJAB NATIONAL BANK": "PNB",
    "CANARA": "CANBK",
    "CANARA BANK": "CANBK",
    "BANK OF BARODA": "BANKBARODA",
    "BOB": "BANKBARODA",
    "TCS": "TCS",
    "TATA CONSULTANCY": "TCS",
    "TATA CONSULTANCY SERVICES": "TCS",
    "INFY": "INFY",
    "INFOSYS": "INFY",
    "WIPRO": "WIPRO",
    "WIRPO": "WIPRO",  # common typo
    "HCLTECH": "HCLTECH",
    "HCL TECH": "HCLTECH",
    "HCL": "HCLTECH",
    "TECH MAHINDRA": "TECHM",
    "TECHM": "TECHM",
    "LTIM": "LTIM",
    "LTIMINDTREE": "LTIM",
    "RELIANCE": "RELIANCE",
    "RIL": "RELIANCE",
    "TATA MOTORS": "TATAMOTORS",
    "TATAMOTORS": "TATAMOTORS",
    "MARUTI": "MARUTI",
    "MARUTI SUZUKI": "MARUTI",
    "BAJAJ FINANCE": "BAJFINANCE",
    "BAJFINANCE": "BAJFINANCE",
    "BAJAJ FINSERV": "BAJAJFINSV",
    "BAJAJFINSV": "BAJAJFINSV",
    "L&T": "LT",
    "LARSEN": "LT",
    "LT": "LT",
    "LARSEN & TOUBRO": "LT",
    "ITC": "ITC",
    "BHARTI": "BHARTIARTL",
    "AIRTEL": "BHARTIARTL",
    "BHARTI AIRTEL": "BHARTIARTL",
    "BHARTIARTL": "BHARTIARTL",
    "SUN PHARMA": "SUNPHARMA",
    "SUNPHARMA": "SUNPHARMA",
    "TITAN": "TITAN",
    "YES BANK": "YESBANK",
    "YESBANK": "YESBANK",
    "ADANI ENT": "ADANIENT",
    "ADANIENT": "ADANIENT",
    "ADANI PORTS": "ADANIPORTS",
    "ADANIPORTS": "ADANIPORTS",
    "TATA STEEL": "TATASTEEL",
    "TATASTEEL": "TATASTEEL",
    "JSW STEEL": "JSWSTEEL",
    "JSWSTEEL": "JSWSTEEL",
    "NTPC": "NTPC",
    "POWERGRID": "POWERGRID",
    "ONGC": "ONGC",
    "COAL INDIA": "COALINDIA",
    "COALINDIA": "COALINDIA",
}

# Augment with comprehensive master directory from symbol_resolver (excluding generic/conglomerate words)
for _alias_text, _official_ticker in _ALIAS_LOOKUP_MAP.items():
    if _alias_text.lower() in GENERIC_FINANCIAL_WORDS or _alias_text.lower() in CONGLOMERATE_GROUPS:
        continue
    _up_key = _alias_text.upper()
    if _up_key not in STOCK_ALIASES:
        STOCK_ALIASES[_up_key] = _official_ticker


# Recognized global technology & multinational equities
GLOBAL_STOCKS: dict[str, dict[str, str]] = {
    "IBM": {
        "ticker": "IBM",
        "exchange": "NYSE",
        "currency": "USD",
        "name": "International Business Machines Corp",
    },
    "CAPGEMINI": {
        "ticker": "CAP.PA",
        "exchange": "Euronext Paris",
        "currency": "EUR",
        "name": "Capgemini SE",
    },
    "ACCENTURE": {
        "ticker": "ACN",
        "exchange": "NYSE",
        "currency": "USD",
        "name": "Accenture plc",
    },
    "COGNIZANT": {
        "ticker": "CTSH",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Cognizant Technology Solutions",
    },
    "MICROSOFT": {
        "ticker": "MSFT",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Microsoft Corporation",
    },
    "APPLE": {
        "ticker": "AAPL",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Apple Inc.",
    },
    "GOOGLE": {
        "ticker": "GOOGL",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Alphabet Inc.",
    },
    "AMAZON": {
        "ticker": "AMZN",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Amazon.com Inc.",
    },
    "TESLA": {
        "ticker": "TSLA",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Tesla Inc.",
    },
    "TSLA": {
        "ticker": "TSLA",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Tesla Inc.",
    },
    "NVIDIA": {
        "ticker": "NVDA",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "NVIDIA Corporation",
    },
    "NVDA": {
        "ticker": "NVDA",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "NVIDIA Corporation",
    },
    "META": {
        "ticker": "META",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Meta Platforms Inc.",
    },
    "FACEBOOK": {
        "ticker": "META",
        "exchange": "NASDAQ",
        "currency": "USD",
        "name": "Meta Platforms Inc.",
    },
}

STOP_WORDS = {
    "COMPARE", "THIS", "THESE", "COMPAINES", "COMPANIES", "COMPANY",
    "STOCKS", "STOCK", "THE", "AND", "VS", "VERSUS", "WITH", "BETWEEN",
    "OF", "FOR", "ME", "PLEASE", "GIVE", "RANDOM", "RANDONG", "NUMBER",
    "N", "WHAT", "IS", "TELL", "ABOUT", "TODAY", "ALL", "LIST",
    "LATEST", "NEWS", "MARKET", "CURRENT", "RECENT", "UPDATES", "UPDATE",
    "CAN", "YOU", "HOW", "MUCH", "PRICE", "PRICES", "SHOW", "GET", "ANY",
    "SOME", "ARE", "WHICH", "BEST", "TOP", "SHARE", "SHARES", "HEADLINES",
    "BUY", "BUYING", "SELL", "SELLING", "HOLD", "HOLDING", "EXIT", "EXITING",
    "REDUCE", "AVERAGE", "AVERAGING", "KEEP", "KEEPING", "OWN", "OWNING",
    "BOUGHT", "PURCHASED", "SHOULD", "WOULD", "COULD", "GOOD", "TIME",
    "DOWN", "UP", "AT", "IT", "IN", "AN", "A", "OR", "IF", "MY", "I",
    "NOW", "ANOTHER", "YEAR", "YEARS", "MONTH", "MONTHS", "CORRECTION",
    "INVESTMENT", "DECISION", "CONTINUE", "PORTFOLIO", "STACK", "UP", "AGAINST",
    "SIDE", "BY", "PUT", "COMPARED", "TO", "TELL", "CHECK", "DOING",
}


def extract_symbols_from_query(query: str) -> list[str]:
    """Extract known or probable symbols from a user query string.

    Supports:
    - Multiple comma-separated or 'and'/'vs' joined companies
    - Typo-tolerant fuzzy matching (e.g., 'Wirpo' -> 'WIPRO', 'infosis' -> 'INFY')
    - Master directory of NSE listed equities, colloquial aliases, and global tech firms
    - Natural language inputs (e.g., 'Tata Power', 'Tata Motors Company', 'Reliance Industries', 'YES Bank')
    - Unlimited arbitrary N number of companies
    - Strict generic word protection (never extracts 'Bank', 'Power', etc. alone)
    """
    found: list[str] = []
    q_upper = query.upper()

    # Check for direct ambiguity or pure generic queries first (e.g. user just asks about 'Tata' or 'Bank')
    cleaned_full = re.sub(r"[^\w\s&]", " ", query.lower()).strip()
    words_full = [w for w in cleaned_full.split() if w.upper() not in STOP_WORDS]
    if len(words_full) == 1 and (words_full[0] in CONGLOMERATE_GROUPS or words_full[0] in GENERIC_FINANCIAL_WORDS):
        return []

    # Identify explicit negative exclusions (e.g. "not ITC", "excluding TCS", "except RELIANCE")
    excluded_symbols: set[str] = set()
    for ex_match in re.finditer(r"\b(?:not|excluding|except)\s+([a-zA-Z0-9&]+(?:\s+[a-zA-Z0-9&]+)?)", query, flags=re.IGNORECASE):
        candidate_ex = ex_match.group(1).strip().upper()
        res_ex = resolve_nse_symbol(candidate_ex, allow_online_lookup=False)
        if res_ex.get("symbol"):
            excluded_symbols.add(res_ex["symbol"])
        elif candidate_ex in STOCK_ALIASES:
            excluded_symbols.add(STOCK_ALIASES[candidate_ex])

    # 1. Multi-word phrase scan first (longest multi-word aliases first, e.g. 'TATA MOTORS', 'STATE BANK OF INDIA', 'YES BANK')
    matched_spans: list[tuple[int, int]] = []
    for alias, ticker in sorted(STOCK_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if len(alias.split()) < 2:
            continue
        pattern = r"\b" + re.escape(alias) + r"\b"
        for m in re.finditer(pattern, q_upper):
            if any(max(m.start(), s[0]) < min(m.end(), s[1]) for s in matched_spans):
                continue
            matched_spans.append((m.start(), m.end()))
            if ticker not in found and ticker not in excluded_symbols:
                found.append(ticker)

    for g_alias in sorted(GLOBAL_STOCKS.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(g_alias) + r"\b"
        for m in re.finditer(pattern, q_upper):
            if any(max(m.start(), s[0]) < min(m.end(), s[1]) for s in matched_spans):
                continue
            matched_spans.append((m.start(), m.end()))
            if g_alias not in found and g_alias not in excluded_symbols:
                found.append(g_alias)

    # 2. Segment-based parser (handles comma-separated lists, 'and'/'or'/'vs' joined, em-dashes, or carrier phrases)
    segments = re.split(r'[,;\n/—–\-]|(?:\s+(?:and|or|vs|versus|with|against|compared to|side by side with|between)\s+)', query, flags=re.IGNORECASE)
    all_known_keys = [
        k for k in (list(STOCK_ALIASES.keys()) + list(GLOBAL_STOCKS.keys()))
        if k.lower() not in GENERIC_FINANCIAL_WORDS and k.lower() not in CONGLOMERATE_GROUPS
    ]

    for seg in segments:
        clean = seg.strip()
        words = [w for w in re.split(r'\s+', clean) if w.upper() not in STOP_WORDS and not w.isdigit()]
        if not words:
            continue
        phrase = " ".join(words)
        phrase_upper = phrase.upper()
        phrase_lower = phrase.lower()

        # Disallow isolated generic sector words or conglomerate names
        if phrase_lower in GENERIC_FINANCIAL_WORDS or phrase_lower in CONGLOMERATE_GROUPS:
            continue

        # Check exact alias match
        if phrase_upper in STOCK_ALIASES:
            sym = STOCK_ALIASES[phrase_upper]
            if sym not in found and sym not in excluded_symbols:
                found.append(sym)
            continue

        if phrase_upper in GLOBAL_STOCKS:
            if phrase_upper not in found and phrase_upper not in excluded_symbols:
                found.append(phrase_upper)
            continue

        # Resolve via symbol_resolver service (handles suffix stripping, concatenations, tokens)
        res = resolve_nse_symbol(phrase, allow_online_lookup=False)
        if res.get("symbol") and res.get("confidence", 0.0) >= 0.75 and not res.get("is_ambiguous"):
            sym = res["symbol"]
            if sym not in found and sym not in excluded_symbols:
                found.append(sym)
            continue

        # Fuzzy match against known aliases (only for non-generic phrases of sufficient length)
        if len(phrase_upper) >= 4 and not is_generic_query(phrase_lower):
            close = difflib.get_close_matches(phrase_upper, all_known_keys, n=1, cutoff=0.78)
            if close:
                match_key = close[0]
                sym = STOCK_ALIASES.get(match_key, match_key)
                if sym not in found and sym not in excluded_symbols:
                    found.append(sym)
                continue

        # If phrase was not resolved as a cohesive entity and has multiple words, check individual words
        for w in words:
            w_up = w.upper()
            w_low = w.lower()

            # STRICT GENERIC SUPPRESSION: Words like 'Bank', 'Power', 'Steel' alone MUST NOT match!
            if w_low in GENERIC_FINANCIAL_WORDS or w_low in CONGLOMERATE_GROUPS:
                continue

            if w_up in STOCK_ALIASES:
                sym = STOCK_ALIASES[w_up]
                if sym not in found and sym not in excluded_symbols:
                    found.append(sym)
            elif w_up in GLOBAL_STOCKS:
                if w_up not in found and w_up not in excluded_symbols:
                    found.append(w_up)
            else:
                res_w = resolve_nse_symbol(w, allow_online_lookup=False)
                if res_w.get("symbol") and res_w.get("confidence", 0.0) >= 0.8 and not res_w.get("is_ambiguous"):
                    sym = res_w["symbol"]
                    if sym not in found and sym not in excluded_symbols:
                        found.append(sym)
                else:
                    if len(w_up) >= 4:
                        close_w = difflib.get_close_matches(w_up, all_known_keys, n=1, cutoff=0.78)
                        if close_w:
                            match_key = close_w[0]
                            sym = STOCK_ALIASES.get(match_key, match_key)
                            if sym not in found:
                                found.append(sym)
                                continue
                    if len(w_up) >= 2 and w_up.isalnum():
                        if w_up in INDIAN_STOCK_MASTER:
                            if w_up not in found:
                                found.append(w_up)
                        elif w.isupper() and len(w_up) <= 10:
                            try:
                                norm = normalize_nse_symbol(w_up)
                                if norm in INDIAN_STOCK_MASTER and norm not in found:
                                    found.append(norm)
                            except Exception:
                                pass

    return found


def format_currency(value: Any, currency: str = "INR") -> str:
    """Format numeric currency value with appropriate currency symbol."""
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    curr_upper = (currency or "INR").upper()
    if curr_upper == "INR":
        return f"₹{value:,.2f}"
    elif curr_upper == "USD":
        return f"${value:,.2f}"
    elif curr_upper == "EUR":
        return f"€{value:,.2f}"
    elif curr_upper == "GBP":
        return f"£{value:,.2f}"
    return f"{value:,.2f} {curr_upper}"


def run_demo_comparison(symbols: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Execute live tools and generate a side-by-side comparison report for N companies."""
    tool_calls: list[dict[str, Any]] = []
    stock_data: list[dict[str, Any]] = []
    exchange_notes: list[str] = []

    for sym in symbols:
        # Check if recognized global stock
        if sym in GLOBAL_STOCKS:
            meta = GLOBAL_STOCKS[sym]
            ticker_id = meta["ticker"]
            exchange_name = meta["exchange"]
            curr = meta["currency"]
            comp_name = meta["name"]

            tool_calls.append({"name": "get_stock_price", "args": {"symbol": sym, "exchange": exchange_name}})
            try:
                t = yf.Ticker(ticker_id)
                price = t.fast_info.last_price
                prev = t.fast_info.previous_close
                chg = (price - prev) if price and prev else 0.0
                chg_pct = (chg / prev * 100) if prev else 0.0
                h52 = t.fast_info.year_high
                l52 = t.fast_info.year_low
            except Exception:
                price = prev = chg = chg_pct = h52 = l52 = None

            stock_data.append({
                "symbol": sym,
                "company": comp_name,
                "exchange": exchange_name,
                "current_price": price,
                "change": chg,
                "change_percent": chg_pct,
                "currency": curr,
                "sentiment": "N/A (Global)",
                "sentiment_badge": "🌐 Overseas",
                "trailing_pe": "N/A",
                "52_week_high": h52,
                "52_week_low": l52,
                "is_global": True,
            })
            exchange_notes.append(f"**{comp_name} ({sym})** is listed on **{exchange_name}** (`{ticker_id}`) in {curr}.")

        else:
            # Query Indian NSE tool
            tool_calls.append({"name": "get_stock_price", "args": {"symbol": sym}})
            try:
                price_res = get_stock_price(sym)
            except Exception as e:
                price_res = {"symbol": sym, "error": str(e), "current_price": None}

            # Query company fundamentals
            tool_calls.append({"name": "get_company_info", "args": {"symbol": sym}})
            try:
                info_res = get_company_info(sym)
            except Exception:
                info_res = {}

            # Query technical sentiment
            tool_calls.append({"name": "get_stock_sentiment", "args": {"symbol": sym}})
            try:
                sent_res = get_stock_sentiment(sym)
                sentiment_label = sent_res.get("sentiment", "Neutral")
            except Exception:
                sentiment_label = "Neutral"

            badge = "🟢 Bullish" if sentiment_label == "Bullish" else ("🔴 Bearish" if sentiment_label == "Bearish" else "⚪ Neutral")

            # Check if NSE returned no data, try raw global ticker as fallback
            cur_price = price_res.get("current_price")
            curr = price_res.get("currency") or "INR"
            comp_name = price_res.get("company")
            exch = "NSE (India)"
            is_global = False

            if cur_price is None:
                # Try raw ticker on Yahoo Finance for foreign companies (e.g. IBM, AAPL)
                try:
                    raw_t = yf.Ticker(sym)
                    raw_p = raw_t.fast_info.last_price
                    if raw_p is not None:
                        cur_price = round(raw_p, 2)
                        raw_prev = raw_t.fast_info.previous_close
                        price_res["previous_close"] = raw_prev
                        if raw_prev:
                            price_res["change"] = round(cur_price - raw_prev, 2)
                            price_res["change_percent"] = round((price_res["change"] / raw_prev) * 100, 2)
                        curr = raw_t.fast_info.currency or "USD"
                        price_res["currency"] = curr
                        price_res["52_week_high"] = raw_t.fast_info.year_high
                        price_res["52_week_low"] = raw_t.fast_info.year_low
                        comp_name = raw_t.info.get("shortName") or raw_t.info.get("longName") or sym
                        exch = "Global Exchange"
                        is_global = True
                        badge = "🌐 Overseas"
                        exchange_notes.append(f"**{comp_name} ({sym})** was resolved via global ticker ({curr}).")
                except Exception:
                    pass

            if not comp_name:
                comp_name = sym

            stock_data.append({
                "symbol": sym,
                "company": comp_name,
                "exchange": exch,
                "current_price": cur_price,
                "change": price_res.get("change"),
                "change_percent": price_res.get("change_percent"),
                "currency": curr,
                "sentiment": sentiment_label,
                "sentiment_badge": badge,
                "trailing_pe": info_res.get("trailing_pe", "N/A"),
                "52_week_high": price_res.get("52_week_high"),
                "52_week_low": price_res.get("52_week_low"),
                "sector": info_res.get("sector", "N/A"),
                "is_global": is_global,
            })

    # Build Markdown Comparison Report
    comp_symbols = [s["symbol"] for s in stock_data]
    lines = [
        f"### ⚖️ Multi-Company Comparison ({len(comp_symbols)} Companies): {' vs '.join(comp_symbols)}\n",
        "*(Data retrieved live via Real-Time Market Tools)*\n",
        "| Metric | " + " | ".join(f"**{s['symbol']}**" for s in stock_data) + " |",
        "| :--- | " + " | ".join(":---" for _ in stock_data) + " |",
    ]

    # Company name
    lines.append("| **Company** | " + " | ".join(s["company"] for s in stock_data) + " |")

    # Exchange
    lines.append("| **Primary Exchange** | " + " | ".join(s["exchange"] for s in stock_data) + " |")

    # Current price
    price_cols = []
    for s in stock_data:
        p = s["current_price"]
        curr = s["currency"]
        if p is not None:
            price_cols.append(format_currency(p, curr))
        else:
            price_cols.append("Not listed / No quote")
    lines.append("| **Current Price** | " + " | ".join(price_cols) + " |")

    # Day Change %
    change_cols = []
    for s in stock_data:
        chg = s.get("change")
        chg_pct = s.get("change_percent")
        if chg is not None and chg_pct is not None:
            sign = "+" if chg >= 0 else ""
            color = "🟢" if chg >= 0 else "🔴"
            change_cols.append(f"{color} {sign}{chg:.2f} ({sign}{chg_pct:.2f}%)")
        else:
            change_cols.append("N/A")
    lines.append("| **Day Change** | " + " | ".join(change_cols) + " |")

    # Technical Sentiment
    sent_cols = [s["sentiment_badge"] for s in stock_data]
    lines.append("| **Technical Sentiment** | " + " | ".join(sent_cols) + " |")

    # Trailing P/E
    pe_cols = []
    for s in stock_data:
        pe = s.get("trailing_pe")
        pe_cols.append(f"{pe:.2f}" if isinstance(pe, (int, float)) else str(pe or "N/A"))
    lines.append("| **Trailing P/E** | " + " | ".join(pe_cols) + " |")

    # 52 Week High / Low
    range_cols = []
    for s in stock_data:
        h52 = s.get("52_week_high")
        l52 = s.get("52_week_low")
        curr = s["currency"]
        if h52 is not None and l52 is not None:
            range_cols.append(f"{format_currency(l52, curr)} – {format_currency(h52, curr)}")
        else:
            range_cols.append("N/A")
    lines.append("| **52-Week Range** | " + " | ".join(range_cols) + " |")

    # Observations
    lines.append("\n#### 📊 Key Takeaways & Observations\n")
    for s in stock_data:
        sym = s["symbol"]
        comp = s["company"]
        p = s["current_price"]
        curr = s["currency"]
        chg_pct = s.get("change_percent", 0.0) or 0.0
        exch = s["exchange"]
        if p is not None:
            price_str = format_currency(p, curr)
            lines.append(f"* **{comp} ({sym})**: Trading at **{price_str}** ({chg_pct:+.2f}%) on {exch}. Technical stance: **{s['sentiment']}**.")
        else:
            lines.append(f"* **{comp} ({sym})**: No live market price available on the National Stock Exchange of India (NSE).")

    if exchange_notes:
        lines.append("\n> [!NOTE]\n> **Exchange & Currency Context:**\n> " + "\n> ".join(f"- {n}" for n in exchange_notes))

    return "\n".join(lines), tool_calls


def format_news_section(
    news_items: list[dict[str, Any]],
    company_name: str = "",
    symbol: str = "",
    include_header: bool = True,
) -> list[str]:
    """Format news stories with comprehensive summaries and clean external links.

    Ensures the user can read the complete executive summary directly in the chat
    interface without navigating away to external sites, while providing a clear
    external link if they wish to inspect the original article.
    """
    if not news_items:
        return []

    lines: list[str] = []
    if include_header:
        lines.extend([
            "\n#### 📰 Recent Market News & Executive Summaries:\n",
            "*(Read full summaries below or follow the external links for complete coverage)*\n",
        ])
    else:
        lines.append("*(Read full summaries below or follow the external links for complete coverage)*\n")

    for idx, item in enumerate(news_items, start=1):
        title = item.get("title", "Market Update")
        src = item.get("source") or "Financial News"
        url = item.get("url")
        pub_at = item.get("published_at")
        summary = item.get("summary")

        # Story title
        lines.append(f"{idx}. **{title}**")

        # Metadata: publisher and date
        meta_parts = [f"**Publisher:** {src}"]
        if pub_at:
            meta_parts.append(f"**Published:** {pub_at}")
        lines.append(f"   * {' | '.join(meta_parts)}")

        # Summary resolution: present full executive summary
        if summary and summary.strip():
            clean_summary = summary.strip()
            lines.append(f"   * **Executive Summary:** {clean_summary}")
        else:
            entity = f"{company_name} ({symbol})".strip() or "the company"
            lines.append(
                f"   * **Executive Summary:** Coverage and financial analysis regarding recent developments and trading activity for {entity}."
            )

        # External Link
        if url and url != "#":
            lines.append(f"   * 🔗 [Read full article on {src} ↗]({url})\n")
        else:
            lines.append(f"   * 🔗 *Direct link unavailable*\n")

    return lines


def run_demo_single_stock(
    sym: str,
    is_news_query: bool = False,
    comprehensive: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Execute live tools and generate a single-stock analysis report.

    Supports quick market overview mode or comprehensive fundamental, financial,
    and governance deep-dive report adhering to the full Section V layout.
    """
    tool_calls: list[dict[str, Any]] = []

    entity = get_security_entity(sym)
    ident_banner = entity.format_banner() if entity else ""

    if comprehensive:
        tool_calls.append({"name": "get_stock_price", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_company_info", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_stock_sentiment", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_quarterly_financials", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_cash_flow", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_debt_metrics", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_shareholding_pattern", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_board_meetings", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_corporate_actions", "args": {"symbol": sym}})
        tool_calls.append({"name": "get_market_news", "args": {"company_or_symbol": sym, "limit": 5}})
        tool_calls.append({"name": "get_company_analysis", "args": {"symbol": sym}})

        try:
            analysis_data = get_company_analysis(sym)
            report_md = analysis_data.get("report_markdown", "")
        except Exception as exc:
            logger.error("Comprehensive company analysis failed for %s: %s", sym, exc)
            report_md = f"⚠️ Comprehensive analysis could not be completed for **{sym}**: {exc}"

        if ident_banner and not report_md.startswith("🏢"):
            report_md = f"{ident_banner}\n\n---\n\n" + report_md

        return report_md, tool_calls

    tool_calls.append({"name": "get_stock_price", "args": {"symbol": sym}})
    price_res = get_stock_price(sym)

    tool_calls.append({"name": "get_company_info", "args": {"symbol": sym}})
    info_res = get_company_info(sym)

    tool_calls.append({"name": "get_stock_sentiment", "args": {"symbol": sym}})
    sent_res = get_stock_sentiment(sym)

    news_limit = 5 if is_news_query else 3
    tool_calls.append({"name": "get_market_news", "args": {"company_or_symbol": sym, "limit": news_limit}})
    try:
        news_res = get_market_news(sym, limit=news_limit)
    except Exception:
        news_res = []

    comp = price_res.get("company", sym)
    price = price_res.get("current_price")
    chg = price_res.get("change", 0.0)
    chg_pct = price_res.get("change_percent", 0.0)
    sentiment = sent_res.get("sentiment", "Neutral")
    h52 = price_res.get("52_week_high")
    l52 = price_res.get("52_week_low")

    lines = []
    if ident_banner:
        lines.append(ident_banner + "\n\n---\n")

    lines.extend([
        f"### 📈 {comp} ({sym}) — Market Overview\n",
        f"* **Current Price:** {format_currency(price)} ({chg:+.2f}, {chg_pct:+.2f}%)",
        f"* **52-Week Range:** {format_currency(l52)} – {format_currency(h52)}",
        f"* **Technical Sentiment:** **{sentiment}** (Quantitative Score: {sent_res.get('score', 0):.1f})",
        f"* **Sector:** {info_res.get('sector', 'N/A')} | **Industry:** {info_res.get('industry', 'N/A')}",
        f"* **Trailing P/E:** {info_res.get('trailing_pe', 'N/A')}",
    ])

    if news_res:
        lines.extend(format_news_section(news_res, company_name=comp, symbol=sym))

    lines.append("\n> [!NOTE]\n> *Generated with live NSE market data.*")
    return "\n".join(lines), tool_calls


def run_demo_agent(
    query: str,
    conversation_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Execute live market analysis without requiring an LLM API key.

    Delegates to the modular run_market_assistant pipeline, supporting natural-language
    intent understanding, catalyst discovery, 52-week high scans, stock comparisons,
    and multi-turn conversation context.
    """
    from services.market_assistant_service import run_market_assistant

    return run_market_assistant(
        query=query,
        conversation_id=conversation_id,
        db_path=db_path,
    )


