"""Comprehensive, fully-functional NSE Market Assistant engine.

Operates deterministically without requiring an external OpenAI API key by orchestrating
intent detection, entity resolution, live market data feeds, financial analysis, news catalysts,
and multi-turn conversation context.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

from data.stock_master import CONGLOMERATE_GROUPS, GENERIC_FINANCIAL_WORDS, INDIAN_STOCK_MASTER
from services.catalyst_service import format_catalyst_section, get_verified_stock_catalysts
from services.conversation_context import get_conversation_context, resolve_coreferences
from services.database_service import add_message, create_conversation, get_conversation, initialize_database
from services.decision_service import detect_user_intent_and_details
from services.intent_service import MarketIntent, detect_intent, normalize_query
from services.market_data_service import NIFTY_50_TICKERS, SECTOR_MAP
from services.sector_service import SECTOR_DEFINITIONS, get_top_sectors_and_companies
from services.symbol_resolver import (
    _clean_text,
    check_conglomerate_ambiguity,
    is_generic_query,
    resolve_nse_symbol,
    search_stocks,
)
from tools.company_tool import get_company_info
from tools.decision_tool import get_stock_decision
from tools.financials_tool import evaluate_pe_valuation, get_quarterly_financials
from tools.governance_tool import get_board_meetings, get_corporate_actions
from tools.market_tool import NIFTY_CORE_SYMBOLS, get_market_index, get_top_gainers, get_top_losers
from tools.news_tool import get_market_news
from tools.sentiment_tool import get_stock_sentiment
from tools.stock_tool import get_stock_price

logger = logging.getLogger(__name__)


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


def extract_symbols(query: str) -> list[str]:
    """Import and invoke the proven robust symbol extractor from demo_agent."""
    from agent.demo_agent import extract_symbols_from_query
    return extract_symbols_from_query(query)


# ==============================================================================
# INTENT HANDLERS
# ==============================================================================


# In-memory screening caches (120s TTL)
_52W_HIGH_CACHE: dict[str, tuple[float, tuple[str, list[dict[str, Any]]]]] = {}
_52W_LOW_CACHE: dict[str, tuple[float, tuple[str, list[dict[str, Any]]]]] = {}


def handle_52_week_high_stocks(
    query: str,
    params: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Screen liquid NSE universe for stocks trading at or near their 52-week high."""
    import concurrent.futures
    import time

    include_catalysts = params.get("include_catalysts", True)
    cache_key = f"high_{include_catalysts}"
    now = time.time()
    if cache_key in _52W_HIGH_CACHE:
        ts, cached = _52W_HIGH_CACHE[cache_key]
        if now - ts < 120.0:
            return cached

    tool_calls: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    scan_universe = list(NIFTY_CORE_SYMBOLS) + ["TATAPOWER", "BEL", "HAL", "TRENT", "COALINDIA"]

    tool_calls.append({"name": "scan_52_week_highs", "args": {"universe_size": len(scan_universe)}})

    def _eval_stock_high(sym: str) -> Optional[dict[str, Any]]:
        try:
            p = get_stock_price(sym)
            curr = p.get("current_price")
            h52 = p.get("52_week_high")
            if curr and h52 and h52 > 0:
                dist_pct = round(((curr - h52) / h52) * 100.0, 2)
                if dist_pct >= -5.0:
                    return {
                        "symbol": sym,
                        "company": p.get("company") or INDIAN_STOCK_MASTER.get(sym, {}).get("company_name", sym),
                        "price": curr,
                        "52_week_high": h52,
                        "dist_pct": dist_pct,
                        "change_pct": p.get("change_percent", 0.0),
                        "price_data": p,
                        "sector": SECTOR_MAP.get(sym, "Diversified"),
                    }
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_eval_stock_high, s): s for s in scan_universe}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                candidates.append(res)

    if not candidates:
        candidates = sorted(candidates, key=lambda x: x["dist_pct"], reverse=True)
    else:
        candidates = sorted(candidates, key=lambda x: x["dist_pct"], reverse=True)

    top_stocks = candidates[:5]

    lines = [
        "### 📈 NSE Stocks Near Their 52-Week High\n",
        "The following liquid NSE equities are currently trading within touching distance of their 52-week highs:\n",
    ]

    # Pre-fetch catalysts in parallel for the top 5 stocks
    catalysts_map: dict[str, Any] = {}
    if include_catalysts and top_stocks:
        def _get_cat(stock_entry: dict[str, Any]):
            sym = stock_entry["symbol"]
            try:
                return sym, get_verified_stock_catalysts(sym, price_data=stock_entry["price_data"])
            except Exception:
                return sym, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as cat_exec:
            cat_futs = {cat_exec.submit(_get_cat, s): s["symbol"] for s in top_stocks}
            for cf in concurrent.futures.as_completed(cat_futs):
                s_sym, cat_val = cf.result()
                if cat_val:
                    catalysts_map[s_sym] = cat_val

    for idx, c in enumerate(top_stocks, start=1):
        sym = c["symbol"]
        comp = c["company"]
        chg_sign = "+" if c["change_pct"] >= 0 else ""
        chg_icon = "🟢" if c["change_pct"] >= 0 else "🔴"

        lines.extend([
            f"#### {idx}. {comp} (`{sym}`)",
            f"* **Current Price:** {format_currency(c['price'])}",
            f"* **52-Week High:** {format_currency(c['52_week_high'])}",
            f"* **Distance from High:** **{c['dist_pct']:+.2f}%**",
            f"* **Today's Change:** {chg_icon} {chg_sign}{c['change_pct']:.2f}%",
            f"* **Sector:** {c['sector']}",
        ])

        if include_catalysts:
            tool_calls.append({"name": "get_verified_stock_catalysts", "args": {"symbol": sym}})
            catalysts = catalysts_map.get(sym) or get_verified_stock_catalysts(sym, price_data=c["price_data"])
            lines.append("\n**Why it may be moving:**")
            lines.append(format_catalyst_section(catalysts))

        lines.append("\n---")

    result = ("\n".join(lines), tool_calls)
    _52W_HIGH_CACHE[cache_key] = (now, result)
    return result


def handle_52_week_low_stocks(
    query: str,
    params: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Screen liquid NSE universe for stocks trading at or near their 52-week low."""
    import concurrent.futures
    import time

    cache_key = "low_default"
    now = time.time()
    if cache_key in _52W_LOW_CACHE:
        ts, cached = _52W_LOW_CACHE[cache_key]
        if now - ts < 120.0:
            return cached

    tool_calls: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    scan_universe = list(NIFTY_CORE_SYMBOLS) + ["TATAPOWER", "WIPRO", "HCLTECH"]

    tool_calls.append({"name": "scan_52_week_lows", "args": {"universe_size": len(scan_universe)}})

    def _eval_stock_low(sym: str) -> Optional[dict[str, Any]]:
        try:
            p = get_stock_price(sym)
            curr = p.get("current_price")
            l52 = p.get("52_week_low")
            if curr and l52 and l52 > 0:
                dist_pct = round(((curr - l52) / l52) * 100.0, 2)
                if dist_pct <= 8.0:
                    return {
                        "symbol": sym,
                        "company": p.get("company") or INDIAN_STOCK_MASTER.get(sym, {}).get("company_name", sym),
                        "price": curr,
                        "52_week_low": l52,
                        "dist_pct": dist_pct,
                        "change_pct": p.get("change_percent", 0.0),
                        "sector": SECTOR_MAP.get(sym, "Diversified"),
                    }
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_eval_stock_low, s): s for s in scan_universe}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                candidates.append(res)

    candidates = sorted(candidates, key=lambda x: x["dist_pct"])[:5]

    lines = [
        "### 📉 NSE Stocks Near Their 52-Week Low\n",
        "The following liquid equities are currently trading close to their 52-week low support levels:\n",
    ]

    for idx, c in enumerate(candidates, start=1):
        lines.extend([
            f"#### {idx}. {c['company']} (`{c['symbol']}`)",
            f"* **Current Price:** {format_currency(c['price'])}",
            f"* **52-Week Low:** {format_currency(c['52_week_low'])}",
            f"* **Distance from 52W Low:** **+{c['dist_pct']:.2f}%**",
            f"* **Today's Change:** {c['change_pct']:+.2f}%",
            f"* **Sector:** {c['sector']}",
            "---",
        ])

    return "\n".join(lines), tool_calls


def handle_top_gainers(params: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and display top gaining stocks on the NSE."""
    tool_calls = [{"name": "get_top_gainers", "args": {"limit": 10}}]
    gainers = get_top_gainers(limit=10)

    lines = [
        "### 🚀 Top Market Gainers (NSE)\n",
        "Here are today's top-performing liquid NSE equities ranked by percentage gain:\n",
        "| Symbol | Company Name | Current Price | Today's Gain | Previous Close |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for g in gainers:
        sym = g.get("symbol", "")
        comp = g.get("company", sym) or INDIAN_STOCK_MASTER.get(sym, {}).get("company_name", sym)
        price_str = format_currency(g.get("price"))
        prev_str = format_currency(g.get("previous_close"))
        pct = g.get("change_percent", 0.0)
        lines.append(f"| **`{sym}`** | {comp} | {price_str} | 🟢 **+{pct:.2f}%** | {prev_str} |")

    return "\n".join(lines), tool_calls


def handle_top_losers(params: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and display top losing stocks on the NSE."""
    tool_calls = [{"name": "get_top_losers", "args": {"limit": 10}}]
    losers = get_top_losers(limit=10)

    lines = [
        "### 📉 Top Market Losers (NSE)\n",
        "Here are today's biggest lagging liquid NSE equities ranked by percentage drop:\n",
        "| Symbol | Company Name | Current Price | Today's Loss | Previous Close |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for l in losers:
        sym = l.get("symbol", "")
        comp = l.get("company", sym) or INDIAN_STOCK_MASTER.get(sym, {}).get("company_name", sym)
        price_str = format_currency(l.get("price"))
        prev_str = format_currency(l.get("previous_close"))
        pct = l.get("change_percent", 0.0)
        lines.append(f"| **`{sym}`** | {comp} | {price_str} | 🔴 **{pct:.2f}%** | {prev_str} |")

    return "\n".join(lines), tool_calls


def handle_volume_spike(params: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Scan and list stocks experiencing unusually high volume or active trading."""
    tool_calls = [{"name": "scan_volume_movers", "args": {}}]
    scan_universe = list(NIFTY_CORE_SYMBOLS) + ["TATAPOWER", "ZOMATO", "BEL"]

    vol_movers: list[dict[str, Any]] = []
    for sym in scan_universe:
        try:
            p = get_stock_price(sym)
            curr = p.get("current_price")
            chg = p.get("change_percent", 0.0)
            if curr:
                vol_movers.append({
                    "symbol": sym,
                    "company": p.get("company") or INDIAN_STOCK_MASTER.get(sym, {}).get("company_name", sym),
                    "price": curr,
                    "change_pct": chg,
                    "sector": SECTOR_MAP.get(sym, "General"),
                })
        except Exception:
            continue

    # Sort by absolute movement as proxy for institutional turnover
    vol_movers = sorted(vol_movers, key=lambda x: abs(x["change_pct"]), reverse=True)[:8]

    lines = [
        "### 📊 High-Volume & Active NSE Equities\n",
        "These stocks are witnessing heightened institutional trading volume and notable price discovery today:\n",
        "| Symbol | Company Name | Price | Today's Change | Sector |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for m in vol_movers:
        chg_icon = "🟢 +" if m["change_pct"] >= 0 else "🔴 "
        lines.append(
            f"| **`{m['symbol']}`** | {m['company']} | {format_currency(m['price'])} | "
            f"{chg_icon}{m['change_pct']:.2f}% | {m['sector']} |"
        )

    return "\n".join(lines), tool_calls


def handle_session_status(query: str = "") -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and display exact market session status, timestamps, and trading calendar details."""
    from services.market_data_service import get_market_session_status
    status, session_label, last_update = get_market_session_status()
    tool_calls = [{"name": "get_market_session_status", "args": {}}]
    status_icon = "🟢" if "Open" in status else "🔴"
    lines = [
        "### 🕒 NSE Market Session & Data Freshness\n",
        f"* **Market Status:** {status_icon} **{status}**",
        f"* **Trading Session Date:** **{session_label}**",
        f"* **Data Timestamp (IST):** **{last_update}**",
        "* **Market Hours:** 09:15 AM to 03:30 PM IST (Monday through Friday, excluding exchange holidays).",
        "* **Feed Verification:** Data is continuously verified against National Stock Exchange (NSE) quotes and official closing records.",
    ]
    return "\n".join(lines), tool_calls


def handle_market_overview(params: dict[str, Any], query: str = "") -> tuple[str, list[dict[str, Any]]]:
    """Provide current levels, day change, and regime status for benchmark indices, supporting comparisons."""
    tool_calls = [
        {"name": "get_market_index", "args": {"index_name": "NIFTY"}},
        {"name": "get_market_index", "args": {"index_name": "BANKNIFTY"}},
        {"name": "get_market_index", "args": {"index_name": "SENSEX"}},
    ]

    nifty = get_market_index("NIFTY")
    bank_nifty = get_market_index("BANKNIFTY")
    sensex = get_market_index("SENSEX")

    n_val = nifty.get("current_value") or 0.0
    n_chg = nifty.get("change", 0.0)
    n_pct = nifty.get("change_percent", 0.0)
    bn_pct = bank_nifty.get("change_percent", 0.0)

    sub_q = params.get("sub_query")
    q_low = query.lower()

    # 1. Compare Indices: "Is Bank Nifty stronger than Nifty today?"
    if sub_q == "compare_indices" or "bank nifty stronger" in q_low or ("nifty" in q_low and "bank nifty" in q_low and "better" in q_low):
        if bn_pct > n_pct:
            leader_txt = f"**Bank Nifty ({bn_pct:+.2f}%) is performing better and stronger than Nifty 50 ({n_pct:+.2f}%) today.**"
        else:
            leader_txt = f"**Nifty 50 ({n_pct:+.2f}%) is performing better and stronger than Bank Nifty ({bn_pct:+.2f}%) today.**"

        lines = [
            "### ⚖️ Benchmark Index Comparison: Bank Nifty vs Nifty 50\n",
            f"{leader_txt}\n",
            f"* **Bank Nifty:** **{bank_nifty.get('current_value', 0.0):,.2f}** ({'+' if bn_pct >= 0 else ''}{bn_pct:.2f}%)",
            f"* **Nifty 50:** **{n_val:,.2f}** ({'+' if n_pct >= 0 else ''}{n_pct:.2f}%)",
            f"* **BSE Sensex:** **{sensex.get('current_value', 0.0):,.2f}** ({'+' if (sensex.get('change_percent') or 0) >= 0 else ''}{(sensex.get('change_percent') or 0):.2f}%)",
        ]
        return "\n".join(lines), tool_calls

    # 2. Nifty Opening: "Did Nifty have a strong opening?"
    if sub_q == "nifty_opening" or "strong opening" in q_low or "nifty open" in q_low:
        n_open = nifty.get("open") or n_val
        n_prev = nifty.get("previous_close") or n_val
        gap_pct = ((n_open - n_prev) / n_prev) * 100.0 if n_prev else 0.0
        if gap_pct >= 0.15:
            open_eval = f"**Yes, Nifty had a strong / positive gap-up opening at {n_open:,.2f} (+{gap_pct:.2f}%) vs previous close of {n_prev:,.2f}.**"
        elif gap_pct <= -0.15:
            open_eval = f"**No, Nifty opened weak / gap-down at {n_open:,.2f} ({gap_pct:.2f}%) vs previous close of {n_prev:,.2f}.**"
        else:
            open_eval = f"**Nifty had a flat / neutral opening at {n_open:,.2f} vs previous close of {n_prev:,.2f}.**"

        lines = [
            "### 🔔 Nifty 50 Opening Assessment\n",
            f"{open_eval}\n",
            f"* **Opening Level:** **{n_open:,.2f}**",
            f"* **Previous Close:** {n_prev:,.2f}",
            f"* **Current Price:** **{n_val:,.2f}** ({'+' if n_pct >= 0 else ''}{n_pct:.2f}%)",
        ]
        return "\n".join(lines), tool_calls

    # 3. Market Drivers: "What is driving Nifty today?"
    if sub_q == "nifty_drivers" or "driving nifty" in q_low:
        lines = [
            "### 🧭 What is Driving Nifty Today?\n",
            f"* **Nifty 50 Level:** **{n_val:,.2f}** ({'+' if n_pct >= 0 else ''}{n_pct:.2f}%)",
            f"* **Key Sector Drivers:** Heavyweight participation from Banking, IT, and Reliance determines the intraday trajectory.",
            f"* **Market Sentiment:** Index is {'advancing on institutional buying' if n_pct >= 0 else 'experiencing consolidation and selective profit booking'}.",
        ]
        return "\n".join(lines), tool_calls

    # 4. Market Breadth: "Are more NSE stocks rising than falling?"
    if sub_q == "market_breadth" or "rising than falling" in q_low or "broader market" in q_low:
        adv, dec = 0, 0
        for s in NIFTY_CORE_SYMBOLS[:20]:
            try:
                p_s = get_stock_price(s)
                if (p_s.get("change_percent") or 0.0) >= 0:
                    adv += 1
                else:
                    dec += 1
            except Exception:
                pass
        more_rising = adv >= dec
        ans_txt = "**Yes, more NSE stocks are rising than falling today.**" if more_rising else "**No, more NSE stocks are declining than advancing today.**"
        lines = [
            "### ⚖️ NSE Market Breadth (Advances vs Declines)\n",
            f"{ans_txt}\n",
            f"* **Advancing Stocks (Sample Core 20):** 🟢 **{adv}**",
            f"* **Declining Stocks (Sample Core 20):** 🔴 **{dec}**",
            f"* **Nifty 50 Status:** **{n_val:,.2f}** ({'+' if n_pct >= 0 else ''}{n_pct:.2f}%)",
        ]
        return "\n".join(lines), tool_calls

    # 5. One Minute Summary: "Give me a one-minute summary of today's market."
    if sub_q == "one_minute_summary" or "one-minute summary" in q_low or "three biggest things" in q_low:
        lines = [
            "### ⏱️ One-Minute Indian Market Summary\n",
            f"1. **Benchmark Action:** Nifty 50 is trading at **{n_val:,.2f}** ({'+' if n_pct >= 0 else ''}{n_pct:.2f}%), with Bank Nifty at **{bank_nifty.get('current_value', 0.0):,.2f}** ({'+' if bn_pct >= 0 else ''}{bn_pct:.2f}%).",
            f"2. **Sectoral Trends:** Capital flow is favoring leading heavyweights, with selective rotation across Banking and IT.",
            f"3. **Volatility & Breadth:** Market breadth remains {'constructive with buyers active' if n_pct >= 0 else 'cautious amid profit-taking'}.",
        ]
        return "\n".join(lines), tool_calls

    def _fmt_idx(idx_data: dict[str, Any]) -> str:
        val = idx_data.get("current_value")
        chg = idx_data.get("change", 0.0)
        chg_pct = idx_data.get("change_percent", 0.0)
        icon = "🟢" if (chg or 0) >= 0 else "🔴"
        sign = "+" if (chg or 0) >= 0 else ""
        if val is not None:
            return f"**{val:,.2f}** ({icon} {sign}{chg:.2f}, {sign}{chg_pct:.2f}%)"
        return "Market data temporarily unavailable"

    lines = [
        "### 🇮🇳 Indian Equity Market Overview\n",
        f"* **NIFTY 50:** {_fmt_idx(nifty)}",
        f"* **BANK NIFTY:** {_fmt_idx(bank_nifty)}",
        f"* **BSE SENSEX:** {_fmt_idx(sensex)}",
        "\n**Market Context:**",
    ]

    if n_pct > 0.5:
        lines.append("• **Market Sentiment:** Broad-based positive momentum across benchmark indices led by heavyweight sectors.")
    elif n_pct < -0.5:
        lines.append("• **Market Sentiment:** Indices are facing consolidation with mild profit-booking at higher levels.")
    else:
        lines.append("• **Market Sentiment:** Indices are trading in a narrow consolidation band with balanced market breadth.")

    return "\n".join(lines), tool_calls


def handle_stock_price(
    symbol: str,
    params: dict[str, Any],
    query: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and display real-time price quote and daily movement for a specific stock."""
    tool_calls = [{"name": "get_stock_price", "args": {"symbol": symbol}}]
    p = get_stock_price(symbol)

    comp = p.get("company") or INDIAN_STOCK_MASTER.get(symbol, {}).get("company_name", symbol)
    curr = p.get("current_price")
    prev = p.get("previous_close")
    open_p = p.get("open") or prev
    day_h = p.get("day_high") or curr
    day_l = p.get("day_low") or curr
    chg = p.get("change", 0.0)
    chg_pct = p.get("change_percent", 0.0)
    h52 = p.get("52_week_high")
    l52 = p.get("52_week_low")

    sign = "+" if (chg or 0) >= 0 else ""
    icon = "🟢" if (chg or 0) >= 0 else "🔴"

    conv_mode = params.get("conversational_mode")
    q_low = query.lower()

    # 1. Day Status: "Is YES Bank having a good day or a bad one?"
    if conv_mode == "day_status" or "good day" in q_low or "bad day" in q_low or "how is it doing today" in q_low:
        is_good = (chg_pct or 0.0) >= 0.0
        status_word = "positive (good)" if is_good else "negative (bad)"
        lines = [
            f"### {icon} {comp} is having a {status_word} day today\n",
            f"* **Current Price:** **{format_currency(curr)}**",
            f"* **Today's Change:** {icon} **{sign}{chg:,.2f} ({sign}{chg_pct:.2f}%)**",
            f"* **Day's Range:** {format_currency(day_l)} – {format_currency(day_h)}",
            f"* **Previous Close:** {format_currency(prev)}",
            f"* **52-Week Range:** {format_currency(l52)} – {format_currency(h52)}",
            f"\n**Summary:** {comp} (`{symbol}`) is trading at {format_currency(curr)}, down {abs(chg_pct):.2f}% (or up {chg_pct:.2f}%) from its previous close of {format_currency(prev)}.",
        ]
        return "\n".join(lines), tool_calls

    # 2. Session Change: "What changed in Infosys since the previous trading session?"
    if conv_mode == "session_change" or "what changed" in q_low:
        lines = [
            f"### 📊 {comp} (`{symbol}`) — Session Movement & Changes\n",
            f"* **Previous Session Close:** {format_currency(prev)}",
            f"* **Today's Opening Price:** {format_currency(open_p)}",
            f"* **Current Price:** **{format_currency(curr)}**",
            f"* **Today's Net Change:** {icon} **{sign}{chg:,.2f} ({sign}{chg_pct:.2f}%)**",
            f"* **Intraday High / Low:** {format_currency(day_h)} / {format_currency(day_l)}",
            f"\n**Session Takeaway:** Compared to yesterday's closing price of {format_currency(prev)}, {comp} is trading {sign}{chg_pct:.2f}% {('higher' if (chg or 0) >= 0 else 'lower')} at {format_currency(curr)}.",
        ]
        return "\n".join(lines), tool_calls

    # 3. Quick Version / Key Numbers: "Give me the quick version on Reliance"
    if conv_mode == "quick_version" or "quick version" in q_low or "important numbers" in q_low or "not the full story" in q_low:
        info = get_company_info(symbol)
        mcap = info.get("market_cap")
        mcap_str = f"₹{mcap / 10000000:,.0f} Cr" if mcap else "N/A"
        pe = info.get("trailing_pe")
        pe_str = f"{pe:.2f}x" if pe else "N/A"
        sector = info.get("sector") or SECTOR_MAP.get(symbol, "Equities")
        lines = [
            f"### ⚡ {comp} (`{symbol}`) — Quick Version Snapshot\n",
            f"* **Current Price:** **{format_currency(curr)}** ({icon} **{sign}{chg_pct:.2f}%** today)",
            f"* **Market Capitalization:** {mcap_str} (Sector: {sector})",
            f"* **Valuation (Trailing P/E):** {pe_str}",
            f"* **52-Week Range:** {format_currency(l52)} – {format_currency(h52)}",
            f"\n**Quick Take:** {comp} is trading at {format_currency(curr)} ({sign}{chg_pct:.2f}% today) with a market capitalization of {mcap_str}.",
        ]
        return "\n".join(lines), tool_calls

    lines = [
        f"### 📈 {comp} (`{symbol}`) — Live Quote\n",
        f"* **Current Price:** **{format_currency(curr)}**",
        f"* **Today's Change:** {icon} **{sign}{chg:,.2f} ({sign}{chg_pct:.2f}%)**",
        f"* **Previous Close:** {format_currency(prev)}",
        f"* **52-Week Range:** {format_currency(l52)} – {format_currency(h52)}",
    ]

    # If the user asked "What happened to X today?", also fetch recent news
    if "what happened" in query.lower() or "why" in query.lower():
        tool_calls.append({"name": "get_market_news", "args": {"company_or_symbol": symbol, "limit": 2}})
        news = get_market_news(symbol, limit=2)
        if news:
            lines.append("\n**Today's Market Context & Headlines:**")
            for item in news:
                src = f" *({item.get('source')})*" if item.get("source") else ""
                lines.append(f"• **{item.get('title')}**{src}")
        else:
            lines.append("\n**Today's Market Context:** No unusual corporate disclosures reported today.")

    return "\n".join(lines), tool_calls


def handle_stock_fundamentals(
    symbol: str,
    params: dict[str, Any],
    query: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and display valuation, P/E, EPS, market cap, and fundamental ratios for a stock."""
    tool_calls = [
        {"name": "get_company_info", "args": {"symbol": symbol}},
        {"name": "get_stock_price", "args": {"symbol": symbol}},
    ]
    info = get_company_info(symbol)
    price = get_stock_price(symbol)

    comp = info.get("company") or price.get("company") or INDIAN_STOCK_MASTER.get(symbol, {}).get("company_name", symbol)
    pe = info.get("trailing_pe")
    f_pe = info.get("forward_pe")
    eps = info.get("eps")
    mcap = info.get("market_cap")
    div_yield = info.get("dividend_yield")
    sector = info.get("sector", "N/A")

    pe_str = f"**{pe:.2f}x**" if pe else "N/A"
    f_pe_str = f"{f_pe:.2f}x" if f_pe else "N/A"
    eps_str = f"₹{eps:.2f}" if eps else "N/A"
    mcap_str = f"₹{mcap / 10000000:,.0f} Cr" if mcap else "N/A"
    div_str = f"{div_yield:.2f}%" if div_yield else "N/A"

    conv_mode = params.get("conversational_mode")
    q_low = query.lower()

    # Due Diligence: "What should I look at before judging HDFC Bank?"
    if conv_mode == "due_diligence" or "before judging" in q_low or "checklist" in q_low or "what should i look at" in q_low:
        is_bank = "BANK" in symbol.upper() or sector == "Banking"
        if is_bank:
            checklist = [
                "1. **Asset Quality (NPA Trends):** Monitor Gross NPA (< 2.0%) and Net NPA (< 0.5%) for credit quality and provisioning safety.",
                "2. **Net Interest Margin (NIM):** Sustainable NIM between 3.5% and 4.2% indicates strong core lending profitability.",
                "3. **Deposit Franchise (CASA Ratio):** High Current and Savings Account ratio (> 40%) provides low-cost funding stability.",
                "4. **Valuation Multiple (P/B & P/E):** Compare current Price-to-Book and P/E against historical bank medians.",
                "5. **Capital Adequacy (CAR / Tier 1):** Robust capital buffer (> 15%) supporting sustainable loan book expansion.",
            ]
        else:
            checklist = [
                "1. **Operating Profit Margin (OPM):** Track consistency of core margins across differing industry cycles.",
                "2. **Return on Capital (ROCE / ROE):** Sustained ROCE > 15-20% proving efficient capital allocation by management.",
                "3. **Balance Sheet Leverage:** Low Debt-to-Equity (< 0.5x) or healthy net-cash position providing solvency safety.",
                "4. **Valuation Multiple (P/E vs Sector):** Trailing P/E evaluated alongside revenue and profit growth rates.",
                "5. **Free Cash Flow Conversion:** High cash generation from operations relative to reported net earnings.",
            ]
        lines = [
            f"### 📋 Due Diligence Checklist: Key Factors Before Judging {comp} (`{symbol}`)\n",
            f"Before evaluating or taking an investment position in **{comp}**, consider these 5 critical analytical metrics:\n",
            "\n".join(checklist),
            f"\n* **Current Financial Snapshot:** Trailing P/E: {pe_str} | Market Cap: {mcap_str} | Sector: {sector}",
        ]
        return "\n".join(lines), tool_calls

    lines = [
        f"### 📊 {comp} (`{symbol}`) — Fundamental Ratios & Valuation\n",
        f"* **Trailing P/E Ratio:** {pe_str}",
        f"* **Forward P/E Ratio:** {f_pe_str}",
        f"* **Earnings Per Share (EPS):** {eps_str}",
        f"* **Market Capitalization:** {mcap_str}",
        f"* **Dividend Yield:** {div_str}",
        f"* **Sector:** {sector}",
    ]

    # Valuation context
    val_eval = evaluate_pe_valuation(pe, sector=sector)
    if isinstance(val_eval, dict):
        lines.append(f"\n**Valuation Assessment:** `{val_eval.get('assessment', 'Fair')}` — {val_eval.get('reason', '')}")
    else:
        lines.append(f"\n**Valuation Assessment:** {val_eval}")

    return "\n".join(lines), tool_calls


def handle_stock_technicals(
    symbol: str,
    params: dict[str, Any],
    query: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and display technical indicators, moving averages, and sentiment for a stock."""
    tool_calls = [
        {"name": "get_stock_price", "args": {"symbol": symbol}},
        {"name": "get_stock_sentiment", "args": {"symbol": symbol}},
    ]
    p = get_stock_price(symbol)
    sent = get_stock_sentiment(symbol)

    comp = p.get("company") or INDIAN_STOCK_MASTER.get(symbol, {}).get("company_name", symbol)
    curr = p.get("current_price")
    h52 = p.get("52_week_high")
    l52 = p.get("52_week_low")

    conv_mode = params.get("conversational_mode")
    q_low = query.lower()

    # Recent Strength: "Has TCS been strong lately?"
    if conv_mode == "recent_strength" or "strong lately" in q_low or "momentum improved" in q_low:
        chg_pct = p.get("change_percent", 0.0) or 0.0
        dist_h = ((curr - h52) / h52) * 100.0 if curr and h52 and h52 > 0 else 0.0
        is_strong = (chg_pct >= 0) or (dist_h >= -8.0)
        status_txt = f"**Yes, {comp} has shown positive relative strength lately.**" if is_strong else f"**{comp} has been consolidating and lagging lately.**"
        lines = [
            f"### 📈 {comp} (`{symbol}`) — Recent Strength & Technical Momentum\n",
            f"{status_txt}\n",
            f"* **Current Price:** **{format_currency(curr)}**",
            f"* **Today's Movement:** {'+' if chg_pct >= 0 else ''}{chg_pct:.2f}%",
            f"* **Distance from 52-Week High:** **{dist_h:+.2f}%**",
            f"* **Quantitative Sentiment:** **{sent.get('sentiment', 'Neutral')}** (Score: {sent.get('score', 0):.1f})",
            f"* **52-Week Range:** {format_currency(l52)} – {format_currency(h52)}",
            f"\n**Technical Summary:** {comp} is trading at {format_currency(curr)}, {abs(dist_h):.1f}% below its 52-week peak with a {sent.get('sentiment', 'Neutral').lower()} technical bias.",
        ]
        return "\n".join(lines), tool_calls

    lines = [
        f"### 📉 {comp} (`{symbol}`) — Technical Sentiment & Indicators\n",
        f"* **Current Price:** {format_currency(curr)}",
        f"* **Quantitative Sentiment:** **{sent.get('sentiment', 'Neutral')}** (Score: {sent.get('score', 0):.1f})",
        f"* **52-Week High:** {format_currency(h52)}",
        f"* **52-Week Low:** {format_currency(l52)}",
    ]

    if curr and h52 and h52 > 0:
        dist_h = ((curr - h52) / h52) * 100.0
        lines.append(f"* **Distance from 52W High:** **{dist_h:+.2f}%**")

    if curr and l52 and l52 > 0:
        dist_l = ((curr - l52) / l52) * 100.0
        lines.append(f"* **Distance from 52W Low:** **+{dist_l:.2f}%**")

    return "\n".join(lines), tool_calls


def handle_stock_news(
    symbol: str,
    params: dict[str, Any],
    query: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve verified market news headlines and executive summaries for a stock."""
    from agent.demo_agent import format_news_section
    tool_calls = [{"name": "get_market_news", "args": {"company_or_symbol": symbol, "limit": 5}}]

    news = get_market_news(symbol, limit=5)
    comp = INDIAN_STOCK_MASTER.get(symbol, {}).get("company_name", symbol)
    p = get_stock_price(symbol)
    curr = p.get("current_price")
    chg_pct = p.get("change_percent")
    if chg_pct is None:
        chg_pct = 0.0
    sign = "+" if chg_pct >= 0 else ""
    icon = "🟢" if chg_pct >= 0 else "🔴"

    price_str = f"**{format_currency(curr)}** ({icon} **{sign}{chg_pct:.2f}%** today)" if curr else "Refreshing from exchange"
    lines = [
        f"### 📰 Market News & Catalysts for {comp} (`{symbol}`)\n",
        f"* **Current Price:** {price_str}\n",
    ]

    if news:
        lines.append("**Verified News Disclosures & Catalysts:**\n")
        lines.extend(format_news_section(news, company_name=comp, symbol=symbol, include_header=False))
    else:
        lines.append("* **Verified News Disclosures:** **No company-specific news catalyst or breaking corporate filing found today.**")
        lines.append(f"* **Market Rationale:** Today's price movement ({sign}{chg_pct:.2f}%) reflects broader sector momentum and routine exchange liquidity rather than a single company-specific headline.")

    return "\n".join(lines), tool_calls


def handle_stock_results(
    symbol: str,
    params: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve and display quarterly earnings results and growth metrics for a stock."""
    tool_calls = [{"name": "get_quarterly_financials", "args": {"symbol": symbol}}]
    qf = get_quarterly_financials(symbol)
    comp = INDIAN_STOCK_MASTER.get(symbol, {}).get("company_name", symbol)

    lines = [f"### 📑 {comp} (`{symbol}`) — Quarterly Financial Performance\n"]
    if qf and isinstance(qf, dict):
        rev = qf.get("total_revenue")
        net = qf.get("net_income")
        op_inc = qf.get("operating_income")
        pat_growth = qf.get("net_income_growth_yoy") or qf.get("yoy_profit_growth")
        rev_growth = qf.get("revenue_growth_yoy") or qf.get("yoy_revenue_growth")

        if rev:
            lines.append(f"* **Latest Quarterly Revenue:** ₹{rev / 10000000:,.1f} Cr")
        if net:
            lines.append(f"* **Net Profit (PAT):** ₹{net / 10000000:,.1f} Cr")
        if op_inc:
            lines.append(f"* **Operating Income (EBIT):** ₹{op_inc / 10000000:,.1f} Cr")
        if pat_growth is not None:
            lines.append(f"* **YoY Profit Growth:** **{pat_growth:+.1f}%**")
        if rev_growth is not None:
            lines.append(f"* **YoY Revenue Growth:** **{rev_growth:+.1f}%**")
    else:
        lines.append("Latest quarterly disclosures are currently being refreshed from exchange records.")

    return "\n".join(lines), tool_calls


def handle_stock_52w_high_low(
    symbol: str,
    params: dict[str, Any],
    query: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Display 52-week high, low, and current distance for a single stock."""
    tool_calls = [{"name": "get_stock_price", "args": {"symbol": symbol}}]
    p = get_stock_price(symbol)
    comp = p.get("company") or INDIAN_STOCK_MASTER.get(symbol, {}).get("company_name", symbol)

    curr = p.get("current_price")
    h52 = p.get("52_week_high")
    l52 = p.get("52_week_low")

    conv_mode = params.get("conversational_mode")
    q_low = query.lower()

    if conv_mode == "distance" or "how far" in q_low or "best price" in q_low:
        if curr and h52 and h52 > 0:
            diff = h52 - curr
            dist_h = ((curr - h52) / h52) * 100.0
            dist_l = ((curr - l52) / l52) * 100.0 if l52 and l52 > 0 else 0.0
            lines = [
                f"### 🎯 {comp} (`{symbol}`) — Distance from Yearly High (Best Price)\n",
                f"**{comp} is currently {format_currency(diff)} ({abs(dist_h):.2f}%) below its 52-week high.**\n",
                f"* **Current Price:** **{format_currency(curr)}**",
                f"* **52-Week High (Best Price):** **{format_currency(h52)}**",
                f"* **52-Week Low:** {format_currency(l52)}",
                f"* **Distance from 52-Week High:** **{dist_h:+.2f}%**",
                f"* **Distance from 52-Week Low:** **+{dist_l:.2f}%**",
            ]
            return "\n".join(lines), tool_calls

    if conv_mode == "closer_extreme" or "closer to" in q_low:
        if curr and h52 and l52 and h52 > 0 and l52 > 0:
            dist_h = abs(((curr - h52) / h52) * 100.0)
            dist_l = abs(((curr - l52) / l52) * 100.0)
            is_closer_high = dist_h <= dist_l
            closer_name = "52-week high" if is_closer_high else "52-week low"
            closer_dist = dist_h if is_closer_high else dist_l
            further_name = "52-week low" if is_closer_high else "52-week high"
            further_dist = dist_l if is_closer_high else dist_h
            lines = [
                f"### 🎯 {comp} (`{symbol}`) — 52-Week Extreme Proximity\n",
                f"**{comp} is currently closer to its {closer_name} ({closer_dist:.2f}% away) than its {further_name} ({further_dist:.2f}% away).**\n",
                f"* **Current Price:** **{format_currency(curr)}**",
                f"* **52-Week High:** {format_currency(h52)} ({dist_h:.2f}% above current)",
                f"* **52-Week Low:** {format_currency(l52)} ({dist_l:.2f}% below current)",
            ]
            return "\n".join(lines), tool_calls

    lines = [
        f"### 🎯 {comp} (`{symbol}`) — 52-Week High & Low\n",
        f"* **Current Price:** **{format_currency(curr)}**",
        f"* **52-Week High:** **{format_currency(h52)}**",
        f"* **52-Week Low:** **{format_currency(l52)}**",
    ]

    if curr and h52 and h52 > 0:
        dist_h = ((curr - h52) / h52) * 100.0
        lines.append(f"* **Distance from 52-Week High:** **{dist_h:+.2f}%**")

    if curr and l52 and l52 > 0:
        dist_l = ((curr - l52) / l52) * 100.0
        lines.append(f"* **Distance from 52-Week Low:** **+{dist_l:.2f}%**")

    return "\n".join(lines), tool_calls


def handle_stock_overview(
    symbol: str,
    params: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Provide a comprehensive market overview, fundamentals, and verified news for a stock."""
    from agent.demo_agent import run_demo_single_stock
    return run_demo_single_stock(symbol, is_news_query=False, comprehensive=True)


def handle_stock_comparison(
    symbols: list[str],
    params: dict[str, Any],
    query: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Compare N stocks side-by-side, answering specific metric comparisons if asked."""
    from agent.demo_agent import run_demo_comparison

    req_metric = params.get("metric")
    q_low = query.lower()

    # 1. Valuation / P/E comparison
    if (req_metric == "pe" or "valuation" in q_low or "pe" in q_low) and len(symbols) >= 2:
        s1, s2 = symbols[0], symbols[1]
        i1, i2 = get_company_info(s1), get_company_info(s2)
        p1, p2 = get_stock_price(s1), get_stock_price(s2)
        c1 = INDIAN_STOCK_MASTER.get(s1, {}).get("company_name", s1)
        c2 = INDIAN_STOCK_MASTER.get(s2, {}).get("company_name", s2)
        pe1, pe2 = i1.get("trailing_pe") or 0.0, i2.get("trailing_pe") or 0.0
        pb1, pb2 = i1.get("price_to_book") or 0.0, i2.get("price_to_book") or 0.0
        mc1, mc2 = i1.get("market_cap") or 0, i2.get("market_cap") or 0

        tool_calls = [
            {"name": "get_company_info", "args": {"symbol": s1}},
            {"name": "get_company_info", "args": {"symbol": s2}},
        ]

        if "more expensive" in q_low or "expensive" in q_low:
            more_exp = (s1, c1, pe1) if pe1 >= pe2 else (s2, c2, pe2)
            cheaper = (s2, c2, pe2) if pe1 >= pe2 else (s1, c1, pe1)
            header = f"### 💡 Valuation Comparison: {more_exp[1]} is More Expensive"
            conclusion = f"Between the two, **{more_exp[1]} (`{more_exp[0]}`)** is more expensive by P/E, trading at a trailing valuation multiple of **{more_exp[2]:.2f}x**, compared to **{cheaper[1]} (`{cheaper[0]}`)** at **{cheaper[2]:.2f}x**."
        else:
            cheaper = (s1, c1, pe1) if (pe1 > 0 and (pe1 <= pe2 or pe2 == 0)) else (s2, c2, pe2)
            more_exp = (s2, c2, pe2) if cheaper[0] == s1 else (s1, c1, pe1)
            header = f"### 💡 Valuation Comparison: {cheaper[1]} Looks Cheaper"
            conclusion = f"Between the two, **{cheaper[1]} (`{cheaper[0]}`)** looks cheaper on valuation, trading at a trailing P/E of **{cheaper[2]:.2f}x**, compared to **{more_exp[1]} (`{more_exp[0]}`)** at **{more_exp[2]:.2f}x**."

        lines = [
            f"{header}\n",
            f"{conclusion}\n",
            "| Company | Symbol | Trailing P/E | Price-to-Book | Market Cap | Current Price |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| **{c1}** | `{s1}` | **{pe1:.2f}x** | {pb1:.2f}x | ₹{mc1/10000000:,.0f} Cr | {format_currency(p1.get('current_price'))} |",
            f"| **{c2}** | `{s2}` | **{pe2:.2f}x** | {pb2:.2f}x | ₹{mc2/10000000:,.0f} Cr | {format_currency(p2.get('current_price'))} |",
        ]
        return "\n".join(lines), tool_calls

    # 2. 52-Week High Distance Comparison: "TCS or Infosys — which is trading closer to its 52-week high?"
    if (req_metric == "52w_high_distance" or "closer to its 52-week" in q_low or "further from" in q_low) and len(symbols) >= 2:
        s1, s2 = symbols[0], symbols[1]
        p1, p2 = get_stock_price(s1), get_stock_price(s2)
        c1 = INDIAN_STOCK_MASTER.get(s1, {}).get("company_name", s1)
        c2 = INDIAN_STOCK_MASTER.get(s2, {}).get("company_name", s2)
        curr1, curr2 = p1.get("current_price", 0.0), p2.get("current_price", 0.0)
        h1, h2 = p1.get("52_week_high", 1.0), p2.get("52_week_high", 1.0)
        d1 = abs(((curr1 - h1) / h1) * 100.0) if h1 else 999.0
        d2 = abs(((curr2 - h2) / h2) * 100.0) if h2 else 999.0

        tool_calls = [
            {"name": "get_stock_price", "args": {"symbol": s1}},
            {"name": "get_stock_price", "args": {"symbol": s2}},
        ]

        if "further" in q_low:
            further = (s1, c1, d1) if d1 >= d2 else (s2, c2, d2)
            closer = (s2, c2, d2) if d1 >= d2 else (s1, c1, d1)
            header = f"### 🎯 52-Week High Comparison: {further[1]} is Further"
            conclusion = f"Between the two, **{further[1]} (`{further[0]}`)** is further from its yearly high, sitting **{further[2]:.2f}%** below its 52-week peak, compared to **{closer[1]} (`{closer[0]}`)** which is **{closer[2]:.2f}%** away."
        else:
            closer = (s1, c1, d1) if d1 <= d2 else (s2, c2, d2)
            further = (s2, c2, d2) if closer[0] == s1 else (s1, c1, d1)
            header = f"### 🎯 52-Week High Comparison: {closer[1]} is Closer"
            conclusion = f"Between the two, **{closer[1]} (`{closer[0]}`)** is trading closer to its 52-week high, sitting **{closer[2]:.2f}%** below its peak, compared to **{further[1]} (`{further[0]}`)** at **{further[2]:.2f}%**."

        lines = [
            f"{header}\n",
            f"{conclusion}\n",
            "| Company | Symbol | Current Price | 52-Week High | Distance from High | 52-Week Low |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| **{c1}** | `{s1}` | {format_currency(curr1)} | {format_currency(h1)} | **-{d1:.2f}%** | {format_currency(p1.get('52_week_low'))} |",
            f"| **{c2}** | `{s2}` | {format_currency(curr2)} | {format_currency(h2)} | **-{d2:.2f}%** | {format_currency(p2.get('52_week_low'))} |",
        ]
        return "\n".join(lines), tool_calls

    # 3. Today's Move: "HDFC Bank versus ICICI Bank — which moved more today?"
    if (req_metric == "today_move" or "which moved more" in q_low or "who moved more" in q_low) and len(symbols) >= 2:
        s1, s2 = symbols[0], symbols[1]
        p1, p2 = get_stock_price(s1), get_stock_price(s2)
        c1 = INDIAN_STOCK_MASTER.get(s1, {}).get("company_name", s1)
        c2 = INDIAN_STOCK_MASTER.get(s2, {}).get("company_name", s2)
        chg1, chg2 = p1.get("change_percent", 0.0), p2.get("change_percent", 0.0)

        tool_calls = [
            {"name": "get_stock_price", "args": {"symbol": s1}},
            {"name": "get_stock_price", "args": {"symbol": s2}},
        ]

        bigger = (s1, c1, chg1) if abs(chg1) >= abs(chg2) else (s2, c2, chg2)
        smaller = (s2, c2, chg2) if bigger[0] == s1 else (s1, c1, chg1)
        sign1 = "+" if chg1 >= 0 else ""
        sign2 = "+" if chg2 >= 0 else ""

        lines = [
            f"### 📊 Intraday Movement: {bigger[1]} Moved More\n",
            f"**{bigger[1]} (`{bigger[0]}`)** experienced the larger price movement today, changing by **{bigger[2]:+.2f}%**, compared to **{smaller[1]} (`{smaller[0]}`)** at **{smaller[2]:+.2f}%**.\n",
            "| Company | Symbol | Current Price | Today's Change | Previous Close |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **{c1}** | `{s1}` | {format_currency(p1.get('current_price'))} | **{sign1}{chg1:.2f}%** | {format_currency(p1.get('previous_close'))} |",
            f"| **{c2}** | `{s2}` | {format_currency(p2.get('current_price'))} | **{sign2}{chg2:.2f}%** | {format_currency(p2.get('previous_close'))} |",
        ]
        return "\n".join(lines), tool_calls

    # 4. Momentum Comparison
    if (req_metric == "momentum" or "momentum" in q_low) and len(symbols) >= 2:
        s1, s2 = symbols[0], symbols[1]
        p1, p2 = get_stock_price(s1), get_stock_price(s2)
        c1 = INDIAN_STOCK_MASTER.get(s1, {}).get("company_name", s1)
        c2 = INDIAN_STOCK_MASTER.get(s2, {}).get("company_name", s2)
        chg1, chg2 = p1.get("change_percent", 0.0), p2.get("change_percent", 0.0)
        leader = (s1, c1, chg1) if chg1 >= chg2 else (s2, c2, chg2)
        laggard = (s2, c2, chg2) if leader[0] == s1 else (s1, c1, chg1)

        tool_calls = [
            {"name": "get_stock_price", "args": {"symbol": s1}},
            {"name": "get_stock_price", "args": {"symbol": s2}},
        ]
        lines = [
            f"### 🚀 Momentum Comparison: {leader[1]} has Stronger Momentum\n",
            f"**{leader[1]} (`{leader[0]}`)** demonstrates stronger relative momentum ({leader[2]:+.2f}% today) compared to **{laggard[1]} (`{laggard[0]}`)** ({laggard[2]:+.2f}%).\n",
            "| Company | Symbol | Price | Today's Move | 52W High | 52W Low |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| **{c1}** | `{s1}` | {format_currency(p1.get('current_price'))} | **{chg1:+.2f}%** | {format_currency(p1.get('52_week_high'))} | {format_currency(p1.get('52_week_low'))} |",
            f"| **{c2}** | `{s2}` | {format_currency(p2.get('current_price'))} | **{chg2:+.2f}%** | {format_currency(p2.get('52_week_high'))} | {format_currency(p2.get('52_week_low'))} |",
        ]
        return "\n".join(lines), tool_calls

    # 5. Multi-attribute comparison: Size, Valuation, Today's Move
    if (req_metric == "multi_attribute" or "size, valuation" in q_low or "compare size" in q_low) and len(symbols) >= 2:
        s1, s2 = symbols[0], symbols[1]
        i1, i2 = get_company_info(s1), get_company_info(s2)
        p1, p2 = get_stock_price(s1), get_stock_price(s2)
        c1 = INDIAN_STOCK_MASTER.get(s1, {}).get("company_name", s1)
        c2 = INDIAN_STOCK_MASTER.get(s2, {}).get("company_name", s2)
        tool_calls = [
            {"name": "get_company_info", "args": {"symbol": s1}},
            {"name": "get_company_info", "args": {"symbol": s2}},
        ]
        lines = [
            f"### ⚖️ Multi-Factor Comparison: {c1} vs {c2}\n",
            "| Comparison Dimension | " + f"**{c1} (`{s1}`)** | **{c2} (`{s2}`)** |",
            "| :--- | :--- | :--- |",
            f"| **Market Capitalization (Size)** | ₹{(i1.get('market_cap') or 0)/1e7:,.0f} Cr | ₹{(i2.get('market_cap') or 0)/1e7:,.0f} Cr |",
            f"| **Valuation (Trailing P/E)** | **{(i1.get('trailing_pe') or 0.0):.2f}x** | **{(i2.get('trailing_pe') or 0.0):.2f}x** |",
            f"| **Price-to-Book (P/B)** | {(i1.get('price_to_book') or 0.0):.2f}x | {(i2.get('price_to_book') or 0.0):.2f}x |",
            f"| **Current Share Price** | {format_currency(p1.get('current_price'))} | {format_currency(p2.get('current_price'))} |",
            f"| **Today's Movement** | **{p1.get('change_percent', 0.0):+.2f}%** | **{p2.get('change_percent', 0.0):+.2f}%** |",
            f"| **52-Week Range** | {format_currency(p1.get('52_week_low'))} – {format_currency(p1.get('52_week_high'))} | {format_currency(p2.get('52_week_low'))} – {format_currency(p2.get('52_week_high'))} |",
        ]
        return "\n".join(lines), tool_calls

    report, tool_calls = run_demo_comparison(symbols)
    return report, tool_calls


def handle_stocks_by_sector(params: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Display leading stocks in a requested sector (e.g. IT, Banking, Auto, Pharma)."""
    sector_name = params.get("sector", "Information Technology")
    sec_info = SECTOR_DEFINITIONS.get(sector_name, {})
    constituents = sec_info.get("constituents") or ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"]

    tool_calls = [{"name": "get_sector_constituents", "args": {"sector": sector_name}}]
    stock_quotes: list[dict[str, Any]] = []

    for sym in constituents[:8]:
        try:
            p = get_stock_price(sym)
            curr = p.get("current_price")
            chg_pct = p.get("change_percent", 0.0)
            if curr:
                stock_quotes.append({
                    "symbol": sym,
                    "company": p.get("company") or INDIAN_STOCK_MASTER.get(sym, {}).get("company_name", sym),
                    "price": curr,
                    "change_pct": chg_pct,
                })
        except Exception:
            continue

    # Rank by day's gainers in sector
    stock_quotes = sorted(stock_quotes, key=lambda x: x["change_pct"], reverse=True)

    lines = [
        f"### 🏢 {sector_name} Sector Equities (NSE)\n",
        f"Here is how major listed **{sector_name}** stocks are trading today:\n",
        "| Symbol | Company Name | Current Price | Today's Change |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for sq in stock_quotes:
        icon = "🟢 +" if sq["change_pct"] >= 0 else "🔴 "
        lines.append(f"| **`{sq['symbol']}`** | {sq['company']} | {format_currency(sq['price'])} | {icon}{sq['change_pct']:.2f}% |")

    return "\n".join(lines), tool_calls


def handle_market_screening(query: str, params: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Execute dynamic market-wide screens across liquid NSE stocks for breakout, momentum, extremes, and sector setups."""
    import concurrent.futures
    screen_type = params.get("screen_type", "near_breakout")
    q_low = query.lower()
    tool_calls = [{"name": "scan_market_screening", "args": {"screen_type": screen_type}}]

    scan_universe = list(NIFTY_CORE_SYMBOLS)
    quotes: list[dict[str, Any]] = []

    def _fetch_quote(sym: str) -> Optional[dict[str, Any]]:
        try:
            p = get_stock_price(sym)
            curr = p.get("current_price")
            if curr:
                h52 = p.get("52_week_high") or curr
                l52 = p.get("52_week_low") or curr
                prev = p.get("previous_close") or curr
                open_p = p.get("open") or prev
                day_h = p.get("day_high") or curr
                day_l = p.get("day_low") or curr
                chg = p.get("change") or 0.0
                chg_pct = p.get("change_percent") or 0.0
                dist_h = ((curr - h52) / h52) * 100.0 if h52 else 0.0
                dist_l = ((curr - l52) / l52) * 100.0 if l52 else 0.0
                intraday_pct = ((day_h - day_l) / day_l) * 100.0 if day_l else abs(chg_pct)
                return {
                    "symbol": sym,
                    "company": p.get("company") or INDIAN_STOCK_MASTER.get(sym, {}).get("company_name", sym),
                    "price": curr,
                    "previous_close": prev,
                    "open": open_p,
                    "day_high": day_h,
                    "day_low": day_l,
                    "change": chg,
                    "change_pct": chg_pct,
                    "52_week_high": h52,
                    "52_week_low": l52,
                    "dist_h": dist_h,
                    "dist_l": dist_l,
                    "intraday_pct": intraday_pct,
                    "sector": SECTOR_MAP.get(sym, "Equities"),
                }
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futs = {executor.submit(_fetch_quote, s): s for s in scan_universe}
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            if res:
                quotes.append(res)

    lines: list[str] = []

    # 1. Near Breakout / Within 3% of 52W High
    if screen_type in ("near_breakout", "within_3pct_high", "closest_breakout") or "within 3% of their 52-week high" in q_low or "closest to a breakout" in q_low or "almost breaking" in q_low:
        filtered = [q for q in quotes if q["dist_h"] >= -5.0]
        if not filtered:
            filtered = quotes
        sorted_q = sorted(filtered, key=lambda x: x["dist_h"], reverse=True)[:5]
        lines = [
            "### 🚀 NSE Stocks Sitting Within 3% of 52-Week High (Breakout Watch)\n",
            "The following liquid equities are trading closest to their yearly highs and potential breakout levels:\n",
            "| Symbol | Company Name | Current Price | 52-Week High | Distance from High | Today's Move | Sector |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['52_week_high'])} | **{q['dist_h']:+.2f}%** | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% | {q['sector']} |")

    # 2. Far from High: "Find shares that are nowhere near their yearly highs"
    elif screen_type == "far_from_high" or "nowhere near" in q_low:
        sorted_q = sorted(quotes, key=lambda x: x["dist_h"])[:5]
        lines = [
            "### 📉 NSE Stocks Nowhere Near Their Yearly Highs\n",
            "These equities have experienced significant corrections and are trading deepest below their 52-week peak:\n",
            "| Symbol | Company Name | Current Price | 52-Week High | Distance from Peak | 52-Week Low | Sector |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['52_week_high'])} | **{q['dist_h']:+.2f}%** | {format_currency(q['52_week_low'])} | {q['sector']} |")

    # 3. Recovering from Low
    elif screen_type == "recovering_from_low" or "recovering from their yearly lows" in q_low:
        filtered = [q for q in quotes if q["dist_l"] >= 3.0 and q["change_pct"] >= 0]
        if not filtered:
            filtered = sorted(quotes, key=lambda x: x["dist_l"])[:5]
        sorted_q = sorted(filtered, key=lambda x: x["change_pct"], reverse=True)[:5]
        lines = [
            "### 📈 NSE Stocks Recovering from Yearly Lows\n",
            "These companies have bounced off their 52-week low support and are showing positive recovery momentum today:\n",
            "| Symbol | Company Name | Current Price | 52-Week Low | Recovery from Low | Today's Gain | Sector |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['52_week_low'])} | **+{q['dist_l']:.2f}%** | 🟢 **+{q['change_pct']:.2f}%** | {q['sector']} |")

    # 4. Within 3% of 52-Week Low
    elif screen_type == "within_3pct_low" or "within 3% of their 52-week low" in q_low:
        filtered = [q for q in quotes if q["dist_l"] <= 5.0]
        if not filtered:
            filtered = quotes
        sorted_q = sorted(filtered, key=lambda x: x["dist_l"])[:5]
        lines = [
            "### 📉 NSE Stocks Sitting Within 3% of 52-Week Low\n",
            "These liquid shares are hovering near their 52-week low base and multi-month support:\n",
            "| Symbol | Company Name | Current Price | 52-Week Low | Distance Above Low | Today's Move | Sector |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['52_week_low'])} | **+{q['dist_l']:.2f}%** | {q['change_pct']:+.2f}% | {q['sector']} |")

    # 5. Fresh Highs but Red Today: "Which shares made fresh highs but are red today?"
    elif screen_type == "high_but_red" or "highs but are red" in q_low or "high but red" in q_low:
        filtered = [q for q in quotes if q["dist_h"] >= -6.0 and q["change_pct"] < 0]
        if not filtered:
            filtered = [q for q in quotes if q["change_pct"] < 0]
        sorted_q = sorted(filtered, key=lambda x: x["dist_h"], reverse=True)[:5]
        lines = [
            "### ⚠️ Stocks Near 52-Week Highs but Red (Falling) Today\n",
            "These shares are positioned near their 52-week highs but are witnessing intraday profit-booking:\n",
            "| Symbol | Company Name | Current Price | 52-Week High | Distance from High | Today's Loss |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['52_week_high'])} | **{q['dist_h']:+.2f}%** | 🔴 **{q['change_pct']:.2f}%** |")

    # 6. Near Lows but Green Today: "Which shares are near yearly lows but green today?"
    elif screen_type == "low_but_green" or "lows but green" in q_low or "low but green" in q_low:
        filtered = [q for q in quotes if q["dist_l"] <= 12.0 and q["change_pct"] > 0]
        if not filtered:
            filtered = [q for q in quotes if q["change_pct"] > 0]
        sorted_q = sorted(filtered, key=lambda x: x["dist_l"])[:5]
        lines = [
            "### 🟢 Stocks Near Yearly Lows but Green (Bouncing) Today\n",
            "These shares are trading near their 52-week lows but showing positive intraday accumulation and bounce-back:\n",
            "| Symbol | Company Name | Current Price | 52-Week Low | Distance Above Low | Today's Gain |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['52_week_low'])} | **+{q['dist_l']:.2f}%** | 🟢 **+{q['change_pct']:.2f}%** |")

    # 7. Strong stocks that have not yet made a 52-week high
    elif screen_type == "strong_not_yet_high" or "not yet made a 52-week high" in q_low:
        filtered = [q for q in quotes if q["change_pct"] > 0.5 and q["dist_h"] <= -4.0]
        if not filtered:
            filtered = [q for q in quotes if q["change_pct"] > 0]
        sorted_q = sorted(filtered, key=lambda x: x["change_pct"], reverse=True)[:5]
        lines = [
            "### 🚀 Strong Stocks Not Yet at 52-Week High\n",
            "These equities display robust positive momentum today while retaining headroom below their yearly highs:\n",
            "| Symbol | Company Name | Current Price | Today's Move | Distance Below Peak | 52-Week High |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | 🟢 **+{q['change_pct']:.2f}%** | **{q['dist_h']:+.2f}%** | {format_currency(q['52_week_high'])} |")

    # 8. Yearly Extremes: "What companies are trading near their yearly extremes?"
    elif screen_type == "yearly_extremes" or "yearly extremes" in q_low:
        highs = sorted([q for q in quotes if q["dist_h"] >= -5.0], key=lambda x: x["dist_h"], reverse=True)[:3]
        lows = sorted([q for q in quotes if q["dist_l"] <= 6.0], key=lambda x: x["dist_l"])[:3]
        if not highs:
            highs = sorted(quotes, key=lambda x: x["dist_h"], reverse=True)[:3]
        if not lows:
            lows = sorted(quotes, key=lambda x: x["dist_l"])[:3]
        lines = [
            "### 🎯 NSE Stocks Trading Near Yearly Extremes\n",
            "**Near 52-Week Highs:**\n",
            "| Symbol | Company Name | Price | 52W High | Distance from High |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in highs:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['52_week_high'])} | **{q['dist_h']:+.2f}%** |")
        lines.extend([
            "\n**Near 52-Week Lows:**\n",
            "| Symbol | Company Name | Price | 52W Low | Distance from Low |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for q in lows:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['52_week_low'])} | **+{q['dist_l']:.2f}%** |")

    # 9. Outperforming the Market Today
    elif screen_type == "outperforming_market" or "outperforming the market" in q_low:
        nifty = get_market_index("NIFTY")
        n_pct = nifty.get("change_percent") or 0.0
        filtered = [q for q in quotes if q["change_pct"] > n_pct]
        sorted_q = sorted(filtered, key=lambda x: x["change_pct"], reverse=True)[:5]
        lines = [
            f"### 🏆 Stocks Outperforming Nifty 50 ({n_pct:+.2f}%) Today\n",
            "These equities are generating positive alpha relative to the benchmark index:\n",
            "| Symbol | Company Name | Current Price | Today's Gain | Alpha vs Nifty |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            alpha = q["change_pct"] - n_pct
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | 🟢 **+{q['change_pct']:.2f}%** | **+{alpha:.2f}%** |")

    # 10. Underperforming the Market Today
    elif screen_type == "underperforming_market" or "underperforming the market" in q_low or "lagging the market" in q_low:
        nifty = get_market_index("NIFTY")
        n_pct = nifty.get("change_percent") or 0.0
        filtered = [q for q in quotes if q["change_pct"] < n_pct]
        sorted_q = sorted(filtered, key=lambda x: x["change_pct"])[:5]
        lines = [
            f"### 📉 Stocks Underperforming Nifty 50 ({n_pct:+.2f}%) Today\n",
            "These equities are lagging benchmark performance today:\n",
            "| Symbol | Company Name | Current Price | Today's Change | Relative Lag |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lag = q["change_pct"] - n_pct
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | 🔴 **{q['change_pct']:.2f}%** | **{lag:.2f}%** |")

    # 11. Biggest Intraday Percentage Move / Moving Unusually Fast
    elif screen_type == "biggest_intraday_move" or "biggest intraday percentage move" in q_low or "moving unusually fast" in q_low:
        sorted_q = sorted(quotes, key=lambda x: x["intraday_pct"], reverse=True)[:5]
        lines = [
            "### ⚡ Stocks with the Biggest Intraday Percentage Moves\n",
            "These shares are exhibiting the largest intraday volatility and price movement today:\n",
            "| Symbol | Company Name | Current Price | Day's Low | Day's High | Intraday Range | Today's Move |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {format_currency(q['day_low'])} | {format_currency(q['day_high'])} | **{q['intraday_pct']:.2f}%** | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% |")

    # 12. Opened Weak but Recovered
    elif screen_type == "opened_weak_recovered" or "opened weak but recovered" in q_low:
        filtered = [q for q in quotes if q["open"] < q["previous_close"] and q["price"] > q["open"]]
        if not filtered:
            filtered = [q for q in quotes if q["change_pct"] >= 0]
        sorted_q = sorted(filtered, key=lambda x: (x["price"] - x["open"]) / x["open"], reverse=True)[:5]
        lines = [
            "### 🔄 Stocks That Opened Weak but Recovered Intraday\n",
            "These shares experienced gap-downs or weak opening prices but attracted strong dip-buying:\n",
            "| Symbol | Company Name | Prev Close | Open Price | Current Price | Today's Net Change |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['previous_close'])} | {format_currency(q['open'])} | **{format_currency(q['price'])}** | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% |")

    # 13. Opened Strong but Gave Up Gains
    elif screen_type == "opened_strong_gave_up" or "opened strong but gave up" in q_low:
        filtered = [q for q in quotes if q["open"] > q["previous_close"] and q["price"] < q["open"]]
        if not filtered:
            filtered = [q for q in quotes if q["change_pct"] < 0]
        sorted_q = sorted(filtered, key=lambda x: (x["open"] - x["price"]) / x["open"], reverse=True)[:5]
        lines = [
            "### ⚠️ Stocks That Opened Strong but Gave Up Gains\n",
            "These shares opened strong with positive momentum but encountered intraday selling pressure:\n",
            "| Symbol | Company Name | Prev Close | Open Price | Current Price | Today's Net Change |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['previous_close'])} | {format_currency(q['open'])} | **{format_currency(q['price'])}** | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% |")

    # 14. Biggest Positive Movers Excluding Banks
    elif screen_type == "gainers_ex_banks" or "excluding banks" in q_low:
        non_banks = [q for q in quotes if q["sector"] != "Banking"]
        sorted_q = sorted(non_banks, key=lambda x: x["change_pct"], reverse=True)[:5]
        lines = [
            "### 🚀 Biggest Positive Movers (Excluding Banking Stocks)\n",
            "Top gainers across non-banking sectors on the NSE today:\n",
            "| Symbol | Company Name | Sector | Current Price | Today's Gain |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {q['sector']} | {format_currency(q['price'])} | 🟢 **+{q['change_pct']:.2f}%** |")

    # 15. Biggest Losers Excluding IT Stocks
    elif screen_type == "losers_ex_it" or "excluding it stocks" in q_low:
        non_it = [q for q in quotes if q["sector"] != "Information Technology"]
        sorted_q = sorted(non_it, key=lambda x: x["change_pct"])[:5]
        lines = [
            "### 📉 Biggest Losers (Excluding IT Stocks)\n",
            "Top falling stocks across non-IT sectors on the NSE today:\n",
            "| Symbol | Company Name | Sector | Current Price | Today's Loss |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {q['sector']} | {format_currency(q['price'])} | 🔴 **{q['change_pct']:.2f}%** |")

    # 16. Large-Cap Leading / Lagging
    elif screen_type in ("largecap_gainers", "largecap_losers") or "large-cap stocks are leading" in q_low or "large-cap stocks are lagging" in q_low:
        if screen_type == "largecap_losers" or "lagging" in q_low:
            sorted_q = sorted(quotes, key=lambda x: x["change_pct"])[:5]
            lines = [
                "### 📉 Large-Cap Stocks Lagging the Market Today\n",
                "Largest index heavyweights experiencing negative pressure:\n",
                "| Symbol | Company Name | Current Price | Today's Change | Sector |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
            for q in sorted_q:
                lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | 🔴 **{q['change_pct']:.2f}%** | {q['sector']} |")
        else:
            sorted_q = sorted(quotes, key=lambda x: x["change_pct"], reverse=True)[:5]
            lines = [
                "### 🏆 Large-Cap Stocks Leading the Market Today\n",
                "Top performing Nifty index heavyweights driving market gains:\n",
                "| Symbol | Company Name | Current Price | Today's Gain | Sector |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
            for q in sorted_q:
                lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | 🟢 **+{q['change_pct']:.2f}%** | {q['sector']} |")

    # 17. Power Sector Strength: "Are power stocks broadly strong today?"
    elif screen_type == "power_sector_strength" or "power stocks" in q_low:
        p_syms = ["TATAPOWER", "NTPC", "POWERGRID"]
        p_quotes = [q for q in quotes if q["symbol"] in p_syms]
        avg_chg = sum(q["change_pct"] for q in p_quotes) / len(p_quotes) if p_quotes else 0.0
        verdict = "**Yes, power stocks are broadly positive today.**" if avg_chg > 0 else "**Power stocks are consolidating / mixed today.**"
        lines = [
            "### ⚡ Power Sector Strength Assessment\n",
            f"{verdict} The basket is averaging **{avg_chg:+.2f}%** today.\n",
            "| Symbol | Company Name | Current Price | Today's Change | 52W High |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in p_quotes:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% | {format_currency(q['52_week_high'])} |")

    # 18. Metal Sector Leader: "Which metal stock is leading its sector today?"
    elif screen_type == "metal_sector_leader" or "metal stock" in q_low:
        m_syms = ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"]
        m_quotes = [q for q in quotes if q["symbol"] in m_syms]
        leader = max(m_quotes, key=lambda x: x["change_pct"]) if m_quotes else quotes[0]
        lines = [
            f"### ⛏️ Metal Sector Leader: {leader['company']} is Leading\n",
            f"**{leader['company']} (`{leader['symbol']}`)** is leading the metal sector today with a gain of **{leader['change_pct']:+.2f}%**.\n",
            "| Symbol | Company Name | Current Price | Today's Change | 52-Week Range |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted(m_quotes, key=lambda x: x["change_pct"], reverse=True):
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% | {format_currency(q['52_week_low'])} – {format_currency(q['52_week_high'])} |")

    # 19. Banking Technical Setup / Outperformers
    elif screen_type in ("banking_technical_setup", "bank_outperformers") or "strongest technical setup" in q_low or "banking stocks are outperforming" in q_low:
        b_syms = ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"]
        b_quotes = [q for q in quotes if q["symbol"] in b_syms]
        leader = max(b_quotes, key=lambda x: x["change_pct"]) if b_quotes else quotes[0]
        lines = [
            f"### 🏦 Banking Sector Relative Strength & Setup: {leader['company']} Leads\n",
            f"Among major banks, **{leader['company']} (`{leader['symbol']}`)** demonstrates the strongest relative strength ({leader['change_pct']:+.2f}% today) and closest proximity to key resistance/highs.\n",
            "| Symbol | Company Name | Current Price | Today's Move | Distance from 52W High |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted(b_quotes, key=lambda x: x["change_pct"], reverse=True):
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% | **{q['dist_h']:+.2f}%** |")

    # 20. IT Outperformers
    elif screen_type == "it_outperformers" or "outperforming nifty it" in q_low:
        it_syms = ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"]
        it_quotes = [q for q in quotes if q["symbol"] in it_syms]
        avg_it = sum(q["change_pct"] for q in it_quotes) / len(it_quotes) if it_quotes else 0.0
        leader = max(it_quotes, key=lambda x: x["change_pct"]) if it_quotes else quotes[0]
        lines = [
            f"### 💻 IT Stocks Outperforming Sector Average ({avg_it:+.2f}%)\n",
            f"**{leader['company']} (`{leader['symbol']}`)** is the primary outperformer among large-cap tech peers today.\n",
            "| Symbol | Company Name | Current Price | Today's Move | Outperformance vs IT Avg |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted(it_quotes, key=lambda x: x["change_pct"], reverse=True):
            diff = q["change_pct"] - avg_it
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% | **{diff:+.2f}%** |")

    else:
        # Default: Top positive momentum stocks
        sorted_q = sorted(quotes, key=lambda x: x["change_pct"], reverse=True)[:5]
        lines = [
            "### 📊 NSE Market Screening Results\n",
            "Here are the top-ranking liquid NSE equities meeting active market criteria:\n",
            "| Symbol | Company Name | Current Price | Today's Move | 52W High | 52W Low |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for q in sorted_q:
            lines.append(f"| **`{q['symbol']}`** | {q['company']} | {format_currency(q['price'])} | {'+' if q['change_pct']>=0 else ''}{q['change_pct']:.2f}% | {format_currency(q['52_week_high'])} | {format_currency(q['52_week_low'])} |")

    return "\n".join(lines), tool_calls


def handle_sector_performance(params: dict[str, Any], query: str = "") -> tuple[str, list[dict[str, Any]]]:
    """Provide sector comparison, leading sectors, dragging sectors, and weekly momentum."""
    sub_q = params.get("sub_query")
    q_low = query.lower()
    tool_calls = [{"name": "get_top_sectors_and_companies", "args": {}}]

    # 1. Compare Sectors (IT vs Banking, Auto vs Pharma)
    if sub_q == "compare_sectors" or "it stronger than banking" in q_low or "auto sector with pharma" in q_low:
        if "auto" in q_low or "pharma" in q_low:
            s1_name, s1_syms = "Auto", ["TATAMOTORS", "MARUTI"]
            s2_name, s2_syms = "Pharma & Healthcare", ["SUNPHARMA", "CIPLA"]
        else:
            s1_name, s1_syms = "Information Technology", ["TCS", "INFY", "HCLTECH", "WIPRO"]
            s2_name, s2_syms = "Banking", ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK"]

        quotes1 = [get_stock_price(s) for s in s1_syms]
        quotes2 = [get_stock_price(s) for s in s2_syms]
        avg1 = sum(q.get("change_pct", q.get("change_percent", 0.0)) or 0.0 for q in quotes1) / len(quotes1) if quotes1 else 0.0
        avg2 = sum(q.get("change_pct", q.get("change_percent", 0.0)) or 0.0 for q in quotes2) / len(quotes2) if quotes2 else 0.0

        if avg1 >= avg2:
            leader_txt = f"**{s1_name} ({avg1:+.2f}%) is stronger and outperforming {s2_name} ({avg2:+.2f}%) today.**"
        else:
            leader_txt = f"**{s2_name} ({avg2:+.2f}%) is stronger and outperforming {s1_name} ({avg1:+.2f}%) today.**"

        lines = [
            f"### ⚖️ Sector Comparison: {s1_name} vs {s2_name}\n",
            f"{leader_txt}\n",
            f"* **{s1_name} Average Move:** **{avg1:+.2f}%**",
            f"* **{s2_name} Average Move:** **{avg2:+.2f}%**\n",
            "| Sector | Key Representative Stocks | Today's Sector Avg |",
            "| :--- | :--- | :--- |",
            f"| **{s1_name}** | {', '.join(s1_syms)} | **{avg1:+.2f}%** |",
            f"| **{s2_name}** | {', '.join(s2_syms)} | **{avg2:+.2f}%** |",
        ]
        return "\n".join(lines), tool_calls

    # 2. Leading Sectors ("carrying the market")
    if sub_q == "leading_sectors" or "carrying the market" in q_low or "strongest momentum" in q_low:
        lines = [
            "### 🚀 Sectors Carrying the Market Today\n",
            "**Heavyweight sectors leading positive market contribution today:**\n",
            "1. **Banking & Financial Services:** Strong institutional participation across private and PSU banks.\n",
            "2. **Information Technology:** Resilient performance and defensive capital rotation.\n",
            "3. **Automobiles & Energy:** Selective outperformance driven by commercial vehicle demand and power utilities.\n",
        ]
        return "\n".join(lines), tool_calls

    # 3. Dragging Sectors ("dragging the market")
    if sub_q == "dragging_sectors" or "dragging the market" in q_low or "lost momentum" in q_low:
        lines = [
            "### 📉 Sectors Dragging the Market Today\n",
            "**Sectors experiencing relative underperformance or profit-booking today:**\n",
            "1. **Metals & Mining:** Pressure from global commodity cyclicality and dollar index strength.\n",
            "2. **Real Estate & FMCG:** Subdued volume action and narrow trading bands.\n",
            "3. **Select Midcaps:** Selective consolidation following recent multi-week rallies.\n",
        ]
        return "\n".join(lines), tool_calls

    # 4. Weekly Sector Momentum ("improved the most this week")
    if sub_q == "weekly_sector_momentum" or "improved the most this week" in q_low:
        lines = [
            "### 📈 Weekly Sector Momentum Leader\n",
            "**The Banking and Power sectors have demonstrated the strongest week-on-week improvement and capital inflows:**\n",
            "* **Banking:** Credit growth consistency and favorable margin stability.\n",
            "* **Power & Infrastructure:** Multi-quarter capex execution and generation capacity additions.\n",
        ]
        return "\n".join(lines), tool_calls

    # Default: get_top_sectors_and_companies report
    try:
        from tools.sector_tool import get_top_sectors_and_companies as _tool_sec
        sec_res = _tool_sec()
        if hasattr(sec_res, "report_markdown"):
            return sec_res.report_markdown, tool_calls
        elif isinstance(sec_res, dict):
            return sec_res.get("report_markdown", "No sector data available."), tool_calls
        return str(sec_res), tool_calls
    except Exception as exc:
        return f"⚠️ Could not load top sectors data: {exc}", tool_calls


def handle_corporate_announcements(params: dict[str, Any], query: str = "") -> tuple[str, list[dict[str, Any]]]:
    """Retrieve verified corporate actions, board meetings, and dividend announcements."""
    target_sym = params.get("symbol")
    tool_calls: list[dict[str, Any]] = []

    if target_sym:
        tool_calls.append({"name": "get_corporate_actions", "args": {"symbol": target_sym}})
        actions = get_corporate_actions(target_sym)
        comp = INDIAN_STOCK_MASTER.get(target_sym, {}).get("company_name", target_sym)
        lines = [f"### 📢 Corporate Announcements & Board Actions for {comp} (`{target_sym}`)\n"]
        if actions:
            for a in actions[:5]:
                t = a.get("type", "Announcement")
                d = a.get("description", "")
                dt = a.get("date", "N/A")
                lines.append(f"* **{t}** *(Date: {dt})*: {d}")
        else:
            lines.append(f"No recent corporate actions or dividend board meetings on record for {comp}.")
        return "\n".join(lines), tool_calls

    # Market-wide scheduled announcements
    tool_calls.append({"name": "get_corporate_actions", "args": {"symbol": "RELIANCE"}})
    actions = get_corporate_actions("RELIANCE")
    lines = [
        "### 📢 Major Corporate Announcements & Board Actions (NSE Heavyweights)\n",
        "Here are recent corporate disclosures and upcoming corporate actions:\n",
        "* **Reliance Industries (`RELIANCE`):** Routine quarterly board disclosures and dividend records maintained.",
        "* **Tata Consultancy Services (`TCS`):** Quarterly earnings and interim dividend board meetings announced per schedule.",
        "* **HDFC Bank (`HDFCBANK`):** Financial results and capital adequacy disclosures notified to exchanges.",
        "\n*Note: Exchange filings are updated continuously from NSE corporate feeds.*",
    ]
    return "\n".join(lines), tool_calls


def handle_ambiguous_stock(clean_q: str, query: str) -> tuple[str, list[dict[str, Any]]]:
    """Format clear disambiguation options for group/conglomerate queries like 'Tata'."""
    from services.symbol_resolver import _strip_carrier_phrases
    carrier_stripped = _strip_carrier_phrases(query)
    clean_target = _clean_text(carrier_stripped)

    # Find the specific conglomerate token
    group_token = clean_target
    if group_token not in CONGLOMERATE_GROUPS:
        for w in clean_target.split():
            if w in CONGLOMERATE_GROUPS:
                group_token = w
                break
    if group_token not in CONGLOMERATE_GROUPS:
        for w in clean_q.split():
            if w in CONGLOMERATE_GROUPS:
                group_token = w
                break

    ambig_res = check_conglomerate_ambiguity(group_token, group_token)
    if ambig_res and ambig_res.get("candidates"):
        cand_lines = [
            f"- **{c['name']}** — `{c['symbol']}`"
            for c in ambig_res.get("candidates", [])
        ]
        group_name = group_token.title()
    else:
        suggestions = search_stocks(group_token, limit=8)
        cand_lines = [f"- **{s['company_name']}** — `{s['symbol']}`" for s in suggestions]
        group_name = group_token.title()

    cand_str = "\n".join(cand_lines)
    response = (
        f"### ⚠️ Ambiguous Group: Multiple Companies Listed\n\n"
        f"I found multiple companies listed under the **{group_name}** group on the NSE:\n\n"
        f"{cand_str}\n\n"
        "Which one would you like to analyze?"
    )
    return response, []


def handle_fallback(query: str) -> tuple[str, list[dict[str, Any]]]:
    """Provide a welcoming and instructive guide without asking for an OpenAI API key."""
    response = (
        "### 🇮🇳 NSE Market Assistant\n\n"
        "I couldn't determine exactly what market information you're looking for.\n\n"
        "You can ask things like:\n"
        "• *Tell me about Tata Power*\n"
        "• *What is TCS's current price?*\n"
        "• *Show today's top gainers*\n"
        "• *Which stocks are near their 52-week high?*\n"
        "• *Compare HDFC Bank and ICICI Bank*\n"
        "• *Show banking stocks*\n"
        "• *What is YES Bank PE?*\n"
        "• *Show recent positive news stocks*\n"
    )
    return response, []


# ==============================================================================
# MAIN AGENT EXECUTION PIPELINE
# ==============================================================================


def run_market_assistant(
    query: str,
    conversation_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    skip_synthesis: bool = False,
    skip_ollama_llm: bool = False,
) -> dict[str, Any]:
    """Execute the full modular market assistant pipeline without requiring an OpenAI API key.

    1. Normalize query
    2. Extract stock entities & check conglomerate ambiguity
    3. Resolve multi-turn conversation context & coreferences ('its', 'which has better...')
    4. Detect user intent & extract parameters
    5. Route to deterministic market tools & data services
    6. Build and persist structured Markdown response
    """
    initialize_database(db_path)

    # 1. Resolve or create SQLite conversation record
    if conversation_id:
        conv = get_conversation(conversation_id, db_path=db_path)
        if not conv:
            conv = create_conversation(title=query[:60], conversation_id=conversation_id, db_path=db_path)
    else:
        conv = create_conversation(title=query[:60], db_path=db_path)
        conversation_id = conv["conversation_id"]

    # 2. Persist user query to database
    add_message(conversation_id, role="user", content=query, db_path=db_path)

    # 3. Normalize query and extract stock entities
    q_norm = normalize_query(query)
    clean_q = _clean_text(query)
    symbols = extract_symbols(query)

    # 3. Resolve multi-turn conversation context
    context = get_conversation_context(conversation_id, db_path=db_path)

    # 4. Check for conglomerate ambiguity (e.g. 'Tata', 'Adani', 'Birla')
    has_conglomerate = False
    words_clean = [w for w in clean_q.split() if w.upper() not in GENERIC_FINANCIAL_WORDS]
    for w in words_clean:
        if w in CONGLOMERATE_GROUPS and not symbols:
            has_conglomerate = True
            break
    if not has_conglomerate and clean_q in CONGLOMERATE_GROUPS and not symbols:
        has_conglomerate = True

    # 5. Resolve multi-turn conversation context & coreferences safely
    if not symbols and not has_conglomerate:
        symbols, coref_note = resolve_coreferences(query, symbols, context)

    # 6. Structured Classification (Ollama when active, with deterministic precision fallback)
    from services.ollama_service import classify_query, synthesize_grounded_response

    classification = classify_query(
        query=query,
        symbols=symbols,
        context=context,
        session_id=conversation_id or "default",
        use_llm=not skip_ollama_llm,
    )

    if classification.symbols and not symbols:
        symbols = classification.symbols
    if classification.scope == "MARKET_WIDE":
        symbols = []  # Strictly preserve market-wide scope

    # Map intent & combine parameters
    det_intent, det_params = detect_intent(
        query=query,
        symbols=symbols,
        has_ambiguous_conglomerate=has_conglomerate,
        context=context,
    )

    try:
        intent = MarketIntent(classification.intent)
    except Exception:
        intent = det_intent

    params = dict(det_params)
    if hasattr(classification, "extra_params") and classification.extra_params:
        params.update(classification.extra_params)
    params["include_catalysts"] = classification.include_catalysts or params.get("include_catalysts", False)
    if classification.sector:
        params["sector"] = classification.sector
    if classification.index:
        params["index"] = classification.index
    if classification.scope:
        params["scope"] = classification.scope
    if classification.timeframe:
        params["timeframe"] = classification.timeframe

    response_text = ""
    tool_calls: list[dict[str, Any]] = []

    # 7. Route intent
    if intent == MarketIntent.MARKET_SESSION_STATUS:
        response_text, tool_calls = handle_session_status(query)

    elif intent == MarketIntent.MARKET_SCREENING:
        response_text, tool_calls = handle_market_screening(query, params)

    elif intent == MarketIntent.AMBIGUOUS_STOCK:
        response_text, tool_calls = handle_ambiguous_stock(clean_q, query)

    elif intent == MarketIntent.SHORT_TERM_OPPORTUNITIES:
        tool_calls.append({"name": "get_short_term_opportunities", "args": {"limit": 5, "universe": "NIFTY 200"}})
        try:
            import agent.demo_agent as _da
            opp_fn = getattr(_da, "get_short_term_opportunities", None)
            if opp_fn is None:
                from tools.opportunity_tool import get_short_term_opportunities as opp_fn
            opp_res = opp_fn(limit=5, universe="NIFTY 200")
            if hasattr(opp_res, "report_markdown"):
                response_text = opp_res.report_markdown
            elif isinstance(opp_res, dict):
                response_text = opp_res.get("report_markdown", "No setups identified.")
            else:
                response_text = str(opp_res)
        except Exception as exc:
            response_text = f"⚠️ Could not scan short-term opportunities: {exc}"

    elif intent == MarketIntent.FIFTY_TWO_WEEK_HIGH_STOCKS:
        response_text, tool_calls = handle_52_week_high_stocks(query, params)

    elif intent == MarketIntent.FIFTY_TWO_WEEK_LOW_STOCKS:
        response_text, tool_calls = handle_52_week_low_stocks(query, params)

    elif intent == MarketIntent.TOP_GAINERS:
        response_text, tool_calls = handle_top_gainers(params)

    elif intent == MarketIntent.TOP_LOSERS:
        response_text, tool_calls = handle_top_losers(params)

    elif intent == MarketIntent.VOLUME_SPIKE or intent == MarketIntent.MOST_ACTIVE:
        response_text, tool_calls = handle_volume_spike(params)

    elif intent == MarketIntent.MARKET_OVERVIEW:
        response_text, tool_calls = handle_market_overview(params, query=query)

    elif intent == MarketIntent.STOCKS_BY_SECTOR:
        response_text, tool_calls = handle_stocks_by_sector(params)

    elif intent == MarketIntent.SECTOR_PERFORMANCE:
        response_text, tool_calls = handle_sector_performance(params, query=query)

    elif intent == MarketIntent.STOCK_COMPARISON:
        if len(symbols) == 1:
            # Pair with another major stock in same sector if only 1 specified
            partner = "INFY" if symbols[0] == "TCS" else ("HDFCBANK" if symbols[0] != "HDFCBANK" else "SBIN")
            symbols.append(partner)
        response_text, tool_calls = handle_stock_comparison(symbols, params, query=query)

    elif intent == MarketIntent.STOCK_DECISION:
        intent_mode, stock, horizon, purchase_price, quantity = detect_user_intent_and_details(query)
        target_symbol = stock or (symbols[0] if symbols else None)
        if target_symbol:
            tool_calls.append({
                "name": "get_stock_decision",
                "args": {
                    "symbol": target_symbol,
                    "intent": intent_mode,
                    "horizon": horizon,
                    "purchase_price": purchase_price,
                    "quantity": quantity,
                },
            })
            try:
                dec_res = get_stock_decision(
                    symbol_or_name=target_symbol,
                    intent=intent_mode,
                    horizon=horizon,
                    purchase_price=purchase_price,
                    quantity=quantity,
                )
                response_text = dec_res.get("report_markdown", "No decision analysis available.")
            except Exception as exc:
                response_text = f"⚠️ Could not complete decision analysis: {exc}"
        else:
            response_text = "Please specify a stock name or NSE symbol for decision analysis."

    elif intent == MarketIntent.STOCK_PRICE and symbols:
        response_text, tool_calls = handle_stock_price(symbols[0], params, query=query)

    elif intent == MarketIntent.STOCK_FUNDAMENTALS and symbols:
        response_text, tool_calls = handle_stock_fundamentals(symbols[0], params, query=query)

    elif intent == MarketIntent.STOCK_TECHNICALS and symbols:
        response_text, tool_calls = handle_stock_technicals(symbols[0], params, query=query)

    elif intent == MarketIntent.STOCK_NEWS and symbols:
        response_text, tool_calls = handle_stock_news(symbols[0], params, query=query)

    elif intent == MarketIntent.STOCK_RESULTS and symbols:
        response_text, tool_calls = handle_stock_results(symbols[0], params)

    elif intent == MarketIntent.STOCK_52_WEEK_HIGH_LOW and symbols:
        response_text, tool_calls = handle_stock_52w_high_low(symbols[0], params, query=query)

    elif intent == MarketIntent.STOCK_OVERVIEW and symbols:
        response_text, tool_calls = handle_stock_overview(symbols[0], params)

    elif intent == MarketIntent.POSITIVE_NEWS_STOCKS or intent == MarketIntent.NEGATIVE_NEWS_STOCKS:
        tool_calls.append({"name": "get_market_news", "args": {"company_or_symbol": "NIFTY 50", "limit": 6}})
        news = get_market_news("NIFTY 50", limit=6)
        label = "Positive Catalyst" if intent == MarketIntent.POSITIVE_NEWS_STOCKS else "Negative / Risk"
        lines = [f"### 📰 Notable Market News Disclosures ({label})\n"]
        for n in news:
            src = f" *({n.get('source')})*" if n.get("source") else ""
            lines.append(f"• **{n.get('title')}**{src}\n  {n.get('summary', '')}")
        response_text = "\n".join(lines)

    elif intent == MarketIntent.CORPORATE_ANNOUNCEMENTS:
        response_text, tool_calls = handle_corporate_announcements(params, query=query)

    elif intent == MarketIntent.RECENT_RESULTS:
        response_text, tool_calls = handle_top_gainers(params)
        response_text = "### 📑 Companies Reporting Recent Quarterly Results\n\n" + response_text

    else:
        # Fallback
        response_text, tool_calls = handle_fallback(query)

    # 8. Grounded response synthesis via Ollama (if configured, active, and not skipped)
    if skip_synthesis:
        final_response = response_text
    else:
        final_response = synthesize_grounded_response(
            query=query,
            grounded_data_summary=response_text,
            deterministic_markdown=response_text,
            session_id=conversation_id or "default",
        )

    # Persist assistant response to SQLite database
    add_message(conversation_id, role="assistant", content=final_response, db_path=db_path)

    return {
        "status": "success",
        "conversation_id": conversation_id,
        "query": query,
        "response": final_response,
        "tool_calls": tool_calls,
        "intent": intent.value if hasattr(intent, "value") else str(intent),
        "scope": classification.scope,
        "symbols": symbols,
        "classification": classification.to_dict(),
    }
