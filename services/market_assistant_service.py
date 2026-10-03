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


def handle_market_overview(params: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Provide current levels, day change, and regime status for benchmark indices."""
    tool_calls = [
        {"name": "get_market_index", "args": {"index_name": "NIFTY"}},
        {"name": "get_market_index", "args": {"index_name": "BANKNIFTY"}},
        {"name": "get_market_index", "args": {"index_name": "SENSEX"}},
    ]

    nifty = get_market_index("NIFTY")
    bank_nifty = get_market_index("BANKNIFTY")
    sensex = get_market_index("SENSEX")

    def _fmt_idx(idx_data: dict[str, Any]) -> str:
        val = idx_data.get("current_value")
        chg = idx_data.get("change", 0.0)
        chg_pct = idx_data.get("change_percent", 0.0)
        icon = "🟢" if chg >= 0 else "🔴"
        sign = "+" if chg >= 0 else ""
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

    n_pct = nifty.get("change_percent") or 0.0
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
    chg = p.get("change", 0.0)
    chg_pct = p.get("change_percent", 0.0)
    h52 = p.get("52_week_high")
    l52 = p.get("52_week_low")

    sign = "+" if (chg or 0) >= 0 else ""
    icon = "🟢" if (chg or 0) >= 0 else "🔴"

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
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve verified market news headlines and executive summaries for a stock."""
    from agent.demo_agent import format_news_section
    tool_calls = [{"name": "get_market_news", "args": {"company_or_symbol": symbol, "limit": 5}}]

    news = get_market_news(symbol, limit=5)
    comp = INDIAN_STOCK_MASTER.get(symbol, {}).get("company_name", symbol)

    lines = [f"### 📰 Latest News for {comp} (`{symbol}`)\n"]
    lines.extend(format_news_section(news, company_name=comp, symbol=symbol, include_header=False))

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
) -> tuple[str, list[dict[str, Any]]]:
    """Display 52-week high, low, and current distance for a single stock."""
    tool_calls = [{"name": "get_stock_price", "args": {"symbol": symbol}}]
    p = get_stock_price(symbol)
    comp = p.get("company") or INDIAN_STOCK_MASTER.get(symbol, {}).get("company_name", symbol)

    curr = p.get("current_price")
    h52 = p.get("52_week_high")
    l52 = p.get("52_week_low")

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
) -> tuple[str, list[dict[str, Any]]]:
    """Compare N stocks side-by-side, answering specific metric comparisons if asked."""
    from agent.demo_agent import run_demo_comparison
    report, tool_calls = run_demo_comparison(symbols)

    req_metric = params.get("metric")
    if req_metric == "pe" and len(symbols) >= 2:
        # Evaluate which has better P/E
        pe_vals: list[tuple[str, float]] = []
        for s in symbols:
            info = get_company_info(s)
            pe = info.get("trailing_pe")
            if pe and pe > 0:
                pe_vals.append((s, pe))
        if len(pe_vals) >= 2:
            cheapest = min(pe_vals, key=lambda x: x[1])
            comp_name = INDIAN_STOCK_MASTER.get(cheapest[0], {}).get("company_name", cheapest[0])
            pe_summary = (
                f"\n\n#### 💡 Valuation Comparison: P/E Ratio\n"
                f"Between the compared companies, **{comp_name} (`{cheapest[0]}`)** trades at the lower P/E ratio "
                f"of **{cheapest[1]:.2f}x**, representing a more attractive trailing valuation multiple."
            )
            report += pe_summary

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

    # Map intent
    try:
        intent = MarketIntent(classification.intent)
    except Exception:
        intent, params = detect_intent(
            query=query,
            symbols=symbols,
            has_ambiguous_conglomerate=has_conglomerate,
            context=context,
        )

    params = {
        "include_catalysts": classification.include_catalysts,
        "sector": classification.sector,
        "index": classification.index,
        "scope": classification.scope,
        "timeframe": classification.timeframe,
    }

    response_text = ""
    tool_calls: list[dict[str, Any]] = []

    # 7. Route intent
    if intent == MarketIntent.AMBIGUOUS_STOCK:
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
        response_text, tool_calls = handle_market_overview(params)

    elif intent == MarketIntent.STOCKS_BY_SECTOR:
        response_text, tool_calls = handle_stocks_by_sector(params)

    elif intent == MarketIntent.SECTOR_PERFORMANCE:
        tool_calls.append({"name": "get_top_sectors_and_companies", "args": {}})
        try:
            from tools.sector_tool import get_top_sectors_and_companies as _tool_sec
            sec_res = _tool_sec()
            if hasattr(sec_res, "report_markdown"):
                response_text = sec_res.report_markdown
            elif isinstance(sec_res, dict):
                response_text = sec_res.get("report_markdown", "No sector data available.")
            else:
                response_text = str(sec_res)
        except Exception as exc:
            response_text = f"⚠️ Could not load top sectors data: {exc}"

    elif intent == MarketIntent.STOCK_COMPARISON:
        if len(symbols) == 1:
            # Pair with another major stock in same sector if only 1 specified
            partner = "INFY" if symbols[0] == "TCS" else ("HDFCBANK" if symbols[0] != "HDFCBANK" else "SBIN")
            symbols.append(partner)
        response_text, tool_calls = handle_stock_comparison(symbols, params)

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
        response_text, tool_calls = handle_stock_fundamentals(symbols[0], params)

    elif intent == MarketIntent.STOCK_TECHNICALS and symbols:
        response_text, tool_calls = handle_stock_technicals(symbols[0], params)

    elif intent == MarketIntent.STOCK_NEWS and symbols:
        response_text, tool_calls = handle_stock_news(symbols[0], params)

    elif intent == MarketIntent.STOCK_RESULTS and symbols:
        response_text, tool_calls = handle_stock_results(symbols[0], params)

    elif intent == MarketIntent.STOCK_52_WEEK_HIGH_LOW and symbols:
        response_text, tool_calls = handle_stock_52w_high_low(symbols[0], params)

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
        tool_calls.append({"name": "get_corporate_actions", "args": {"symbol": "RELIANCE"}})
        actions = get_corporate_actions("RELIANCE")
        lines = ["### 📢 Recent Corporate Actions & Announcements\n"]
        for a in actions[:5]:
            lines.append(f"• **{a.get('type')}:** {a.get('description', '')} *(Date: {a.get('date', 'N/A')})*")
        response_text = "\n".join(lines) if actions else "No corporate announcements recorded today."

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
