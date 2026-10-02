"""Orchestrating comprehensive company analysis tool for the NSE AI Agent."""

from __future__ import annotations

import logging
from typing import Any, Optional

from tools.company_tool import get_company_info
from tools.financials_tool import (
    calculate_financial_health,
    evaluate_pe_valuation,
    evaluate_roce,
    evaluate_roe,
    get_cash_flow,
    get_debt_metrics,
    get_quarterly_financials,
    is_financial_company,
)
from tools.governance_tool import (
    get_board_meetings,
    get_corporate_actions,
    get_shareholding_pattern,
)
from tools.news_tool import get_market_news
from tools.sentiment_tool import get_stock_sentiment
from tools.stock_tool import get_stock_price
from utils.helpers import normalize_nse_symbol

logger = logging.getLogger(__name__)


def format_currency_inr(value: Any) -> str:
    """Format numeric values into standard Indian rupee representations (Crores or Rupees)."""
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 10_000_000:
        cr = abs_val / 10_000_000
        return f"{sign}₹{cr:,.2f} Cr"
    elif abs_val >= 100_000:
        lakh = abs_val / 100_000
        return f"{sign}₹{lakh:,.2f} Lakh"
    return f"{sign}₹{value:,.2f}"


def get_company_analysis(symbol: str) -> dict[str, Any]:
    """Execute a comprehensive fundamental, technical, financial, and governance analysis for an NSE company.

    Integrates real-time price data, 4-quarter financials, cash flow conversion, debt metrics,
    deterministic financial health scoring, shareholding patterns, board meetings, corporate actions,
    and verified market news into an authoritative, structured report.

    Args:
        symbol: NSE stock symbol (e.g. 'TCS', 'INFY', 'RELIANCE', 'HDFCBANK').

    Returns:
        Dictionary containing:
            - symbol: Clean NSE ticker symbol
            - company: Company legal name
            - report_markdown: Formatted comprehensive Markdown report
            - raw_data: Dictionary of underlying tool outputs
            - financial_health: Structured health score and factors
            - limitations: List of unavailable or unparseable fields
    """
    limitations: list[str] = []

    try:
        from services.symbol_resolver import resolve_nse_symbol
        res = resolve_nse_symbol(symbol)
        if res.get("symbol"):
            symbol = res["symbol"]
    except Exception:
        pass

    try:
        clean_symbol = normalize_nse_symbol(symbol, target_format="clean")
    except Exception as exc:
        logger.warning("Symbol normalization failed for '%s': %s", symbol, exc)
        clean_symbol = str(symbol).strip().upper()

    raw_data: dict[str, Any] = {}

    # 1. Stock Price
    try:
        price_data = get_stock_price(clean_symbol)
        raw_data["stock_price"] = price_data
    except Exception as exc:
        logger.warning("Price retrieval failed for '%s': %s", clean_symbol, exc)
        price_data = {"symbol": clean_symbol, "current_price": None}
        limitations.append("Live stock price could not be retrieved from exchange feeds.")

    # 2. Company Info
    try:
        comp_info = get_company_info(clean_symbol)
        raw_data["company_info"] = comp_info
    except Exception as exc:
        logger.warning("Company info retrieval failed for '%s': %s", clean_symbol, exc)
        comp_info = {"symbol": clean_symbol, "company": clean_symbol}
        limitations.append("Company fundamental profile could not be fully loaded.")

    comp_name = comp_info.get("company") or price_data.get("company") or clean_symbol
    sector = comp_info.get("sector")
    industry = comp_info.get("industry")
    is_fin = is_financial_company(sector, industry)

    # 3. Technical Sentiment
    try:
        sentiment_data = get_stock_sentiment(clean_symbol)
        raw_data["sentiment"] = sentiment_data
    except Exception as exc:
        logger.warning("Sentiment analysis failed for '%s': %s", clean_symbol, exc)
        sentiment_data = {"sentiment": "Neutral", "score": 0.0, "signals": []}
        limitations.append("Technical sentiment analysis unavailable.")

    # 4. Quarterly Financials
    try:
        fin_data = get_quarterly_financials(clean_symbol)
        raw_data["quarterly_financials"] = fin_data
    except Exception as exc:
        logger.warning("Quarterly financials failed for '%s': %s", clean_symbol, exc)
        fin_data = {"quarters": []}
        limitations.append("Quarterly reported financials unavailable.")

    # 5. Cash Flow
    try:
        cf_data = get_cash_flow(clean_symbol)
        raw_data["cash_flow"] = cf_data
    except Exception as exc:
        logger.warning("Cash flow retrieval failed for '%s': %s", clean_symbol, exc)
        cf_data = {"quarterly_reported": False, "annual_ocf_history": []}
        limitations.append("Cash flow statements unavailable.")

    # 6. Debt Metrics
    try:
        debt_data = get_debt_metrics(clean_symbol)
        raw_data["debt_metrics"] = debt_data
    except Exception as exc:
        logger.warning("Debt metrics retrieval failed for '%s': %s", clean_symbol, exc)
        debt_data = {"is_financial": is_fin, "assessment": "Debt metrics unavailable."}
        limitations.append("Debt and balance sheet metrics unavailable.")

    # 7. Shareholding Pattern
    try:
        shareholding_data = get_shareholding_pattern(clean_symbol)
        raw_data["shareholding"] = shareholding_data
    except Exception as exc:
        logger.warning("Shareholding pattern retrieval failed for '%s': %s", clean_symbol, exc)
        shareholding_data = {"promoter_names": [], "as_of_date": "Unavailable"}
        limitations.append("Shareholding pattern filings unavailable.")

    # 8. Board Meetings
    try:
        meetings_data = get_board_meetings(clean_symbol)
        raw_data["board_meetings"] = meetings_data
    except Exception as exc:
        logger.warning("Board meetings retrieval failed for '%s': %s", clean_symbol, exc)
        meetings_data = []
        limitations.append("Recent board meeting announcements unavailable.")

    # 9. Corporate Actions
    try:
        actions_data = get_corporate_actions(clean_symbol)
        raw_data["corporate_actions"] = actions_data
    except Exception as exc:
        logger.warning("Corporate actions retrieval failed for '%s': %s", clean_symbol, exc)
        actions_data = []
        limitations.append("Historical corporate actions unavailable.")

    # 10. News Stories
    try:
        news_data = get_market_news(clean_symbol, limit=5)
        raw_data["news"] = news_data
    except Exception as exc:
        logger.warning("News retrieval failed for '%s': %s", clean_symbol, exc)
        news_data = []
        limitations.append("Recent market news stories temporarily unavailable.")

    # Contextual Evaluations
    trailing_pe = comp_info.get("trailing_pe")
    forward_pe = comp_info.get("forward_pe")
    yoy_profit_growth = fin_data.get("yoy_profit_growth")
    qoq_profit_growth = fin_data.get("qoq_profit_growth")
    profit_growth = yoy_profit_growth if yoy_profit_growth is not None else qoq_profit_growth

    pe_eval = evaluate_pe_valuation(
        trailing_pe,
        sector=sector,
        earnings_growth=profit_growth,
        forward_pe=forward_pe,
    )

    roce_val = comp_info.get("roce")
    roce_eval = evaluate_roce(roce_val, is_financial=is_fin)

    roe_val = comp_info.get("roe")
    debt_to_eq = debt_data.get("debt_to_equity")
    roe_eval = evaluate_roe(roe_val, debt_to_equity=debt_to_eq, is_financial=is_fin)

    # Financial Health Scoring
    health_data = calculate_financial_health(
        qoq_revenue_growth=fin_data.get("qoq_revenue_growth"),
        yoy_revenue_growth=fin_data.get("yoy_revenue_growth"),
        qoq_profit_growth=fin_data.get("qoq_profit_growth"),
        yoy_profit_growth=fin_data.get("yoy_profit_growth"),
        operating_margin=fin_data["quarters"][-1].get("operating_margin") if fin_data.get("quarters") else None,
        margin_trend=fin_data.get("margin_trend"),
        cash_conversion_ratio=cf_data.get("cash_conversion_ratio"),
        net_debt=debt_data.get("net_debt"),
        debt_to_equity=debt_to_eq,
        roce=roce_val,
        roe=roe_val,
        is_financial=is_fin,
    )

    # BUILD MARKDOWN REPORT
    lines: list[str] = [
        f"# {comp_name} ({clean_symbol})\n",
        "## Snapshot\n",
    ]

    # Section A: Company Overview / Snapshot
    cur_p = price_data.get("current_price")
    chg = price_data.get("change")
    chg_pct = price_data.get("change_percent")
    p_str = f"₹{cur_p:,.2f}" if cur_p is not None else "N/A"
    if chg is not None and chg_pct is not None:
        p_str += f" ({chg:+.2f}, {chg_pct:+.2f}%)"

    lines.append(f"* **Current Price:** {p_str}")
    lines.append(f"* **Market Cap:** {format_currency_inr(comp_info.get('market_cap'))}")
    h52 = comp_info.get("52_week_high") or price_data.get("52_week_high")
    l52 = comp_info.get("52_week_low") or price_data.get("52_week_low")
    lines.append(f"* **52-Week High:** {f'₹{h52:,.2f}' if h52 else 'N/A'}")
    lines.append(f"* **52-Week Low:** {f'₹{l52:,.2f}' if l52 else 'N/A'}")
    lines.append(f"* **Sector:** {sector or 'N/A'} | **Industry:** {industry or 'N/A'}")
    bv = comp_info.get("book_value")
    lines.append(f"* **Book Value Per Share:** {f'₹{bv:,.2f}' if bv else 'N/A'}")
    div_y = comp_info.get("dividend_yield")
    lines.append(f"* **Dividend Yield:** {f'{div_y * 100:.2f}%' if div_y and div_y < 1.0 else (f'{div_y:.2f}%' if div_y else 'N/A')}\n")

    # Section B, C, D: Valuation & Quality
    lines.append("## Valuation & Quality\n")
    pe_str = f"{trailing_pe:.1f}x" if trailing_pe else "N/A"
    lines.append(f"* **Stock P/E:** {pe_str} | **Assessment:** **{pe_eval['assessment']}**")
    lines.append(f"  *Reason:* {pe_eval['reason']}")

    if is_fin:
        lines.append(f"* **ROCE:** Not Applicable *(Financial Institution — customer deposits/debt form core operational capital)*")
    else:
        roce_str = f"{roce_val:.1f}%" if roce_val is not None else "N/A"
        lines.append(f"* **ROCE:** {roce_str} | **Assessment:** **{roce_eval['assessment']}**")
        lines.append(f"  *Reason:* {roce_eval['reason']}")

    roe_str = f"{roe_val:.1f}%" if roe_val is not None else "N/A"
    lines.append(f"* **ROE:** {roe_str} | **Assessment:** **{roe_eval['assessment']}**")
    lines.append(f"  *Reason:* {roe_eval['reason']}\n")

    # Section E: Technical Sentiment
    lines.append("## Technical Sentiment\n")
    sentiment_label = sentiment_data.get("sentiment", "Neutral")
    sent_score = sentiment_data.get("score", 0.0)
    lines.append(f"* **Technical Sentiment:** **{sentiment_label}** (Score: {sent_score:+.1f})")
    signals = sentiment_data.get("signals", [])
    if signals:
        lines.append("* **Reasons:**")
        for sig in signals:
            if isinstance(sig, dict):
                s_name = sig.get("name", "Signal")
                s_val = sig.get("value")
                s_unit = sig.get("unit", "")
                s_status = sig.get("status", "")
                if s_val is not None:
                    if s_name == "price vs 20 DMA":
                        direction = "above" if s_val > 0 else "below"
                        lines.append(f"  - Price is {direction} 20 DMA by {abs(s_val):.2f}%")
                    elif s_name == "price vs 50 DMA":
                        direction = "above" if s_val > 0 else "below"
                        lines.append(f"  - Price is {direction} 50 DMA by {abs(s_val):.2f}%")
                    elif s_name == "1-day return":
                        lines.append(f"  - 1-day return is {s_val:+.2f}% ({s_status})")
                    elif s_name == "5-day return":
                        lines.append(f"  - 5-day return is {s_val:+.2f}% ({s_status})")
                    elif "dma" in s_name.lower():
                        lines.append(f"  - {s_name}: ₹{s_val:,.2f}")
                else:
                    lines.append(f"  - {s_name}: Insufficient historical data")
            else:
                lines.append(f"  - {sig}")
    lines.append("* *Note: Technical sentiment reflects mathematical historical price/volume momentum and is not a prediction of future returns.*\n")

    # Section F: Last Four Quarters Financial Performance
    lines.append("## Last 4 Quarters Financial Performance\n")
    quarters = fin_data.get("quarters", [])
    if quarters:
        lines.append("| Quarter | Revenue / Sales | Expenses | Operating Profit | OPM (%) | Net Profit | EPS |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for q in quarters:
            q_date = q.get("date", "N/A")
            q_rev = format_currency_inr(q.get("revenue"))
            q_exp = format_currency_inr(q.get("expenses"))
            q_op = format_currency_inr(q.get("operating_profit"))
            q_opm = f"{q.get('operating_margin'):.2f}%" if q.get("operating_margin") is not None else "N/A"
            q_np = format_currency_inr(q.get("net_profit"))
            q_eps = f"₹{q.get('eps'):.2f}" if q.get("eps") is not None else "N/A"
            lines.append(f"| **{q_date}** | {q_rev} | {q_exp} | {q_op} | {q_opm} | {q_np} | {q_eps} |")

        lines.append("")
        qoq_r = fin_data.get("qoq_revenue_growth")
        yoy_r = fin_data.get("yoy_revenue_growth")
        qoq_p = fin_data.get("qoq_profit_growth")
        yoy_p = fin_data.get("yoy_profit_growth")
        lines.append(f"* **Revenue Trend:** QoQ: {f'{qoq_r:+.2f}%' if qoq_r is not None else 'N/A'} | YoY: {f'{yoy_r:+.2f}%' if yoy_r is not None else 'Comparable prior quarter not reported'}")
        lines.append(f"* **Profit Trend:** QoQ: {f'{qoq_p:+.2f}%' if qoq_p is not None else 'N/A'} | YoY: {f'{yoy_p:+.2f}%' if yoy_p is not None else 'Comparable prior quarter not reported'}")
        lines.append(f"* **Margin Trend:** {fin_data.get('margin_trend', 'Stable')}\n")
    else:
        lines.append("*Quarterly financial statements could not be loaded from public filings.*\n")

    # Section G & H: Cash Flow & Debt
    lines.append("## Cash Flow & Debt\n")
    annual_ocf = cf_data.get("annual_ocf_history", [])
    if annual_ocf:
        ocf_strs = [f"{item['period']}: {format_currency_inr(item['ocf'])}" for item in annual_ocf[:2]]
        lines.append(f"* **Operating Cash Flow (Annual):** {', '.join(ocf_strs)}")
    if not cf_data.get("quarterly_reported"):
        lines.append("* **Latest quarterly OCF:** Not separately reported by company in quarterly results.")
    lines.append(f"* **Cash Conversion:** {cf_data.get('cash_conversion_assessment')}")

    if is_fin:
        lines.append(f"* **Debt & Capital:** {debt_data.get('assessment')}\n")
    else:
        tot_d = debt_data.get("total_debt")
        cash_val = debt_data.get("cash_and_equivalents")
        net_d = debt_data.get("net_debt")
        lines.append(f"* **Total Debt:** {format_currency_inr(tot_d)}")
        lines.append(f"* **Cash & Equivalents:** {format_currency_inr(cash_val)}")
        lines.append(f"* **Net Debt:** {format_currency_inr(net_d)} ({debt_data.get('assessment')})")
        d_to_e_val = debt_data.get("debt_to_equity")
        lines.append(f"* **Debt / Equity:** {f'{d_to_e_val:.2f}x' if d_to_e_val is not None else 'N/A'}\n")

    # Section I: Financial Health
    lines.append("## Financial Health\n")
    lines.append(f"* **Rating:** **{health_data['rating']}** | **Score:** **{health_data['score']} / 10**")
    if health_data["positive_factors"]:
        lines.append("* **Positive Factors:**")
        for pos in health_data["positive_factors"]:
            lines.append(f"  + {pos}")
    if health_data["concerns"]:
        lines.append("* **Concerns / Risks:**")
        for con in health_data["concerns"]:
            lines.append(f"  - {con}")
    lines.append("")

    # Section J to P: Shareholding Pattern
    lines.append("## Shareholding Pattern\n")
    lines.append(f"* **As of:** {shareholding_data.get('as_of_date', 'Latest Filing')}")
    prom_p = shareholding_data.get("promoters_percent")
    lines.append(f"* **Promoters:** {f'{prom_p:.2f}%' if prom_p is not None else 'N/A'}")
    fii_p = shareholding_data.get("fii_percent")
    lines.append(f"* **FII / FPI:** {f'{fii_p:.2f}%' if fii_p is not None else 'N/A'}")
    dii_p = shareholding_data.get("dii_percent")
    lines.append(f"* **DII:** {f'{dii_p:.2f}%' if dii_p is not None else 'N/A'}")
    govt_p = shareholding_data.get("government_percent", 0.0)
    lines.append(f"* **Government:** {f'{govt_p:.2f}%' if govt_p else 'No government holding reported'}")
    pub_p = shareholding_data.get("public_percent")
    lines.append(f"* **Public / Others:** {f'{pub_p:.2f}%' if pub_p is not None else 'N/A'}")

    prom_names = shareholding_data.get("promoter_names", [])
    if prom_names:
        lines.append("\n### Major Disclosed Promoter Entities")
        lines.append("| Entity / Holder | Holding (%) |")
        lines.append("| :--- | :--- |")
        for p_entity in prom_names:
            h_p = p_entity.get("holding_percent")
            lines.append(f"| {p_entity.get('name')} | {f'{h_p:.2f}%' if h_p is not None else 'Disclosed in filings'} |")

    lines.append(f"\n* **Trend:** {shareholding_data.get('trend', 'Stable')}\n")

    # Section Q: Recent Board Meetings
    lines.append("## Recent Board Meetings\n")
    if meetings_data:
        for idx, m in enumerate(meetings_data, start=1):
            lines.append(f"{idx}. **{m.get('date', 'Date Not Disclosed')}**")
            lines.append(f"   * **Purpose:** {m.get('purpose')}")
            lines.append(f"   * **Status / Outcome:** {m.get('outcome')}")
            if m.get("url"):
                lines.append(f"   * 🔗 [Filing Source ↗]({m.get('url')})")
        lines.append("")
    else:
        lines.append("*No recent board meeting notices retrieved.*\n")

    # Section R: Last 5 Corporate Actions
    lines.append("## Last 5 Corporate Actions\n")
    if actions_data:
        lines.append("| Date | Action | Details | Source |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for act in actions_data[:5]:
            d_url = act.get("url", "https://www.nseindia.com")
            lines.append(f"| {act.get('date')} | **{act.get('action')}** | {act.get('details')} | [NSE Link ↗]({d_url}) |")
        lines.append("")
    else:
        lines.append("*No recent corporate actions recorded.*\n")

    # Section S: Recent Verified News
    lines.append("## Recent Verified News\n")
    if news_data:
        for idx, item in enumerate(news_data[:5], start=1):
            n_title = item.get("title", "Market Update")
            n_src = item.get("source") or "Financial News"
            n_url = item.get("url")
            n_date = item.get("published_at")
            n_summary = item.get("summary")

            d_str = f" • *{n_date}*" if n_date else ""
            lines.append(f"{idx}. **{n_title}**")
            lines.append(f"   * **Publisher:** {n_src}{d_str}")
            if n_summary and n_summary.strip():
                lines.append(f"   * **Executive Summary:** {n_summary.strip()}")
            else:
                lines.append(f"   * **Executive Summary:** Recent verified news and corporate developments concerning {comp_name}.")
            if n_url and n_url != "#":
                lines.append(f"   * 🔗 [Read full article on {n_src} ↗]({n_url})")
            lines.append("")
    else:
        lines.append("*Recent news stories are temporarily unavailable.*\n")

    # Section: Data Availability / Limitations
    lines.append("## Data Availability / Limitations\n")
    if limitations:
        for lim in limitations:
            lines.append(f"* ⚠️ {lim}")
    else:
        lines.append("* All primary corporate fundamentals, quarterly financials, debt metrics, and governance records were successfully retrieved.")

    report_md = "\n".join(lines)

    return {
        "symbol": clean_symbol,
        "company": comp_name,
        "report_markdown": report_md,
        "raw_data": raw_data,
        "financial_health": health_data,
        "limitations": limitations,
    }
