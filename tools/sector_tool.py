"""Tool for retrieving top performing NSE sectors and multi-factor ranked constituent companies."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


SUPPORTED_TIMEFRAMES = ("1 Day", "1 Week", "1 Month")
SUPPORTED_SORTS = ("Performance", "Volume", "Market Cap", "RSI", "Momentum")


def _get_top_sectors_and_companies(*args: Any, **kwargs: Any) -> Any:
    from services.sector_service import (
        get_top_sectors_and_companies as _impl,
    )
    return _impl(*args, **kwargs)



def format_top_sectors_markdown(result: dict[str, Any]) -> str:
    """Format Top Sectors and Companies results into clean, professional Markdown."""
    lines: list[str] = []

    lines.append(f"# 🔥 Current Top Sectors — Indian Stock Market (NSE)")
    lines.append(f"*Last Updated: {result.get('generated_at', 'Latest Market Data')}*\n")

    top_sectors = result.get("top_sectors", [])
    if top_sectors:
        lines.append("## 📊 Top 5 Performing Sectors")
        lines.append("| Sector | Performance | 1-Day Change | 1-Week Change | Market Trend |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for sec in top_sectors:
            trend_emoji = "🟢" if sec.get("trend") == "Bullish" else "🔴" if sec.get("trend") == "Bearish" else "⚪"
            lines.append(
                f"| **{sec.get('display_name')}** | {sec.get('performance_score', 0):+.2f}% | "
                f"{sec.get('change_1d', 0):+.2f}% | {sec.get('change_1w', 0):+.2f}% | "
                f"{trend_emoji} {sec.get('trend')} |"
            )
        lines.append("")

    sel_sector_name = result.get("selected_sector", "Banking")
    top_companies = result.get("top_companies", [])

    # Find sector performance details for the selected sector
    sel_perf = next((s for s in result.get("all_sectors", []) if s.get("name") == sel_sector_name), None)

    if sel_perf:
        lines.append(f"## 📋 Sector Summary: {sel_sector_name}")
        lines.append(f"**Why moving today:** {sel_perf.get('why_moving')}\n")
        lines.append(f"**FII / DII Activity:** {sel_perf.get('institutional_activity')}\n")
        lines.append(f"**Government & Policy Impact:** {sel_perf.get('policy_impact')}\n")
        lines.append(f"**Major Earnings:** {sel_perf.get('major_earnings')}\n")

    lines.append(f"## 🔥 Top 10 Stocks — {sel_sector_name}")
    lines.append(f"*Filtered by: {result.get('timeframe', '1 Day')} | Sorted by: {result.get('sort_by', 'Momentum')}*\n")

    if top_companies:
        lines.append("| Rank | Company | Symbol | Price (₹) | 1D % | 1W % | Mkt Cap | P/E | RSI | Trend | Why in Top 10 |")
        lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        for c in top_companies:
            trend_icon = "🟢" if c.get("trend") == "Bullish" else "🔴" if c.get("trend") == "Bearish" else "⚪"
            pe_str = f"{c.get('pe_ratio'):.1f}" if c.get("pe_ratio") else "N/A"
            mcap_cr = f"₹{c.get('market_cap') / 10000000:,.0f} Cr" if c.get("market_cap") else "N/A"
            rsi_str = f"{c.get('rsi_14'):.1f}" if c.get("rsi_14") else "N/A"
            reasons = "; ".join(c.get("why_in_top_10", []))

            lines.append(
                f"| #{c.get('rank')} | **{c.get('company_name')}** | `{c.get('symbol')}` | "
                f"₹{c.get('current_price', 0):,.2f} | {c.get('day_change_percent', 0):+.2f}% | "
                f"{c.get('week_change_percent', 0):+.2f}% | {mcap_cr} | {pe_str} | {rsi_str} | "
                f"{trend_icon} {c.get('trend')} | {reasons} |"
            )

    lines.append("\n*Note: Rankings are derived deterministically using multi-factor momentum, relative volume, earnings, catalysts, and technical indicators. This analysis is for educational and research purposes.*")

    return "\n".join(lines)


def get_top_sectors_and_companies(
    sector_name: Optional[str] = None,
    timeframe: str = "1 Day",
    sort_by: str = "Momentum",
    top_n: int = 10,
) -> dict[str, Any]:
    """Retrieve the top performing NSE sectors and multi-factor ranked constituent companies.

    Identifies the top 5 performing sectors in the Indian stock market and ranks the top
    companies in the chosen sector using price momentum, 1d & 1w performance, relative volume,
    sector-relative strength, technical indicators, earnings, and news catalysts.

    Args:
        sector_name: Name of sector (e.g. 'Banking', 'Information Technology', 'Auto', etc.)
                     If omitted, defaults to the #1 top performing sector.
        timeframe: '1 Day', '1 Week', or '1 Month'.
        sort_by: 'Performance', 'Volume', 'Market Cap', 'RSI', or 'Momentum'.
        top_n: Number of companies to return (default: 10).

    Returns:
        dict: TopSectorsResult dictionary with top sectors, selected sector, company rankings,
              and formatted markdown.
    """
    logger.info("Executing get_top_sectors_and_companies: sector=%r, timeframe=%r, sort_by=%r", sector_name, timeframe, sort_by)

    # Normalize timeframe and sort_by
    tf_norm = "1 Day"
    for tf in SUPPORTED_TIMEFRAMES:
        if tf.lower() == str(timeframe).lower():
            tf_norm = tf
            break

    sort_norm = "Momentum"
    for s in SUPPORTED_SORTS:
        if s.lower() == str(sort_by).lower():
            sort_norm = s
            break

    try:
        res = _get_top_sectors_and_companies(
            sector_name=sector_name,
            timeframe=tf_norm,
            sort_by=sort_norm,
            top_sectors_count=5,
            top_companies_count=top_n,
        )
        res_dict = res.to_dict()
        res_dict["status"] = "success"
        res_dict["report_markdown"] = format_top_sectors_markdown(res_dict)
        return res_dict
    except Exception as exc:
        logger.error("Error retrieving top sectors and companies: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "top_sectors": [],
            "top_companies": [],
            "selected_sector": sector_name or "Banking",
            "report_markdown": f"⚠️ Could not retrieve top sectors data: {exc}",
        }
