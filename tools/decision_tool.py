"""Tool for Stock Decision Assistant (Buy / Hold / Sell / Reduce / Exit / Wait analysis)."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)



def get_stock_decision(
    symbol_or_name: str = "",
    intent: str = "new_investment",
    horizon: str = "1 Year",
    purchase_price: Optional[float] = None,
    quantity: Optional[int] = None,
    stock: Optional[str] = None,
) -> dict[str, Any]:
    """Analyze whether an investor should consider BUY | HOLD | REDUCE | EXIT | WAIT for an Indian stock.

    Performs a complete multi-factor evaluation covering company fundamentals, sector outlook,
    valuation multiples, latest quarterly results, technical setup, news catalysts, peer positioning,
    and company-specific risks based on the user's intended holding period.

    Args:
        symbol_or_name: NSE ticker symbol or natural company name (e.g. 'Tata Power', 'TCS', 'India Cements').
        intent: 'new_investment' (thinking of buying) or 'existing_investment' (already own).
        horizon: Intended holding period ('1 Month', '3 Months', '6 Months', '1 Year', '2 Years', '3+ Years').
        purchase_price: Optional historical buy price if user already owns the stock.
        quantity: Optional number of shares held.
        stock: Alias for symbol_or_name.

    Returns:
        dict: Complete structured decision result including 'decision_indicator', 'snapshot',
              'why_decision', and 'report_markdown'.
    """
    target = stock or symbol_or_name
    logger.info(
        "Executing get_stock_decision: target=%r, intent=%r, horizon=%r, price=%r",
        target, intent, horizon, purchase_price,
    )

    try:
        from services.decision_service import analyze_stock_decision
        res = analyze_stock_decision(
            symbol_or_name=target,
            intent=intent,
            horizon=horizon,
            purchase_price=purchase_price,
            quantity=quantity,
        )
        res_dict = res.to_dict()
        res_dict["status"] = "success"
        return res_dict
    except Exception as exc:
        logger.error("Failed to execute stock decision analysis for %r: %s", symbol_or_name, exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "symbol": symbol_or_name,
            "decision_indicator": "WAIT",
            "report_markdown": f"⚠️ Could not complete stock decision analysis for **{symbol_or_name}**: {exc}",
        }
