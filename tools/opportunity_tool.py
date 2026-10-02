"""Tool for scanning short-term (1-2 day) trading opportunities on the National Stock Exchange of India (NSE)."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


SUPPORTED_UNIVERSES = ("NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200")


def scan_short_term_opportunities(*args: Any, **kwargs: Any) -> Any:
    from services.opportunity_service import scan_short_term_opportunities as _impl
    return _impl(*args, **kwargs)


def get_short_term_opportunities(
    limit: int = 5,
    universe: str = "NIFTY 200",
) -> dict[str, Any]:
    """Scan and rank NSE stocks showing favorable short-term momentum and catalyst setups for the next 1–2 trading sessions.

    Applies a deterministic multi-factor pipeline:
      1. Liquid Universe Screening (NIFTY 50, NIFTY NEXT 50, NIFTY 100, NIFTY 200)
      2. Technical Momentum & Moving Averages (20 DMA, 50 DMA, 5d return, breakouts, RSI)
      3. Volume Confirmation (relative to 20-day average)
      4. Market & Sector Alignment (NIFTY 50, Bank Nifty, Sector trends)
      5. Catalysts & Corporate Events (order wins, earnings beats, capacity additions, buybacks)
      6. Risk & Overextension Checks (RSI overbought, distance from 20 DMA, opening gap risk)

    Strict Compliance Note:
      This tool provides research-oriented technical and catalyst watchlists. It does NOT
      provide guaranteed returns, buy/sell recommendations, or promises of profit.

    Args:
        limit: Number of top watchlist candidates to return (default: 5, max: 20).
        universe: The liquid NSE universe to scan (default: 'NIFTY 200').
                  Supported options: 'NIFTY 50', 'NIFTY NEXT 50', 'NIFTY 100', 'NIFTY 200'.

    Returns:
        dict: Structured result containing:
            - status: 'success' or 'error'
            - generated_at: Timestamp of the scan
            - market_session: 'Open' or 'Closed'
            - target_horizon: e.g. 'Next 1–2 NSE trading sessions (Mon, 05 Oct – Tue, 06 Oct 2026)'
            - market_regime: Deterministic market regime analysis (Bullish / Neutral / Bearish)
            - universe: Scanned universe name
            - total_scanned: Number of equities evaluated
            - candidates: List of ranked candidate setup dictionaries
            - report_markdown: Complete formatted markdown report adhering to compliance standards
    """
    logger.info("Executing get_short_term_opportunities with universe=%r, limit=%r", universe, limit)

    # 1. Normalize and validate arguments
    univ_norm = str(universe).strip().upper() if universe else "NIFTY 200"
    if univ_norm not in [u.upper() for u in SUPPORTED_UNIVERSES]:
        univ_norm = "NIFTY 200"

    try:
        limit_val = int(limit)
    except (ValueError, TypeError):
        limit_val = 5
    limit_val = max(1, min(limit_val, 20))

    # 2. Execute deterministic scan
    try:
        scan_res = scan_short_term_opportunities(universe=univ_norm, limit=limit_val)
        result_dict = scan_res.to_dict()
        result_dict["status"] = "success"
        logger.debug("Successfully scanned %d candidates for %s", len(scan_res.candidates), univ_norm)
        return result_dict
    except Exception as exc:
        logger.error("Failed to scan short-term opportunities: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "universe": univ_norm,
            "candidates": [],
            "report_markdown": f"⚠️ Could not complete short-term opportunity scan: {exc}",
        }
