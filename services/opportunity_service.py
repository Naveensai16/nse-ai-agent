"""Deterministic opportunity scanning, catalyst identification, risk evaluation, and scoring service."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

import pandas as pd
import yfinance as yf

from models.opportunity import (
    CatalystItem,
    MarketRegime,
    OpportunityScanResult,
    OpportunitySetup,
    RiskItem,
    ScoreBreakdown,
    TechnicalSignals,
)
from services.market_data_service import (
    SECTOR_MAP,
    UNIVERSES,
    calculate_stock_technical_signals,
    get_market_regime,
    get_market_session_status,
)
from services.symbol_resolver import NSE_MASTER_DIRECTORY, resolve_nse_symbol
from tools.financials_tool import get_quarterly_financials
from tools.governance_tool import get_corporate_actions
from tools.news_tool import get_market_news

logger = logging.getLogger(__name__)

# Configurable Scoring Weights (Total 100 points)
WEIGHT_TECHNICAL_MOMENTUM = 25.0
WEIGHT_VOLUME_CONFIRMATION = 15.0
WEIGHT_MARKET_SECTOR_STRENGTH = 15.0
WEIGHT_POSITIVE_CATALYSTS = 20.0
WEIGHT_FINANCIAL_RESULTS = 15.0
WEIGHT_RISK_ADJUSTMENT = 10.0

# Neutral, compliant confidence classification thresholds
CONFIDENCE_THRESHOLDS: list[tuple[float, str]] = [
    (80.0, "High-Confidence Setup"),
    (65.0, "Moderate Setup"),
    (50.0, "Watch Only"),
    (0.0, "Weak Setup"),
]

# Positive catalyst detection keywords
POSITIVE_CATALYST_PATTERNS: list[tuple[str, str, int]] = [
    (r"\b(?:order win|secures order|bags order|awarded contract|new contract|major contract|deal worth)\b", "order_win", 8),
    (r"\b(?:revenue jumps|profit jumps|net profit up|profit rises|earnings beat|strong q[1-4]|pat up|results beat)\b", "earnings", 8),
    (r"\b(?:approval|gets nod|cleared by|usfda|cdsco|rbi clears|regulatory green light)\b", "regulatory", 6),
    (r"\b(?:expansion|capacity addition|new plant|capex|commissions plant|manufacturing unit)\b", "capacity", 5),
    (r"\b(?:acquisition|acquires|takeover|buys stake|strategic stake|merger)\b", "acquisition", 5),
    (r"\b(?:partnership|tie-up|collaborates|joint venture|signs agreement|mou)\b", "partnership", 4),
    (r"\b(?:buyback|dividend|bonus share|bonus issue|special dividend)\b", "corporate_action", 4),
    (r"\b(?:credit rating upgrade|rating upgraded|upgraded to)\b", "rating", 4),
    (r"\b(?:analyst upgrade|target raised|brokerage raises|overweight|outperform|buy rating)\b", "analyst", 4),
    (r"\b(?:debt free|reduces debt|debt reduction|prepayment)\b", "debt", 4),
]

# Negative risk / catalyst detection keywords
NEGATIVE_CATALYST_PATTERNS: list[tuple[str, str, int]] = [
    (r"\b(?:loss widens|profit falls|profit drops|revenue dips|earnings miss|pat declines|results miss)\b", "earnings_miss", -8),
    (r"\b(?:investigation|sebi probe|raid|cbi|ed search|notice|fraud|penalty|irregularities)\b", "regulatory_probe", -10),
    (r"\b(?:promoter sells|promoter stake cut|pledge increased|promoter offloads)\b", "promoter_selling", -7),
    (r"\b(?:downgrade|credit rating cut|rating downgraded|underweight)\b", "downgrade", -6),
    (r"\b(?:resigns|resignation|management churn|ceo quits|cfo steps down)\b", "management_exit", -5),
    (r"\b(?:order cancelled|contract terminated|deal terminated|project halted)\b", "order_cancellation", -7),
    (r"\b(?:debt concerns|default|insolvency|nclt)\b", "debt_distress", -10),
]


def map_score_to_confidence(score: float) -> str:
    """Map numeric score (0-100) to an explainable research confidence label.

    Strictly avoids guaranteed profit or predictive promises.
    """
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if score >= threshold:
            return label
    return "Weak Setup"


def evaluate_technical_score(signals: TechnicalSignals) -> tuple[float, list[str]]:
    """Calculate deterministic Technical Momentum score (Max 25 points)."""
    score = 0.0
    evidence: list[str] = []

    # 1. Price vs 20 DMA (up to +7 pts)
    if signals.is_above_20dma:
        score += 7.0
        evidence.append(f"Price is trading above 20 DMA (₹{signals.dma_20:,.2f}) by +{signals.distance_from_20dma:.1f}%")

    # 2. Price vs 50 DMA (up to +6 pts)
    if signals.is_above_50dma:
        score += 6.0
        evidence.append(f"Price is holding above intermediate 50 DMA (₹{signals.dma_50:,.2f})")

    # 3. Moving Average alignment (Golden cross / bullish slope: +4 pts)
    if signals.is_dma_bullish_alignment:
        score += 4.0
        evidence.append("Bullish moving average alignment (20 DMA is above 50 DMA)")

    # 4. Multi-day price momentum (up to +5 pts)
    # Healthy momentum is between +1% and +7%. Move > 10% is extended.
    ret_5d = signals.return_5d or 0.0
    if 1.0 <= ret_5d <= 7.0:
        score += 5.0
        evidence.append(f"Steady 5-day accumulation with +{ret_5d:.1f}% price return")
    elif 0.0 < ret_5d < 1.0:
        score += 2.5
        evidence.append(f"Mild positive 5-day momentum (+{ret_5d:.1f}%)")
    elif ret_5d > 7.0:
        score += 2.0  # extended, lower score
        evidence.append(f"High 5-day return (+{ret_5d:.1f}%), momentum is extended")

    # 5. Breakout detection (up to +3 pts)
    if signals.is_breakout:
        score += 3.0
        evidence.append(f"Trading at or near 20-day resistance high (₹{signals.high_20d:,.2f})")

    return min(score, WEIGHT_TECHNICAL_MOMENTUM), evidence


def evaluate_volume_score(signals: TechnicalSignals) -> tuple[float, list[str]]:
    """Calculate deterministic Volume Confirmation score (Max 15 points)."""
    vr = signals.volume_ratio
    evidence: list[str] = []

    if vr >= 2.0:
        score = 15.0
        evidence.append(f"Strong volume expansion of {vr:.1f}x relative to 20-day average")
    elif vr >= 1.5:
        score = 12.0
        evidence.append(f"Above-average volume participation of {vr:.1f}x 20-day average")
    elif vr >= 1.2:
        score = 8.0
        evidence.append(f"Moderate volume confirmation ({vr:.1f}x 20-day average)")
    elif vr >= 0.9:
        score = 5.0
        evidence.append(f"Normal market volume ({vr:.1f}x average)")
    else:
        score = 1.0
        evidence.append(f"Below-average trading volume ({vr:.1f}x 20-day average)")

    return min(score, WEIGHT_VOLUME_CONFIRMATION), evidence


def evaluate_market_and_sector_score(
    symbol: str,
    market_regime: MarketRegime,
    signals: TechnicalSignals,
) -> tuple[float, list[str]]:
    """Calculate Market & Sector Strength score (Max 15 points)."""
    score = 0.0
    evidence: list[str] = []

    # Market regime contribution (up to 8 pts)
    if market_regime.regime == "Bullish":
        score += 8.0
        evidence.append("Broader market (NIFTY 50) is in a confirmed Bullish regime")
    elif market_regime.regime == "Neutral":
        score += 4.0
        evidence.append("Broader market is Neutral, requiring selective stock-specific setups")
    else:
        score += 1.0
        evidence.append("Broader market is Bearish, representing an overarching macro headwind")

    # Sector relative strength (up to 7 pts)
    sector = SECTOR_MAP.get(symbol, "Diversified")
    stock_day_chg = signals.day_change_percent

    # Stock outperforming benchmark index
    if stock_day_chg > market_regime.nifty_change_1d:
        score += 5.0
        diff = stock_day_chg - market_regime.nifty_change_1d
        evidence.append(f"Outperforming benchmark by +{diff:.2f}% today ({sector} sector)")
    else:
        score += 2.0

    if signals.return_5d and signals.return_5d > market_regime.nifty_return_5d:
        score += 2.0
        evidence.append("Demonstrating 5-day relative strength against the NIFTY 50")

    return min(score, WEIGHT_MARKET_SECTOR_STRENGTH), evidence


def evaluate_financial_results_score(financials: dict[str, Any]) -> tuple[float, list[str]]:
    """Calculate Financial Results score based on reported YoY growth (Max 15 points)."""
    score = 0.0
    evidence: list[str] = []

    if not financials or (not financials.get("quarters") and not financials.get("latest_quarter") and "yoy_revenue_growth" not in financials and "yoy_profit_growth" not in financials):
        return 5.0, ["Quarterly financial statements data unavailable in exchange summary"]

    rev_yoy = financials.get("yoy_revenue_growth")
    if rev_yoy is None and "latest_quarter" in financials:
        rev_yoy = financials["latest_quarter"].get("revenue_yoy")

    profit_yoy = financials.get("yoy_profit_growth")
    if profit_yoy is None and "latest_quarter" in financials:
        profit_yoy = financials["latest_quarter"].get("net_profit_yoy")

    margin_trend = financials.get("margin_trend", "N/A")

    # Revenue YoY (up to 5 pts)
    if rev_yoy is not None:
        if rev_yoy >= 15.0:
            score += 5.0
            evidence.append(f"Strong top-line growth with Revenue up +{rev_yoy:.1f}% YoY")
        elif rev_yoy >= 8.0:
            score += 3.5
            evidence.append(f"Healthy Revenue expansion of +{rev_yoy:.1f}% YoY")
        elif rev_yoy >= 0.0:
            score += 2.0
            evidence.append(f"Stable Revenue growth of +{rev_yoy:.1f}% YoY")
        else:
            evidence.append(f"Revenue contraction of {rev_yoy:.1f}% YoY")
    else:
        score += 2.0

    # Net Profit YoY (up to 6 pts)
    if profit_yoy is not None:
        if profit_yoy >= 20.0:
            score += 6.0
            evidence.append(f"Robust earnings acceleration with Net Profit up +{profit_yoy:.1f}% YoY")
        elif profit_yoy >= 10.0:
            score += 4.0
            evidence.append(f"Solid Net Profit growth of +{profit_yoy:.1f}% YoY")
        elif profit_yoy >= 0.0:
            score += 2.0
            evidence.append(f"Positive Net Profit growth (+{profit_yoy:.1f}% YoY)")
        else:
            evidence.append(f"Net profit declined {profit_yoy:.1f}% YoY in recent filings")
    else:
        score += 2.0

    # Margin trend (up to 4 pts)
    if margin_trend == "Expanding":
        score += 4.0
        evidence.append("Operating margin (OPM) is expanding quarter-on-quarter")
    elif margin_trend == "Stable":
        score += 2.5
        evidence.append("Operating profit margins remain stable")
    else:
        score += 1.0

    return min(score, WEIGHT_FINANCIAL_RESULTS), evidence


def detect_catalysts_and_news(
    symbol: str,
    news_items: list[dict[str, Any]],
    corporate_actions: list[dict[str, Any]],
) -> tuple[float, list[CatalystItem], list[RiskItem]]:
    """Scan verified news stories and corporate actions for positive catalysts and negative risks."""
    catalyst_score = 0.0
    positive_catalysts: list[CatalystItem] = []
    risks: list[RiskItem] = []
    seen_headlines: set[str] = set()

    # 1. Corporate Actions Catalysts (Dividends, Bonus, Buybacks)
    for act in corporate_actions[:3]:
        detail = act.get("detail", "")
        dt_str = act.get("date", "")
        if any(w in detail.lower() for w in ("dividend", "bonus", "split", "buyback")):
            positive_catalysts.append(
                CatalystItem(
                    catalyst_type="positive",
                    category="corporate_action",
                    headline=detail,
                    summary=f"Corporate action announced on {dt_str}: {detail}",
                    source="NSE Corporate Disclosures",
                    date=dt_str,
                )
            )
            catalyst_score += 4.0
            break

    # 2. News Catalyst and Risk Scanner
    for item in news_items:
        title = item.get("title", "")
        clean_title = re.sub(r"\s+", " ", title).strip()
        norm_title = clean_title.lower()

        if norm_title in seen_headlines:
            continue
        seen_headlines.add(norm_title)

        summary = item.get("summary") or clean_title
        src = item.get("source") or "Financial News"
        pub_at = item.get("published_at")
        url = item.get("url")

        # Scan for positive catalysts
        matched_pos = False
        for pat, cat, pts in POSITIVE_CATALYST_PATTERNS:
            if re.search(pat, norm_title, re.IGNORECASE):
                positive_catalysts.append(
                    CatalystItem(
                        catalyst_type="positive",
                        category=cat,
                        headline=clean_title,
                        summary=summary,
                        source=src,
                        date=pub_at,
                        url=url,
                    )
                )
                catalyst_score += pts
                matched_pos = True
                break

        # Scan for negative catalysts
        for pat, cat, penalty in NEGATIVE_CATALYST_PATTERNS:
            if re.search(pat, norm_title, re.IGNORECASE):
                risks.append(
                    RiskItem(
                        risk_type=cat,
                        description=f"Adverse headline: '{clean_title}' reported by {src}",
                        severity="high",
                    )
                )
                catalyst_score += penalty  # negative value
                break

    # Baseline if no specific catalyst found
    if not positive_catalysts:
        catalyst_score = 4.0  # baseline neutral rating

    return max(0.0, min(catalyst_score, WEIGHT_POSITIVE_CATALYSTS)), positive_catalysts, risks


def evaluate_risk_penalties(
    signals: TechnicalSignals,
    market_regime: MarketRegime,
    additional_risks: list[RiskItem],
) -> tuple[float, list[RiskItem], Optional[str]]:
    """Calculate deterministic risk penalties (Base 10 points, reduced for elevated risks)."""
    risk_points = WEIGHT_RISK_ADJUSTMENT  # Starts at 10.0
    risks = list(additional_risks)
    gap_warning = None

    # 1. Extremely overbought RSI penalty
    rsi = signals.rsi_14 or 50.0
    if rsi >= 80.0:
        risk_points -= 7.0
        risks.append(
            RiskItem(
                risk_type="overbought_rsi",
                description=f"RSI(14) is at {rsi:.1f} (extremely overbought zone), high probability of mean-reversion pause",
                severity="high",
            )
        )
    elif rsi >= 72.0:
        risk_points -= 4.0
        risks.append(
            RiskItem(
                risk_type="overbought_rsi",
                description=f"RSI(14) is elevated at {rsi:.1f} approaching overbought territory",
                severity="medium",
            )
        )
    elif rsi <= 35.0:
        risk_points -= 4.0
        risks.append(
            RiskItem(
                risk_type="oversold_downtrend",
                description=f"RSI(14) is weak at {rsi:.1f}, indicating persistent selling pressure",
                severity="medium",
            )
        )

    # 2. Overextended distance from 20 DMA penalty
    dist_20 = signals.distance_from_20dma or 0.0
    if dist_20 >= 12.0:
        risk_points -= 6.0
        risks.append(
            RiskItem(
                risk_type="overextended",
                description=f"Stock is currently {dist_20:.1f}% above its 20 DMA (abnormally stretched from short-term baseline)",
                severity="high",
            )
        )
    elif dist_20 >= 7.5:
        risk_points -= 3.0
        risks.append(
            RiskItem(
                risk_type="overextended",
                description=f"Stock is trading {dist_20:.1f}% above its 20 DMA, entering extended zone",
                severity="medium",
            )
        )

    # 3. Large opening gap-up risk
    gap_pct = signals.gap_percent or 0.0
    if gap_pct >= 3.0:
        risk_points -= 3.0
        risks.append(
            RiskItem(
                risk_type="gap_exhaustion",
                description=f"Significant gap-up of +{gap_pct:.1f}% already registered, elevating morning gap-fill risk",
                severity="medium",
            )
        )
        gap_warning = (
            f"Note: A gap-up of +{gap_pct:.1f}% occurred recently. Setup should be reassessed after market open "
            f"as large opening gaps can alter the risk-to-reward dynamic."
        )

    # 4. Weak volume participation on positive day
    if signals.day_change_percent > 0.5 and signals.volume_ratio < 0.65:
        risk_points -= 3.0
        risks.append(
            RiskItem(
                risk_type="weak_volume",
                description=f"Price rise occurred on low volume ({signals.volume_ratio:.2f}x average), lacking institutional backing",
                severity="medium",
            )
        )

    # 5. Price below 50 DMA
    if not signals.is_above_50dma:
        risk_points -= 3.0
        risks.append(
            RiskItem(
                risk_type="downtrend",
                description=f"Stock remains below its 50 DMA (₹{signals.dma_50:,.2f}), trading against intermediate trend",
                severity="medium",
            )
        )

    # 6. Overhead resistance proximity
    if signals.potential_upside_pct is not None and signals.potential_upside_pct < 1.0:
        risks.append(
            RiskItem(
                risk_type="overhead_resistance",
                description=f"Trading within {signals.potential_upside_pct:.1f}% of key resistance (₹{signals.nearest_resistance:,.2f})",
                severity="medium",
            )
        )

    # Ensure minimum risk points do not drop below 0
    final_risk_score = max(0.0, min(risk_points, WEIGHT_RISK_ADJUSTMENT))
    return final_risk_score, risks, gap_warning


def identify_catalysts_and_risks(
    symbol: str,
    signals: TechnicalSignals,
    financials: dict[str, Any],
    news_items: list[dict[str, Any]],
    corporate_actions: list[dict[str, Any]],
) -> tuple[list[CatalystItem], list[RiskItem]]:
    """Scan and return all positive catalysts and identified risk factors for a stock."""
    _, cats, news_risks = detect_catalysts_and_news(symbol, news_items, corporate_actions)
    if financials:
        _, fin_evidence = evaluate_financial_results_score(financials)
        for ev in fin_evidence:
            if any(k in ev.lower() for k in ("growth", "jump", "strong", "robust", "profit", "expansion")):
                cats.append(
                    CatalystItem(
                        catalyst_type="positive",
                        category="strong_financials",
                        headline=ev,
                        summary=ev,
                        source="Quarterly Financials",
                    )
                )
    _, all_risks, _ = evaluate_risk_penalties(signals, MarketRegime(regime="Neutral"), news_risks)
    return cats, all_risks


def compute_opportunity_score(
    signals: TechnicalSignals,
    market_regime: MarketRegime,
    catalysts: list[CatalystItem],
    risks: list[RiskItem],
    financials: dict[str, Any],
    sector: str = "Diversified",
    symbol: str = "TEST",
) -> tuple[float, ScoreBreakdown, str]:
    """Compute and return total opportunity score, breakdown, and confidence classification."""
    tech_score, _ = evaluate_technical_score(signals)
    vol_score, _ = evaluate_volume_score(signals)
    mkt_score, _ = evaluate_market_and_sector_score(symbol, market_regime, signals)
    fin_score, _ = evaluate_financial_results_score(financials)
    cat_score = sum(c.score_impact if hasattr(c, "score_impact") else 4.0 for c in catalysts)
    cat_score = min(max(cat_score, 0.0), WEIGHT_POSITIVE_CATALYSTS)
    risk_score, _, _ = evaluate_risk_penalties(signals, market_regime, risks)

    total = round(tech_score + vol_score + mkt_score + fin_score + cat_score + risk_score, 1)
    total = max(0.0, min(100.0, total))
    conf = map_score_to_confidence(total)

    breakdown = ScoreBreakdown(
        technical_momentum=round(tech_score, 1),
        volume_confirmation=round(vol_score, 1),
        market_sector_strength=round(mkt_score, 1),
        positive_catalysts=round(cat_score, 1),
        financial_results=round(fin_score, 1),
        risk_adjustment=round(risk_score, 1),
        total_score=total,
    )
    return total, breakdown, conf


def build_candidate_opportunity(
    rank: int,
    symbol: str,
    signals: TechnicalSignals,
    market_regime: MarketRegime,
    financials: dict[str, Any],
    news_items: list[dict[str, Any]],
    corporate_actions: list[dict[str, Any]],
) -> OpportunitySetup:
    """Evaluate and construct a complete OpportunitySetup object with explainable scoring."""
    master_info = NSE_MASTER_DIRECTORY.get(symbol, (symbol, []))
    company_name = master_info[0]
    sector = SECTOR_MAP.get(symbol, "Diversified")

    # 1. Technical Score
    tech_score, tech_evidence = evaluate_technical_score(signals)

    # 2. Volume Score
    vol_score, vol_evidence = evaluate_volume_score(signals)

    # 3. Market & Sector Score
    mkt_score, mkt_evidence = evaluate_market_and_sector_score(symbol, market_regime, signals)

    # 4. Financial Results Score
    fin_score, fin_evidence = evaluate_financial_results_score(financials)

    # 5. Catalysts & News
    cat_score, positive_catalysts, news_risks = detect_catalysts_and_news(symbol, news_items, corporate_actions)

    # 6. Risk Adjustments
    risk_score, all_risks, gap_warning = evaluate_risk_penalties(signals, market_regime, news_risks)

    # Total Opportunity Score
    total_score = round(tech_score + vol_score + mkt_score + cat_score + fin_score + risk_score, 1)
    total_score = max(0.0, min(100.0, total_score))

    confidence_label = map_score_to_confidence(total_score)

    # Compile "Why on Watchlist" summary points
    why_points: list[str] = []
    if positive_catalysts:
        why_points.append(positive_catalysts[0].headline)
    why_points.extend(tech_evidence[:2])
    why_points.extend(vol_evidence[:1])
    if fin_evidence and "unavailable" not in fin_evidence[0].lower():
        why_points.append(fin_evidence[0])
    why_points.extend(mkt_evidence[:1])

    score_breakdown = ScoreBreakdown(
        technical_momentum=round(tech_score, 1),
        volume_confirmation=round(vol_score, 1),
        market_sector_strength=round(mkt_score, 1),
        positive_catalysts=round(cat_score, 1),
        financial_results=round(fin_score, 1),
        risk_adjustment=round(risk_score, 1),
        total_score=total_score,
    )

    return OpportunitySetup(
        rank=rank,
        symbol=symbol,
        company_name=company_name,
        sector=sector,
        current_price=signals.current_price,
        day_change_percent=signals.day_change_percent,
        volume_ratio=signals.volume_ratio,
        opportunity_score=total_score,
        confidence_label=confidence_label,
        score_breakdown=score_breakdown,
        technical_signals=signals,
        why_on_watchlist=why_points[:5],
        positive_catalysts=positive_catalysts,
        risks=all_risks,
        recent_news=news_items[:4],
        recent_filings=corporate_actions[:3],
        gap_risk_warning=gap_warning,
    )


def format_opportunity_report_markdown(result: OpportunityScanResult) -> str:
    """Render the complete research watchlist into high-quality GitHub-flavored Markdown."""
    lines: list[str] = [
        "# 📈 2-Day Trading Opportunities",
        f"**Target Horizon:** {result.target_horizon}",
        f"**Generated At:** {result.generated_at} | **Market Session:** `{result.market_session}`\n",
        f"> [!NOTE]\n> **Short-Term Research Watchlist**: Evaluates price momentum, volume confirmation, "
        f"corporate catalysts, financial results, and risk checks for the upcoming 1–2 NSE sessions. "
        f"This output reflects technical setups and does not assure returns or profit.\n",
        f"### 🌐 Market Regime: **{result.market_regime.regime}**",
        f"*{result.market_regime.summary}*\n",
        "---",
        "### 🏆 Ranked Opportunity Summary\n",
        "| Rank | Symbol | Company | Price | Day Change | Vol Ratio | Score | Setup Confidence |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for c in result.candidates:
        p_str = f"₹{c.current_price:,.2f}"
        chg_str = f"{c.day_change_percent:+.2f}%"
        chg_icon = "🟢" if c.day_change_percent >= 0 else "🔴"
        lines.append(
            f"| **#{c.rank}** | **{c.symbol}** | {c.company_name} | {p_str} | {chg_icon} {chg_str} | "
            f"{c.volume_ratio:.1f}x | **{c.opportunity_score:.0f} / 100** | `{c.confidence_label}` |"
        )

    lines.append("\n---\n")

    # Detailed candidate profiles
    for c in result.candidates:
        sig = c.technical_signals
        lines.extend([
            f"## #{c.rank}. {c.company_name} ({c.symbol}) — Potential Bullish Setup",
            f"**Sector:** {c.sector} | **Current Price:** ₹{c.current_price:,.2f} ({c.day_change_percent:+.2f}%)",
            f"**Opportunity Score:** **{c.opportunity_score:.0f} / 100** — `{c.confidence_label}` · Watchlist Candidate\n",
            "#### 📊 Score Breakdown:",
            f"- Technical Momentum: **{c.score_breakdown.technical_momentum} / {WEIGHT_TECHNICAL_MOMENTUM:.0f}**",
            f"- Volume Confirmation: **{c.score_breakdown.volume_confirmation} / {WEIGHT_VOLUME_CONFIRMATION:.0f}**",
            f"- Market & Sector Strength: **{c.score_breakdown.market_sector_strength} / {WEIGHT_MARKET_SECTOR_STRENGTH:.0f}**",
            f"- Positive Catalysts: **{c.score_breakdown.positive_catalysts} / {WEIGHT_POSITIVE_CATALYSTS:.0f}**",
            f"- Financial Results: **{c.score_breakdown.financial_results} / {WEIGHT_FINANCIAL_RESULTS:.0f}**",
            f"- Risk Adjustment: **{c.score_breakdown.risk_adjustment} / {WEIGHT_RISK_ADJUSTMENT:.0f}**\n",
            "#### 🎯 Technical Setup Levels:",
            f"* **Nearest Support:** ₹{sig.nearest_support:,.2f} (Downside: -{sig.potential_downside_pct:.1f}%)",
            f"* **Nearest Resistance:** ₹{sig.nearest_resistance:,.2f} (Potential Upside: +{sig.potential_upside_pct:.1f}%)",
            f"* **14-Period ATR:** ₹{sig.atr_14:,.2f} | **Approx Technical R/R:** {sig.risk_reward_ratio:.1f}x",
            f"* **Moving Averages:** 20 DMA: ₹{sig.dma_20:,.2f} | 50 DMA: ₹{sig.dma_50:,.2f} | RSI(14): {sig.rsi_14:.1f}\n",
            "#### 💡 Why It Is On The Watchlist:",
        ])

        for pt in c.why_on_watchlist:
            lines.append(f"- {pt}")

        if c.positive_catalysts:
            lines.append("\n#### 🚀 Positive Catalysts Detected:")
            for cat in c.positive_catalysts:
                source_part = f" *({cat.source})*" if cat.source else ""
                lines.append(f"- **{cat.headline}**{source_part}")
        else:
            lines.append("\n#### 🚀 Positive Catalysts Detected:")
            lines.append("- *No significant verified corporate catalyst identified; selection driven primarily by quantitative technical momentum.*")

        lines.append("\n#### ⚠️ Risks / Setup Invalidation Factors:")
        if c.risks:
            for r in c.risks:
                lines.append(f"- {r.description}")
        else:
            lines.append("- *Standard equity market volatility applies.*")

        if c.gap_risk_warning:
            lines.append(f"\n> [!WARNING]\n> **Opening Gap Caution**: {c.gap_risk_warning}")

        if c.recent_news:
            lines.append("\n#### 📰 Recent Verified News:")
            for n in c.recent_news[:3]:
                t = n.get("title", "")
                src = n.get("source", "News")
                url = n.get("url")
                link_str = f" [Read ↗]({url})" if url and url != "#" else ""
                lines.append(f"- **{t}** — *{src}*{link_str}")

        lines.append("\n---\n")

    lines.append(
        "> [!IMPORTANT]\n> **Risk Disclaimer**: Short-term setups are subject to overnight gap risk, macro headlines, "
        "and market volatility. Always adhere to strict risk management and position sizing. Past performance or "
        "technical momentum does not guarantee future price action."
    )

    return "\n".join(lines)


def scan_short_term_opportunities(
    universe: str = "NIFTY 200",
    limit: int = 5,
    history_provider: Optional[Callable[[str], pd.DataFrame]] = None,
    news_provider: Optional[Callable[[str], list[dict[str, Any]]]] = None,
    financials_provider: Optional[Callable[[str], dict[str, Any]]] = None,
) -> OpportunityScanResult:
    """Execute two-stage quantitative and fundamental scanning for 2-day NSE trading setups.

    Args:
        universe: Target liquid universe ('NIFTY 50', 'NIFTY NEXT 50', 'NIFTY 100', 'NIFTY 200').
        limit: Number of top candidates to return (defaults to 5, max 10).
        history_provider: Optional callable for injecting mock OHLCV DataFrames.
        news_provider: Optional callable for injecting mock news items.
        financials_provider: Optional callable for injecting mock financials.

    Returns:
        OpportunityScanResult with market regime, ranked candidates, and markdown report.
    """
    tickers = UNIVERSES.get(universe, UNIVERSES["NIFTY 200"])
    session_status, timestamp_str, horizon_str = get_market_session_status()
    market_regime = get_market_regime()

    # Stage 1: Quantitative screening of the universe
    scanned_candidates: list[tuple[float, str, TechnicalSignals]] = []

    # If mock history provider is passed, use it directly
    if history_provider is not None:
        for sym in tickers:
            try:
                df = history_provider(sym)
                sig = calculate_stock_technical_signals(sym, df)
                if sig and sig.current_price >= 10.0 and sig.avg_volume_20d >= 1000:
                    prelim_score = (
                        (6.0 if sig.is_above_20dma else 0.0) +
                        (5.0 if sig.is_above_50dma else 0.0) +
                        (min(sig.volume_ratio * 4.0, 12.0)) +
                        (4.0 if 1.0 <= (sig.return_5d or 0.0) <= 8.0 else 0.0) +
                        (3.0 if sig.is_breakout else 0.0)
                    )
                    scanned_candidates.append((prelim_score, sym, sig))
            except Exception as exc:
                logger.debug("Failed technical check for %s: %s", sym, exc)
    else:
        # Live batch fetch
        formatted_symbols = [f"{s}.NS" for s in tickers]
        batch_size = 35
        for i in range(0, len(formatted_symbols), batch_size):
            chunk = formatted_symbols[i : i + batch_size]
            try:
                df_all = yf.download(chunk, period="3mo", interval="1d", progress=False, group_by="ticker")
                for s in tickers[i : i + batch_size]:
                    sig = None
                    try:
                        s_ticker = f"{s}.NS"
                        if isinstance(df_all.columns, pd.MultiIndex) and s_ticker in df_all.columns.levels[0]:
                            sub_df = df_all[s_ticker]
                            sig = calculate_stock_technical_signals(s, sub_df)
                    except Exception:
                        pass
                    if not sig:
                        # Fallback to individual ticker history if batch extraction missed it
                        try:
                            t = yf.Ticker(f"{s}.NS")
                            sig = calculate_stock_technical_signals(s, t.history(period="3mo"))
                        except Exception:
                            pass
                    if sig and sig.current_price >= 15.0 and sig.avg_volume_20d >= 10000:
                        prelim_score = (
                            (6.0 if sig.is_above_20dma else 0.0) +
                            (5.0 if sig.is_above_50dma else 0.0) +
                            (min(sig.volume_ratio * 4.0, 12.0)) +
                            (4.0 if 1.0 <= (sig.return_5d or 0.0) <= 8.0 else 0.0) +
                            (3.0 if sig.is_breakout else 0.0)
                        )
                        scanned_candidates.append((prelim_score, s, sig))
            except Exception as exc:
                logger.warning("Batch download failed for chunk: %s", exc)

    # Sort stage 1 candidates by preliminary score descending
    scanned_candidates.sort(key=lambda x: x[0], reverse=True)

    # Stage 2: Deep catalyst, filing, earnings, and risk inspection for top shortlist (top 12)
    shortlist = scanned_candidates[:12]
    final_setups: list[OpportunitySetup] = []

    for idx, (_, sym, signals) in enumerate(shortlist, start=1):
        # 1. Fetch Financials
        fin = {}
        try:
            if financials_provider is not None:
                fin = financials_provider(sym)
            else:
                fin = get_quarterly_financials(sym)
        except Exception:
            pass

        # 2. Fetch News
        news = []
        try:
            if news_provider is not None:
                news = news_provider(sym)
            else:
                news = get_market_news(sym, limit=4)
        except Exception:
            pass

        # 3. Fetch Corporate Actions
        corp = []
        try:
            corp = get_corporate_actions(sym)
        except Exception:
            pass

        setup = build_candidate_opportunity(
            rank=idx,
            symbol=sym,
            signals=signals,
            market_regime=market_regime,
            financials=fin,
            news_items=news,
            corporate_actions=corp,
        )
        final_setups.append(setup)

    # Re-rank by total opportunity score descending
    final_setups.sort(key=lambda s: s.opportunity_score, reverse=True)
    for rank_idx, setup in enumerate(final_setups, start=1):
        setup.rank = rank_idx

    top_candidates = final_setups[:limit]

    result = OpportunityScanResult(
        generated_at=timestamp_str,
        market_session=session_status,
        target_horizon=horizon_str,
        market_regime=market_regime,
        universe=universe,
        total_scanned=len(scanned_candidates),
        candidates=top_candidates,
    )
    result.report_markdown = format_opportunity_report_markdown(result)
    return result
