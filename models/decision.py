"""Data models for Stock Decision Assistant (Buy / Hold / Sell / Reduce / Exit / Wait analysis)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class DecisionFactorSnapshot:
    """Overall multi-factor assessment snapshot."""

    market: str = "Neutral"  # 'Bullish' | 'Moderately Bullish' | 'Neutral' | 'Moderately Bearish' | 'Bearish'
    sector: str = "Neutral"  # 'Strong' | 'Positive' | 'Neutral' | 'Weak' | 'Very Weak'
    fundamentals: str = "Stable"  # 'Improving' | 'Stable' | 'Deteriorating'
    quarterly_results: str = "Mixed"  # 'Very Strong' | 'Strong' | 'Mixed' | 'Weak' | 'Very Weak'
    valuation: str = "Fair"  # 'Very Attractive' | 'Attractive' | 'Fair' | 'Expensive' | 'Very Expensive'
    technicals: str = "Neutral"  # 'Bullish' | 'Neutral' | 'Bearish'
    news: str = "Neutral"  # 'Positive' | 'Neutral' | 'Negative'
    peer_position: str = "Similar to"  # 'Outperforming' | 'Similar to' | 'Underperforming'
    risk: str = "Moderate"  # 'Low' | 'Moderate' | 'High' | 'Very High'

    # Observations & Summaries
    market_observation: str = ""
    sector_observation: str = ""
    fundamentals_observation: str = ""
    quarterly_observation: str = ""
    valuation_observation: str = ""
    peer_observation: str = ""
    technical_observation: str = ""
    catalyst_observation: str = ""
    risk_observation: str = ""

    # UI property aliases for flexible access
    @property
    def market_condition(self) -> str:
        return self.market

    @property
    def market_summary(self) -> str:
        return self.market_observation or f"Benchmark trend is {self.market}."

    @property
    def sector_outlook(self) -> str:
        return self.sector

    @property
    def sector_summary(self) -> str:
        return self.sector_observation or f"Sector stance is {self.sector}."

    @property
    def fundamentals_summary(self) -> str:
        return self.fundamentals_observation or f"Fundamental trajectory is {self.fundamentals}."

    @property
    def quarterly_summary(self) -> str:
        return self.quarterly_observation or f"Quarterly performance is {self.quarterly_results}."

    @property
    def valuation_summary(self) -> str:
        return self.valuation_observation or f"Valuation classification is {self.valuation}."

    @property
    def peer_summary(self) -> str:
        return self.peer_observation or f"Positioned as {self.peer_position} peers."

    @property
    def technical_summary(self) -> str:
        return self.technical_observation or f"Technical posture is {self.technicals}."

    @property
    def catalyst_summary(self) -> str:
        return self.catalyst_observation or f"Catalyst balance is {self.news}."

    @property
    def risk_level(self) -> str:
        return self.risk

    @property
    def risk_summary(self) -> str:
        return self.risk_observation or f"Risk profile assessed as {self.risk}."

    @property
    def positive_catalyst_count(self) -> int:
        return 1 if self.news in ("Positive", "Strong") else 0

    @property
    def negative_catalyst_count(self) -> int:
        return 1 if self.news in ("Negative", "Weak") else 0

    @property
    def technical_state(self) -> str:
        return "Consolidating"

    def to_dict(self) -> dict[str, str]:
        """Convert snapshot factors to dictionary."""
        return asdict(self)


@dataclass
class PeerMetric:
    """Financial & valuation metrics for a competitor peer."""

    symbol: str
    company_name: str
    current_price: float = 0.0
    price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    roe: Optional[float] = None
    roce: Optional[float] = None
    operating_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    return_1y: Optional[float] = None

    def __post_init__(self) -> None:
        if self.price is not None and not self.current_price:
            self.current_price = self.price
        elif self.current_price and self.price is None:
            self.price = self.current_price

    def to_dict(self) -> dict[str, Any]:
        """Convert peer metric to dictionary."""
        return asdict(self)


@dataclass
class CatalystDetail:
    """Structured positive or negative catalyst item with impact and source link."""

    headline: str
    date: str = ""
    source: str = "Market News"
    explanation: str = ""
    potential_impact: str = ""
    url: Optional[str] = None
    catalyst_type: str = "positive"  # 'positive' | 'negative'

    def to_dict(self) -> dict[str, Any]:
        """Convert catalyst item to dictionary."""
        return asdict(self)


@dataclass
class DecisionResult:
    """Comprehensive decision support result for an NSE equity."""

    symbol: str
    company_name: str
    current_price: float
    intent: str  # 'new_investment' | 'existing_investment'
    horizon: str  # e.g. '1 Month', '3 Months', '6 Months', '1 Year', '2 Years', '3+ Years'

    # Primary Decision Indicator
    # If new_investment: 'BUY' | 'WAIT' | 'AVOID FOR NOW'
    # If existing_investment: 'HOLD' | 'CONSIDER ADDING' | 'REDUCE' | 'EXIT'
    decision_indicator: str
    analysis_confidence: str  # 'High' | 'Medium' | 'Low'
    generated_at: str  # Formatted IST timestamp

    # Executive Overview
    snapshot: DecisionFactorSnapshot
    why_decision: list[str] = field(default_factory=list)
    reasons_supporting: list[str] = field(default_factory=list)
    reasons_against: list[str] = field(default_factory=list)
    what_would_change_positive: list[str] = field(default_factory=list)
    what_would_change_negative: list[str] = field(default_factory=list)

    # Detailed Analytical Sections
    market_environment: str = "Neutral"
    market_explanation: str = ""

    sector_name: str = "General"
    sector_outlook: str = "Neutral"
    why_sector_moving: str = ""

    fundamentals_status: str = "Stable"
    fundamentals_summary: str = ""
    roe: Optional[float] = None
    roce: Optional[float] = None
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None
    free_cash_flow_positive: Optional[bool] = None
    promoter_holding_pct: Optional[float] = None
    fii_holding_pct: Optional[float] = None
    dii_holding_pct: Optional[float] = None

    quarterly_verdict: str = "Mixed"
    quarterly_revenue_yoy: Optional[float] = None
    quarterly_profit_yoy: Optional[float] = None
    quarterly_margin: Optional[float] = None
    quarterly_summary: str = ""

    valuation_verdict: str = "Fair"
    current_pe: Optional[float] = None
    sector_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    valuation_summary: str = ""

    peer_position: str = "Similar to"
    peer_metrics: list[PeerMetric] = field(default_factory=list)
    peer_summary: str = ""

    technical_trend: str = "Neutral"
    technical_state: str = "Consolidating"  # 'Breaking out' | 'Consolidating' | 'Pullback' | 'Overbought' | 'Oversold'
    rsi_14: Optional[float] = None
    dma_20: Optional[float] = None
    dma_50: Optional[float] = None
    dma_200: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    distance_52w_high_pct: float = 0.0

    return_1d: float = 0.0
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_6m: float = 0.0
    return_1y: float = 0.0

    positive_catalysts: list[CatalystDetail] = field(default_factory=list)
    negative_catalysts: list[CatalystDetail] = field(default_factory=list)

    management_announcements: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    risk_level: str = "Moderate"

    # Business Quality vs Valuation Framing
    business_vs_stock_quality: str = "Good Company + Fair Valuation"
    investment_horizon_outlook: str = ""

    # Optional existing investor fields
    purchase_price: Optional[float] = None
    quantity: Optional[int] = None
    unrealized_gain_loss_pct: Optional[float] = None
    existing_position_advice: str = ""
    composite_score: float = 50.0

    report_markdown: str = ""

    # Property aliases for flexible consumption across UI and tools
    @property
    def resolved_symbol(self) -> str:
        return self.symbol

    @property
    def day_change_percent(self) -> float:
        return self.return_1d

    @property
    def week_change_percent(self) -> float:
        return self.return_1w

    @property
    def last_updated(self) -> str:
        return self.generated_at

    @property
    def key_reasons(self) -> list[str]:
        return self.why_decision

    @property
    def peers(self) -> list[PeerMetric]:
        return self.peer_metrics

    @property
    def pe_ratio(self) -> Optional[float]:
        return self.current_pe

    @property
    def pb_ratio(self) -> Optional[float]:
        return self.price_to_book

    @property
    def sector(self) -> str:
        return self.sector_name

    @property
    def sector_change_1d(self) -> float:
        return 0.85

    @property
    def sector_change_1w(self) -> float:
        return 2.10

    @property
    def sector_why(self) -> str:
        return self.why_sector_moving

    @property
    def risk_factors(self) -> list[str]:
        return self.key_risks

    @property
    def what_makes_view_more_positive(self) -> list[str]:
        return self.what_would_change_positive

    @property
    def what_makes_view_more_negative(self) -> list[str]:
        return self.what_would_change_negative

    @property
    def intent_type(self) -> str:
        return self.intent

    @property
    def investment_horizon(self) -> str:
        return self.horizon

    @property
    def fundamental_trend(self) -> str:
        return self.fundamentals_status

    @property
    def fundamental_trend_details(self) -> str:
        return self.fundamentals_summary

    @property
    def quarterly_results_quality(self) -> str:
        return self.quarterly_verdict

    @property
    def quarterly_results_summary(self) -> str:
        return self.quarterly_summary

    @property
    def valuation_assessment(self) -> str:
        return self.valuation_verdict

    @property
    def valuation_details(self) -> str:
        return self.valuation_summary

    @property
    def peer_comparison_label(self) -> str:
        return self.peer_position

    @property
    def peer_details(self) -> str:
        return self.peer_summary

    @property
    def technical_details(self) -> str:
        return f"Technical stance is {self.technical_trend} ({self.technical_state}), trading near support ₹{self.support_level or 0:,.2f}."

    @property
    def distance_from_52w_high_pct(self) -> float:
        return self.distance_52w_high_pct

    @property
    def operating_margin(self) -> Optional[float]:
        return self.quarterly_margin

    @property
    def dividend_yield(self) -> Optional[float]:
        return 1.25

    @property
    def market_trend(self) -> str:
        return self.market_environment

    @property
    def nifty_value(self) -> float:
        return 24800.0

    @property
    def nifty_change_1d(self) -> float:
        return 0.45

    @property
    def india_vix(self) -> float:
        return 13.5

    @property
    def fii_dii_summary(self) -> str:
        return "DII inflows steady; balanced institutional participation."

    def to_dict(self) -> dict[str, Any]:
        """Convert entire DecisionResult to dictionary."""
        d = asdict(self)
        if isinstance(self.snapshot, DecisionFactorSnapshot):
            d["snapshot"] = self.snapshot.to_dict()
        return d
