"""Data models and structures for the 2-Day Trading Opportunities scanner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class TechnicalSignals:
    """Quantitative technical indicators and price action signals."""

    current_price: float
    previous_close: float
    day_change_percent: float
    symbol: str = ""
    return_2d: Optional[float] = None
    return_5d: Optional[float] = None
    return_20d: Optional[float] = None
    dma_20: Optional[float] = None
    dma_50: Optional[float] = None
    distance_from_20dma: Optional[float] = None
    distance_from_50dma: Optional[float] = None
    volume: Optional[float] = None
    avg_volume_20d: Optional[float] = None
    volume_ratio: float = 1.0
    rsi_14: Optional[float] = None
    atr_14: Optional[float] = None
    high_20d: Optional[float] = None
    low_20d: Optional[float] = None
    is_breakout: bool = False
    is_above_20dma: bool = False
    is_above_50dma: bool = False
    is_dma_bullish_alignment: bool = False
    gap_percent: Optional[float] = None
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    potential_upside_pct: Optional[float] = None
    potential_downside_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    @property
    def return_1d(self) -> float:
        return self.day_change_percent

    @property
    def dist_from_20dma_pct(self) -> float:
        return self.distance_from_20dma or 0.0

    @property
    def gap_pct(self) -> float:
        return self.gap_percent or 0.0

    @property
    def support_level(self) -> float:
        return self.nearest_support or round(self.current_price * 0.97, 2)

    @property
    def resistance_level(self) -> float:
        return self.nearest_resistance or round(self.current_price * 1.04, 2)

    @property
    def downside_risk_pct(self) -> float:
        return self.potential_downside_pct or 3.0

    @property
    def upside_potential_pct(self) -> float:
        return self.potential_upside_pct or 4.0


@dataclass
class CatalystItem:
    """Individual factual positive or negative catalyst identified for a company."""

    catalyst_type: str  # 'positive' | 'negative'
    category: str  # 'earnings', 'order_win', 'contract', 'regulatory', 'corporate_action', 'news', 'management'
    headline: str
    summary: str
    source: str = "NSE Filings / Financial News"
    date: Optional[str] = None
    url: Optional[str] = None
    score_impact: float = 4.0


@dataclass
class RiskItem:
    """Specific technical, market, or corporate risk factor that could invalidate the setup."""

    risk_type: str  # 'overbought', 'extended_from_ma', 'gap_exhaustion', 'weak_volume', 'downtrend', 'negative_catalyst', 'market_regime'
    description: str
    severity: str = "medium"  # 'high', 'medium', 'low'

    @property
    def category(self) -> str:
        return self.risk_type


@dataclass
class ScoreBreakdown:
    """Explainable deterministic components contributing to the opportunity score."""

    technical_momentum: float = 0.0  # Max 25
    volume_confirmation: float = 0.0  # Max 15
    market_sector_strength: float = 0.0  # Max 15
    positive_catalysts: float = 0.0  # Max 20
    financial_results: float = 0.0  # Max 15
    risk_adjustment: float = 10.0  # Base 10, reduced for risks
    total_score: float = 0.0  # Sum of components (0 to 100)

    @property
    def momentum(self) -> float:
        return self.technical_momentum

    @property
    def volume(self) -> float:
        return self.volume_confirmation

    @property
    def market_sector(self) -> float:
        return self.market_sector_strength

    @property
    def catalysts(self) -> float:
        return self.positive_catalysts

    @property
    def financials(self) -> float:
        return self.financial_results

    @property
    def risk_penalty(self) -> float:
        return self.risk_adjustment


@dataclass
class OpportunitySetup:
    """Comprehensive evaluated short-term setup candidate for an NSE equity."""

    rank: int
    symbol: str
    company_name: str
    sector: str
    current_price: float
    day_change_percent: float
    volume_ratio: float
    opportunity_score: float
    confidence_label: str  # 'High-Confidence Setup', 'Moderate Setup', 'Watch Only', 'Weak Setup'
    score_breakdown: ScoreBreakdown
    technical_signals: TechnicalSignals
    why_on_watchlist: list[str] = field(default_factory=list)
    positive_catalysts: list[CatalystItem] = field(default_factory=list)
    risks: list[RiskItem] = field(default_factory=list)
    recent_news: list[dict[str, Any]] = field(default_factory=list)
    recent_filings: list[dict[str, Any]] = field(default_factory=list)
    gap_risk_warning: Optional[str] = None

    @property
    def signals(self) -> TechnicalSignals:
        return self.technical_signals

    @property
    def setup_title(self) -> str:
        return "Potential Bullish Setup"

    @property
    def opening_gap_risk(self) -> str:
        if self.gap_risk_warning:
            return self.gap_risk_warning
        return "If the stock opens gap-up > 1.5%–2.0% above previous close, risk-reward worsens significantly. Avoid chasing extended opening spikes."

    @property
    def risk_factors(self) -> list[RiskItem]:
        return self.risks

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to standard Python dictionary."""
        return asdict(self)


@dataclass
class MarketRegime:
    """Deterministic market environment classification based on benchmark indices."""

    regime: str  # 'Bullish' | 'Neutral' | 'Bearish'
    nifty_change_1d: float = 0.0
    nifty_return_5d: float = 0.0
    nifty_above_20dma: bool = False
    nifty_above_50dma: bool = False
    nifty_current_value: Optional[float] = None
    bank_nifty_change_1d: Optional[float] = None
    it_change_1d: Optional[float] = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to standard Python dictionary."""
        return asdict(self)


@dataclass
class OpportunityScanResult:
    """Full scan output containing market context and ranked watchlist candidates."""

    generated_at: str
    market_session: str  # 'Open' | 'Closed'
    target_horizon: str  # 'Next 1–2 NSE trading sessions'
    market_regime: MarketRegime
    universe: str
    total_scanned: int
    candidates: list[OpportunitySetup] = field(default_factory=list)
    report_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to standard Python dictionary."""
        return asdict(self)
