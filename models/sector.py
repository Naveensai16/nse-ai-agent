"""Data models for Top Sectors and Sector-Constituent Rankings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SectorPerformance:
    """Performance metrics and technical trend for an NSE market sector."""

    name: str
    display_name: str
    index_symbol: str
    performance_score: float  # Composite performance score
    change_1d: float  # 1-day percentage change
    change_1w: float  # 1-week percentage change (5 trading days)
    change_1m: float = 0.0  # 1-month percentage change (20 trading days)
    trend: str = "Neutral"  # 'Bullish' | 'Neutral' | 'Bearish'
    constituents_count: int = 0
    advance_decline_ratio: float = 1.0  # Proportion of green vs red stocks

    # Sector Narrative & Institutional Context
    why_moving: str = ""
    important_news: list[str] = field(default_factory=list)
    institutional_activity: str = ""
    policy_impact: str = ""
    major_earnings: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to standard Python dictionary."""
        return asdict(self)


@dataclass
class CompanySectorRanking:
    """Evaluated and ranked company within a specific NSE sector."""

    rank: int
    symbol: str
    company_name: str
    sector: str
    current_price: float
    day_change_percent: float
    week_change_percent: float
    month_change_percent: float = 0.0
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    volume: Optional[float] = None
    relative_volume: float = 1.0
    rsi_14: Optional[float] = None
    trend: str = "Neutral"  # 'Bullish' | 'Neutral' | 'Bearish'
    distance_from_52w_high_pct: float = 0.0  # Negative or 0 percentage from high
    latest_catalyst: str = "Normal market trading activity"
    quarterly_result_summary: str = "Quarterly results in line with sector averages"
    composite_score: float = 0.0  # 0 to 100 multi-factor ranking score
    why_in_top_10: list[str] = field(default_factory=list)

    @property
    def distance_from_52w_high(self) -> float:
        """Alias for distance from 52-week high."""
        return self.distance_from_52w_high_pct

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to standard Python dictionary."""
        return asdict(self)


@dataclass
class TopSectorsResult:
    """Comprehensive result containing Top 5 sectors and ranked companies in selected sector."""

    generated_at: str
    top_sectors: list[SectorPerformance]
    all_sectors: list[SectorPerformance]
    selected_sector: str
    top_companies: list[CompanySectorRanking]
    timeframe: str = "1 Day"  # '1 Day' | '1 Week' | '1 Month'
    sort_by: str = "Momentum"  # 'Performance' | 'Volume' | 'Market Cap' | 'RSI' | 'Momentum'

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to standard Python dictionary."""
        return asdict(self)
