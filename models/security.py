"""Security entity data models for Indian listed equities (NSE / BSE)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SecurityEntity:
    """Authoritative representation of an identified Indian listed security."""

    symbol: Optional[str] = None  # Primary NSE ticker, e.g. 'YESBANK'
    company_name: Optional[str] = None  # Official company name, e.g. 'Yes Bank Limited'
    yahoo_symbol: Optional[str] = None  # Yahoo Finance ticker, e.g. 'YESBANK.NS'
    isin: Optional[str] = None  # ISIN code, e.g. 'INE528G01035'
    sector: Optional[str] = None  # e.g. 'Financial Services'
    industry: Optional[str] = None  # e.g. 'Private Sector Bank'
    bse_code: Optional[str] = None  # e.g. '532648'
    exchange: str = "NSE"
    aliases: list[str] = field(default_factory=list)
    confidence: float = 1.0
    match_type: str = "exact_symbol"  # 'exact_symbol' | 'exact_company_name' | 'exact_alias' | 'normalized_match' | 'token_match' | 'fuzzy_alias' | 'online_search'
    is_ambiguous: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.symbol:
            self.symbol = self.symbol.upper().strip()
            if not self.yahoo_symbol:
                self.yahoo_symbol = f"{self.symbol}.NS"
        if not self.company_name and self.symbol:
            self.company_name = self.symbol

    @property
    def is_resolved(self) -> bool:
        """True if the security was unambiguously identified."""
        return bool(self.symbol and not self.is_ambiguous and not self.error)

    def to_dict(self) -> dict[str, Any]:
        """Convert security entity to dictionary format matching resolve_nse_symbol return schema."""
        return {
            "input": getattr(self, "_input_query", self.symbol),
            "symbol": self.symbol if not self.is_ambiguous and not self.error else (self.symbol or None),
            "yahoo_symbol": self.yahoo_symbol if not self.is_ambiguous and not self.error else (self.yahoo_symbol or None),
            "company_name": self.company_name if not self.is_ambiguous and not self.error else (self.company_name or None),
            "isin": self.isin,
            "sector": self.sector,
            "industry": self.industry,
            "bse_code": self.bse_code,
            "exchange": self.exchange,
            "confidence": self.confidence,
            "match_type": self.match_type,
            "is_ambiguous": self.is_ambiguous,
            "candidates": self.candidates,
            "error": self.error,
        }

    def format_banner(self) -> str:
        """Format an identification header banner for display in chat responses and UI."""
        if not self.company_name or not self.symbol:
            return ""

        lines = [f"🏢 **{self.company_name}**"]
        
        identifiers = [f"**NSE:** `{self.symbol}`"]
        if self.bse_code:
            identifiers.append(f"**BSE:** `{self.bse_code}`")
        if self.isin:
            identifiers.append(f"**ISIN:** `{self.isin}`")
        lines.append(" | ".join(identifiers))

        sec_ind = []
        if self.sector:
            sec_ind.append(f"**Sector:** {self.sector}")
        if self.industry:
            sec_ind.append(f"**Industry:** {self.industry}")
        if sec_ind:
            lines.append(" | ".join(sec_ind))

        return "\n".join(lines)
