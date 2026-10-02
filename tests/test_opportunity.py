"""Unit tests for NSE 2-day trading opportunities scanning, technical signals, market regime, catalysts, and scoring."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

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
from services.opportunity_service import (
    CONFIDENCE_THRESHOLDS,
    WEIGHT_FINANCIAL_RESULTS,
    WEIGHT_MARKET_SECTOR_STRENGTH,
    WEIGHT_POSITIVE_CATALYSTS,
    WEIGHT_RISK_ADJUSTMENT,
    WEIGHT_TECHNICAL_MOMENTUM,
    WEIGHT_VOLUME_CONFIRMATION,
    build_candidate_opportunity,
    compute_opportunity_score,
    format_opportunity_report_markdown,
    identify_catalysts_and_risks,
    scan_short_term_opportunities,
)
from tools.opportunity_tool import get_short_term_opportunities


def _make_dummy_ohlcv(
    n_days: int = 60,
    base_price: float = 1000.0,
    trend: float = 1.0,
    volume: float = 100000.0,
    last_gap: float = 0.0,
) -> pd.DataFrame:
    """Helper to create deterministic mock OHLCV pandas DataFrame for testing."""
    dates = pd.date_range(end=datetime.date.today(), periods=n_days, freq="B")
    prices = [base_price + i * trend for i in range(n_days)]
    if last_gap != 0.0 and len(prices) > 1:
        prices[-1] = prices[-2] * (1.0 + last_gap)

    closes = np.array(prices, dtype=float)
    opens = closes * 0.995
    highs = closes * 1.01
    lows = closes * 0.99
    volumes = np.full(n_days, volume, dtype=float)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


# ==============================================================================
# 1. Universe Loading Tests
# ==============================================================================


def test_universe_loading_all_standard_universes():
    """Verify standard liquid universes are defined and contain expected constituents."""
    assert "NIFTY 50" in UNIVERSES
    assert "NIFTY NEXT 50" in UNIVERSES
    assert "NIFTY 100" in UNIVERSES
    assert "NIFTY 200" in UNIVERSES

    assert len(UNIVERSES["NIFTY 50"]) == 50
    assert len(UNIVERSES["NIFTY NEXT 50"]) == 50
    assert len(UNIVERSES["NIFTY 100"]) == 100
    assert len(UNIVERSES["NIFTY 200"]) >= 100

    # Ensure key benchmark stocks are present
    assert "TCS" in UNIVERSES["NIFTY 50"]
    assert "INFY" in UNIVERSES["NIFTY 50"]
    assert "RELIANCE" in UNIVERSES["NIFTY 50"]
    assert "HDFCBANK" in UNIVERSES["NIFTY 50"]


def test_universe_sector_mapping():
    """Verify sector mapping covers major stocks."""
    assert SECTOR_MAP.get("TCS") == "IT"
    assert SECTOR_MAP.get("HDFCBANK") == "Banking"
    assert "Energy" in SECTOR_MAP.get("RELIANCE")
    assert SECTOR_MAP.get("MARUTI") == "Automobile"


def test_unknown_universe_fallback_in_tool():
    """Verify get_short_term_opportunities falls back gracefully when given unknown universe."""
    with patch("tools.opportunity_tool.scan_short_term_opportunities") as mock_scan:
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"candidates": [], "report_markdown": "Test Report"}
        mock_result.candidates = []
        mock_scan.return_value = mock_result

        res = get_short_term_opportunities(limit=5, universe="UNKNOWN_UNIVERSE_XYZ")
        assert res["status"] == "success"
        # Check it normalized to NIFTY 200
        mock_scan.assert_called_once_with(universe="NIFTY 200", limit=5)


# ==============================================================================
# 2. Technical Indicators & Momentum Calculations
# ==============================================================================


def test_technical_signals_calculations_uptrend():
    """Verify return, moving averages, distance, RSI, and breakout on an uptrending stock."""
    df = _make_dummy_ohlcv(n_days=60, base_price=100.0, trend=2.0, volume=50000.0)
    # Double the last day's volume to test volume ratio
    df.iloc[-1, df.columns.get_loc("Volume")] = 100000.0

    sig = calculate_stock_technical_signals("TESTSTOCK", df)
    assert sig is not None
    assert sig.symbol == "TESTSTOCK"
    assert sig.current_price > 200.0
    assert sig.return_1d is not None and sig.return_1d > 0.0
    assert sig.return_5d is not None and sig.return_5d > 0.0
    assert sig.dma_20 is not None
    assert sig.dma_50 is not None
    assert sig.is_above_20dma is True
    assert sig.is_above_50dma is True
    assert sig.dist_from_20dma_pct > 0.0
    assert sig.volume_ratio > 1.2
    assert sig.rsi_14 is not None and sig.rsi_14 > 50.0
    assert sig.is_breakout is True
    assert sig.atr_14 > 0.0
    assert sig.support_level < sig.current_price
    assert sig.resistance_level > sig.current_price


def test_technical_signals_missing_volume_handling():
    """Verify zero or missing volume does not crash and defaults volume_ratio gracefully."""
    df = _make_dummy_ohlcv(n_days=40, base_price=500.0, trend=1.0, volume=0.0)
    sig = calculate_stock_technical_signals("NOVOL", df)
    assert sig is not None
    assert sig.volume_ratio == 1.0  # safe default when avg volume is 0


def test_technical_signals_empty_dataframe():
    """Verify empty or insufficient DataFrame returns None gracefully."""
    empty_df = pd.DataFrame()
    assert calculate_stock_technical_signals("EMPTY", empty_df) is None

    small_df = _make_dummy_ohlcv(n_days=5)
    assert calculate_stock_technical_signals("SMALL", small_df) is None


def test_technical_signals_gap_calculation():
    """Verify gap percentage is calculated accurately from previous close to today's open."""
    # Last day opens 3% above previous day's close
    df = _make_dummy_ohlcv(n_days=30, base_price=100.0, trend=0.5)
    prev_close = df.iloc[-2]["Close"]
    df.iloc[-1, df.columns.get_loc("Open")] = prev_close * 1.03

    sig = calculate_stock_technical_signals("GAPSTOCK", df)
    assert sig is not None
    assert abs(sig.gap_pct - 3.0) < 0.2


# ==============================================================================
# 3. Market Regime & Session Tests
# ==============================================================================


def test_market_regime_bullish():
    """Verify market regime evaluates to Bullish when Nifty is above 20 and 50 DMA with positive momentum."""
    nifty_df = _make_dummy_ohlcv(n_days=60, base_price=24000.0, trend=20.0)
    bank_df = _make_dummy_ohlcv(n_days=60, base_price=51000.0, trend=30.0)

    regime = get_market_regime(nifty_history=nifty_df, bank_nifty_history=bank_df)
    assert regime.regime == "Bullish"
    assert regime.nifty_above_20dma is True
    assert regime.nifty_above_50dma is True
    assert regime.nifty_return_5d > 0.0
    assert "Bullish" in regime.summary


def test_market_regime_bearish():
    """Verify market regime evaluates to Bearish when Nifty is below moving averages with negative momentum."""
    nifty_df = _make_dummy_ohlcv(n_days=60, base_price=25000.0, trend=-30.0)
    bank_df = _make_dummy_ohlcv(n_days=60, base_price=53000.0, trend=-40.0)

    regime = get_market_regime(nifty_history=nifty_df, bank_nifty_history=bank_df)
    assert regime.regime == "Bearish"
    assert regime.nifty_above_20dma is False
    assert regime.nifty_return_5d < 0.0


def test_market_session_status_and_target_horizon():
    """Verify market session status and horizon format."""
    sess, _, horizon = get_market_session_status()
    assert sess in ("Open", "Closed")
    assert "Next 1–2 NSE trading sessions" in horizon


# ==============================================================================
# 4. Catalyst and Risk Detection Tests
# ==============================================================================


def test_positive_catalyst_detection():
    """Verify detection of various corporate positive news patterns."""
    news = [
        {"title": "Company secures major order win worth ₹1,200 crore from Indian Railways", "source": "NSE Disclosures"},
        {"title": "Net profit jumps 45% YoY in Q2 earnings beat", "source": "BSE Filings"},
        {"title": "Board approves share buyback at ₹4,200 per share", "source": "Company PR"},
        {"title": "Gets USFDA approval for key formulation manufacturing unit", "source": "Moneycontrol"},
    ]
    signals = calculate_stock_technical_signals("TCS", _make_dummy_ohlcv(40))
    cats, risks = identify_catalysts_and_risks("TCS", signals, financials={}, news_items=news, corporate_actions=[])

    cat_types = [c.category for c in cats]
    assert "order_win" in cat_types
    assert "earnings" in cat_types
    assert "corporate_action" in cat_types
    assert "regulatory" in cat_types
    assert len(cats) >= 4


def test_negative_catalyst_and_risk_detection():
    """Verify negative news trigger risk items."""
    news = [
        {"title": "SEBI initiates probe into accounting irregularities at firm", "source": "LiveMint"},
        {"title": "Promoter entity sells 4% stake in open market block deal", "source": "Economic Times"},
    ]
    signals = calculate_stock_technical_signals("ABC", _make_dummy_ohlcv(40))
    cats, risks = identify_catalysts_and_risks("ABC", signals, financials={}, news_items=news, corporate_actions=[])

    risk_cats = [r.category for r in risks]
    assert "regulatory_probe" in risk_cats or "promoter_selling" in risk_cats or "negative_catalyst" in risk_cats


def test_quarterly_financials_positive_catalyst():
    """Verify strong YoY profit growth in quarterly financials adds a positive catalyst."""
    fin = {
        "latest_quarter": {
            "net_profit": 500.0,
            "net_profit_yoy": 28.5,
            "revenue": 3000.0,
            "revenue_yoy": 18.2,
            "operating_margin": 24.0,
        }
    }
    signals = calculate_stock_technical_signals("TCS", _make_dummy_ohlcv(40))
    cats, _ = identify_catalysts_and_risks("TCS", signals, financials=fin, news_items=[], corporate_actions=[])
    fin_cats = [c for c in cats if c.category == "strong_financials"]
    assert len(fin_cats) >= 1
    assert any("28.5%" in c.headline for c in fin_cats)


def test_news_deduplication():
    """Verify identical or duplicate news stories are deduplicated in catalyst identification."""
    news = [
        {"title": "Secures order worth ₹500 crore from defense ministry", "source": "Moneycontrol"},
        {"title": "Secures order worth ₹500 crore from defense ministry", "source": "Economic Times"},
    ]
    signals = calculate_stock_technical_signals("BEL", _make_dummy_ohlcv(40))
    cats, _ = identify_catalysts_and_risks("BEL", signals, financials={}, news_items=news, corporate_actions=[])
    order_cats = [c for c in cats if c.category == "order_win"]
    assert len(order_cats) == 1  # Deduplicated!


# ==============================================================================
# 5. Scoring, Weighting, and Risk Penalties Tests
# ==============================================================================


def test_scoring_within_bounds_and_breakdown():
    """Verify score breakdown sums correctly and is bounded between 0 and 100."""
    sig = calculate_stock_technical_signals("STOCK1", _make_dummy_ohlcv(40, trend=1.5))
    cats = [CatalystItem(headline="Order win", category="order_win", source="NSE", score_impact=8.0, summary="Order win", catalyst_type="positive")]
    risks = []
    fin = {"latest_quarter": {"net_profit_yoy": 20.0}}
    market_regime = MarketRegime(regime="Bullish", nifty_above_20dma=True, nifty_above_50dma=True)

    score, breakdown, conf = compute_opportunity_score(sig, market_regime, cats, risks, fin, sector="IT")

    assert 0.0 <= score <= 100.0
    assert breakdown.momentum <= WEIGHT_TECHNICAL_MOMENTUM
    assert breakdown.volume <= WEIGHT_VOLUME_CONFIRMATION
    assert breakdown.market_sector <= WEIGHT_MARKET_SECTOR_STRENGTH
    assert breakdown.catalysts <= WEIGHT_POSITIVE_CATALYSTS
    assert breakdown.financials <= WEIGHT_FINANCIAL_RESULTS
    assert breakdown.risk_penalty <= WEIGHT_RISK_ADJUSTMENT
    assert conf in [label for _, label in CONFIDENCE_THRESHOLDS]


def test_overbought_rsi_penalty():
    """Verify high RSI (>75) incurs a risk penalty."""
    # Create extreme uptrend to pump RSI > 80
    df = _make_dummy_ohlcv(40, base_price=100.0, trend=8.0)
    sig = calculate_stock_technical_signals("OVERBOUGHT", df)
    assert sig.rsi_14 > 75.0

    cats, risks = identify_catalysts_and_risks("OVERBOUGHT", sig, financials={}, news_items=[], corporate_actions=[])
    rsi_risks = [r for r in risks if r.category == "overbought_rsi"]
    assert len(rsi_risks) == 1
    assert rsi_risks[0].severity in ("medium", "high")


def test_overextended_20dma_penalty():
    """Verify stock extended >8% above 20 DMA incurs a risk penalty."""
    df = _make_dummy_ohlcv(40, base_price=100.0, trend=0.5)
    # Spike the last day 15% above
    df.iloc[-1, df.columns.get_loc("Close")] = df.iloc[-2]["Close"] * 1.15
    sig = calculate_stock_technical_signals("EXTENDED", df)
    assert sig.dist_from_20dma_pct > 8.0

    _, risks = identify_catalysts_and_risks("EXTENDED", sig, financials={}, news_items=[], corporate_actions=[])
    ext_risks = [r for r in risks if r.category == "overextended"]
    assert len(ext_risks) == 1


# ==============================================================================
# 6. Candidate Building & Strict Compliance Tests
# ==============================================================================


def test_build_candidate_opportunity_and_gap_risk_warning():
    """Verify candidate opportunity object construction, why points, and gap warning."""
    sig = calculate_stock_technical_signals("RELIANCE", _make_dummy_ohlcv(40, trend=2.0))
    market_regime = MarketRegime(regime="Bullish", nifty_above_20dma=True, nifty_above_50dma=True)

    cand = build_candidate_opportunity(
        rank=1,
        symbol="RELIANCE",
        signals=sig,
        market_regime=market_regime,
        financials={},
        news_items=[{"title": "Reliance commissions new solar unit", "source": "Reuters"}],
        corporate_actions=[],
    )

    assert cand.symbol == "RELIANCE"
    assert cand.company_name == "Reliance Industries Limited"
    assert "Energy" in cand.sector
    assert len(cand.why_on_watchlist) >= 1
    assert "Opening Gap Risk" in cand.opening_gap_risk or "gap-up" in cand.opening_gap_risk.lower()
    assert cand.signals.risk_reward_ratio > 0.0


def test_no_guaranteed_profit_language_compliance():
    """CRITICAL COMPLIANCE TEST: Ensure forbidden words are NEVER present in generated reports or setups."""
    sig = calculate_stock_technical_signals("TCS", _make_dummy_ohlcv(40, trend=1.5))
    market_regime = MarketRegime(regime="Bullish", nifty_above_20dma=True, nifty_above_50dma=True)

    cand = build_candidate_opportunity(
        rank=1,
        symbol="TCS",
        signals=sig,
        market_regime=market_regime,
        financials={},
        news_items=[],
        corporate_actions=[],
    )

    scan_res = OpportunityScanResult(
        generated_at="2026-10-02 15:30:00 IST",
        market_session="Closed",
        target_horizon="Next 1–2 NSE trading sessions (Mon, 05 Oct – Tue, 06 Oct 2026)",
        market_regime=market_regime,
        universe="NIFTY 50",
        total_scanned=50,
        candidates=[cand],
    )
    report_md = format_opportunity_report_markdown(scan_res)

    forbidden_phrases = [
        "guaranteed profit",
        "guaranteed return",
        "will definitely rise",
        "sure buy",
        "100% buy",
        "tomorrow winner",
        "expected profit",
        "guaranteed gain",
        "sure profit",
    ]

    report_lower = report_md.lower()
    for phrase in forbidden_phrases:
        assert phrase not in report_lower, f"Forbidden phrase '{phrase}' found in opportunity report markdown!"

    # Ensure required compliant phrases are present
    assert (
        "potential bullish setup" in report_lower
        or "watchlist candidate" in report_lower
        or "momentum setup" in report_lower
        or "higher-confidence" in report_lower
    )


# ==============================================================================
# 7. End-to-End Deterministic Scanner Test
# ==============================================================================


def test_scan_short_term_opportunities_offline_providers():
    """Verify scan_short_term_opportunities executes end-to-end with injected offline providers."""

    def mock_history_provider(sym: str) -> pd.DataFrame:
        trend = 3.0 if sym in ("TCS", "INFY") else 0.5
        return _make_dummy_ohlcv(45, base_price=2000.0, trend=trend)

    def mock_news_provider(sym: str) -> list[dict]:
        return [
            {
                "title": f"{sym} bags $100M cloud migration deal",
                "source": "NSE",
                "url": f"https://example.com/news/{sym.lower()}",
                "summary": f"{sym} announced an order win.",
            }
        ]

    def mock_fin_provider(sym: str) -> dict:
        return {"latest_quarter": {"net_profit_yoy": 15.0, "revenue_yoy": 12.0}}

    result = scan_short_term_opportunities(
        universe="NIFTY 50",
        limit=3,
        history_provider=mock_history_provider,
        news_provider=mock_news_provider,
        financials_provider=mock_fin_provider,
    )

    assert isinstance(result, OpportunityScanResult)
    assert result.universe == "NIFTY 50"
    assert len(result.candidates) == 3
    assert result.candidates[0].rank == 1
    assert result.candidates[1].rank == 2
    assert result.candidates[2].rank == 3
    assert result.candidates[0].opportunity_score >= result.candidates[1].opportunity_score >= result.candidates[2].opportunity_score
    assert len(result.report_markdown) > 100
