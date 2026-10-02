"""Unit tests for the market index and historical price retrieval tools."""

from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from tools.market_tool import (
    INDEX_MAPPING,
    SUPPORTED_PERIODS,
    PriceHistory,
    get_market_index,
    get_price_history,
    resolve_index,
)


class TestIndexMappingAndResolution:
    """Test suite for the centralized index mapping."""

    @pytest.mark.parametrize(
        ("input_alias", "expected_ticker", "expected_name"),
        [
            ("NIFTY", "^NSEI", "NIFTY 50"),
            ("NIFTY50", "^NSEI", "NIFTY 50"),
            ("NIFTY 50", "^NSEI", "NIFTY 50"),
            ("nifty", "^NSEI", "NIFTY 50"),
            ("nifty 50", "^NSEI", "NIFTY 50"),
            ("BANKNIFTY", "^NSEBANK", "BANK NIFTY"),
            ("BANK NIFTY", "^NSEBANK", "BANK NIFTY"),
            ("bank nifty", "^NSEBANK", "BANK NIFTY"),
            ("SENSEX", "^BSESN", "SENSEX"),
            ("sensex", "^BSESN", "SENSEX"),
            ("BSE SENSEX", "^BSESN", "SENSEX"),
            ("NIFTY IT", "^CNXIT", "NIFTY IT"),
            ("NIFTYIT", "^CNXIT", "NIFTY IT"),
            ("nifty it", "^CNXIT", "NIFTY IT"),
            ("^NSEI", "^NSEI", "NIFTY 50"),
        ],
    )
    def test_resolve_index_aliases(
        self, input_alias: str, expected_ticker: str, expected_name: str
    ) -> None:
        resolved = resolve_index(input_alias)
        assert resolved is not None
        assert resolved["ticker"] == expected_ticker
        assert resolved["name"] == expected_name

    def test_centralized_mapping_keys(self) -> None:
        """Verify centralized index mapping covers all required indices."""
        required_indices = ["NIFTY", "NIFTY50", "NIFTY 50", "BANKNIFTY", "BANK NIFTY", "SENSEX", "NIFTY IT"]
        for key in required_indices:
            assert key in INDEX_MAPPING


class TestGetMarketIndex:
    """Test suite for get_market_index function."""

    def test_valid_index_quote(self) -> None:
        """Test complete quote retrieval for NIFTY 50."""
        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": 24850.50,
                "previous_close": 24700.00,
            }
            mock_instance.info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_market_index("NIFTY 50")

            mock_ticker_cls.assert_called_once_with("^NSEI")
            assert res["index"] == "NIFTY 50"
            assert res["ticker"] == "^NSEI"
            assert res["current_value"] == 24850.50
            assert res["previous_close"] == 24700.00
            assert res["change"] == 150.50
            assert res["change_percent"] == 0.61

    @pytest.mark.parametrize(
        ("input_alias", "expected_ticker", "expected_name"),
        [
            ("NIFTY", "^NSEI", "NIFTY 50"),
            ("BANKNIFTY", "^NSEBANK", "BANK NIFTY"),
            ("SENSEX", "^BSESN", "SENSEX"),
            ("NIFTY IT", "^CNXIT", "NIFTY IT"),
        ],
    )
    def test_all_supported_indices(
        self, input_alias: str, expected_ticker: str, expected_name: str
    ) -> None:
        """Test that all required index types are queried correctly."""
        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": 50000.0,
                "previous_close": 49500.0,
            }
            mock_instance.info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_market_index(input_alias)

            mock_ticker_cls.assert_called_once_with(expected_ticker)
            assert res["index"] == expected_name
            assert res["ticker"] == expected_ticker
            assert res["current_value"] == 50000.0
            assert res["previous_close"] == 49500.0

    @pytest.mark.parametrize(
        "unsupported_input",
        ["UNKNOWN_INDEX", "DOWJONES", "", "   ", None, 12345],
    )
    def test_unsupported_index(self, unsupported_input: object) -> None:
        """Test unsupported or invalid index queries return None safely."""
        res = get_market_index(unsupported_input)  # type: ignore[arg-type]
        assert res["index"] is None
        assert res["ticker"] is None
        assert res["current_value"] is None
        assert res["previous_close"] is None
        assert res["change"] is None
        assert res["change_percent"] is None

    def test_missing_price(self) -> None:
        """Test missing current price results in None for change metrics."""
        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {"previous_close": 24000.0}
            mock_instance.info = {}
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_instance

            res = get_market_index("NIFTY")

            assert res["index"] == "NIFTY 50"
            assert res["current_value"] is None
            assert res["previous_close"] == 24000.0
            assert res["change"] is None
            assert res["change_percent"] is None

    def test_missing_previous_close(self) -> None:
        """Test missing previous close results in None for change metrics."""
        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {"last_price": 24500.0}
            mock_instance.info = {}
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_instance

            res = get_market_index("NIFTY")

            assert res["index"] == "NIFTY 50"
            assert res["current_value"] == 24500.0
            assert res["previous_close"] is None
            assert res["change"] is None
            assert res["change_percent"] is None

    def test_previous_close_zero(self) -> None:
        """Test previous_close = 0 safely sets change_percent to None."""
        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": 25000.0,
                "previous_close": 0.0,
            }
            mock_instance.info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_market_index("NIFTY")

            assert res["current_value"] == 25000.0
            assert res["previous_close"] == 0.0
            assert res["change"] == 25000.0
            assert res["change_percent"] is None

    def test_history_fallback_for_index(self) -> None:
        """Test fallback to history(period='5d') when fast_info lacks price data."""
        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {}
            mock_instance.info = {}
            df = pd.DataFrame({"Close": [24500.0, 24650.0]})
            mock_instance.history.return_value = df
            mock_ticker_cls.return_value = mock_instance

            res = get_market_index("NIFTY")

            assert res["current_value"] == 24650.0
            assert res["previous_close"] == 24500.0
            assert res["change"] == 150.0
            assert res["change_percent"] == 0.61

    def test_network_and_provider_errors(self) -> None:
        """Test graceful handling of network or yfinance exceptions."""
        with patch(
            "tools.market_tool.yf.Ticker",
            side_effect=Exception("Failed to reach Yahoo Finance"),
        ):
            res = get_market_index("NIFTY")

            assert res["index"] == "NIFTY 50"
            assert res["ticker"] == "^NSEI"
            assert res["current_value"] is None
            assert res["previous_close"] is None


class TestGetPriceHistory:
    """Test suite for get_price_history function."""

    @pytest.mark.parametrize("period", SUPPORTED_PERIODS)
    def test_supported_periods(self, period: str) -> None:
        """Verify all supported periods are accepted and passed to yfinance."""
        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_instance

            res = get_price_history("TCS", period=period)

            mock_instance.history.assert_called_once_with(period=period)
            assert isinstance(res, PriceHistory)
            assert res.period == period

    def test_unsupported_period_raises_value_error(self) -> None:
        """Verify unsupported periods raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported period"):
            get_price_history("TCS", period="10y")

    def test_equity_price_history(self) -> None:
        """Test successful parsing of historical OHLCV data for an equity."""
        dates = pd.date_range("2026-09-01", periods=3, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [3400.0, 3450.0, 3480.0],
                "High": [3460.0, 3490.0, 3510.0],
                "Low": [3390.0, 3440.0, 3470.0],
                "Close": [3450.0, 3480.0, 3500.0],
                "Volume": [1000000, 1200000, 1150000],
            },
            index=dates,
        )

        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_instance

            res = get_price_history("tcs", period="1mo")

            mock_ticker_cls.assert_called_once_with("TCS.NS")
            assert len(res) == 3
            assert res.symbol == "TCS"
            assert res.yahoo_symbol == "TCS.NS"
            assert res.period == "1mo"

            # Check individual bar structure
            first_bar = res[0]
            assert first_bar["date"] == "2026-09-01"
            assert first_bar["open"] == 3400.0
            assert first_bar["high"] == 3460.0
            assert first_bar["low"] == 3390.0
            assert first_bar["close"] == 3450.0
            assert first_bar["volume"] == 1000000

    def test_index_price_history(self) -> None:
        """Test price history retrieval when passed an index alias."""
        dates = pd.date_range("2026-09-01", periods=2, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [24500.0, 24600.0],
                "High": [24650.0, 24700.0],
                "Low": [24480.0, 24550.0],
                "Close": [24600.0, 24680.0],
                "Volume": [0, 0],
            },
            index=dates,
        )

        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.history.return_value = mock_df
            mock_ticker_cls.return_value = mock_instance

            res = get_price_history("NIFTY 50", period="5d")

            mock_ticker_cls.assert_called_once_with("^NSEI")
            assert len(res) == 2
            assert res.symbol == "NIFTY 50"
            assert res.yahoo_symbol == "^NSEI"
            assert res[0]["close"] == 24600.0

    def test_empty_history(self) -> None:
        """Test handling of empty historical DataFrame."""
        with patch("tools.market_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_instance

            res = get_price_history("TCS", period="1mo")

            assert isinstance(res, PriceHistory)
            assert len(res) == 0
            assert res.symbol == "TCS"

    def test_network_or_provider_error(self) -> None:
        """Test handling when yfinance history raises an exception."""
        with patch(
            "tools.market_tool.yf.Ticker",
            side_effect=Exception("Network error on history"),
        ):
            res = get_price_history("INFY", period="1mo")

            assert isinstance(res, PriceHistory)
            assert len(res) == 0

    def test_unsupported_symbol(self) -> None:
        """Test invalid symbol input returns empty PriceHistory without raising."""
        res = get_price_history("INVALID$$$", period="1mo")
        assert isinstance(res, PriceHistory)
        assert len(res) == 0
        assert res.symbol is None

    def test_price_history_container_features(self) -> None:
        """Test PriceHistory list, dict-like indexing, and to_dict() conversions."""
        records = [
            {"date": "2026-09-01", "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 500}
        ]
        history = PriceHistory(records, symbol="TEST", yahoo_symbol="TEST.NS", period="1mo")

        # List behavior
        assert len(history) == 1
        assert history[0]["close"] == 105.0
        assert [r["date"] for r in history] == ["2026-09-01"]

        # Attribute access
        assert history.symbol == "TEST"
        assert history.yahoo_symbol == "TEST.NS"
        assert history.period == "1mo"

        # Dict-like access
        assert history["symbol"] == "TEST"
        assert history["period"] == "1mo"
        assert len(history["data"]) == 1
        assert history["count"] == 1

        # Dict conversion
        d = history.to_dict()
        assert isinstance(d, dict)
        assert d["symbol"] == "TEST"
        assert d["count"] == 1


class TestGainersAndLosers:
    """Test suite for get_top_gainers and get_top_losers with isolated provider."""

    def test_top_gainers_and_losers_ranking(self) -> None:
        """Test that top gainers are sorted descending and losers ascending by percentage change."""
        from tools.market_tool import GainersLosersProvider, get_top_gainers, get_top_losers

        mock_provider = GainersLosersProvider(symbols=("TCS", "INFY", "RELIANCE", "WIPRO"))

        mock_movers_gainers = [
            {"symbol": "TCS", "price": 3600.0, "change": 100.0, "change_percent": 2.86},
            {"symbol": "INFY", "price": 1500.0, "change": 25.0, "change_percent": 1.69},
        ]
        mock_movers_losers = [
            {"symbol": "WIPRO", "price": 450.0, "change": -15.0, "change_percent": -3.23},
            {"symbol": "RELIANCE", "price": 2800.0, "change": -20.0, "change_percent": -0.71},
        ]

        with patch.object(
            mock_provider,
            "fetch_market_movers",
            return_value=(mock_movers_gainers, mock_movers_losers),
        ):
            gainers = get_top_gainers(limit=2, provider=mock_provider)
            losers = get_top_losers(limit=2, provider=mock_provider)

            assert len(gainers) == 2
            assert gainers[0]["symbol"] == "TCS"
            assert gainers[0]["change_percent"] == 2.86
            assert gainers[1]["symbol"] == "INFY"

            assert len(losers) == 2
            assert losers[0]["symbol"] == "WIPRO"
            assert losers[0]["change_percent"] == -3.23
            assert losers[1]["symbol"] == "RELIANCE"

    def test_provider_error_handling(self) -> None:
        """Test graceful return of empty list if provider encounters failure."""
        from tools.market_tool import GainersLosersProvider, get_top_gainers, get_top_losers

        mock_provider = GainersLosersProvider()
        with patch.object(
            mock_provider,
            "fetch_market_movers",
            side_effect=Exception("Failed to scan movers"),
        ):
            assert get_top_gainers(provider=mock_provider) == []
            assert get_top_losers(provider=mock_provider) == []

