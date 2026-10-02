"""Unit tests for the get_stock_price stock price tool."""

from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from tools.stock_tool import get_stock_price


class TestGetStockPrice:
    """Test suite for get_stock_price tool function."""

    def test_valid_symbol(self) -> None:
        """Test with a valid symbol and complete market data."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": 3500.50,
                "previous_close": 3450.00,
                "year_high": 4000.00,
                "year_low": 3000.00,
                "currency": "INR",
            }
            mock_instance.info = {
                "longName": "Tata Consultancy Services Limited",
            }
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("TCS")

            mock_ticker_cls.assert_called_once_with("TCS.NS")
            assert res["symbol"] == "TCS"
            assert res["yahoo_symbol"] == "TCS.NS"
            assert res["company"] == "Tata Consultancy Services Limited"
            assert res["current_price"] == 3500.50
            assert res["previous_close"] == 3450.00
            assert res["change"] == 50.50
            assert res["change_percent"] == 1.46
            assert res["52_week_high"] == 4000.00
            assert res["52_week_low"] == 3000.00
            assert res["currency"] == "INR"

    def test_lowercase_symbol(self) -> None:
        """Test normalization of lowercase symbol input."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": 2500.0,
                "previous_close": 2400.0,
            }
            mock_instance.info = {"shortName": "Reliance Industries"}
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("reliance")

            mock_ticker_cls.assert_called_once_with("RELIANCE.NS")
            assert res["symbol"] == "RELIANCE"
            assert res["yahoo_symbol"] == "RELIANCE.NS"
            assert res["current_price"] == 2500.0

    def test_ns_already_supplied(self) -> None:
        """Test input that already contains .NS extension."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": 1500.0,
                "previous_close": 1490.0,
            }
            mock_instance.info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("INFY.NS")

            mock_ticker_cls.assert_called_once_with("INFY.NS")
            assert res["symbol"] == "INFY"
            assert res["yahoo_symbol"] == "INFY.NS"
            assert res["current_price"] == 1500.0

    def test_unavailable_fast_info(self) -> None:
        """Test fallback to info when fast_info raises an exception."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            type(mock_instance).fast_info = PropertyMock(
                side_effect=Exception("FastInfo unavailable")
            )
            mock_instance.info = {
                "longName": "HDFC Bank Limited",
                "currentPrice": 1600.0,
                "previousClose": 1580.0,
                "fiftyTwoWeekHigh": 1750.0,
                "fiftyTwoWeekLow": 1380.0,
                "currency": "INR",
            }
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("HDFCBANK")

            assert res["symbol"] == "HDFCBANK"
            assert res["yahoo_symbol"] == "HDFCBANK.NS"
            assert res["company"] == "HDFC Bank Limited"
            assert res["current_price"] == 1600.0
            assert res["previous_close"] == 1580.0
            assert res["change"] == 20.0
            assert res["change_percent"] == 1.27
            assert res["52_week_high"] == 1750.0
            assert res["52_week_low"] == 1380.0
            assert res["currency"] == "INR"

    def test_missing_field(self) -> None:
        """Test handling of partially missing fields without manufacturing data."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            # Only current price is available; all other fields missing
            mock_instance.fast_info = {"last_price": 120.0}
            mock_instance.info = {}
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("TCS")

            assert res["symbol"] == "TCS"
            assert res["current_price"] == 120.0
            assert res["previous_close"] is None
            assert res["change"] is None
            assert res["change_percent"] is None
            assert res["52_week_high"] is None
            assert res["52_week_low"] is None
            assert res["company"] is None
            assert res["currency"] is None

    def test_invalid_stock(self) -> None:
        """Test with a stock symbol that normalizes but has no market data."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {}
            mock_instance.info = {}
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("NOTREALSTOCK")

            assert res["symbol"] == "NOTREALSTOCK"
            assert res["yahoo_symbol"] == "NOTREALSTOCK.NS"
            assert res["current_price"] is None
            assert res["previous_close"] is None
            assert res["change"] is None
            assert res["change_percent"] is None
            assert res["company"] is None

    @pytest.mark.parametrize(
        "invalid_input",
        ["", "   ", "INVALID$$$", None, 999],
    )
    def test_invalid_and_empty_symbol_inputs(self, invalid_input: object) -> None:
        """Test invalid or unparseable symbol inputs return safe None dictionary."""
        res = get_stock_price(invalid_input)  # type: ignore[arg-type]
        assert res["symbol"] is None
        assert res["yahoo_symbol"] is None
        assert res["current_price"] is None
        assert res["previous_close"] is None

    def test_yfinance_exception(self) -> None:
        """Test graceful recovery when yfinance raises an exception (e.g. network failure)."""
        with patch(
            "tools.stock_tool.yf.Ticker",
            side_effect=Exception("Connection timed out"),
        ):
            res = get_stock_price("TCS")

            assert res["symbol"] == "TCS"
            assert res["yahoo_symbol"] == "TCS.NS"
            assert res["current_price"] is None
            assert res["previous_close"] is None
            assert res["company"] is None

    @pytest.mark.parametrize(
        ("current", "previous", "expected_change", "expected_percent"),
        [
            (110.0, 100.0, 10.0, 10.0),
            (90.0, 100.0, -10.0, -10.0),
            (105.5, 100.0, 5.5, 5.5),
            (3450.0, 3400.0, 50.0, 1.47),
        ],
    )
    def test_percentage_calculation(
        self,
        current: float,
        previous: float,
        expected_change: float,
        expected_percent: float,
    ) -> None:
        """Test calculation of change and change_percent."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": current,
                "previous_close": previous,
            }
            mock_instance.info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("TCS")

            assert res["change"] == expected_change
            assert res["change_percent"] == expected_percent

    def test_previous_close_zero(self) -> None:
        """Test that previous_close = 0 avoids ZeroDivisionError and sets change_percent to None."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": 150.0,
                "previous_close": 0.0,
            }
            mock_instance.info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("TCS")

            assert res["current_price"] == 150.0
            assert res["previous_close"] == 0.0
            assert res["change"] == 150.0
            assert res["change_percent"] is None

    def test_nan_values(self) -> None:
        """Test that NaN values in yfinance data are converted to None."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {
                "last_price": float("nan"),
                "previous_close": float("nan"),
                "year_high": float("nan"),
                "year_low": float("nan"),
            }
            mock_instance.info = {}
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("TCS")

            assert res["current_price"] is None
            assert res["previous_close"] is None
            assert res["change"] is None
            assert res["change_percent"] is None
            assert res["52_week_high"] is None
            assert res["52_week_low"] is None

    def test_history_fallback(self) -> None:
        """Test fallback to history() when fast_info and info have no price data."""
        with patch("tools.stock_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.fast_info = {}
            mock_instance.info = {}
            df = pd.DataFrame({"Close": [3400.0, 3450.0]})
            mock_instance.history.return_value = df
            mock_ticker_cls.return_value = mock_instance

            res = get_stock_price("TCS")

            assert res["current_price"] == 3450.0
            assert res["previous_close"] == 3400.0
            assert res["change"] == 50.0
            assert res["change_percent"] == 1.47
