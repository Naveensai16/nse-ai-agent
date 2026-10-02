"""Unit tests for the get_company_info company information tool."""

from unittest.mock import MagicMock, patch

import pytest

from tools.company_tool import get_company_info


class TestGetCompanyInfo:
    """Test suite for get_company_info tool function."""

    def test_valid_company(self) -> None:
        """Test with a valid company returning complete fundamental data."""
        with patch("tools.company_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.info = {
                "longName": "Tata Consultancy Services Limited",
                "sector": "Technology",
                "industry": "Information Technology Services",
                "marketCap": 14500000000000,
                "trailingPE": 30.25,
                "forwardPE": 27.50,
                "trailingEps": 115.40,
                "dividendYield": 0.0142,
                "fiftyTwoWeekHigh": 4200.00,
                "fiftyTwoWeekLow": 3100.00,
                "website": "https://www.tcs.com",
                "longBusinessSummary": "Tata Consultancy Services Limited provides IT and consulting services.",
            }
            mock_instance.fast_info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_company_info("TCS")

            mock_ticker_cls.assert_called_once_with("TCS.NS")
            assert res["symbol"] == "TCS"
            assert res["company"] == "Tata Consultancy Services Limited"
            assert res["sector"] == "Technology"
            assert res["industry"] == "Information Technology Services"
            assert res["market_cap"] == 14500000000000
            assert res["trailing_pe"] == 30.25
            assert res["forward_pe"] == 27.50
            assert res["eps"] == 115.40
            assert res["dividend_yield"] == 0.0142
            assert res["52_week_high"] == 4200.00
            assert res["52_week_low"] == 3100.00
            assert res["website"] == "https://www.tcs.com"
            assert (
                res["business_summary"]
                == "Tata Consultancy Services Limited provides IT and consulting services."
            )

    def test_missing_pe(self) -> None:
        """Test company information when trailing and forward PE ratios are missing."""
        with patch("tools.company_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.info = {
                "longName": "Zomato Limited",
                "sector": "Consumer Cyclical",
                "industry": "Internet Retail",
                "trailingPE": None,
                "forwardPE": None,
                "marketCap": 2000000000000,
            }
            mock_instance.fast_info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_company_info("ZOMATO")

            assert res["symbol"] == "ZOMATO"
            assert res["company"] == "Zomato Limited"
            assert res["trailing_pe"] is None
            assert res["forward_pe"] is None
            assert res["market_cap"] == 2000000000000

    def test_missing_dividend(self) -> None:
        """Test company information when dividend yield is missing."""
        with patch("tools.company_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.info = {
                "longName": "Adani Enterprises Limited",
                "sector": "Energy",
                "dividendYield": None,
            }
            mock_instance.fast_info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_company_info("ADANIENT")

            assert res["symbol"] == "ADANIENT"
            assert res["company"] == "Adani Enterprises Limited"
            assert res["dividend_yield"] is None

    def test_missing_sector(self) -> None:
        """Test company information when sector and industry are missing."""
        with patch("tools.company_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.info = {
                "longName": "Unknown Entity Limited",
                "sector": None,
                "industry": None,
            }
            mock_instance.fast_info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_company_info("RELIANCE")

            assert res["symbol"] == "RELIANCE"
            assert res["company"] == "Unknown Entity Limited"
            assert res["sector"] is None
            assert res["industry"] is None

    def test_missing_info_dictionary(self) -> None:
        """Test behavior when ticker.info raises an exception or returns non-dict."""
        with patch("tools.company_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.info = None
            mock_instance.fast_info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_company_info("INFY")

            assert res["symbol"] == "INFY"
            assert res["company"] is None
            assert res["sector"] is None
            assert res["industry"] is None
            assert res["market_cap"] is None
            assert res["trailing_pe"] is None
            assert res["forward_pe"] is None
            assert res["eps"] is None
            assert res["dividend_yield"] is None
            assert res["52_week_high"] is None
            assert res["52_week_low"] is None
            assert res["website"] is None
            assert res["business_summary"] is None

    @pytest.mark.parametrize(
        "invalid_symbol",
        ["", "   ", "INVALID$$$", None, 12345],
    )
    def test_invalid_symbol(self, invalid_symbol: object) -> None:
        """Test invalid symbol inputs return dictionary with all None fields."""
        res = get_company_info(invalid_symbol)  # type: ignore[arg-type]

        assert res["symbol"] is None
        assert res["company"] is None
        assert res["sector"] is None
        assert res["market_cap"] is None

    def test_yfinance_exception(self) -> None:
        """Test graceful error handling when yfinance.Ticker raises an exception."""
        with patch(
            "tools.company_tool.yf.Ticker",
            side_effect=Exception("API connection timeout"),
        ):
            res = get_company_info("TCS")

            assert res["symbol"] == "TCS"
            assert res["company"] is None
            assert res["sector"] is None
            assert res["market_cap"] is None

    def test_lowercase_symbol_normalization(self) -> None:
        """Test lowercase input symbol is properly normalized to uppercase and Yahoo ticker."""
        with patch("tools.company_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.info = {"longName": "Infosys Limited"}
            mock_instance.fast_info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_company_info("infy")

            mock_ticker_cls.assert_called_once_with("INFY.NS")
            assert res["symbol"] == "INFY"
            assert res["company"] == "Infosys Limited"

    def test_nan_values_in_info(self) -> None:
        """Test that NaN values in info fields are sanitized to None."""
        with patch("tools.company_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.info = {
                "trailingPE": float("nan"),
                "forwardPE": float("nan"),
                "dividendYield": float("nan"),
                "marketCap": float("nan"),
            }
            mock_instance.fast_info = {}
            mock_ticker_cls.return_value = mock_instance

            res = get_company_info("TCS")

            assert res["trailing_pe"] is None
            assert res["forward_pe"] is None
            assert res["dividend_yield"] is None
            assert res["market_cap"] is None

    def test_fast_info_fallback(self) -> None:
        """Test fallback to fast_info for market_cap and 52-week high/low when info lacks them."""
        with patch("tools.company_tool.yf.Ticker") as mock_ticker_cls:
            mock_instance = MagicMock()
            mock_instance.info = {
                "longName": "HDFC Bank Limited",
            }
            mock_instance.fast_info = {
                "market_cap": 12000000000000,
                "year_high": 1750.0,
                "year_low": 1380.0,
            }
            mock_ticker_cls.return_value = mock_instance

            res = get_company_info("HDFCBANK")

            assert res["symbol"] == "HDFCBANK"
            assert res["company"] == "HDFC Bank Limited"
            assert res["market_cap"] == 12000000000000
            assert res["52_week_high"] == 1750.0
            assert res["52_week_low"] == 1380.0
