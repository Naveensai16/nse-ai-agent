"""Unit tests for the helper utilities and symbol normalization."""

import pytest

from utils.helpers import is_valid_nse_symbol, normalize_nse_symbol


class TestNormalizeNseSymbol:
    """Test suite for normalize_nse_symbol function."""

    @pytest.mark.parametrize(
        ("input_symbol", "expected"),
        [
            ("reliance", "RELIANCE"),
            ("RELIANCE", "RELIANCE"),
            ("  infy  ", "INFY"),
            ("\ttcs\n", "TCS"),
            ("3mindia", "3MINDIA"),
            ("m&m", "M&M"),
            ("L&TFH", "L&TFH"),
            ("bajaj-auto", "BAJAJ-AUTO"),
        ],
    )
    def test_basic_normalization(self, input_symbol: str, expected: str) -> None:
        assert normalize_nse_symbol(input_symbol) == expected

    @pytest.mark.parametrize(
        ("input_symbol", "expected"),
        [
            ("NSE:RELIANCE", "RELIANCE"),
            ("nse:infy", "INFY"),
            ("NSE_TCS", "TCS"),
            ("  NSE:HDFCBANK  ", "HDFCBANK"),
        ],
    )
    def test_prefix_removal(self, input_symbol: str, expected: str) -> None:
        assert normalize_nse_symbol(input_symbol) == expected

    @pytest.mark.parametrize(
        ("input_symbol", "expected"),
        [
            ("RELIANCE.NS", "RELIANCE"),
            ("infy.ns", "INFY"),
            ("tcs.bo", "TCS"),
            ("  WIPRO.NS  ", "WIPRO"),
        ],
    )
    def test_exchange_suffix_removal(self, input_symbol: str, expected: str) -> None:
        assert normalize_nse_symbol(input_symbol) == expected

    @pytest.mark.parametrize(
        ("input_symbol", "expected"),
        [
            ("HDFCBANK-EQ", "HDFCBANK"),
            ("INFY.EQ", "INFY"),
            ("IDEA-BE", "IDEA"),
            ("TCS.BE", "TCS"),
            ("nse:hdfcbank-eq", "HDFCBANK"),
        ],
    )
    def test_series_suffix_removal(self, input_symbol: str, expected: str) -> None:
        assert normalize_nse_symbol(input_symbol) == expected

    @pytest.mark.parametrize(
        ("input_symbol", "expected"),
        [
            ("reliance", "RELIANCE.NS"),
            ("INFY.NS", "INFY.NS"),
            ("NSE:TCS", "TCS.NS"),
            ("m&m", "M&M.NS"),
        ],
    )
    def test_target_format_yahoo(self, input_symbol: str, expected: str) -> None:
        assert normalize_nse_symbol(input_symbol, target_format="yahoo") == expected

    @pytest.mark.parametrize(
        ("input_symbol", "expected"),
        [
            ("reliance", "NSE:RELIANCE"),
            ("NSE:INFY", "NSE:INFY"),
            ("tcs.ns", "NSE:TCS"),
            ("m&m", "NSE:M&M"),
        ],
    )
    def test_target_format_tradingview(self, input_symbol: str, expected: str) -> None:
        assert normalize_nse_symbol(input_symbol, target_format="tradingview") == expected

    def test_invalid_target_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid target_format"):
            normalize_nse_symbol("RELIANCE", target_format="bloomberg")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "invalid_symbol",
        [
            "",
            "   ",
            "\n\t",
            "RELIANCE$%",
            "STOCK NAME WITH SPACES",
            "!!!",
            "@",
        ],
    )
    def test_invalid_symbol_strings(self, invalid_symbol: str) -> None:
        with pytest.raises(ValueError):
            normalize_nse_symbol(invalid_symbol)

    @pytest.mark.parametrize(
        "invalid_type",
        [
            None,
            123,
            ["RELIANCE"],
            {"symbol": "RELIANCE"},
            12.34,
        ],
    )
    def test_invalid_types(self, invalid_type: object) -> None:
        with pytest.raises(TypeError, match="Symbol must be a string"):
            normalize_nse_symbol(invalid_type)  # type: ignore[arg-type]


class TestIsValidNseSymbol:
    """Test suite for is_valid_nse_symbol helper function."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "RELIANCE",
            "reliance",
            "NSE:INFY",
            "TCS.NS",
            "HDFCBANK-EQ",
            "M&M",
            "3MINDIA",
            "BAJAJ-AUTO",
        ],
    )
    def test_valid_symbols(self, symbol: str) -> None:
        assert is_valid_nse_symbol(symbol) is True

    @pytest.mark.parametrize(
        "symbol",
        [
            "",
            "   ",
            "INVALID SYMBOL",
            "$$$",
            None,
            123,
            [],
        ],
    )
    def test_invalid_symbols(self, symbol: object) -> None:
        assert is_valid_nse_symbol(symbol) is False  # type: ignore[arg-type]
