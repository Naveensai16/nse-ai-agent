"""Unit tests for the NSE symbol resolver service."""

import pytest
from services.symbol_resolver import resolve_nse_symbol, NSE_MASTER_DIRECTORY


class TestSymbolResolver:
    """Test suite for multi-level NSE symbol resolution."""

    @pytest.mark.parametrize(
        "query,expected_symbol",
        [
            # Tata Power variations
            ("TATAPOWER", "TATAPOWER"),
            ("TATAPOWER.NS", "TATAPOWER"),
            ("Tata Power", "TATAPOWER"),
            ("TataPower", "TATAPOWER"),
            ("Tata Power Company", "TATAPOWER"),
            ("Tata Power Ltd", "TATAPOWER"),
            ("Tata Power Company Limited", "TATAPOWER"),
            ("tata power co ltd", "TATAPOWER"),
            ("  Tata  Power  ", "TATAPOWER"),
            # Tata Motors variations
            ("TATAMOTORS", "TATAMOTORS"),
            ("TATAMOTORS.NS", "TATAMOTORS"),
            ("Tata Motors", "TATAMOTORS"),
            ("TataMotors", "TATAMOTORS"),
            ("Tata Motors Ltd", "TATAMOTORS"),
            ("Tata Motors Limited", "TATAMOTORS"),
            ("Tata Motors company", "TATAMOTORS"),
            ("tata motor", "TATAMOTORS"),
            # Other prominent Indian enterprises
            ("TCS", "TCS"),
            ("tcs.ns", "TCS"),
            ("Tata Consultancy Services", "TCS"),
            ("Tata Consultancy Services Ltd", "TCS"),
            ("Tata Consultancy Services Limited", "TCS"),
            ("Infosys", "INFY"),
            ("INFY", "INFY"),
            ("INFY.NS", "INFY"),
            ("Infosys Limited", "INFY"),
            ("Infosys Ltd", "INFY"),
            ("Reliance", "RELIANCE"),
            ("Reliance Industries", "RELIANCE"),
            ("Reliance Industries Ltd", "RELIANCE"),
            ("Reliance Industries Limited", "RELIANCE"),
            ("HDFC Bank", "HDFCBANK"),
            ("HDFC Bank Ltd", "HDFCBANK"),
            ("HDFCBANK", "HDFCBANK"),
            ("ICICI Bank", "ICICIBANK"),
            ("ICICI Bank Limited", "ICICIBANK"),
            ("ICICIBANK", "ICICIBANK"),
            ("State Bank of India", "SBIN"),
            ("SBI", "SBIN"),
            ("SBI Bank", "SBIN"),
            ("Wipro", "WIPRO"),
            ("Wipro Ltd", "WIPRO"),
            ("WIRPO", "WIPRO"),  # typo tolerance
            ("Zomato", "ZOMATO"),
            ("Paytm", "PAYTM"),
        ],
    )
    def test_resolve_nse_symbol_variations(self, query: str, expected_symbol: str):
        """Verify that various natural language company names, legal entity variants, and tickers resolve accurately."""
        res = resolve_nse_symbol(query, allow_online_lookup=False)
        assert res is not None
        assert res["symbol"] == expected_symbol
        assert res["yahoo_symbol"] == f"{expected_symbol}.NS"
        assert res["exchange"] == "NSE"
        assert res["confidence"] >= 0.75
        assert isinstance(res["company_name"], str) and len(res["company_name"]) > 0

    def test_structured_return_fields(self):
        """Verify the exact dictionary structure returned by resolve_nse_symbol."""
        res = resolve_nse_symbol("Tata Power")
        assert res["input"] == "Tata Power"
        assert res["symbol"] == "TATAPOWER"
        assert res["yahoo_symbol"] == "TATAPOWER.NS"
        assert "Tata Power Company" in res["company_name"]
        assert res["exchange"] == "NSE"
        assert res["confidence"] == 1.0

    @pytest.mark.parametrize(
        "invalid_query",
        [
            "",
            "   ",
            None,
            12345,
            "NonExistentStockEntity12345xyz",
            "Random gibberish company that does not exist anywhere",
        ],
    )
    def test_unresolvable_and_invalid_queries(self, invalid_query):
        """Verify graceful failure for empty, non-string, or non-existent company queries."""
        res = resolve_nse_symbol(invalid_query, allow_online_lookup=False)
        assert res["symbol"] is None
        assert res["yahoo_symbol"] is None
        assert res["confidence"] == 0.0

    def test_ambiguous_group_names_do_not_resolve_to_arbitrary_stocks(self):
        """Verify that generic group names like 'Tata' or 'Bank' do not falsely map to specific single stocks."""
        res_tata = resolve_nse_symbol("Tata", allow_online_lookup=False)
        assert res_tata["symbol"] is None

        res_bank = resolve_nse_symbol("Bank", allow_online_lookup=False)
        assert res_bank["symbol"] is None
