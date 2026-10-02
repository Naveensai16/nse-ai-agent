"""Comprehensive tests for NSE Stock Master, Entity Resolution, and False-Positive Prevention.

Validates that:
1. "Tell me about YES Bank" and "Should I hold YES Bank stock for 1 year?" NEVER resolve to SBIN.
2. Similar-sounding banks and PSUs (Indian Bank, Bank of India, Central Bank of India, SBI, Yes Bank)
   resolve strictly to their canonical entities and never cross-contaminate.
3. Corporate parent-child and conglomerate distinctions (Tata Power vs Power Grid, Tata Motors vs
   Tata Technologies, India Cements vs UltraTech Cement, HDFC Bank vs HDFC AMC vs HDFC Life).
4. Conglomerate group queries (Tata, Adani, Birla, Bajaj) trigger ambiguity flags with candidate options.
5. Generic financial words (Bank, Power, Steel, Cement, Finance, India) are suppressed.
6. Canonical identification banners render correct symbols, BSE codes, ISINs, and sector/industry.
7. Autocomplete search (search_stocks) accurately finds matching equities.
"""

from __future__ import annotations

import pytest

from agent.demo_agent import extract_symbols_from_query, run_demo_agent
from data.stock_master import STOCK_MASTER, get_stock_by_symbol, search_stock_master
from models.security import SecurityEntity
from services.symbol_resolver import (
    check_conglomerate_ambiguity,
    get_security_entity,
    is_generic_query,
    resolve_nse_symbol,
    search_stocks,
)


class TestYesBankVsSbiDistinction:
    """Verify the critical bug fix: YES Bank must never resolve to State Bank of India (SBIN)."""

    @pytest.mark.parametrize(
        "query",
        [
            "YESBANK",
            "YESBANK.NS",
            "Yes Bank",
            "YES Bank",
            "yes bank",
            "YesBank",
            "Yes Bank Ltd",
            "Yes Bank Limited",
            "YES Bank Limited",
            "Tell me about YES Bank",
            "tell me about yes bank",
            "Should I hold YES Bank stock for 1 year?",
            "Is YES Bank a good buy?",
            "What is the target for Yes Bank?",
        ],
    )
    def test_yes_bank_resolves_to_yesbank(self, query: str):
        # 1. Test direct symbol resolver
        res = resolve_nse_symbol(query, allow_online_lookup=False)
        assert res["symbol"] == "YESBANK", f"Query '{query}' resolved to {res['symbol']} instead of YESBANK"
        assert res["yahoo_symbol"] == "YESBANK.NS"
        assert "Yes Bank" in res["company_name"]
        assert res["symbol"] != "SBIN", f"CRITICAL BUG: Query '{query}' resolved to SBIN!"

        # 2. Test extraction from natural language query
        extracted = extract_symbols_from_query(query)
        assert "YESBANK" in extracted, f"Failed to extract YESBANK from query '{query}': got {extracted}"
        assert "SBIN" not in extracted, f"CRITICAL: SBIN extracted from query '{query}'!"

    @pytest.mark.parametrize(
        "query",
        [
            "SBIN",
            "SBIN.NS",
            "SBI",
            "sbi",
            "SBI Bank",
            "State Bank",
            "State Bank of India",
            "State Bank of India Ltd",
            "Tell me about State Bank of India",
            "Tell me about SBI",
        ],
    )
    def test_sbi_resolves_to_sbin(self, query: str):
        res = resolve_nse_symbol(query, allow_online_lookup=False)
        assert res["symbol"] == "SBIN", f"Query '{query}' resolved to {res['symbol']} instead of SBIN"
        assert res["yahoo_symbol"] == "SBIN.NS"
        assert "State Bank of India" in res["company_name"]
        assert res["symbol"] != "YESBANK"

        extracted = extract_symbols_from_query(query)
        assert "SBIN" in extracted
        assert "YESBANK" not in extracted


class TestBankingEntitiesPrecision:
    """Verify distinct Indian banking entities never cross-contaminate."""

    @pytest.mark.parametrize(
        "query,expected_symbol,forbidden_symbol",
        [
            ("Indian Bank", "INDIANB", "BANKINDIA"),
            ("Bank of India", "BANKINDIA", "INDIANB"),
            ("Central Bank of India", "CENTRALBK", "SBIN"),
            ("Central Bank", "CENTRALBK", "SBIN"),
            ("Union Bank of India", "UNIONBANK", "BANKINDIA"),
            ("Union Bank", "UNIONBANK", "BANKINDIA"),
            ("Bank of Baroda", "BANKBARODA", "BANKINDIA"),
            ("BOB", "BANKBARODA", "BANKINDIA"),
            ("Canara Bank", "CANBK", "INDIANB"),
            ("Punjab National Bank", "PNB", "SBIN"),
            ("PNB", "PNB", "SBIN"),
            ("UCO Bank", "UCOBANK", "UNIONBANK"),
            ("Indian Overseas Bank", "IOB", "INDIANB"),
            ("IOB", "IOB", "INDIANB"),
            ("IDFC First Bank", "IDFCFIRSTB", "HDFCBANK"),
            ("IDFC First", "IDFCFIRSTB", "HDFCBANK"),
            ("IndusInd Bank", "INDUSINDBK", "INDIANB"),
            ("Kotak Mahindra Bank", "KOTAKBANK", "HDFCBANK"),
            ("Kotak Bank", "KOTAKBANK", "HDFCBANK"),
            ("Axis Bank", "AXISBANK", "ICICIBANK"),
            ("Federal Bank", "FEDERALBNK", "SBIN"),
            ("Bandhan Bank", "BANDHANBNK", "CANBK"),
            ("AU Small Finance Bank", "AUBANK", "AXISBANK"),
            ("AU Bank", "AUBANK", "AXISBANK"),
        ],
    )
    def test_bank_entities_resolution(self, query: str, expected_symbol: str, forbidden_symbol: str):
        res = resolve_nse_symbol(query, allow_online_lookup=False)
        assert res["symbol"] == expected_symbol, (
            f"Query '{query}' expected {expected_symbol} but got {res['symbol']}"
        )
        assert res["symbol"] != forbidden_symbol


class TestCorporateEntitiesDistinction:
    """Verify multi-entity families and corporate variations resolve accurately."""

    @pytest.mark.parametrize(
        "query,expected_symbol",
        [
            # Tata family
            ("Tata Power", "TATAPOWER"),
            ("Power Grid", "POWERGRID"),
            ("Power Grid Corporation", "POWERGRID"),
            ("Tata Motors", "TATAMOTORS"),
            ("Tata Technologies", "TATATECH"),
            ("Tata Tech", "TATATECH"),
            ("Tata Steel", "TATASTEEL"),
            ("Tata Consumer", "TATACONSUM"),
            ("Tata Consumer Products", "TATACONSUM"),
            ("Tata Elxsi", "TATAELXSI"),
            ("Tata Communications", "TATACOMM"),
            ("Titan", "TITAN"),
            ("Titan Company", "TITAN"),
            ("Trent", "TRENT"),
            ("Voltas", "VOLTAS"),
            # Cement sector
            ("India Cements", "INDIACEM"),
            ("India Cement", "INDIACEM"),
            ("UltraTech Cement", "ULTRACEMCO"),
            ("UltraTech", "ULTRACEMCO"),
            ("Ambuja Cements", "AMBUJACEM"),
            ("Ambuja Cement", "AMBUJACEM"),
            ("ACC", "ACC"),
            ("Shree Cement", "SHREECEM"),
            ("Dalmia Bharat", "DALBHARAT"),
            ("Ramco Cements", "RAMCOCEM"),
            # Oil & Gas
            ("Indian Oil", "IOC"),
            ("Indian Oil Corporation", "IOC"),
            ("IOC", "IOC"),
            ("Oil India", "OIL"),
            ("Oil India Ltd", "OIL"),
            ("Reliance", "RELIANCE"),
            ("Reliance Industries", "RELIANCE"),
            ("Bharat Petroleum", "BPCL"),
            ("BPCL", "BPCL"),
            ("Hindustan Petroleum", "HPCL"),
            ("HPCL", "HPCL"),
            ("ONGC", "ONGC"),
            ("GAIL", "GAIL"),
            # HDFC family
            ("HDFC Bank", "HDFCBANK"),
            ("HDFC Life", "HDFCLIFE"),
            ("HDFC Life Insurance", "HDFCLIFE"),
            ("HDFC AMC", "HDFCAMC"),
            ("HDFC Asset Management", "HDFCAMC"),
            # SBI subsidiaries
            ("SBI Life", "SBILIFE"),
            ("SBI Life Insurance", "SBILIFE"),
            ("SBI Cards", "SBICARD"),
            ("SBI Cards and Payment Services", "SBICARD"),
            # Bajaj family
            ("Bajaj Finance", "BAJFINANCE"),
            ("Bajaj Finserv", "BAJAJFINSV"),
            ("Bajaj Auto", "BAJAJ-AUTO"),
            # Infrastructure
            ("PNC Infratech", "PNCINFRA"),
            ("Larsen & Toubro", "LT"),
            ("L&T", "LT"),
            ("GMR Airports", "GMRINFRA"),
            ("GMR Infra", "GMRINFRA"),
        ],
    )
    def test_corporate_entities_resolution(self, query: str, expected_symbol: str):
        res = resolve_nse_symbol(query, allow_online_lookup=False)
        assert res["symbol"] == expected_symbol, (
            f"Query '{query}' expected {expected_symbol} but got {res['symbol']}"
        )


class TestGenericWordsSuppression:
    """Verify that generic corporate and sector terms alone are never matched to specific stocks."""

    @pytest.mark.parametrize(
        "query",
        [
            "Bank",
            "Banks",
            "Banking",
            "Power",
            "Steel",
            "Cement",
            "Motors",
            "Motor",
            "Finance",
            "Financial",
            "India",
            "Limited",
            "Company",
            "Corporation",
            "Industries",
            "Energy",
            "Holdings",
        ],
    )
    def test_generic_words_suppression(self, query: str):
        # 1. is_generic_query returns True
        assert is_generic_query(query) is True

        # 2. resolve_nse_symbol returns symbol=None
        res = resolve_nse_symbol(query, allow_online_lookup=False)
        assert res["symbol"] is None, f"Generic query '{query}' matched to {res['symbol']}!"
        assert res["confidence"] == 0.0

        # 3. extract_symbols_from_query does not extract any stock
        extracted = extract_symbols_from_query(query)
        assert len(extracted) == 0, f"Generic word '{query}' extracted symbols: {extracted}"


class TestConglomerateAmbiguity:
    """Verify that single conglomerate parent group names trigger ambiguity with candidate options."""

    @pytest.mark.parametrize(
        "group_name,expected_sample_symbol",
        [
            ("Tata", "TATAMOTORS"),
            ("Adani", "ADANIENT"),
            ("Birla", "HINDALCO"),
            ("Bajaj", "BAJFINANCE"),
        ],
    )
    def test_conglomerate_ambiguity_flag(self, group_name: str, expected_sample_symbol: str):
        # 1. Direct ambiguity check
        ambig_res = check_conglomerate_ambiguity(group_name)
        assert ambig_res is not None
        assert ambig_res.get("is_ambiguous") is True
        candidates = ambig_res.get("candidates", [])
        assert len(candidates) > 0
        symbols = [c["symbol"] for c in candidates]
        assert expected_sample_symbol in symbols

        # 2. Symbol resolution handles ambiguity
        res = resolve_nse_symbol(group_name, allow_online_lookup=False)
        assert res["symbol"] is None
        assert res["is_ambiguous"] is True
        assert len(res["candidates"]) > 0

        # 3. Security entity reflects ambiguity
        entity = get_security_entity(group_name)
        assert entity.is_ambiguous is True
        assert entity.symbol is None
        assert len(entity.candidates) > 0


class TestNoMatchBetterThanWrongMatch:
    """Verify that gibberish or unknown entities fail safely rather than guessing wrong."""

    @pytest.mark.parametrize(
        "query",
        [
            "Xyzzqwerty",
            "RandomFakeStock999",
            "Banana Mango Papaya Co",
            "ABCDEFGHIJKLM",
            "Supercalifragilistic",
        ],
    )
    def test_no_match_for_unknown(self, query: str):
        res = resolve_nse_symbol(query, allow_online_lookup=False)
        assert res["symbol"] is None
        assert res["confidence"] == 0.0

        extracted = extract_symbols_from_query(query)
        assert len(extracted) == 0


class TestCanonicalIdentificationBanner:
    """Verify that SecurityEntity and resolve_nse_symbol produce accurate canonical headers."""

    def test_yes_bank_banner_content(self):
        entity = get_security_entity("YESBANK")
        assert entity.is_resolved is True
        assert entity.symbol == "YESBANK"
        assert entity.company_name == "Yes Bank Limited"
        assert entity.isin == "INE528G01035"
        assert entity.bse_code == "532648"
        assert entity.sector == "Financial Services"
        assert entity.industry == "Private Sector Bank"

        banner = entity.format_banner()
        assert "🏢 **Yes Bank Limited**" in banner
        assert "YESBANK" in banner
        assert "532648" in banner
        assert "INE528G01035" in banner
        assert "Financial Services" in banner
        assert "Private Sector Bank" in banner

    def test_sbi_banner_content(self):
        entity = get_security_entity("SBIN")
        assert entity.is_resolved is True
        assert entity.symbol == "SBIN"
        assert entity.company_name == "State Bank of India"
        assert entity.isin == "INE062A01020"
        assert entity.bse_code == "500112"
        assert entity.sector == "Financial Services"
        assert entity.industry == "Public Sector Bank"

        banner = entity.format_banner()
        assert "🏢 **State Bank of India**" in banner
        assert "SBIN" in banner
        assert "500112" in banner
        assert "INE062A01020" in banner
        assert "Financial Services" in banner
        assert "Public Sector Bank" in banner


class TestAutocompleteSearchStocks:
    """Verify search_stocks and search_stock_master return relevant candidates."""

    def test_search_yes(self):
        matches = search_stocks("YES", limit=5)
        assert len(matches) > 0
        symbols = [m["symbol"] for m in matches]
        assert "YESBANK" in symbols

    def test_search_tata(self):
        matches = search_stocks("Tata", limit=10)
        assert len(matches) >= 5
        symbols = [m["symbol"] for m in matches]
        assert "TATAMOTORS" in symbols
        assert "TATAPOWER" in symbols
        assert "TCS" in symbols

    def test_search_bank(self):
        matches = search_stocks("HDFC", limit=5)
        symbols = [m["symbol"] for m in matches]
        assert "HDFCBANK" in symbols


class TestDemoAgentEndToEndIdentification:
    """Verify end-to-end demo agent responses display the proper identification banner and correct stock."""

    def test_demo_agent_yes_bank_query(self):
        res = run_demo_agent("Tell me about YES Bank")
        assert res["status"] == "success"
        response_text = res["response"]

        # MUST contain YESBANK details
        assert "YESBANK" in response_text
        assert "Yes Bank Limited" in response_text
        assert "532648" in response_text  # BSE code
        assert "INE528G01035" in response_text  # ISIN

        # MUST NOT contain SBIN details
        assert "SBIN" not in response_text
        assert "State Bank of India" not in response_text

    def test_demo_agent_tata_ambiguity_guidance(self):
        res = run_demo_agent("Tell me about Tata")
        assert res["status"] == "success"
        response_text = res["response"]

        assert "ambiguous" in response_text.lower() or "multiple companies" in response_text.lower()
        assert "TATAMOTORS" in response_text or "TATAPOWER" in response_text
