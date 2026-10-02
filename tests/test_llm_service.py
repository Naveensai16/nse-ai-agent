"""Unit tests for LLM Service, LangChain tools, and LLM configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from services.llm_service import (
    ALL_TOOLS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROVIDER,
    LLMConfigurationError,
    company_info_tool,
    get_all_tools,
    get_company_info_tool,
    get_llm,
    get_llm_with_tools,
    get_market_index_tool,
    get_market_news_tool,
    get_stock_price_tool,
    get_stock_sentiment_tool,
    get_tool_by_name,
    get_tools,
    get_top_gainers_tool,
    get_top_losers_tool,
    market_index_tool,
    market_news_tool,
    stock_price_tool,
    stock_sentiment_tool,
    top_gainers_tool,
    top_losers_tool,
)


class TestToolsRegistry:
    """Tests verifying tool list, naming, schemas, and lookup."""

    EXPECTED_NAMES = {
        "get_stock_price",
        "get_company_info",
        "get_market_news",
        "get_market_index",
        "get_top_gainers",
        "get_top_losers",
        "get_stock_sentiment",
    }

    def test_tool_list_contains_expected_tools(self):
        """Verify get_all_tools returns the expected list of 7 BaseTool instances."""
        tools = get_all_tools()
        assert isinstance(tools, list)
        assert len(tools) == 7
        for t in tools:
            assert isinstance(t, BaseTool)

    def test_get_tools_alias_and_all_tools_constant(self):
        """Verify get_tools alias and ALL_TOOLS constant exist and match get_all_tools."""
        assert get_tools() == get_all_tools()
        assert len(ALL_TOOLS) == 7

    def test_tool_names_are_correct(self):
        """Verify each tool possesses its canonical name matching the requirements."""
        tools = get_all_tools()
        names = {t.name for t in tools}
        assert names == self.EXPECTED_NAMES

        assert get_stock_price_tool.name == "get_stock_price"
        assert get_company_info_tool.name == "get_company_info"
        assert get_market_news_tool.name == "get_market_news"
        assert get_market_index_tool.name == "get_market_index"
        assert get_top_gainers_tool.name == "get_top_gainers"
        assert get_top_losers_tool.name == "get_top_losers"
        assert get_stock_sentiment_tool.name == "get_stock_sentiment"

    def test_tool_aliases(self):
        """Verify convenience aliases reference the exact same tool objects."""
        assert stock_price_tool is get_stock_price_tool
        assert company_info_tool is get_company_info_tool
        assert market_news_tool is get_market_news_tool
        assert market_index_tool is get_market_index_tool
        assert top_gainers_tool is get_top_gainers_tool
        assert top_losers_tool is get_top_losers_tool
        assert stock_sentiment_tool is get_stock_sentiment_tool

    def test_tool_descriptions_are_informative(self):
        """Verify every tool has a non-empty, detailed description for LLM tool selection."""
        for t in get_all_tools():
            assert t.description is not None
            assert len(t.description.strip()) > 30

    def test_get_tool_by_name_success(self):
        """Verify get_tool_by_name retrieves the correct tool by canonical name."""
        for name in self.EXPECTED_NAMES:
            tool_obj = get_tool_by_name(name)
            assert tool_obj.name == name

    def test_get_tool_by_name_missing_raises_key_error(self):
        """Verify get_tool_by_name raises KeyError when given an unknown name."""
        with pytest.raises(KeyError) as exc_info:
            get_tool_by_name("non_existent_tool")
        assert "non_existent_tool" in str(exc_info.value)


class TestToolDelegation:
    """Tests verifying that AI-callable tools delegate correctly without duplicating logic."""

    @patch("services.llm_service._get_stock_price")
    def test_stock_price_tool_delegation(self, mock_func):
        """Verify get_stock_price_tool delegates to _get_stock_price."""
        mock_func.return_value = {
            "symbol": "TCS",
            "current_price": 3500.0,
            "change": 25.0,
            "change_percent": 0.72,
        }
        result = get_stock_price_tool.invoke({"symbol": "TCS"})
        mock_func.assert_called_once_with("TCS")
        assert result["symbol"] == "TCS"
        assert result["current_price"] == 3500.0

    @patch("services.llm_service._get_company_info")
    def test_company_info_tool_delegation(self, mock_func):
        """Verify get_company_info_tool delegates to _get_company_info."""
        mock_func.return_value = {
            "symbol": "INFY",
            "company": "Infosys Limited",
            "sector": "Technology",
            "trailing_pe": 24.5,
        }
        result = get_company_info_tool.invoke({"symbol": "INFY"})
        mock_func.assert_called_once_with("INFY")
        assert result["company"] == "Infosys Limited"
        assert result["trailing_pe"] == 24.5

    @patch("services.llm_service._get_market_news")
    def test_market_news_tool_delegation(self, mock_func):
        """Verify get_market_news_tool delegates with company_or_symbol and limit."""
        mock_func.return_value = [
            {"title": "Reliance Q2 Results", "source": "Mint", "url": "https://example.com/1"}
        ]
        result = get_market_news_tool.invoke({"company_or_symbol": "RELIANCE", "limit": 3})
        mock_func.assert_called_once_with("RELIANCE", limit=3)
        assert len(result) == 1
        assert result[0]["title"] == "Reliance Q2 Results"

    @patch("services.llm_service._get_market_news")
    def test_market_news_tool_default_limit(self, mock_func):
        """Verify get_market_news_tool passes default limit=5 when unspecified."""
        mock_func.return_value = []
        get_market_news_tool.invoke({"company_or_symbol": "TCS"})
        mock_func.assert_called_once_with("TCS", limit=5)

    @patch("services.llm_service._get_market_index")
    def test_market_index_tool_delegation(self, mock_func):
        """Verify get_market_index_tool delegates to _get_market_index."""
        mock_func.return_value = {
            "index": "NIFTY 50",
            "ticker": "^NSEI",
            "current_value": 24800.0,
            "change_percent": 0.45,
        }
        result = get_market_index_tool.invoke({"index_name": "NIFTY 50"})
        mock_func.assert_called_once_with("NIFTY 50")
        assert result["index"] == "NIFTY 50"
        assert result["current_value"] == 24800.0

    @patch("services.llm_service._get_top_gainers")
    def test_top_gainers_tool_delegation(self, mock_func):
        """Verify get_top_gainers_tool delegates to _get_top_gainers."""
        mock_func.return_value = [
            {"symbol": "TATASTEEL", "change_percent": 4.5},
            {"symbol": "JSWSTEEL", "change_percent": 3.8},
        ]
        result = get_top_gainers_tool.invoke({})
        mock_func.assert_called_once_with()
        assert len(result) == 2
        assert result[0]["symbol"] == "TATASTEEL"

    @patch("services.llm_service._get_top_losers")
    def test_top_losers_tool_delegation(self, mock_func):
        """Verify get_top_losers_tool delegates to _get_top_losers."""
        mock_func.return_value = [
            {"symbol": "WIPRO", "change_percent": -2.8},
        ]
        result = get_top_losers_tool.invoke({})
        mock_func.assert_called_once_with()
        assert len(result) == 1
        assert result[0]["symbol"] == "WIPRO"

    @patch("services.llm_service._get_stock_sentiment")
    def test_stock_sentiment_tool_delegation(self, mock_func):
        """Verify get_stock_sentiment_tool delegates to _get_stock_sentiment."""
        mock_func.return_value = {
            "symbol": "HDFCBANK",
            "score": 0.45,
            "sentiment": "Bullish",
            "signals": ["Price above 20 DMA"],
        }
        result = get_stock_sentiment_tool.invoke({"symbol": "HDFCBANK"})
        mock_func.assert_called_once_with("HDFCBANK")
        assert result["symbol"] == "HDFCBANK"
        assert result["sentiment"] == "Bullish"


class TestLLMConfigurationAndFactory:
    """Tests for environment variable configuration, API key handling, and model instantiation."""

    def test_missing_api_key_raises_understandable_error(self, monkeypatch):
        """Verify that missing API key raises LLMConfigurationError with an understandable message."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "openai")

        with pytest.raises(LLMConfigurationError) as exc_info:
            get_llm()

        error_message = str(exc_info.value)
        assert "OPENAI_API_KEY" in error_message
        assert "missing" in error_message.lower()

    def test_empty_string_api_key_raises_error(self, monkeypatch):
        """Verify that empty string or whitespace API key raises LLMConfigurationError."""
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        monkeypatch.setenv("LLM_PROVIDER", "openai")

        with pytest.raises(LLMConfigurationError) as exc_info:
            get_llm()

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_unsupported_provider_raises_error(self, monkeypatch):
        """Verify that unsupported provider raises LLMConfigurationError."""
        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")

        with pytest.raises(LLMConfigurationError) as exc_info:
            get_llm()

        assert "Unsupported LLM provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)

    def test_explicit_args_instantiation(self):
        """Verify get_llm instantiates ChatOpenAI with explicit parameters."""
        llm = get_llm(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test-explicit-key",
            temperature=0.3,
        )
        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.3

    def test_env_based_instantiation(self, monkeypatch):
        """Verify get_llm reads configuration from environment variables."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-12345")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-08-06")

        llm = get_llm()
        assert llm.model_name == "gpt-4o-2024-08-06"
        assert llm.temperature == 0.0

    def test_default_model_when_openai_model_unset(self, monkeypatch):
        """Verify fallback to DEFAULT_OPENAI_MODEL when OPENAI_MODEL is not set."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-12345")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        llm = get_llm()
        assert llm.model_name == DEFAULT_OPENAI_MODEL

    def test_mocked_llm_invocation(self):
        """Verify mocked LLM invocation returns expected response without external calls."""
        with patch.object(
            ChatOpenAI, "invoke", return_value=AIMessage(content="NIFTY 50 is trending upwards.")
        ) as mock_invoke:
            llm = get_llm(api_key="sk-test-key")
            response = llm.invoke("What is the market outlook?")
            mock_invoke.assert_called_once_with("What is the market outlook?")
            assert isinstance(response, AIMessage)
            assert response.content == "NIFTY 50 is trending upwards."

    def test_get_llm_with_tools_binding(self):
        """Verify get_llm_with_tools returns a runnable with bound tools."""
        with patch.object(ChatOpenAI, "bind_tools", return_value="bound_model") as mock_bind:
            bound = get_llm_with_tools(api_key="sk-test-key")
            mock_bind.assert_called_once()
            args, _ = mock_bind.call_args
            bound_tools = args[0]
            assert len(bound_tools) == 7
            assert bound == "bound_model"
