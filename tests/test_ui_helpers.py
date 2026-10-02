"""Unit tests for UI helper functions, message filtering, title formatting, and error handling."""

import pytest

from utils.ui_helpers import (
    filter_chat_messages,
    format_conversation_title,
    format_error_message,
    format_tool_call_summary,
)


class TestFilterChatMessages:
    """Tests verifying that only user and assistant turns are displayed in the UI."""

    def test_filter_user_and_assistant_messages(self):
        """Verify standard user and assistant messages are preserved."""
        raw = [
            {"role": "user", "content": "What is TCS price?"},
            {"role": "assistant", "content": "TCS is trading at ₹3,500."},
        ]
        result = filter_chat_messages(raw)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What is TCS price?"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "TCS is trading at ₹3,500."

    def test_filter_discards_internal_tool_and_system_messages(self):
        """Verify internal ToolMessages and system messages are excluded."""
        raw = [
            {"role": "system", "content": "You are an NSE stock analyst."},
            {"role": "user", "content": "Analyze TCS"},
            {"role": "tool", "content": '{"symbol": "TCS", "price": 3500.0}'},
            {"role": "tool_call", "content": "get_stock_price(TCS)"},
            {"role": "assistant", "content": "TCS is at ₹3,500."},
        ]
        result = filter_chat_messages(raw)
        assert len(result) == 2
        assert [m["role"] for m in result] == ["user", "assistant"]
        assert all(m["role"] != "tool" for m in result)
        assert all(m["role"] != "system" for m in result)

    def test_filter_discards_empty_messages(self):
        """Verify messages with empty or whitespace-only content are discarded."""
        raw = [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Valid query"},
        ]
        result = filter_chat_messages(raw)
        assert len(result) == 1
        assert result[0]["content"] == "Valid query"

    def test_filter_empty_list(self):
        """Verify empty input list returns empty list."""
        assert filter_chat_messages([]) == []


class TestFormatConversationTitle:
    """Tests verifying conversation title truncation and cleanup."""

    def test_short_title_unchanged(self):
        """Verify concise queries remain untruncated."""
        assert format_conversation_title("Analyze TCS") == "Analyze TCS"

    def test_long_title_truncated_with_ellipsis(self):
        """Verify long queries are truncated cleanly."""
        long_query = "What are the latest fundamental metrics and news stories for Reliance Industries?"
        title = format_conversation_title(long_query, max_length=30)
        assert len(title) <= 33  # 30 chars + "..."
        assert title.endswith("...")

    def test_empty_or_whitespace_query(self):
        """Verify empty query defaults to 'New Chat'."""
        assert format_conversation_title("") == "New Chat"
        assert format_conversation_title("   ") == "New Chat"
        assert format_conversation_title(None) == "New Chat"

    def test_multi_space_normalization(self):
        """Verify repeated internal whitespace is collapsed."""
        assert format_conversation_title("Why   is   INFY    falling?") == "Why is INFY falling?"


class TestFormatToolCallSummary:
    """Tests verifying human-friendly descriptions for executed tools."""

    def test_stock_price_summary(self):
        """Verify get_stock_price tool call label."""
        tc = {"name": "get_stock_price", "args": {"symbol": "TCS"}}
        assert format_tool_call_summary(tc) == "Fetched live stock price for TCS"

    def test_company_info_summary(self):
        """Verify get_company_info tool call label."""
        tc = {"name": "get_company_info", "args": {"symbol": "INFY"}}
        assert format_tool_call_summary(tc) == "Retrieved fundamentals for INFY"

    def test_market_news_summary(self):
        """Verify get_market_news tool call label."""
        tc = {"name": "get_market_news", "args": {"company_or_symbol": "RELIANCE"}}
        assert format_tool_call_summary(tc) == "Searched recent market news for 'RELIANCE'"

    def test_market_index_summary(self):
        """Verify get_market_index tool call label."""
        tc = {"name": "get_market_index", "args": {"index_name": "NIFTY IT"}}
        assert format_tool_call_summary(tc) == "Retrieved NIFTY IT index performance"

    def test_gainers_and_losers_summary(self):
        """Verify top gainers and top losers labels."""
        assert format_tool_call_summary({"name": "get_top_gainers"}) == "Scanned NSE top market gainers"
        assert format_tool_call_summary({"name": "get_top_losers"}) == "Scanned NSE top market losers"

    def test_stock_sentiment_summary(self):
        """Verify get_stock_sentiment tool call label."""
        tc = {"name": "get_stock_sentiment", "args": {"symbol": "HDFCBANK"}}
        assert format_tool_call_summary(tc) == "Computed technical sentiment for HDFCBANK"

    def test_unknown_tool_fallback(self):
        """Verify fallback for unmapped tool names."""
        tc = {"name": "custom_calculation", "args": {}}
        assert format_tool_call_summary(tc) == "Executed tool: custom_calculation"


class TestFormatErrorMessage:
    """Tests verifying friendly error messaging without exposing stack traces."""

    def test_missing_api_key_error(self):
        """Verify friendly explanation for missing OpenAI API key."""
        err = ValueError("OPENAI_API_KEY is missing or empty.")
        msg = format_error_message(err)
        assert "Configuration Error" in msg
        assert "OPENAI_API_KEY" in msg

    def test_connection_error(self):
        """Verify friendly explanation for connection timeout / network issues."""
        err = ConnectionError("Connection reset by peer on Yahoo Finance API")
        msg = format_error_message(err)
        assert "Network Error" in msg
        assert "internet connection" in msg

    def test_rate_limit_error(self):
        """Verify friendly explanation for API rate limits."""
        err = RuntimeError("429 Too Many Requests - rate limit exceeded")
        msg = format_error_message(err)
        assert "Rate Limit Notice" in msg

    def test_generic_error_fallback(self):
        """Verify generic errors are framed cleanly."""
        err = RuntimeError("Symbol normalization failed")
        msg = format_error_message(err)
        assert "Unable to complete analysis" in msg
        assert "Symbol normalization failed" in msg
