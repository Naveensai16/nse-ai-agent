"""UI helper functions for Streamlit interface formatting, message filtering, and error handling."""

from __future__ import annotations

from typing import Any, Mapping


def filter_chat_messages(messages: list[dict[str, Any] | Mapping[str, Any]]) -> list[dict[str, str]]:
    """Filter persisted database messages to only include displayable user and assistant turns.

    Ensures that internal ToolMessages, system prompts, or debug messages
    are never displayed as standard chat messages in the user interface.

    Args:
        messages: List of message records (e.g. from database_service.get_messages).

    Returns:
        list[dict[str, str]]: Filtered list with 'role' and 'content' keys.
    """
    valid_roles = {"user", "assistant"}
    displayable_messages: list[dict[str, str]] = []

    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", "")

        # Skip messages that are not user or assistant
        if role not in valid_roles:
            continue

        # Skip empty messages
        if not content or not str(content).strip():
            continue

        displayable_messages.append({
            "role": role,
            "content": str(content),
            "message_id": str(msg.get("message_id", "")),
            "created_at": str(msg.get("created_at", "")),
        })

    return displayable_messages


def format_conversation_title(query: str, max_length: int = 35) -> str:
    """Format and truncate a user prompt into a concise conversation title.

    Args:
        query: Raw user prompt string.
        max_length: Maximum allowed characters before truncation.

    Returns:
        str: Cleaned, truncated title string.
    """
    if not query or not query.strip():
        return "New Chat"

    cleaned = " ".join(query.strip().split())
    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[:max_length].rstrip() + "..."


def format_tool_call_summary(tool_call: dict[str, Any]) -> str:
    """Format a raw LangChain tool call into a user-friendly label for activity display.

    Args:
        tool_call: Dictionary containing 'name' and 'args' keys.

    Returns:
        str: Human-readable description of the tool operation.
    """
    name = str(tool_call.get("name", "")).strip()
    args = tool_call.get("args", {})
    if not isinstance(args, dict):
        args = {}

    if name == "get_stock_price":
        symbol = args.get("symbol", "").upper()
        return f"Fetched live stock price for {symbol}" if symbol else "Fetched stock price"
    elif name == "get_company_info":
        symbol = args.get("symbol", "").upper()
        return f"Retrieved fundamentals for {symbol}" if symbol else "Retrieved company info"
    elif name == "get_market_news":
        query = args.get("company_or_symbol", "")
        return f"Searched recent market news for '{query}'" if query else "Searched market news"
    elif name == "get_market_index":
        index_name = args.get("index_name", "")
        return f"Retrieved {index_name} index performance" if index_name else "Retrieved market index"
    elif name == "get_top_gainers":
        return "Scanned NSE top market gainers"
    elif name == "get_top_losers":
        return "Scanned NSE top market losers"
    elif name == "get_stock_sentiment":
        symbol = args.get("symbol", "").upper()
        return f"Computed technical sentiment for {symbol}" if symbol else "Computed technical sentiment"

    return f"Executed tool: {name}"


def format_error_message(error: Exception) -> str:
    """Translate technical exceptions into clean, friendly error messages without stack traces.

    Args:
        error: The caught Exception instance.

    Returns:
        str: User-facing explanatory message.
    """
    err_str = str(error).strip()
    err_type = type(error).__name__

    if "LLMConfigurationError" in err_type or "OPENAI_API_KEY" in err_str:
        return (
            "Configuration Error: OpenAI API key is missing or invalid. "
            "Please configure `OPENAI_API_KEY` in your `.env` file to enable AI analysis."
        )

    if isinstance(error, (ConnectionError, TimeoutError)) or "timeout" in err_str.lower():
        return (
            "Network Error: Unable to connect to market data or AI services. "
            "Please verify your internet connection and try again."
        )

    if "rate limit" in err_str.lower() or "429" in err_str:
        return (
            "Rate Limit Notice: The API provider rate limit was reached. "
            "Please wait a few moments and try your request again."
        )

    return f"Unable to complete analysis: {err_str}" if err_str else "An unexpected error occurred."
