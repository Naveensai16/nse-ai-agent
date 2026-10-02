"""Execution runner for the NSE Agentic AI stock analysis application.

Binds persisted SQLite conversation history to LangGraph agent execution,
enabling multi-turn conversational context, coreference resolution, and state tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent.graph import build_nse_agent
from services.database_service import (
    add_message,
    create_conversation,
    get_conversation,
    get_messages,
    initialize_database,
)

logger = logging.getLogger(__name__)

# Sensitive key patterns that must never appear in logs
SENSITIVE_KEY_PATTERNS = ("key", "secret", "auth", "token", "password", "credential", "bearer")


def sanitize_args(args: Any) -> Any:
    """Sanitize arguments dictionary by masking any credential or sensitive keys."""
    if not isinstance(args, dict):
        return args

    sanitized = {}
    for k, v in args.items():
        k_lower = str(k).lower()
        if any(pat in k_lower for pat in SENSITIVE_KEY_PATTERNS):
            sanitized[k] = "******"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_args(v)
        else:
            sanitized[k] = v
    return sanitized


def run_agent(
    query: str,
    conversation_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    agent_graph: Optional[CompiledStateGraph] = None,
    llm: Optional[Any] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Execute the NSE AI agent with conversation context persisted in SQLite.

    1. Resolves or creates a conversation record in SQLite.
    2. Records the current user query in the database.
    3. Loads full conversation history from SQLite into LangChain messages.
    4. Invokes the LangGraph agent state graph with full multi-turn context.
    5. Extracts the assistant's final response and any executed tool calls.
    6. Persists the assistant's response back to SQLite.
    7. Returns a comprehensive execution dictionary.

    Args:
        query: User's input question or instruction.
        conversation_id: Optional UUID string identifying an existing conversation.
        db_path: Optional SQLite database file path.
        agent_graph: Pre-compiled LangGraph agent graph. If None, builds one.
        llm: Chat model or mock model to use if building the graph.
        tools: Tools sequence to bind if building the graph.
        system_prompt: Optional system prompt override.

    Returns:
        dict: Result dictionary containing:
            - 'conversation_id': UUID of the conversation
            - 'query': Original user query
            - 'response': Assistant final answer text
            - 'messages': Complete list of messages from the agent run
            - 'tool_calls': List of tool calls executed during this run
    """
    initialize_database(db_path)

    # 1. Resolve or create conversation
    if conversation_id:
        conv = get_conversation(conversation_id, db_path=db_path)
        if not conv:
            conv = create_conversation(
                title=query[:60], conversation_id=conversation_id, db_path=db_path
            )
    else:
        conv = create_conversation(title=query[:60], db_path=db_path)
        conversation_id = conv["conversation_id"]

    logger.info("Starting agent run for conversation_id=%s, query=%r", conversation_id, query)

    # 2. Persist the current user query to SQLite
    add_message(conversation_id, role="user", content=query, db_path=db_path)

    # 3. Retrieve complete conversation history from SQLite
    db_records = get_messages(conversation_id, db_path=db_path)
    logger.debug(
        "Retrieved %d persisted message(s) from SQLite for conversation_id=%s",
        len(db_records),
        conversation_id,
    )

    # 4. Map SQLite records to LangChain BaseMessage objects
    langchain_messages: list[BaseMessage] = []
    for rec in db_records:
        role = rec.get("role", "user")
        content = rec.get("content", "")
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        elif role == "system":
            langchain_messages.append(SystemMessage(content=content))

    # 5. Resolve or build the agent graph
    graph = agent_graph or build_nse_agent(
        llm=llm, tools=tools, system_prompt=system_prompt, api_key=api_key
    )

    # 6. Invoke the agent graph with historical messages + current turn
    try:
        graph_output = graph.invoke({"messages": langchain_messages})
        result_messages = graph_output.get("messages", [])
    except Exception as exc:
        logger.error(
            "Agent graph execution failed for conversation_id=%s: %s: %s",
            conversation_id,
            type(exc).__name__,
            exc,
        )
        raise

    # 7. Extract final response from the last AIMessage with content
    final_response = ""
    for m in reversed(result_messages):
        if isinstance(m, AIMessage) and m.content:
            final_response = str(m.content)
            break

    # 8. Extract all tool calls executed during this turn and log them securely
    executed_tool_calls: list[dict[str, Any]] = []
    for m in result_messages:
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                executed_tool_calls.append(tc)
                tool_name = tc.get("name", "unknown")
                raw_args = tc.get("args", {})
                safe_args = sanitize_args(raw_args)
                logger.info(
                    "Conversation %s: executed tool '%s' with args %s",
                    conversation_id,
                    tool_name,
                    safe_args,
                )

    # 9. Persist assistant final response to SQLite
    add_message(conversation_id, role="assistant", content=final_response, db_path=db_path)

    logger.info(
        "Agent run succeeded for conversation_id=%s with %d tool call(s) executed",
        conversation_id,
        len(executed_tool_calls),
    )

    return {
        "conversation_id": conversation_id,
        "query": query,
        "response": final_response,
        "messages": result_messages,
        "tool_calls": executed_tool_calls,
    }
