"""LangGraph agent graph implementation for the NSE AI Agent application.

Implements multi-tool reasoning, sequential tool execution, tool-calling loops,
and partial tool failure handling using standard LangGraph patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.prompts import SYSTEM_PROMPT
from services.llm_service import get_all_tools, get_llm

logger = logging.getLogger(__name__)


class AgentState(MessagesState):
    """Agent state container for LangGraph, inheriting messages reducer."""

    pass


def default_tool_error_handler(error: Exception) -> str:
    """Format tool execution errors into informative message strings for the model."""
    error_type = type(error).__name__
    error_msg = str(error)
    logger.error("Tool execution encountered error [%s]: %s", error_type, error_msg)
    return f"Error executing tool: {error_type}: {error_msg}"


def build_nse_agent(
    llm: Optional[Any] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    tool_error_handler: Optional[Callable[[Exception], str]] = None,
    api_key: Optional[str] = None,
) -> CompiledStateGraph:
    """Construct and compile the NSE stock analysis LangGraph agent.

    Args:
        llm: LangChain chat model or mock model supporting .bind_tools() and .invoke().
             If None, resolves via services.llm_service.get_llm().
        tools: Sequence of BaseTool instances. Defaults to get_all_tools().
        system_prompt: Optional override for agent system instructions.
        checkpointer: Optional LangGraph checkpoint saver (e.g. MemorySaver).
        tool_error_handler: Callable to format exceptions into tool messages.
        api_key: Optional API key override for LLM initialization.

    Returns:
        CompiledStateGraph: The compiled, executable agent state graph.
    """
    model = llm if llm is not None else get_llm(api_key=api_key)
    agent_tools = list(tools) if tools is not None else get_all_tools(include_extended=True)
    prompt_text = system_prompt or SYSTEM_PROMPT
    error_handler = tool_error_handler or default_tool_error_handler

    # Configure tool execution node with graceful error handling
    tool_node = ToolNode(agent_tools, handle_tool_errors=error_handler)

    # Bind tools to model
    bound_model = model.bind_tools(agent_tools)

    def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Call the model with system prompt and message history."""
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=prompt_text)] + messages

        logger.debug("Invoking chat model with %d messages in conversation state", len(messages))
        try:
            response = bound_model.invoke(messages)
        except Exception as exc:
            logger.error("Chat model invocation failed: %s: %s", type(exc).__name__, exc)
            raise

        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info("Model requested %d tool call(s): %s", len(response.tool_calls), [t.get("name") for t in response.tool_calls])
        else:
            logger.debug("Model generated direct response without additional tool calls")

        return {"messages": [response]}

    # Build the state graph
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


# Convenience alias
create_nse_agent = build_nse_agent
