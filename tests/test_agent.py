"""Unit tests for the LangGraph agent, multi-tool reasoning, and conversation context."""

from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.graph import build_nse_agent
from agent.prompts import SYSTEM_PROMPT
from agent.runner import run_agent, sanitize_args
from services.database_service import get_messages
from services.llm_service import get_all_tools


class MockChatModel:
    """Mock LangChain chat model to return predetermined responses and record calls."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.call_history: list[list[BaseMessage]] = []
        self.bound_tools: list[BaseTool] = []

    def bind_tools(self, tools: Sequence[BaseTool]) -> "MockChatModel":
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.call_history.append(list(messages))
        if not self.responses:
            return AIMessage(content="Default fallback response")
        return self.responses.pop(0)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database path for isolated test runs."""
    return tmp_path / "test_agent_conversations.sqlite3"


class TestMultiToolReasoning:
    """Tests verifying multi-tool, repeated tool, and sequential tool execution in the agent."""

    @patch("services.llm_service._get_stock_price")
    def test_one_tool_call(self, mock_price):
        """Verify the agent successfully calls a single tool and produces the final answer."""
        mock_price.return_value = {
            "symbol": "TCS",
            "current_price": 3520.0,
            "change": 20.0,
            "change_percent": 0.57,
        }

        mock_llm = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "get_stock_price", "args": {"symbol": "TCS"}}
                ],
            ),
            AIMessage(content="TCS is currently trading at ₹3,520.00 (+0.57%)."),
        ])

        agent = build_nse_agent(llm=mock_llm, tools=get_all_tools())
        result = agent.invoke({"messages": [HumanMessage(content="What is TCS stock price?")]})

        mock_price.assert_called_once_with("TCS")
        messages = result["messages"]
        assert len(messages) == 4
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage) and len(messages[1].tool_calls) == 1
        assert isinstance(messages[2], ToolMessage) and messages[2].name == "get_stock_price"
        assert isinstance(messages[3], AIMessage)
        assert "₹3,520.00" in messages[3].content

    @patch("services.llm_service._get_company_info")
    @patch("services.llm_service._get_stock_price")
    def test_two_tool_calls(self, mock_price, mock_info):
        """Verify the agent can execute two different tools in parallel in response to a broad query."""
        mock_price.return_value = {
            "symbol": "TCS",
            "current_price": 3500.0,
            "change": 15.0,
            "change_percent": 0.43,
        }
        mock_info.return_value = {
            "symbol": "TCS",
            "company": "Tata Consultancy Services Limited",
            "sector": "Technology",
            "trailing_pe": 28.5,
            "market_cap": 12800000000000,
        }

        # Mock LLM generates 2 tool calls at once: price and company fundamentals
        mock_llm = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_p", "name": "get_stock_price", "args": {"symbol": "TCS"}},
                    {"id": "call_i", "name": "get_company_info", "args": {"symbol": "TCS"}},
                ],
            ),
            AIMessage(
                content="TCS is currently trading at ₹3,500 with a trailing P/E of 28.5 in the Technology sector."
            ),
        ])

        agent = build_nse_agent(llm=mock_llm, tools=get_all_tools())
        result = agent.invoke({"messages": [HumanMessage(content="Tell me about TCS")]})

        mock_price.assert_called_once_with("TCS")
        mock_info.assert_called_once_with("TCS")

        messages = result["messages"]
        # HumanMessage -> AIMessage(2 tool calls) -> ToolMessage 1 -> ToolMessage 2 -> AIMessage(final)
        assert len(messages) == 5
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2
        tool_names = {m.name for m in tool_messages}
        assert tool_names == {"get_stock_price", "get_company_info"}

        final_msg = messages[-1]
        assert isinstance(final_msg, AIMessage)
        assert "₹3,500" in final_msg.content
        assert "28.5" in final_msg.content

    @patch("services.llm_service._get_stock_price")
    def test_same_tool_for_two_symbols(self, mock_price):
        """Verify the agent can execute the same tool multiple times for different symbols."""

        def price_side_effect(symbol):
            if symbol.upper() == "TCS":
                return {"symbol": "TCS", "current_price": 3500.0, "change_percent": 1.2}
            elif symbol.upper() == "INFY":
                return {"symbol": "INFY", "current_price": 1850.0, "change_percent": -0.8}
            return {"symbol": symbol, "current_price": 100.0}

        mock_price.side_effect = price_side_effect

        mock_llm = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_tcs", "name": "get_stock_price", "args": {"symbol": "TCS"}},
                    {"id": "call_infy", "name": "get_stock_price", "args": {"symbol": "INFY"}},
                ],
            ),
            AIMessage(
                content="TCS is trading at ₹3,500 (+1.2%), while Infosys is trading at ₹1,850 (-0.8%)."
            ),
        ])

        agent = build_nse_agent(llm=mock_llm, tools=get_all_tools())
        result = agent.invoke({"messages": [HumanMessage(content="Compare TCS with Infosys")]})

        assert mock_price.call_count == 2
        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2
        assert all(m.name == "get_stock_price" for m in tool_messages)

        final_msg = messages[-1]
        assert "₹3,500" in final_msg.content
        assert "₹1,850" in final_msg.content

    @patch("services.llm_service._get_market_index")
    @patch("services.llm_service._get_market_news")
    @patch("services.llm_service._get_stock_price")
    def test_tool_sequence(self, mock_price, mock_news, mock_index):
        """Verify sequential multi-turn tool calling across reasoning steps."""
        mock_price.return_value = {
            "symbol": "INFY",
            "current_price": 1800.0,
            "change_percent": -3.2,
        }
        mock_news.return_value = [
            {"title": "IT sector under pressure amid global client spend cuts"}
        ]
        mock_index.return_value = {
            "index": "NIFTY IT",
            "ticker": "^CNXIT",
            "change_percent": -2.8,
        }

        # Step 1: Model calls get_stock_price for INFY
        # Step 2: Model sees INFY dropped -3.2%, decides to check news and sector index
        # Step 3: Model synthesizes final answer
        mock_llm = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_step1", "name": "get_stock_price", "args": {"symbol": "INFY"}}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_step2_news",
                        "name": "get_market_news",
                        "args": {"company_or_symbol": "INFY"},
                    },
                    {
                        "id": "call_step2_index",
                        "name": "get_market_index",
                        "args": {"index_name": "NIFTY IT"},
                    },
                ],
            ),
            AIMessage(
                content="Infosys dropped 3.2% today, pressured by broad sector headwinds as the NIFTY IT index fell 2.8%."
            ),
        ])

        agent = build_nse_agent(llm=mock_llm, tools=get_all_tools())
        result = agent.invoke(
            {"messages": [HumanMessage(content="Why is Infosys falling today?")]}
        )

        mock_price.assert_called_once_with("INFY")
        mock_news.assert_called_once_with("INFY", limit=5)
        mock_index.assert_called_once_with("NIFTY IT")

        messages = result["messages"]
        assert len(messages) == 7
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 3
        # Sequence of tool names executed
        assert [m.name for m in tool_messages] == [
            "get_stock_price",
            "get_market_news",
            "get_market_index",
        ]

        final_msg = messages[-1]
        assert "fell 2.8%" in final_msg.content

    @patch("services.llm_service._get_stock_price")
    def test_partial_tool_failure(self, mock_price):
        """Verify the agent gracefully handles partial tool failure without crashing."""

        def price_side_effect(symbol):
            if symbol.upper() == "BROKEN":
                raise ConnectionError("NSE network connection timeout for BROKEN")
            return {
                "symbol": "TCS",
                "current_price": 3500.0,
                "change": 10.0,
                "change_percent": 0.29,
            }

        mock_price.side_effect = price_side_effect

        mock_llm = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_fail", "name": "get_stock_price", "args": {"symbol": "BROKEN"}},
                    {"id": "call_succ", "name": "get_stock_price", "args": {"symbol": "TCS"}},
                ],
            ),
            AIMessage(
                content="TCS is trading at ₹3,500.00 (+0.29%). However, real-time data for BROKEN could not be retrieved due to a connection timeout."
            ),
        ])

        agent = build_nse_agent(llm=mock_llm, tools=get_all_tools())
        result = agent.invoke(
            {"messages": [HumanMessage(content="Analyze TCS and BROKEN stocks")]}
        )

        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2

        # One tool succeeded, one caught an error
        error_tool_msg = next(m for m in tool_messages if m.tool_call_id == "call_fail")
        success_tool_msg = next(m for m in tool_messages if m.tool_call_id == "call_succ")

        assert "Error executing tool" in error_tool_msg.content
        assert "ConnectionError" in error_tool_msg.content
        assert "3500" in success_tool_msg.content

        final_msg = messages[-1]
        assert "TCS is trading at ₹3,500.00" in final_msg.content


class TestConversationContextAndPersistence:
    """Tests verifying multi-turn conversation memory, pronoun resolution, and SQLite persistence."""

    @patch("services.llm_service._get_market_news")
    @patch("services.llm_service._get_stock_price")
    def test_follow_up_conversation(self, mock_price, mock_news, temp_db):
        """Verify follow-up questions resolve to the prior context entity using SQLite history."""
        mock_price.return_value = {
            "symbol": "TCS",
            "current_price": 3500.0,
            "change_percent": 0.5,
        }
        mock_news.return_value = [
            {"title": "TCS signs major multi-year digital transformation deal with UK insurer"}
        ]

        # Turn 1: User asks "Analyze TCS."
        mock_llm_turn1 = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "get_stock_price", "args": {"symbol": "TCS"}}
                ],
            ),
            AIMessage(content="TCS is currently trading at ₹3,500 (+0.5%)."),
        ])

        turn1_result = run_agent(
            query="Analyze TCS.",
            db_path=temp_db,
            llm=mock_llm_turn1,
            tools=get_all_tools(),
        )
        cid = turn1_result["conversation_id"]
        assert "₹3,500" in turn1_result["response"]

        # Verify DB has Turn 1 messages
        db_msgs = get_messages(cid, db_path=temp_db)
        assert len(db_msgs) == 2
        assert db_msgs[0]["role"] == "user"
        assert db_msgs[0]["content"] == "Analyze TCS."
        assert db_msgs[1]["role"] == "assistant"

        # Turn 2: User asks "What about recent news?"
        # The mock LLM should inspect previous messages and call news for TCS
        mock_llm_turn2 = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c2",
                        "name": "get_market_news",
                        "args": {"company_or_symbol": "TCS"},
                    }
                ],
            ),
            AIMessage(content="Recent TCS news includes a major deal with a UK insurer."),
        ])

        turn2_result = run_agent(
            query="What about recent news?",
            conversation_id=cid,
            db_path=temp_db,
            llm=mock_llm_turn2,
            tools=get_all_tools(),
        )

        # Verify that the LLM in Turn 2 received the Turn 1 messages in its prompt history
        turn2_prompt_messages = mock_llm_turn2.call_history[0]
        user_queries_in_history = [
            m.content for m in turn2_prompt_messages if isinstance(m, HumanMessage)
        ]
        assert "Analyze TCS." in user_queries_in_history
        assert "What about recent news?" in user_queries_in_history

        mock_news.assert_called_once_with("TCS", limit=5)
        assert "UK insurer" in turn2_result["response"]

        # Verify DB now contains all 4 messages in chronological sequence
        final_db_msgs = get_messages(cid, db_path=temp_db)
        assert len(final_db_msgs) == 4
        assert [m["role"] for m in final_db_msgs] == ["user", "assistant", "user", "assistant"]
        assert final_db_msgs[2]["content"] == "What about recent news?"

    @patch("services.llm_service._get_stock_price")
    def test_pronoun_context_reference(self, mock_price, temp_db):
        """Verify pronoun 'it' in follow-up 'Compare it with Infosys' resolves to TCS."""

        def price_side_effect(symbol):
            if symbol.upper() == "TCS":
                return {"symbol": "TCS", "current_price": 3500.0}
            elif symbol.upper() == "INFY":
                return {"symbol": "INFY", "current_price": 1800.0}
            return {"symbol": symbol, "current_price": 100.0}

        mock_price.side_effect = price_side_effect

        # Turn 1: Analyze TCS
        mock_llm_t1 = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "get_stock_price", "args": {"symbol": "TCS"}}
                ],
            ),
            AIMessage(content="TCS is trading at ₹3,500."),
        ])
        t1 = run_agent("Analyze TCS.", db_path=temp_db, llm=mock_llm_t1, tools=get_all_tools())
        cid = t1["conversation_id"]

        # Turn 2: "Compare it with Infosys."
        # Because 'it' refers to TCS from Turn 1, the LLM calls price for TCS and INFY
        mock_llm_t2 = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t2_tcs", "name": "get_stock_price", "args": {"symbol": "TCS"}},
                    {"id": "t2_infy", "name": "get_stock_price", "args": {"symbol": "INFY"}},
                ],
            ),
            AIMessage(content="TCS (₹3,500) trades at a premium to Infosys (₹1,800)."),
        ])

        t2 = run_agent(
            "Compare it with Infosys.",
            conversation_id=cid,
            db_path=temp_db,
            llm=mock_llm_t2,
            tools=get_all_tools(),
        )

        # Verify Turn 2 LLM saw Turn 1 context
        history = mock_llm_t2.call_history[0]
        assert any(
            isinstance(m, HumanMessage) and "Analyze TCS." in m.content for m in history
        )
        assert any(
            isinstance(m, AIMessage) and "TCS is trading at ₹3,500." in m.content for m in history
        )

        assert t2["response"] == "TCS (₹3,500) trades at a premium to Infosys (₹1,800)."
        assert len(t2["tool_calls"]) == 2
        called_symbols = {tc["args"]["symbol"] for tc in t2["tool_calls"]}
        assert called_symbols == {"TCS", "INFY"}

    def test_custom_system_prompt_propagation(self):
        """Verify custom system prompt is placed at the beginning of the message history."""
        custom_prompt = "Custom financial assistant prompt."
        mock_llm = MockChatModel([
            AIMessage(content="Hello! I am ready."),
        ])

        agent = build_nse_agent(
            llm=mock_llm,
            tools=get_all_tools(),
            system_prompt=custom_prompt,
        )
        agent.invoke({"messages": [HumanMessage(content="Hello")]})

        first_call_messages = mock_llm.call_history[0]
        assert first_call_messages[0].content == custom_prompt

    def test_argument_sanitization_masks_sensitive_keys(self):
        """Verify sanitize_args masks API keys and credentials while preserving normal args."""
        raw_args = {
            "symbol": "TCS",
            "api_key": "secret-12345",
            "auth_token": "bearer-xyz",
            "password": "my_password",
            "limit": 5,
        }
        sanitized = sanitize_args(raw_args)
        assert sanitized["symbol"] == "TCS"
        assert sanitized["limit"] == 5
        assert sanitized["api_key"] == "******"
        assert sanitized["auth_token"] == "******"
        assert sanitized["password"] == "******"
