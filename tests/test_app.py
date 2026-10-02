"""Unit tests validating app.py structure, import safety, and session state initialization."""

from unittest.mock import MagicMock, patch

import pytest


def test_app_imports_cleanly_without_crashing():
    """Verify that app.py can be imported cleanly in any Python environment."""
    import app

    assert hasattr(app, "render_app")
    assert hasattr(app, "init_session_state")
    assert hasattr(app, "render_sidebar")
    assert hasattr(app, "render_chat_messages")
    assert hasattr(app, "process_user_input")
    assert callable(app.render_app)


def test_init_session_state():
    """Verify init_session_state initializes current_conversation_id to None."""
    import app

    fake_session_state = {}
    with patch("streamlit.session_state", fake_session_state):
        app.init_session_state()
        assert "current_conversation_id" in fake_session_state
        assert fake_session_state["current_conversation_id"] is None

        # Existing conversation_id should not be overwritten
        fake_session_state["current_conversation_id"] = "existing-uuid"
        app.init_session_state()
        assert fake_session_state["current_conversation_id"] == "existing-uuid"


def test_process_user_input_with_mocked_backend(tmp_path):
    """Verify process_user_input creates conversation and delegates to run_agent."""
    import app

    fake_session_state = {"current_conversation_id": None}
    with patch("streamlit.session_state", fake_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.spinner"), \
         patch("streamlit.markdown"), \
         patch("streamlit.rerun") as mock_rerun, \
         patch("app.run_agent") as mock_run_agent, \
         patch("app.create_conversation") as mock_create_conv:

        mock_create_conv.return_value = {"conversation_id": "test-new-cid"}
        mock_run_agent.return_value = {
            "conversation_id": "test-new-cid",
            "response": "TCS is trading at ₹3,500.",
            "tool_calls": [{"name": "get_stock_price", "args": {"symbol": "TCS"}}],
        }

        app.process_user_input("What is TCS price?", current_id=None)

        mock_create_conv.assert_called_once()
        mock_run_agent.assert_called_once()
        assert fake_session_state["current_conversation_id"] == "test-new-cid"
        mock_rerun.assert_called_once()


def test_process_user_input_persists_error_on_failure():
    """Verify that when run_agent raises an error, it is persisted as an assistant message."""
    import app
    from services.llm_service import LLMConfigurationError

    fake_session_state = {"current_conversation_id": "test-cid", "openai_api_key": ""}
    with patch("streamlit.session_state", fake_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.spinner"), \
         patch("streamlit.markdown"), \
         patch("streamlit.error"), \
         patch("streamlit.rerun") as mock_rerun, \
         patch("app.run_agent") as mock_run_agent, \
         patch("app.add_message") as mock_add_msg:

        mock_run_agent.side_effect = LLMConfigurationError("OpenAI API key is missing.")

        app.process_user_input("What is TCS price?", current_id="test-cid")

        mock_run_agent.assert_called_once()
        mock_add_msg.assert_called_once()
        call_args = mock_add_msg.call_args[0]
        assert call_args[0] == "test-cid"
        assert mock_add_msg.call_args[1]["role"] == "assistant"
        assert "OpenAI API key is missing" in mock_add_msg.call_args[1]["content"]
        mock_rerun.assert_called_once()


def test_process_user_input_passes_api_key_from_session():
    """Verify that process_user_input forwards session state api key to run_agent."""
    import app

    fake_session_state = {"current_conversation_id": "test-cid", "openai_api_key": "sk-custom-test-key"}
    with patch("streamlit.session_state", fake_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.spinner"), \
         patch("streamlit.markdown"), \
         patch("streamlit.rerun"), \
         patch("app.run_agent") as mock_run_agent:

        mock_run_agent.return_value = {
            "conversation_id": "test-cid",
            "response": "Response",
            "tool_calls": [],
        }

        app.process_user_input("Test query", current_id="test-cid")

        mock_run_agent.assert_called_once_with(
            query="Test query",
            conversation_id="test-cid",
            api_key="sk-custom-test-key",
        )


def test_process_user_input_with_demo_mode():
    """Verify that process_user_input delegates to run_demo_agent when demo_mode is True and key is missing."""
    import app

    fake_session_state = {
        "current_conversation_id": "test-cid",
        "openai_api_key": "",
        "demo_mode": True,
    }
    with patch("streamlit.session_state", fake_session_state), \
         patch("streamlit.chat_message"), \
         patch("streamlit.spinner"), \
         patch("streamlit.markdown"), \
         patch("streamlit.rerun"), \
         patch("app.run_demo_agent") as mock_demo:

        mock_demo.return_value = {
            "conversation_id": "test-cid",
            "response": "Comparison of SBIN and HDFCBANK",
            "tool_calls": [{"name": "get_stock_price", "args": {"symbol": "SBIN"}}],
        }

        app.process_user_input("Compare SBI Bank and HDFC Bank stocks", current_id="test-cid")

        mock_demo.assert_called_once_with(
            query="Compare SBI Bank and HDFC Bank stocks",
            conversation_id="test-cid",
        )


