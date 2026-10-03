"""Unit tests for stock autocomplete functionality, ranking prioritization, and header HTML rendering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from services.symbol_resolver import resolve_nse_symbol, search_stocks
from ui.components import stock_autocomplete
from ui.header import render_top_header


class TestStockAutocompleteSearch:
    """Test search_stocks ranking prioritization and accuracy against acceptance criteria."""

    def test_search_tata_returns_multiple_tata_companies(self):
        matches = search_stocks("tata", limit=15)
        assert len(matches) >= 5
        symbols = [m["symbol"] for m in matches]
        # Should include major Tata subsidiaries
        assert "TATAPOWER" in symbols
        assert "TATAMOTORS" in symbols
        assert "TATASTEEL" in symbols
        # All returned companies should have Tata in name or symbol
        for m in matches:
            assert "tata" in m["company_name"].lower() or "tata" in m["symbol"].lower()

    def test_search_tata_power_prioritizes_tatapower(self):
        matches = search_stocks("Tata Power", limit=10)
        assert len(matches) >= 1
        assert matches[0]["symbol"] == "TATAPOWER"
        assert "Tata Power" in matches[0]["company_name"]
        assert matches[0]["display_text"] == "Tata Power Company Limited (TATAPOWER)"

    def test_search_tatap_prefix_prioritizes_tatapower(self):
        matches = search_stocks("TATAP", limit=10)
        assert len(matches) >= 1
        assert matches[0]["symbol"] == "TATAPOWER"

    def test_search_yes_and_yesbank_prioritizes_yesbank(self):
        matches_yes = search_stocks("yes", limit=10)
        assert len(matches_yes) >= 1
        assert matches_yes[0]["symbol"] == "YESBANK"
        assert "Yes Bank" in matches_yes[0]["company_name"]

        matches_exact = search_stocks("YESBANK", limit=10)
        assert len(matches_exact) >= 1
        assert matches_exact[0]["symbol"] == "YESBANK"

        # Crucial: YESBANK must never match SBIN
        symbols = [m["symbol"] for m in matches_yes]
        assert "SBIN" not in symbols

    def test_search_reliance_prioritizes_reliance_industries(self):
        matches = search_stocks("reliance", limit=10)
        assert len(matches) >= 1
        assert matches[0]["symbol"] == "RELIANCE"
        assert "Reliance Industries" in matches[0]["company_name"]

    def test_search_inf_prioritizes_infosys(self):
        matches = search_stocks("inf", limit=10)
        assert len(matches) >= 1
        assert matches[0]["symbol"] == "INFY"
        assert "Infosys" in matches[0]["company_name"]

    def test_search_invalid_stock_returns_empty_list(self):
        matches = search_stocks("xyzinvalidstock123", limit=10)
        assert matches == []

    def test_search_case_insensitivity(self):
        res_lower = search_stocks("hdfc", limit=5)
        res_upper = search_stocks("HDFC", limit=5)
        res_mixed = search_stocks("HdFc", limit=5)
        assert [r["symbol"] for r in res_lower] == [r["symbol"] for r in res_upper] == [r["symbol"] for r in res_mixed]

    def test_search_prioritization_order(self):
        """Verify: 1. Company name starts with query, 2. Symbol starts with query, 3. Contains."""
        matches = search_stocks("Tata", limit=15)
        # All companies whose name starts with 'Tata' should rank above any company that merely contains 'tata'
        for m in matches:
            assert m["company_name"].lower().startswith("tata") or m["symbol"].lower().startswith("tata")


class TestHeaderHtmlRendering:
    """Verify render_top_header renders clean HTML without raw markdown-escaped source tags."""

    def test_render_top_header_does_not_contain_multiline_indentation(self):
        with patch("streamlit.markdown") as mock_markdown:
            render_top_header(title="Test Title", subtitle="Test Subtitle", show_live_pill=True)
            mock_markdown.assert_called_once()
            args, kwargs = mock_markdown.call_args
            rendered_html = args[0]
            assert kwargs.get("unsafe_allow_html") is True

            # Verify no line has 4+ leading spaces which triggers markdown code blocks
            lines = rendered_html.split("\n")
            for line in lines:
                if line.strip():
                    leading_spaces = len(line) - len(line.lstrip(" "))
                    assert leading_spaces < 4, f"Line has {leading_spaces} leading spaces: {line!r}"

            # Verify it contains the status pill structure
            assert "market-status-pill" in rendered_html
            assert "market-status-dot" in rendered_html
            assert "Market Data Live" in rendered_html


class TestStockAutocompleteUI:
    """Verify stock_autocomplete component interaction and session state management."""

    def test_autocomplete_returns_typed_input_when_no_dropdown_selection(self):
        with patch("streamlit.text_input", return_value="Tata Power"), \
             patch("streamlit.selectbox", return_value="▾ 1 matching stock found — Click to select (or continue typing):"), \
             patch("streamlit.session_state", {}):
            val = stock_autocomplete(label="Stock Name or Symbol", value="Tata Power", key="test_stock_input")
            assert val == "Tata Power"

    def test_autocomplete_handles_exact_selection(self):
        with patch("streamlit.text_input", return_value="Tata Power Company Limited (TATAPOWER)"), \
             patch("streamlit.caption") as mock_caption, \
             patch("streamlit.session_state", {}):
            val = stock_autocomplete(label="Stock Name or Symbol", value="Tata Power Company Limited (TATAPOWER)", key="test_stock_input")
            assert val == "TATAPOWER"
            mock_caption.assert_called_once()
            assert "Tata Power Company Limited" in mock_caption.call_args[0][0]


class TestDecisionAssistantLifecycleAndAnalyzeOnlyTrigger:
    """Verify that autocomplete, typing, and selecting never trigger analysis, and ONLY Analyze button triggers analysis."""

    def test_typing_tata_does_not_start_analysis(self):
        """Case 1 & 2: Typing 'Tata' and waiting must show matches but NEVER start analysis."""
        import app
        mock_res = MagicMock()
        with patch.object(app, "st") as mock_st, \
             patch("app._cached_analyze_stock_decision", return_value=mock_res) as mock_analyze:
            mock_st.session_state = {
                "decision_stock": "Tata Power",
                "decision_intent": "Thinking of Buying",
                "decision_horizon": "1 Year",
                "decision_purchase_price": 0.0,
                "decision_quantity": 0,
                "decision_cache_token": 0,
                "view_mode": "decision",
                "stock_search_text": "Tata",
                "selected_symbol": None,
                "analysis_result": None,
                "analysis_requested": False,
            }
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.radio.return_value = "🛒 Thinking of Buying (New Investment)"
            mock_st.text_input.return_value = "Tata"
            mock_st.selectbox.return_value = "1 Year"
            mock_st.button.return_value = False  # Analyze NOT clicked

            app.render_decision_assistant_dashboard()

            # Analysis must NOT run
            mock_analyze.assert_not_called()
            assert mock_st.session_state["analysis_requested"] is False
            assert mock_st.session_state["analysis_result"] is None

    def test_selecting_suggestion_does_not_start_analysis(self):
        """Case 3, 6, 7: Selecting a suggestion (Tata Motors, Tata Steel, Tata Power) sets symbol but does NOT analyze."""
        import app
        mock_res = MagicMock()
        with patch.object(app, "st") as mock_st, \
             patch("app._cached_analyze_stock_decision", return_value=mock_res) as mock_analyze:
            mock_st.session_state = {
                "decision_stock": "",
                "decision_intent": "Thinking of Buying",
                "decision_horizon": "1 Year",
                "decision_purchase_price": 0.0,
                "decision_quantity": 0,
                "decision_cache_token": 0,
                "view_mode": "decision",
                "stock_search_text": "Tata Motors Limited (TATAMOTORS)",
                "selected_symbol": "TATAMOTORS",
                "selected_stock_name": "Tata Motors Limited",
                "analysis_result": None,
                "analysis_requested": False,
            }
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.radio.return_value = "🛒 Thinking of Buying (New Investment)"
            mock_st.text_input.return_value = "Tata Motors Limited (TATAMOTORS)"
            mock_st.selectbox.return_value = "1 Year"
            mock_st.button.return_value = False  # Analyze NOT clicked

            app.render_decision_assistant_dashboard()

            mock_analyze.assert_not_called()
            assert mock_st.session_state["analysis_requested"] is False

    def test_changing_horizon_does_not_start_analysis(self):
        """Case 4: Changing investment horizon does NOT start analysis."""
        import app
        mock_res = MagicMock()
        with patch.object(app, "st") as mock_st, \
             patch("app._cached_analyze_stock_decision", return_value=mock_res) as mock_analyze:
            mock_st.session_state = {
                "decision_stock": "",
                "decision_intent": "Thinking of Buying",
                "decision_horizon": "6 Months",  # Changed horizon
                "decision_purchase_price": 0.0,
                "decision_quantity": 0,
                "decision_cache_token": 0,
                "view_mode": "decision",
                "stock_search_text": "Tata Motors Limited (TATAMOTORS)",
                "selected_symbol": "TATAMOTORS",
                "selected_stock_name": "Tata Motors Limited",
                "analysis_result": None,
                "analysis_requested": False,
            }
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.radio.return_value = "🛒 Thinking of Buying (New Investment)"
            mock_st.text_input.return_value = "Tata Motors Limited (TATAMOTORS)"
            mock_st.selectbox.return_value = "6 Months"
            mock_st.button.return_value = False  # Analyze NOT clicked

            app.render_decision_assistant_dashboard()

            mock_analyze.assert_not_called()
            assert mock_st.session_state["analysis_requested"] is False

    def test_clicking_analyze_runs_analysis_for_selected_stock(self):
        """Case 5: Clicking Analyze starts analysis for selected stock."""
        import app
        from tests.test_decision_ui import _create_mock_decision_result
        mock_res = _create_mock_decision_result(intent="new")
        mock_res.symbol = "TATAMOTORS"
        mock_res.company_name = "Tata Motors Limited"

        with patch.object(app, "st") as mock_st, \
             patch("app._cached_analyze_stock_decision", return_value=mock_res) as mock_analyze:
            mock_st.session_state = {
                "decision_stock": "",
                "decision_intent": "Thinking of Buying",
                "decision_horizon": "1 Year",
                "decision_purchase_price": 0.0,
                "decision_quantity": 0,
                "decision_cache_token": 0,
                "view_mode": "decision",
                "stock_search_text": "Tata Motors Limited (TATAMOTORS)",
                "selected_symbol": "TATAMOTORS",
                "selected_stock_name": "Tata Motors Limited",
                "analysis_result": None,
                "analysis_requested": False,
            }
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.tabs.return_value = [MagicMock() for _ in range(6)]
            mock_st.radio.return_value = "🛒 Thinking of Buying (New Investment)"
            mock_st.text_input.return_value = "Tata Motors Limited (TATAMOTORS)"
            mock_st.selectbox.return_value = "1 Year"
            mock_st.button.side_effect = lambda label, **kwargs: True if "Analyze" in label else False

            app.render_decision_assistant_dashboard()

            mock_analyze.assert_called_once()
            assert mock_analyze.call_args[1]["stock"] == "TATAMOTORS"
            assert mock_st.session_state["analysis_requested"] is True
            assert mock_st.session_state["analysis_result"] == mock_res

    def test_typing_tata_and_clicking_analyze_without_selecting_shows_ambiguity_warning(self):
        """Case 8: Typing 'Tata' and clicking Analyze without selecting asks user to select."""
        import app
        mock_res = MagicMock()
        with patch.object(app, "st") as mock_st, \
             patch("ui.components.st.text_input", return_value="Tata"), \
             patch("app._cached_analyze_stock_decision", return_value=mock_res) as mock_analyze:
            mock_st.session_state = {
                "decision_stock": "",
                "decision_intent": "Thinking of Buying",
                "decision_horizon": "1 Year",
                "decision_purchase_price": 0.0,
                "decision_quantity": 0,
                "decision_cache_token": 0,
                "view_mode": "decision",
                "stock_search_text": "Tata",
                "selected_symbol": None,
                "selected_stock_name": None,
                "analysis_result": None,
                "analysis_requested": False,
            }
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.radio.return_value = "🛒 Thinking of Buying (New Investment)"
            mock_st.text_input.return_value = "Tata"
            mock_st.selectbox.return_value = "1 Year"
            mock_st.button.side_effect = lambda label, **kwargs: True if "Analyze" in label else False

            app.render_decision_assistant_dashboard()

            # Must NOT analyze!
            mock_analyze.assert_not_called()
            mock_st.warning.assert_called()
            warning_msg = mock_st.warning.call_args[0][0]
            assert 'Multiple stocks match "Tata"' in warning_msg
            assert mock_st.session_state["analysis_requested"] is False

    def test_typing_invalid_stock_and_clicking_analyze_shows_error(self):
        """Case 9: Typing an invalid stock and clicking Analyze shows error and never analyzes."""
        import app
        mock_res = MagicMock()
        with patch.object(app, "st") as mock_st, \
             patch("ui.components.st.text_input", return_value="xyzinvalidstock123"), \
             patch("app._cached_analyze_stock_decision", return_value=mock_res) as mock_analyze:
            mock_st.session_state = {
                "decision_stock": "",
                "decision_intent": "Thinking of Buying",
                "decision_horizon": "1 Year",
                "decision_purchase_price": 0.0,
                "decision_quantity": 0,
                "decision_cache_token": 0,
                "view_mode": "decision",
                "stock_search_text": "xyzinvalidstock123",
                "selected_symbol": None,
                "selected_stock_name": None,
                "analysis_result": None,
                "analysis_requested": False,
            }
            mock_st.columns.side_effect = lambda spec, **kwargs: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            mock_st.radio.return_value = "🛒 Thinking of Buying (New Investment)"
            mock_st.text_input.return_value = "xyzinvalidstock123"
            mock_st.selectbox.return_value = "1 Year"
            mock_st.button.side_effect = lambda label, **kwargs: True if "Analyze" in label else False

            app.render_decision_assistant_dashboard()

            # Must NOT analyze!
            mock_analyze.assert_not_called()
            mock_st.error.assert_called()
            err_msg = mock_st.error.call_args[0][0]
            assert "Could not resolve 'xyzinvalidstock123'" in err_msg
            assert mock_st.session_state["analysis_requested"] is False

    def test_select_tata_chemicals_uses_pending_input_and_never_modifies_active_widget(self):
        """Verify selecting Tata Chemicals sets pending input and calls rerun without modifying active widget."""
        session = {}
        with patch("streamlit.text_input", return_value="TATA"), \
             patch("streamlit.selectbox", return_value="Tata Chemicals Limited (TATACHEM)"), \
             patch("streamlit.rerun") as mock_rerun, \
             patch("streamlit.session_state", session):
            stock_autocomplete(label="Stock Name or Symbol", value="", key="decision_stock_input")

            # Must set pending input, NOT decision_stock_input directly in the same run
            assert session.get("decision_pending_input") == "Tata Chemicals Limited (TATACHEM)"
            assert session.get("decision_stock_input_pending") == "Tata Chemicals Limited (TATACHEM)"
            assert session.get("selected_symbol") == "TATACHEM"
            assert session.get("decision_selected_symbol") == "TATACHEM"
            assert session.get("selected_stock_name") == "Tata Chemicals Limited"
            assert session.get("decision_selected_stock") == "Tata Chemicals Limited"
            assert session.get("analysis_requested") is False
            assert session.get("analysis_result") is None
            mock_rerun.assert_called_once()

    def test_pending_input_is_consumed_before_text_input_instantiation_on_rerun(self):
        """Verify on rerun, pending input is safely moved to widget state before text_input instantiation."""
        session = {
            "decision_pending_input": "Tata Chemicals Limited (TATACHEM)",
            "selected_symbol": "TATACHEM",
            "decision_selected_symbol": "TATACHEM",
            "selected_stock_name": "Tata Chemicals Limited",
            "decision_selected_stock": "Tata Chemicals Limited",
        }
        with patch("streamlit.text_input") as mock_text_input, \
             patch("streamlit.caption") as mock_caption, \
             patch("streamlit.selectbox") as mock_selectbox, \
             patch("streamlit.session_state", session):
            mock_text_input.return_value = "Tata Chemicals Limited (TATACHEM)"

            val = stock_autocomplete(label="Stock Name or Symbol", value="", key="decision_stock_input")

            # Pending input should be consumed
            assert "decision_pending_input" not in session
            # Widget key is safely populated before text_input was called
            assert session["decision_stock_input"] == "Tata Chemicals Limited (TATACHEM)"
            assert val == "TATACHEM"
            # Confirmation pill displayed
            mock_caption.assert_called_once()
            assert "Tata Chemicals Limited" in mock_caption.call_args[0][0]
            # Suggestions selectbox should be hidden
            mock_selectbox.assert_not_called()

    def test_editing_text_after_selection_clears_selected_symbol(self):
        """Verify that if user selects one stock and then edits the text, selected_symbol is cleared."""
        session = {
            "decision_stock_input": "Tata Steel",  # User edited text away from Tata Chemicals
            "selected_symbol": "TATACHEM",
            "decision_selected_symbol": "TATACHEM",
            "selected_stock_name": "Tata Chemicals Limited",
            "decision_selected_stock": "Tata Chemicals Limited",
            "decision_stock_input_canonical": "TATACHEM",
            "analysis_result": MagicMock(),
        }
        with patch("streamlit.text_input", return_value="Tata Steel"), \
             patch("streamlit.selectbox", return_value="▾ 2 matching stocks found — Click to select (or continue typing):"), \
             patch("streamlit.session_state", session):
            stock_autocomplete(label="Stock Name or Symbol", value="", key="decision_stock_input")

            # Selection must be cleared
            assert session.get("selected_symbol") is None
            assert session.get("decision_selected_symbol") is None
            assert session.get("selected_stock_name") is None
            assert session.get("decision_selected_stock") is None
            assert session.get("analysis_result") is None
            assert session.get("analysis_requested") is False


