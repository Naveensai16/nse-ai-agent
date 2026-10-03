---
title: NSE AI Agent
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# NSE Agentic AI Stock Analysis Application

A production-ready, agentic AI stock analysis application designed for research, technical evaluation, fundamental analysis, and news tracking of equities listed on the **National Stock Exchange of India (NSE)**.

Built with **Python 3.12+**, **LangChain**, **LangGraph**, **yfinance**, **SQLite**, and **Streamlit**.

---

## 🏛️ Architecture Overview

The application is structured into decoupled, modular layers ensuring deterministic execution, full testability, and resilient error recovery:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                           │
│  - ChatGPT-style sidebar with chat creation, selection, & deletion    │
│  - Multi-turn conversation display (st.chat_message)                   │
│  - User prompt input (st.chat_input) & loading spinners                │
│  - Collapsible tool activity expander (hiding internal ToolMessages)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Agent Runner (agent/runner.py)                   │
│  - Restores full conversation history from SQLite                      │
│  - Sanitizes arguments and performs secure audit logging               │
│  - Invokes compiled LangGraph agent state graph                        │
│  - Persists user prompt and assistant response to SQLite database      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     LangGraph ReAct State Graph                        │
│                           (agent/graph.py)                             │
│                                                                        │
│       START ──► [ agent ] ◄──────────────┐                             │
│                    │                     │                             │
│             (tools_condition)            │                             │
│              /           \               │                             │
│         [Has Tools]   [No Tools]         │                             │
│             │              │             │                             │
│             ▼              ▼             │                             │
│         [ tools ] ─────────┴─────► END   │                             │
│             │                            │                             │
│             └────────────────────────────┘                             │
└───────────────────┬────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       AI-Callable Tool Layer                           │
│  - stock_tool: yfinance quotes, 52-week high/low, change %             │
│  - company_tool: fundamentals, P/E, EPS, dividend yield, market cap   │
│  - market_tool: benchmark & sectoral indices (NIFTY 50, NIFTY IT)     │
│  - news_tool: RSS feed search, title/URL deduplication, date parsing   │
│  - sentiment_tool: deterministic score (1D/5D return, 20/50 DMA)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Persistence & External Services                     │
│  - SQLite Database: conversations & messages (ON DELETE CASCADE)       │
│  - OpenAI / LLM Provider: function calling & synthesis                 │
│  - Yahoo Finance & Google News RSS: Indian equity & market data feeds  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
nse-ai-agent/
├── app.py                  # Streamlit Web UI (ChatGPT-style application)
├── requirements.txt        # Production dependencies
├── .env.example            # Environment configuration template
├── .gitignore              # Git ignore rules for Python, SQLite, & environment
├── README.md               # Complete architecture & operations documentation
├── agent/                  # LangGraph agent orchestration & prompts
│   ├── __init__.py
│   ├── graph.py            # LangGraph StateGraph, ToolNode, and ReAct loop
│   ├── prompts.py          # Expert financial analyst system prompt
│   └── runner.py           # SQLite-to-Graph execution runner & audit logger
├── tools/                  # Domain market data, technical, & fundamental tools
│   ├── __init__.py
│   ├── stock_tool.py       # Live stock price and trading summary (yfinance)
│   ├── company_tool.py     # Fundamentals, P/E, EPS, dividend, balance profile
│   ├── market_tool.py      # Indian market indices (NIFTY 50, etc.) & price history
│   ├── news_tool.py        # RSS news retrieval, deduplication & date parsing
│   └── sentiment_tool.py   # Deterministic technical sentiment (DMAs, return, volume)
├── services/               # Persistent database and LLM configuration services
│   ├── __init__.py
│   ├── database_service.py # SQLite persistent conversation and message storage
│   └── llm_service.py      # LLM initialization factory & LangChain tool wrappers
├── utils/                  # Utility helpers and UI formatting
│   ├── __init__.py
│   ├── helpers.py          # NSE symbol normalization and validation logic
│   └── ui_helpers.py       # Message filtering, title truncation, error formatters
├── tests/                  # Automated test suite (242 tests)
│   ├── __init__.py
│   ├── test_helpers.py     # Symbol normalization tests
│   ├── test_stock_tool.py  # Stock price tool tests
│   ├── test_company_tool.py# Company fundamentals tests
│   ├── test_market_tool.py # Market indices, history, gainers/losers tests
│   ├── test_news_tool.py   # Financial news parsing and deduplication tests
│   ├── test_sentiment_tool.py # Deterministic technical sentiment tests
│   ├── test_database.py    # SQLite database schema, CRUD, cascade tests
│   ├── test_llm_service.py # LLM factory, tool registry, config error tests
│   ├── test_agent.py       # Multi-tool reasoning, sequential tools, context tests
│   ├── test_ui_helpers.py  # UI message filtering, title formatting tests
│   └── test_app.py         # Streamlit app import safety & session state tests
└── data/                   # Local storage directory (SQLite database)
```

---

## 🛠️ Prerequisites

* **Python 3.12+**
* An active **OpenAI API Key** (or compatible LLM provider)

---

## 🚀 Setup & Virtual Environment

### 1. Clone & Navigate to Repository

```powershell
git clone <repository-url>
cd nse-ai-agent
```

### 2. Create and Activate Virtual Environment

#### Option A: Using `uv` (Recommended)

```powershell
# Create Python 3.12 virtual environment
uv venv --python 3.12

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt
```

#### Option B: Using standard Python `venv` + `pip`

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

Create your local `.env` configuration from the provided template:

```powershell
cp .env.example .env
```

Configure the following variables in `.env`:

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `LLM_PROVIDER` | No | `openai` | Supported LLM provider (`openai`). |
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API authentication key (`sk-...`). |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI chat model (`gpt-4o-mini`, `gpt-4o`). |
| `DATA_DIR` | No | `./data` | Directory for storing SQLite databases. |
| `APP_ENV` | No | `development` | Environment mode (`development`, `production`). |
| `LOG_LEVEL` | No | `INFO` | Python logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

> [!NOTE]
> Never commit `.env` to version control. The `.gitignore` file is configured to strictly exclude `.env`, `.env.local`, and SQLite database files (`*.sqlite3`).

---

## 🖥️ How to Run the Application

Launch the Streamlit web interface:

```powershell
python -m streamlit run app.py
```
*(Or `streamlit run app.py`)*

Access the application in your browser at `http://localhost:8501`.

### Key UI Capabilities:
* **Sidebar Controls**:
  * `➕ New Chat`: Start a fresh conversation session.
  * **Recent Chats**: Click any previous conversation to resume context.
  * **Delete**: Click `🗑️` to permanently delete any conversation.
* **Responsive Chat Feed**: Renders user questions and AI analysis.
* **Tool Activity Inspection**: Click `🛠️ Executed tools` to see what live data feeds the agent called during reasoning.
* **Clean Error Handling**: Clean, actionable messages without exposing raw Python stack traces.

---

## 🧪 How to Test

Run the full automated test suite:

```powershell
pytest
```

Run tests with detailed verbose output:

```powershell
pytest -v
```

Byte-compile the entire project to ensure syntax and import integrity:

```powershell
python -m compileall .
```

Validate clean programmatic imports:

```powershell
python -c "import app; print('Application imported cleanly!')"
```

---

## 🧰 Available Tools

The agent has access to 7 AI-callable domain tools wrapping validated market logic:

| Tool Name | Arguments | Purpose & Output |
| :--- | :--- | :--- |
| `get_stock_price` | `symbol: str` | Fetches live market price, previous close, daily price change, change percentage, and 52-week high/low for an NSE equity. |
| `get_company_info` | `symbol: str` | Retrieves fundamental metrics: market cap, sector, industry, trailing P/E, forward P/E, EPS, dividend yield, and business summary. |
| `get_market_news` | `company_or_symbol: str`, `limit: int = 5` | Searches verified RSS financial feeds for recent, deduplicated news articles, headlines, publication dates, and URLs. |
| `get_market_index` | `index_name: str` | Returns current value and change for major Indian indices (NIFTY 50, BANK NIFTY, SENSEX, NIFTY IT). |
| `get_top_gainers` | *None* | Scans the top performing NSE liquid equities ranked by daily percentage gain. |
| `get_top_losers` | *None* | Scans the worst performing NSE liquid equities ranked by daily percentage decline. |
| `get_stock_sentiment` | `symbol: str` | Calculates deterministic technical sentiment score (-1.0 to 1.0) and label (*Bullish*, *Neutral*, *Bearish*) based on 1D/5D returns, 20/50 DMAs, and volume trends. |

---

## 💬 Example Questions

The agent supports complex single-tool, multi-tool, repeated, and sequential reasoning patterns:

1. **Broad Stock Overview (Parallel multi-tool call)**:
   > *"Tell me about TCS"*
   * Calls `get_stock_price("TCS")` and `get_company_info("TCS")` simultaneously.
2. **Movement & Root Cause Analysis (Sequential multi-stage call)**:
   > *"Why is Infosys falling today?"*
   * Calls `get_stock_price("INFY")`, inspects the drop, then queries `get_market_news("INFY")` and `get_market_index("NIFTY IT")`.
3. **Stock Comparison (Repeated tool calls for multiple symbols)**:
   > *"Compare TCS with Infosys"*
   * Calls `get_stock_price("TCS")` and `get_stock_price("INFY")`.
4. **Market Overview**:
   > *"What are the top gainers and losers in the market today?"*
   * Calls `get_top_gainers()` and `get_top_losers()`.
5. **Technical Sentiment**:
   > *"What is the technical sentiment for RELIANCE?"*
   * Calls `get_stock_sentiment("RELIANCE")`.
6. **Multi-Turn Context & Pronoun Resolution**:
   * Turn 1: *"Analyze TCS."*
   * Turn 2: *"What about recent news?"* *(Resolves automatically to TCS from conversation history)*
   * Turn 3: *"Compare it with Infosys."* *(Resolves 'it' to TCS through conversational context)*

---

## 🔄 LangGraph Reasoning & Orchestration

The agent is orchestrated using a **LangGraph** `StateGraph(MessagesState)` implementing the ReAct pattern:

1. **Message History & System Context**:
   The `agent` node injects the expert financial prompt and appends the stored conversation history.
2. **Dynamic Tool Calling**:
   The chat model inspects user intent and returns tool call requests. If tools are requested, LangGraph routes via `tools_condition` to the `tools` node (`ToolNode`).
3. **Fault-Tolerant Execution**:
   The `ToolNode` wraps execution in `default_tool_error_handler`. If a network timeout or data provider exception occurs on one tool, the error is converted into an informative `ToolMessage` rather than crashing execution.
4. **Iterative Reasoning Loop**:
   The tool output loops back to the `agent` node, allowing the model to either request additional sequential tools or synthesize the final answer.
5. **Audit Logging & Security**:
   The `agent/runner.py` logs conversation IDs, tool invocations, and arguments while masking sensitive keys (e.g. `api_key`, `token`, `password`).

---

## 💾 Conversation Persistence (SQLite)

Conversations and messages are stored persistently in a local SQLite database (`data/conversations.sqlite3`):

* **`conversations` Table**:
  * `conversation_id` (TEXT, Primary Key UUID)
  * `title` (TEXT, user-friendly prompt summary)
  * `created_at` (TEXT, ISO 8601 UTC)
  * `updated_at` (TEXT, ISO 8601 UTC)
* **`messages` Table**:
  * `message_id` (TEXT, Primary Key UUID)
  * `conversation_id` (TEXT, Foreign Key referencing `conversations`)
  * `role` (TEXT: `user`, `assistant`)
  * `content` (TEXT)
  * `created_at` (TEXT, ISO 8601 UTC)
* **Foreign Key Constraints & Indexes**:
  * Enforced `PRAGMA foreign_keys = ON;`
  * `ON DELETE CASCADE` ensures deleting a conversation automatically cleans up its associated messages.
  * Indexed on `conversation_id`, `created_at`, and `updated_at` for high query performance.

---

## ⚠️ Known Limitations

1. **Market Data Latency**: Market data is sourced via `yfinance` and public RSS feeds, which may have short delays compared to direct NSE algorithmic feeds.
2. **Provider Availability**: Rate limits or transient connectivity issues with Yahoo Finance or Google News RSS can occasionally delay response generation; the agent handles these gracefully by reporting partial availability.
3. **Market Hours**: Live quote metrics reflect the last traded price, which outside market hours represents the previous market session's close.
4. **Deterministic Technical Sentiment**: The sentiment score is a purely mathematical heuristic (based on 1D/5D returns, 20/50 DMAs, and volume) and does not predict future prices.

---

## ⚖️ Data Provider & Financial Disclaimer

> [!CAUTION]
> **NOT FINANCIAL ADVICE**
> The information, analysis, and sentiment scores provided by this application are for **educational and informational purposes only**.
> This application does not constitute financial, investment, legal, or tax advice. Market investments are subject to market risks. Always perform your own independent research and consult a certified financial advisor before making investment decisions.
>
> Market data is retrieved via open third-party sources (Yahoo Finance and Google News RSS). The authors and contributors do not guarantee the completeness, timeliness, or accuracy of the data.
