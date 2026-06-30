# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# One-time setup: download dataset + build ChromaDB index (default: 15000 sampled rows)
uv run build-index --sample 15000

# Force re-download of the CSV before indexing
uv run build-index --refresh-data

# Launch the Streamlit chat app
uv run run-app

# Run the MCP server standalone (normally the agent spawns it automatically)
uv run run-mcp

# Inspect MCP tools interactively (requires Node.js)
npx @modelcontextprotocol/inspector uv run run-mcp
```

There is no test suite or linter configured in `pyproject.toml`.

## Environment Setup

Copy `.env.example` to `.env` and set:
- `OPENAI_API_KEY` — required for the EPAM Azure OpenAI proxy (`https://ai-proxy.lab.epam.com`)
- `KAGGLE_USERNAME` / `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`) — required for `download-data`

Optional overrides (defaults are fine for EPAM): `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `LLM_MODEL`.

## Architecture

The chatbot combines two retrieval paths under an Azure OpenAI tool-use loop:

**RAG path** (`rag.py`) — semantic search over the `description` (tasting notes) column. Uses ChromaDB with `sentence-transformers/all-MiniLM-L6-v2` embeddings stored in `chroma_db/` (gitignored). The collection is built by `indexer.py` and queried by `rag.search_wine_descriptions()`.

**MCP path** (`mcp_server.py`) — structured Pandas queries (filter, aggregate, rank) over `data/winemag.csv` (gitignored). Implemented with `FastMCP`; the DataFrame is loaded once and cached via `lru_cache`. Tools: `describe_schema`, `filter_wines`, `aggregate_wines`, `top_wines`.

**Agent** (`agent.py`) — an async Azure OpenAI tool-use loop. On each user turn it spawns the MCP server as a stdio subprocess (`sys.executable -m mcp_server`), converts MCP tool definitions to OpenAI format, then iterates up to 8 turns dispatching calls either to the local `rag.search_wine_descriptions` or to the MCP session. `run_agent()` is the synchronous entry point used by Streamlit.

**UI** (`app.py`) — minimal Streamlit chat. Maintains `st.session_state.messages` and calls `run_agent(history)` on each submission.

**Config** (`config.py`) — single source of truth for all paths, env vars, model settings, and the `AzureOpenAI` client factory. Every other module imports from here.

### Data flow summary

```
Streamlit (app.py) → run_agent(history)
  → spawns MCP server (stdio subprocess)
  → Azure OpenAI tool-use loop (agent.py)
       ├─ search_wine_descriptions → rag.py → chroma_db/
       └─ MCP tools → mcp_server.py → data/winemag.csv
```

### Module layout

All modules live flat under `src/` and are importable as top-level names (hatchling `sources = ["src"]`). `data/` and `chroma_db/` are gitignored and must be created locally via `build-index`.

### Switching LLM providers

The only files to change are `config.py` (client factory) and `agent.py` (the `AzureOpenAI` call). The MCP and RAG layers are provider-agnostic.
