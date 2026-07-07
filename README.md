# 🍷 Wine Reviews Chatbot

A Python chatbot that answers questions about the Kaggle **Wine Reviews** dataset
([`zynicide/wine-reviews`](https://www.kaggle.com/datasets/zynicide/wine-reviews),
file `winemag-data-130k-v2.csv`) by combining two retrieval paths under an LLM agent:

1. **RAG path** — semantic search over the free-text `description` (tasting notes)
   column using **ChromaDB** + Azure OpenAI `text-embedding-3-small-1` embeddings
   (served via the EPAM proxy — same endpoint as the chat model).
2. **MCP server path** — an **MCP** server exposing **Pandas** tools for structured
   queries (filtering, aggregation, ranking) over columns like `country`, `variety`,
   `points`, `price`, `province`, `winery`, and `title`.

The agent (Azure OpenAI `gpt-4o` via the EPAM proxy) decides per question which
tool(s) to call. Some questions need both — e.g. *"find earthy Pinots under \$30 with
at least 90 points"* uses semantic search for *earthy* and structured filters for the rest.

## Architecture

```
                ┌──────────────────────────┐
   user ─────►  │   Streamlit chat (app.py) │
                └─────────────┬─────────────┘
                              │ run_agent(history)
                ┌─────────────▼─────────────┐
                │   Agent (agent.py)        │   Azure OpenAI tool-use loop
                │   AzureOpenAI + tools     │
                └───────┬───────────┬───────┘
            RAG tool    │           │   MCP tools (stdio subprocess)
                ┌───────▼───┐   ┌───▼──────────────┐
                │ rag.py    │   │ mcp_server.py    │
                │ ChromaDB  │   │ Pandas DataFrame │
                └─────┬─────┘   └───────┬──────────┘
                ┌─────▼─────┐   ┌───────▼──────────┐
                │ chroma_db │   │ data/winemag.csv │
                └───────────┘   └──────────────────┘
```

### Folder structure

```
.
├── pyproject.toml          # deps + four `uv run` script entries
├── README.md
├── .env.example
├── .gitignore
├── data/                   # gitignored — populated by `download-data`
├── chroma_db/              # gitignored — populated by `build-index`
└── src/                    # flat module layout
    ├── config.py           # env loading + Azure OpenAI client factories
    ├── embeddings.py       # Chroma embedding function backed by Azure OpenAI
    ├── download.py         # download-data
    ├── indexer.py          # build-index (embeds descriptions into ChromaDB)
    ├── mcp_server.py        # run-mcp  (four Pandas tools)
    ├── rag.py              # ChromaDB query helpers used by the agent
    ├── agent.py            # LLM agent: tool definitions + routing loop
    └── app.py              # run-app (Streamlit chat)
```

## Prerequisites

- **Python 3.11–3.13**
- [**uv**](https://docs.astral.sh/uv/) for package management
- **Kaggle credentials** for `download-data` — set `KAGGLE_USERNAME` / `KAGGLE_KEY`
  or place `~/.kaggle/kaggle.json` (see the [kagglehub docs](https://github.com/Kaggle/kagglehub)).
- An **`OPENAI_API_KEY`** for the EPAM Azure OpenAI proxy (`https://ai-proxy.lab.epam.com`) —
  needed for both `build-index` (embeddings) and `run-app` (chat).
- **Node.js / npx** (only for the MCP Inspector, optional).

## Install

```bash
uv sync
```

This resolves dependencies, creates `.venv/`, and writes `uv.lock`.

Copy the env template and set your key:

```bash
cp .env.example .env
# OPENAI_API_KEY is read from the environment; export it or add it to .env
export OPENAI_API_KEY=...        # PowerShell: $env:OPENAI_API_KEY="..."
```

## Run

The console scripts are registered under `[project.scripts]`:

```bash
# 1. Download (if needed) AND build the ChromaDB index in one step.
#    On first run this pulls data/winemag.csv from Kaggle automatically;
#    re-runs reuse the cached CSV. Default 15000 sampled rows (0 = all).
uv run build-index --sample 15000

# 2. Launch the Streamlit chat app
uv run run-app
```

Extra commands:

```bash
# Force a fresh dataset download before indexing
uv run build-index --refresh-data

# Download only (standalone) — build-index does this for you, so this is optional
uv run download-data

# Run the MCP server standalone (stdio). On its own it just waits for JSON-RPC on
# stdin — mainly useful as a quick "does it start?" check. To interact with it, use
# the MCP Inspector below (which launches this command for you).
uv run run-mcp
```

> **Why is `download-data` still separate?** `build-index` downloads the dataset on
> first run and caches it, so you normally only run `build-index`. `download-data`
> remains available for the rare case where you want to (re)fetch the CSV on its own.

`run-app` opens the chat in your browser. The agent spawns the MCP server itself
(in-process over stdio), so you do **not** need to run `run-mcp` separately for normal use.

## Inspecting the MCP server

Use the official MCP Inspector to list and call the Pandas tools interactively:

```bash
npx @modelcontextprotocol/inspector uv run run-mcp
```

Open the Inspector URL it prints, connect, and you'll see the four tools:

- `describe_schema()` — columns, dtypes, numeric ranges, distinct categorical counts.
- `filter_wines(country, variety, province, min_points, max_points, min_price, max_price, limit=20)`
- `aggregate_wines(group_by, metric, agg="mean", filters=None)`
- `top_wines(by="points", filters=None, n=10)`

## Example usage

> **You:** Find earthy Pinot Noirs under \$30 with at least 90 points.
>
> **Assistant:** *(calls `search_wine_descriptions("earthy")` for the flavour match and
> `filter_wines(variety="Pinot Noir", max_price=30, min_points=90)` for the structure,
> then combines)* Here are a few that fit — *Domaine X 2014* (91 pts, \$28): "earthy,
> forest-floor notes over bright cherry…" …

Other questions to try:

- *"What's the average price of wine by country?"* → `aggregate_wines`
- *"Top 10 highest-rated wines from Italy"* → `top_wines`
- *"Describe a bold, tannic Cabernet"* → `search_wine_descriptions`

## Notes

- The agent backend is **Azure OpenAI** via the EPAM proxy (configured in `config.py`
  and `.env.example`). To switch to another provider, change only `config.py` /
  `agent.py`.
- Re-running `build-index` rebuilds the `wines` collection from scratch.
