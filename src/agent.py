"""The LLM agent: an Azure OpenAI tool-use loop wired to two retrieval paths.

Tools exposed to the model:
  * ``search_wine_descriptions`` — local RAG over ChromaDB (semantic search).
  * the MCP server's Pandas tools — spawned in-process over stdio.

The agent decides per question which tool(s) to call; some questions need both.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
import rag

SRC_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """\
You are a knowledgeable wine assistant answering questions about a dataset of ~130k \
professional wine reviews (Wine Enthusiast). You have two kinds of tools:

1. search_wine_descriptions — semantic search over free-text tasting notes. Use it for \
subjective / flavour / aroma questions ("earthy", "bold tannins", "notes of cherry").

2. Structured Pandas tools (describe_schema, filter_wines, aggregate_wines, top_wines) — \
use these for filtering and aggregation over country, variety, province, winery, points, \
and price.

Many questions need BOTH paths — e.g. "earthy Pinots under $30 with at least 90 points": \
use semantic search for "earthy" and structured filters for the rest, then combine. \
Call describe_schema first if you are unsure which values exist. Ground every claim in \
tool results; never invent wines, scores, or prices. Cite wine titles where helpful."""

# OpenAI tool definition for the local RAG path.
RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "search_wine_descriptions",
        "description": (
            "Semantic search over wine tasting-note descriptions. Returns the most "
            "similar reviews with their metadata (title, country, variety, points, "
            "price, row_id)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of flavours/aromas to match.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def _mcp_tools_to_openai(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions into OpenAI ``tools`` entries."""
    converted: list[dict[str, Any]] = []
    for tool in mcp_tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _mcp_result_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult's content blocks into a string."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts) if parts else "[no content]"


async def _run_agent_async(history: list[dict[str, str]], max_turns: int = 8) -> str:
    """Drive the tool-use loop for one user turn and return the final assistant text."""
    client = config.get_llm_client()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
        cwd=str(SRC_DIR),  # so `-m mcp_server` resolves the flat module
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = [RAG_TOOL, *_mcp_tools_to_openai(mcp_tools)]
            mcp_tool_names = {t.name for t in mcp_tools}

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
            ]

            for _ in range(max_turns):
                response = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    tools=tools,
                )
                msg = response.choices[0].message
                if not msg.tool_calls:
                    return msg.content or ""

                messages.append(msg)  # assistant turn carrying the tool calls
                for call in msg.tool_calls:
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    name = call.function.name
                    try:
                        if name == "search_wine_descriptions":
                            content = json.dumps(rag.search_wine_descriptions(**args))
                        elif name in mcp_tool_names:
                            content = _mcp_result_to_text(
                                await session.call_tool(name, args)
                            )
                        else:
                            content = f"[unknown tool: {name}]"
                    except Exception as exc:  # surface tool errors back to the model
                        content = f"[tool error] {exc}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": content,
                        }
                    )

            return "I wasn't able to complete the request within the tool-call budget."


def run_agent(history: list[dict[str, str]]) -> str:
    """Synchronous wrapper for Streamlit: run one agent turn given the chat history.

    ``history`` is a list of ``{"role": "user"|"assistant", "content": str}`` messages.
    """
    return asyncio.run(_run_agent_async(history))
