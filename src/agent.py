"""Azure OpenAI tool-use loop wired to two retrieval paths: RAG and MCP."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
import rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

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
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for tool in mcp_tools
    ]


def _mcp_result_to_text(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    texts = [b.text for b in blocks if getattr(b, "text", None) is not None]
    return "\n".join(texts) if texts else "[no content]"


async def _call_tool(session: ClientSession, mcp_tool_names: set[str], name: str, args: dict) -> str:
    try:
        if name == "search_wine_descriptions":
            logger.info("Routing to RAG: %s(%s)", name, args)
            return json.dumps(rag.search_wine_descriptions(**args))
        if name in mcp_tool_names:
            logger.info("Routing to MCP: %s(%s)", name, args)
            return _mcp_result_to_text(await session.call_tool(name, args))
        logger.warning("Model called unknown tool: %s", name)
        return f"[unknown tool: {name}]"
    except Exception as exc:
        logger.exception("Tool %s raised an error", name)
        return f"[tool error] {exc}"


async def _run_agent_async(history: list[dict[str, str]], max_turns: int = 8) -> str:
    client = config.get_llm_client()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
        cwd=str(SRC_DIR),
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            mcp_tool_names = {t.name for t in mcp_tools}
            tools = [RAG_TOOL, *_mcp_tools_to_openai(mcp_tools)]

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

                messages.append(msg)
                for call in msg.tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        logger.warning("Bad arguments for %s: %r", name, call.function.arguments)
                        args = {}

                    content = await _call_tool(session, mcp_tool_names, name, args)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

            logger.warning("Hit max tool-call budget (%d iterations).", max_turns)
            return "I wasn't able to complete the request within the tool-call budget."


def run_agent(history: list[dict[str, str]]) -> str:
    """Synchronous entry point for Streamlit: run one agent turn given the chat history."""
    return asyncio.run(_run_agent_async(history))
