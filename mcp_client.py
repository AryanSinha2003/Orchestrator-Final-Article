"""
Shared MCP client utilities.
Supports both SSE and StreamableHTTP transports.

Usage (from any sub-folder app):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mcp_client import call_mcp_tool, list_mcp_tools, to_openai_tools
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

# ── Async ↔ Sync bridge ─────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=4)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def run_async(coro):
    """Thread-safe wrapper for running async code from Streamlit."""
    future = _executor.submit(_run_async, coro)
    return future.result(timeout=120)


# ── Transport-aware connection ───────────────────────────────────────
# URLs ending in /mcp use StreamableHTTP; everything else uses SSE.

def _is_streamable_http(url: str) -> bool:
    return url.rstrip("/").endswith("/mcp")


@asynccontextmanager
async def connect(server_url: str):
    """Open a ClientSession to any MCP server (auto-detects transport)."""
    if _is_streamable_http(server_url):
        async with streamablehttp_client(url=server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        async with sse_client(url=server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


# ── Tool calling ─────────────────────────────────────────────────────
async def _call_mcp_tool(server_url: str, tool_name: str, args: dict) -> str:
    async with connect(server_url) as session:
        result = await session.call_tool(tool_name, args)
        texts = [b.text for b in result.content if hasattr(b, "text")]
        return "\n".join(texts) if texts else str(result.content)


async def _list_mcp_tools(server_url: str):
    async with connect(server_url) as session:
        return (await session.list_tools()).tools


def call_mcp_tool(server_url: str, tool_name: str, args: dict) -> str:
    """Synchronous wrapper — call a single MCP tool."""
    return run_async(_call_mcp_tool(server_url, tool_name, args))


def list_mcp_tools(server_url: str):
    """Synchronous wrapper — list all tools from an MCP server."""
    return run_async(_list_mcp_tools(server_url))


def list_mcp_tools_as_dicts(server_url: str) -> list[dict]:
    """List tools and serialise to plain dicts (Streamlit-cacheable)."""
    raw = list_mcp_tools(server_url)
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.inputSchema or {"type": "object", "properties": {}},
        }
        for t in raw
    ]


# ── OpenAI format conversion ────────────────────────────────────────
def to_openai_tools(
    mcp_tools: list[dict], exclude: set | None = None
) -> list[dict]:
    """Convert MCP tool dicts to OpenAI function-calling format."""
    exclude = exclude or set()
    result = []
    for t in mcp_tools:
        if t["name"] in exclude:
            continue
            
        # Hide openai_api_key from the LLM
        schema = t["inputSchema"].copy()
        if "properties" in schema and "openai_api_key" in schema["properties"]:
            schema["properties"] = schema["properties"].copy()
            del schema["properties"]["openai_api_key"]
            if "required" in schema and "openai_api_key" in schema["required"]:
                schema["required"] = [req for req in schema["required"] if req != "openai_api_key"]
                
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": schema,
            },
        })
    return result
