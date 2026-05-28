"""
Payer Policy specialist agent — retrieves PA / coverage criteria via MCP.

Can be imported by the orchestrator:
    from payer_agent.agent import run_payer_agent, PAYER_MCP_URL
"""

import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_client import call_mcp_tool, list_mcp_tools_as_dicts, to_openai_tools

from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────
PAYER_MCP_URL = os.environ.get(
    "PAYER_MCP_SERVER_URL",
    "https://tanya1280-payer-mcp-server.hf.space/mcp",
)
OPENAI_MODEL="gpt-5.4-mini"

PAYER_SYSTEM = textwrap.dedent("""\
    You are a Payer Policy / Prior Authorisation specialist agent.
    You answer questions about insurance coverage criteria, prior auth
    requirements, step therapy requirements reagding the respective payer policies.

    RULES:
    - ALWAYS use tools to retrieve policy data. NEVER invent policy rules.
    - Base your response strictly on retrieved information.
    - Structure your response clearly
    - Do not make any recommendations/Suggestions regarding anything beyond the scope of information provided.
    - If the user asks for generating a PA auth/ Step therepy Requirement documentation, based on the respective information , give the response that for a respective PA/ST auth for a specific medication/treatment based on the payer policy could/could not be generated since not in policy coverage. 
""")


# ── Cached tool list ─────────────────────────────────────────────────
_payer_tools_cache: list[dict] | None = None


def get_payer_tools() -> list[dict]:
    global _payer_tools_cache
    if _payer_tools_cache is None:
        _payer_tools_cache = list_mcp_tools_as_dicts(PAYER_MCP_URL)
    return _payer_tools_cache


# ── Agent runner ─────────────────────────────────────────────────────
def run_payer_agent(query: str, context: str = "", tool_calls_log: list = None) -> str:
    """Run the Payer specialist: OpenAI + Payer MCP tools → answer."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    payer_tools = to_openai_tools(get_payer_tools())

    user_content = query
    if context:
        user_content = (
            f"Context from another agent:\n{context}\n\nUser question: {query}"
        )

    messages = [
        {"role": "system", "content": PAYER_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, tools=payer_tools or None
    )
    msg = response.choices[0].message

    for _ in range(8):
        if not msg.tool_calls:
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
                
            # Transparently inject the OpenAI API key into the MCP tool arguments
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                args["openai_api_key"] = api_key
                
            try:
                result = call_mcp_tool(PAYER_MCP_URL, tc.function.name, args)
            except Exception as e:
                result = f"⚠️ Error: {e}"
            
            if tool_calls_log is not None:
                tool_calls_log.append({
                    "agent": "Payer Specialist",
                    "tool": tc.function.name,
                    "arguments": args,
                    "response": result
                })

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
        response = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, tools=payer_tools or None
        )
        msg = response.choices[0].message

    return msg.content or "No response from Payer agent."
