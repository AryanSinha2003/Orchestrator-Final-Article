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
    "https://ary-007-payer-policy-mcp-server.hf.space/sse",
)
OPENAI_MODEL="gpt-5.4-mini"

PAYER_SYSTEM = textwrap.dedent("""\
    You are a Payer Policy / Prior Authorisation specialist agent.
    You answer questions about insurance coverage criteria, prior auth requirements, and step therapy requirements regarding payer policies.

    RULES:
    - ALWAYS use tools to retrieve policy data. NEVER invent policy rules.
    - Base your response strictly on retrieved information.
    - Actively look for specific exclusions, contraindications, or strictly enforced diagnostic requirements (e.g., verifying if off-label use is strictly excluded).
    - When evaluating clinical context against policy, explicitly check thresholds (e.g., is a medication "high intensity" or "maximally tolerated"? Do lab values meet exact numerical criteria?).
    - Explicitly identify mandatory step-therapy interventions (e.g., failed trials of preferred agents, specific procedures, or conservative therapies) and flag if they are missing from the patient's history.
    - When discussing preferred vs. non-preferred agents, state the specific criteria required to override the preference (e.g., documented intolerance, severe contraindication, inadequate response).
    - Structure your response clearly.
    - Do not make any recommendations regarding anything beyond the scope of information provided.
    - If asked to generate a PA/ST requirement document, state whether it could or could not be generated based on policy coverage.
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
            f"Context from EHR agent:\n{context}\n\n User question: {query}"
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
                    #"arguments": args,
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
