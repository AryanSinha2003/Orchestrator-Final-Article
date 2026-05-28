"""
EHR specialist agent — queries patient clinical data via FHIR MCP server.

Can be imported by the orchestrator:
    from ehr_agent.agent import run_ehr_agent, EHR_MCP_URL
"""

import json
import os
import sys
import textwrap

# Ensure the parent dir is on sys.path so we can import mcp_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_client import call_mcp_tool, list_mcp_tools_as_dicts, to_openai_tools

from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────
EHR_MCP_URL = os.environ.get(
    "EHR_MCP_SERVER_URL",
    "https://fhir-ehr-mcp-server.onrender.com/sse",
)
OPENAI_MODEL="gpt-5.4-mini"

UPLOAD_ONLY_TOOLS = {"upload_patient_bundle", "upload_patient_bundle_json"}

EHR_SYSTEM = textwrap.dedent("""\
    You are an EHR (Electronic Health Records) specialist agent.
    You answer questions about patient clinical data using FHIR bundle tools.

    RULES:
    - ALWAYS use tools to fetch data. NEVER fabricate medical information.
    - When asked to summarise patient details, use all medical related information such as Patient Name and Demographics, Medications, Conditions, diagnostic reports, procedures, and care plans. 
    - For the patient_name - use tool `get_patient_info` to fetch the full name of the Patient.
    - Be exhaustive with medication history: explicitly extract dosages, frequencies, and duration, and note if therapies failed or were discontinued.
    - Extract specific laboratory values (e.g., lipid panels, baseline vs. recent LDL-C, BMI) when investigating specific conditions.
    - Format results with **headers** and bullet points.
    - Be concise but thorough. Do not make any recommendations or suggestions regarding what to do next.
""")


# ── Cached tool list ─────────────────────────────────────────────────
_ehr_tools_cache: list[dict] | None = None


def get_ehr_tools() -> list[dict]:
    global _ehr_tools_cache
    if _ehr_tools_cache is None:
        _ehr_tools_cache = list_mcp_tools_as_dicts(EHR_MCP_URL)
    return _ehr_tools_cache


# ── Agent runner ─────────────────────────────────────────────────────
def run_ehr_agent(query: str, context: str = "", tool_calls_log: list = None) -> str:
    """Run the EHR specialist: OpenAI + FHIR MCP tools → answer."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    ehr_tools = to_openai_tools(get_ehr_tools(), exclude=UPLOAD_ONLY_TOOLS)

    user_content = query
    if context:
        user_content = (
            f"Context from another agent:\n{context}\n\nUser question: {query}"
        )

    messages = [
        {"role": "system", "content": EHR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, tools=ehr_tools or None
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
            try:
                result = call_mcp_tool(EHR_MCP_URL, tc.function.name, args)
            except Exception as e:
                result = f"⚠️ Error: {e}"
            
            if tool_calls_log is not None:
                tool_calls_log.append({
                    "agent": "EHR Specialist",
                    "tool": tc.function.name,
                    "arguments": args,
                    "response": result
                })

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
        response = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, tools=ehr_tools or None
        )
        msg = response.choices[0].message

    return msg.content or "No response from EHR agent."
