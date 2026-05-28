"""
Multi-Agent Orchestrator — Streamlit App
==========================================
Routes questions to the EHR Agent, Payer Agent, or both.

Run:
    streamlit run multi_agent_system/orchestrator/app.py
"""

import json
import os
import sys
import textwrap

_PARENT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _PARENT)

import streamlit as st
from openai import OpenAI
from mcp_client import call_mcp_tool

# Import specialist agents
from ehr_agent.agent import run_ehr_agent, EHR_MCP_URL, get_ehr_tools
from payer_agent.agent import run_payer_agent, PAYER_MCP_URL, get_payer_tools

OPENAI_MODEL="gpt-5.4-mini"

# =====================================================================
# ORCHESTRATOR LOGIC
# =====================================================================
ORCHESTRATOR_SYSTEM = textwrap.dedent("""\
    You are a clinical orchestrator. You route questions to specialist agents
    and combine their responses into a clear, unified answer.

    You have THREE tools:

    1. call_ehr_agent — for patient clinical data questions:
       demographics, medications, conditions, observations, encounters,
       procedures, immunizations, care plans, claims, etc.

    2. call_payer_agent — for insurance / prior authorisation questions:
       insurance/ medical payer policies, coverage criteria, step therapy, PA requirements.

    3. call_both_agents — for questions that need BOTH patient data AND
       policy data. The EHR agent runs first, then its results are passed
       as context to the Payer agent.

    RULES:
    - ALWAYS use at least one tool. Never answer from your own knowledge.
    - For combined questions, use call_both_agents.
    - Assess if the question can be directly answered by the payer policy without the information from EHR agent then call payer agent only.
    - When synthesizing the final answer from call_both_agents, explicitly compare the patient's clinical data against the payer policy requirements.
    - Highlight any missing information, unfulfilled step-therapy interventions, or reasons for likely PA denial based on clinical thresholds (e.g., medication intensities, missing diagnoses, excluded drug classes).
    - Synthesise a clear final answer. Do not suggest any further clinical recommendations. Just display the final output.
""")

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_ehr_agent",
            "description": (
                "Delegate a question to the EHR specialist agent. "
                "Use for any patient related information regarding personal information, demographics, medications (existing/discontinued and new if any), conditions, "
                "doctor's observations/ notes, medical encounters, procedures, immunizations or injections taken, "
                "care plans given by doctors, insurance claims status, lab results, and patient vitals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The clinical data question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_payer_agent",
            "description": (
                """Delegate a question to the Payer Policy agent for accessing the insurance policy information, medical polices etc.
                Use only for prior authorization and step therepy coverage criteria, step therapy/preferred drugs, formulary lookups, and payer policy questions."""
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The payer policy question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_both_agents",
            "description": (
                "Delegate to BOTH agents when the question requires patient "
                "clinical data AND payer policy data. EHR runs first, then "
                "its results are passed as context to the Payer agent. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ehr_query": {
                        "type": "string",
                        "description": "Question for the EHR agent (runs first).",
                    },
                    "payer_query": {
                        "type": "string",
                        "description": "Question for the Payer agent (gets EHR context).",
                    },
                },
                "required": ["ehr_query", "payer_query"],
            },
        },
    },
]


def _exec_tool(tool_name: str, args: dict, status, tool_calls_log: list = None) -> str:
    if tool_name == "call_ehr_agent":
        status.info("🩺 **EHR Agent** is querying patient records…")
        return run_ehr_agent(args["query"], tool_calls_log=tool_calls_log)
    elif tool_name == "call_payer_agent":
        status.info("📋 **Payer Agent** is looking up policies…")
        return run_payer_agent(args["query"], tool_calls_log=tool_calls_log)
    elif tool_name == "call_both_agents":
        status.info("🩺 **EHR Agent** is querying patient records…")
        ehr = run_ehr_agent(args["ehr_query"], tool_calls_log=tool_calls_log)
        status.info("📋 **Payer Agent** is checking policies (with EHR context)…")
        payer = run_payer_agent(args["payer_query"], context=ehr, tool_calls_log=tool_calls_log)
        return (
            f"=== EHR Agent Response ===\n{ehr}\n\n"
            f"=== Payer Agent Response ===\n{payer}"
        )
    return f"Unknown tool: {tool_name}"


def run_orchestrator(user_msg: str, history: list[dict], status, tool_calls_log: list = None) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM}]
    messages += history
    messages.append({"role": "user", "content": user_msg})

    status.info("🧠 **Orchestrator** is analysing your question…")
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, tools=ORCHESTRATOR_TOOLS
    )
    msg = resp.choices[0].message

    for _ in range(5):
        if not msg.tool_calls:
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            try:
                result = _exec_tool(tc.function.name, args, status, tool_calls_log=tool_calls_log)
            except Exception as e:
                result = f"⚠️ Error in {tc.function.name}: {e}"
            
            if tool_calls_log is not None:
                tool_calls_log.append({
                    "agent": "Clinical Orchestrator",
                    "tool": tc.function.name,
                    "arguments": args,
                    "response": result
                })
            
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        status.info("🧠 **Orchestrator** is combining results…")
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, tools=ORCHESTRATOR_TOOLS
        )
        msg = resp.choices[0].message

    status.empty()
    return msg.content or "I could not generate a response."


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.set_page_config(
    page_title="Multi-Agent Orchestrator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #fff !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important; border: none; border-radius: 8px;
    padding: .5rem 1rem; font-weight: 600;
    transition: transform .15s, box-shadow .15s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(102,126,234,.4);
}
.app-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
}
.app-header h1 { margin: 0; font-size: 1.8rem; }
.app-header p  { margin: .3rem 0 0; opacity: .85; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: .75rem; font-weight: 600; margin-right: 4px; }
.badge-ehr   { background: #0ea5e9; color: #fff; }
.badge-payer { background: #f59e0b; color: #fff; }
.badge-orch  { background: linear-gradient(90deg, #0ea5e9, #f59e0b); color: #fff; }
.status-ok   { display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: .8rem; font-weight: 600; background: #00c853; color: #fff; }
.status-fail { display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: .8rem; font-weight: 600; background: #ff1744; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Orchestrator")
    st.caption("EHR Agent · Payer Agent · Combined")
    st.divider()

    st.markdown("### 🔑 API Configuration")
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        if st.button("Test API Key", use_container_width=True):
            with st.spinner("Testing API Key..."):
                try:
                    client = OpenAI(api_key=user_api_key)
                    client.models.list()
                    st.success("✅ API Key is valid!")
                except Exception as e:
                    st.error(f"❌ Invalid API Key: {e}")
    st.divider()

    # File upload (EHR)
    st.markdown("### 📁 Upload Patient Bundle")
    uploaded = st.file_uploader("FHIR JSON file", type=["json"])
    if uploaded:
        try:
            content = uploaded.read().decode("utf-8")
            data = json.loads(content)
            st.info(f"📋 **{uploaded.name}** — {len(data.get('entry', []))} entries")
            if st.button("🚀 Load Bundle", use_container_width=True):
                with st.spinner("Uploading…"):
                    try:
                        r = call_mcp_tool(EHR_MCP_URL, "upload_patient_bundle_json",
                                          {"json_content": content})
                        st.success(r)
                        st.session_state["bundle_loaded"] = True
                        st.session_state["bundle_name"] = uploaded.name
                    except Exception as e:
                        st.error(f"Failed: {e}")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON")

    if st.session_state.get("bundle_loaded"):
        st.success(f"📂 Active: **{st.session_state.get('bundle_name')}**")

    # Server status
    st.divider()
    st.markdown("### 🔗 Servers")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("EHR", use_container_width=True, key="chk_ehr"):
            with st.spinner("…"):
                try:
                    t = get_ehr_tools()
                    st.session_state["ehr_ok"] = (True, len(t))
                except Exception:
                    st.session_state["ehr_ok"] = (False, 0)
    with c2:
        if st.button("Payer", use_container_width=True, key="chk_payer"):
            with st.spinner("…"):
                try:
                    t = get_payer_tools()
                    st.session_state["payer_ok"] = (True, len(t))
                except Exception:
                    st.session_state["payer_ok"] = (False, 0)

    for key, label in [("ehr_ok", "EHR"), ("payer_ok", "Payer")]:
        val = st.session_state.get(key)
        if val and val[0]:
            st.markdown(f'<span class="status-ok">✅ {label} — {val[1]} tools</span>',
                        unsafe_allow_html=True)
        elif val and not val[0]:
            st.markdown(f'<span class="status-fail">❌ {label}</span>',
                        unsafe_allow_html=True)

    # Quick queries
    st.divider()
    st.markdown("### ⚡ Quick Queries")
    for label, q in [
        ("🩺 Conditions", "What are the patient's medical conditions?"),
        ("💊 Medications", "List the patient's active medications."),
        ("📋 PA — Diabetes", "What are the prior auth policies for diabetes meds?"),
        ("🔄 Step Therapy", "Which patient conditions are covered under step therapy?"),
        ("💊 Med Coverage", "Check which patient medications require prior auth."),
    ]:
        if st.button(label, use_container_width=True, key=f"q_{label}"):
            st.session_state["pending"] = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Main area ────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🏥 Multi-Agent Orchestrator</h1>
    <p>
        <span class="badge badge-ehr">🩺 EHR Agent</span>
        <span class="badge badge-payer">📋 Payer Agent</span>
        <span class="badge badge-orch">🔄 Orchestrator</span>
        &nbsp; Ask anything — the orchestrator routes to the right agent(s)
    </p>
</div>
""", unsafe_allow_html=True)

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pending" in st.session_state:
    st.session_state["messages"].append(
        {"role": "user", "content": st.session_state.pop("pending")}
    )

def render_tool_calls(tool_calls: list, msg_idx: int = 0):
    if not tool_calls:
        return
    with st.expander(f"⚙️ Tool Execution Logs ({len(tool_calls)} calls)", expanded=False):
        import hashlib
        for i, tc in enumerate(tool_calls):
            agent = tc.get("agent", "Unknown Agent")
            tool = tc.get("tool", "unknown_tool")
            args = tc.get("arguments", {})
            resp = tc.get("response", "")
            
            badge_class = "badge-orch" if "Orchestrator" in agent else ("badge-ehr" if "EHR" in agent else "badge-payer")
            
            st.markdown(
                f'**Tool #{i+1}:** <span class="badge {badge_class}">{agent}</span> '
                f'called **`{tool}`**',
                unsafe_allow_html=True
            )
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.caption("Arguments")
                st.json(args)
            with col2:
                st.caption("Response Output")
                if len(resp) > 300:
                    resp_hash = hashlib.md5(f"{msg_idx}_{tool}_{json.dumps(args)}_{i}".encode('utf-8')).hexdigest()[:12]
                    st.text_area("Output", resp, height=120, disabled=True, key=f"resp_ta_{resp_hash}")
                else:
                    st.code(resp)
            if i < len(tool_calls) - 1:
                st.divider()

for idx, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            render_tool_calls(msg["tool_calls"], msg_idx=idx)
        st.markdown(msg["content"])

if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    with st.chat_message("assistant"):
        status = st.empty()
        try:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"][:-1]
            ]
            tool_calls_log = []
            answer = run_orchestrator(
                st.session_state["messages"][-1]["content"], history, status, tool_calls_log=tool_calls_log
            )
        except Exception as e:
            status.empty()
            answer = f"❌ Error: {e}"
            tool_calls_log = []
        
        if tool_calls_log:
            render_tool_calls(tool_calls_log, msg_idx=len(st.session_state["messages"]))
            
        st.markdown(answer)
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "tool_calls": tool_calls_log
    })

if prompt := st.chat_input("Ask about patient records or payer policies…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.rerun()
