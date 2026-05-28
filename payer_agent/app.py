"""
Payer Policy Agent — Standalone Streamlit App
===============================================
Ask questions about insurance payer policies and prior auth criteria.

Run:
    export OPENAI_API_KEY="sk-..."
    streamlit run multi_agent_system/payer_agent/app.py
"""

import os
import sys

_PARENT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _PARENT)

import streamlit as st
from openai import OpenAI
from mcp_client import list_mcp_tools_as_dicts
from payer_agent.agent import run_payer_agent, PAYER_MCP_URL, get_payer_tools

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(page_title="Payer Policy Agent", page_icon="📋", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #2d1b0e 0%, #78350f 50%, #92400e 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #fff !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white !important; border: none; border-radius: 8px;
    padding: .5rem 1rem; font-weight: 600;
}
.app-header {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
}
.app-header h1 { margin: 0; font-size: 1.8rem; }
.app-header p  { margin: .3rem 0 0; opacity: .85; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("## 📋 Payer Policy Agent")
    st.caption("Prior Auth & Coverage Criteria")
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

    # Server check
    if st.button("🔗 Check Server", use_container_width=True):
        with st.spinner("Connecting to Payer MCP server…"):
            try:
                t = get_payer_tools()
                st.success(f"✅ Connected — {len(t)} tools")
            except Exception as e:
                st.error(f"❌ {e}")

    # Quick queries
    st.divider()
    st.markdown("### ⚡ Quick Queries")
    for label, q in [
        ("💊 Diabetes Meds PA", "What are the prior auth policies for diabetes medications?"),
        ("🔄 Step Therapy", "What drugs are covered under step therapy programs?"),
        ("💉 Biologics Coverage", "What are the coverage criteria for biologic drugs?"),
        ("🩺 Cardiology PA", "What are the PA requirements for cardiology treatments?"),
        ("📋 Formulary Info", "Explain the formulary tier structure and coverage rules."),
    ]:
        if st.button(label, use_container_width=True, key=f"q_{label}"):
            st.session_state["pending"] = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# =====================================================================
# MAIN CHAT
# =====================================================================
st.markdown("""
<div class="app-header">
    <h1>📋 Payer Policy Agent</h1>
    <p>Ask about insurance payer policies, prior auth, and coverage criteria</p>
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

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("📋 Looking up payer policies…"):
            try:
                answer = run_payer_agent(st.session_state["messages"][-1]["content"])
            except Exception as e:
                answer = f"❌ Error: {e}"
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

if prompt := st.chat_input("Ask about payer policies…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.rerun()
