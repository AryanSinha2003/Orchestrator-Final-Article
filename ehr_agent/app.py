"""
EHR Agent — Standalone Streamlit App
=====================================
Upload a FHIR patient bundle and query patient clinical data.

Run:
    export OPENAI_API_KEY="sk-..."
    streamlit run multi_agent_system/ehr_agent/app.py
"""

import json
import os
import sys

# ── Path setup ───────────────────────────────────────────────────────
_PARENT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _PARENT)

import streamlit as st
from openai import OpenAI
from mcp_client import call_mcp_tool, list_mcp_tools_as_dicts
from ehr_agent.agent import run_ehr_agent, EHR_MCP_URL, get_ehr_tools

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(page_title="EHR Agent", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0a1628 0%, #1a365d 50%, #1e3a5f 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #fff !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    color: white !important; border: none; border-radius: 8px;
    padding: .5rem 1rem; font-weight: 600;
}
.app-header {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
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
    st.markdown("## 🩺 EHR Agent")
    st.caption("Patient Clinical Data")
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

    # File upload
    st.markdown("### 📁 Upload Patient Bundle")
    uploaded = st.file_uploader("FHIR JSON file", type=["json"])
    if uploaded:
        try:
            content = uploaded.read().decode("utf-8")
            data = json.loads(content)
            entry_count = len(data.get("entry", []))
            st.info(f"📋 **{uploaded.name}** — {entry_count} entries")
            if st.button("🚀 Load Bundle", use_container_width=True):
                with st.spinner("Uploading…"):
                    try:
                        r = call_mcp_tool(EHR_MCP_URL, "upload_patient_bundle_json",
                                          {"json_content": content})
                        st.success(r)
                        st.session_state["bundle_loaded"] = True
                        st.session_state["bundle_name"] = uploaded.name
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON")

    if st.session_state.get("bundle_loaded"):
        st.success(f"📂 Active: **{st.session_state.get('bundle_name')}**")

    # Server check
    st.divider()
    if st.button("🔗 Check Server", use_container_width=True):
        with st.spinner("…"):
            try:
                t = get_ehr_tools()
                st.success(f"✅ Connected — {len(t)} tools")
            except Exception as e:
                st.error(f"❌ {e}")

    # Quick queries
    st.divider()
    st.markdown("### ⚡ Quick Queries")
    for label, q in [
        ("👤 Patient Info", "Show the patient's demographic information."),
        ("🩺 Conditions", "What are all the patient's medical conditions?"),
        ("💊 Medications", "List all active medications with dosage."),
        ("🔬 Lab Results", "Show recent lab results and vital signs."),
        ("🏥 Encounters", "List all clinical encounters."),
        ("💉 Immunizations", "Show vaccination history."),
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
    <h1>🩺 EHR Agent</h1>
    <p>Upload a FHIR patient bundle → ask questions about patient records</p>
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
        with st.spinner("🔍 Querying patient records…"):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"][:-1]
                ]
                answer = run_ehr_agent(st.session_state["messages"][-1]["content"])
            except Exception as e:
                answer = f"❌ Error: {e}"
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

if prompt := st.chat_input("Ask about the patient's records…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.rerun()
