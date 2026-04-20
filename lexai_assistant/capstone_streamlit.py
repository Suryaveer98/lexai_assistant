"""
capstone_streamlit.py — LexAI Legal Assistant UI
==================================================
Run: streamlit run capstone_streamlit.py

Features:
- Login screen with name capture
- Multi-turn chat with persistent memory
- Session stats sidebar
- New Conversation button
- Admin dashboard (password: admin123)
"""

import streamlit as st
import time
import json
from pathlib import Path
from datetime import datetime

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAI — Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load agent (cached — initialises only once per server lifetime) ──────────
@st.cache_resource
def load_agent():
    from agent import app, ask, DOCUMENTS, load_profile
    return app, ask, DOCUMENTS, load_profile

app, ask, DOCUMENTS, load_profile = load_agent()

# ─── Session State Defaults ───────────────────────────────────────────────────
if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
if "user_name"   not in st.session_state: st.session_state.user_name   = ""
if "user_id"     not in st.session_state: st.session_state.user_id     = ""
if "messages"    not in st.session_state: st.session_state.messages    = []
if "thread_id"   not in st.session_state: st.session_state.thread_id   = "session-1"
if "session_num" not in st.session_state: st.session_state.session_num = 1
if "admin_mode"  not in st.session_state: st.session_state.admin_mode  = False

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1a3a5c; }
    .sub-title   { color: #555; font-size: 1rem; margin-bottom: 1.5rem; }
    .stat-card   { background: #f0f4f8; border-radius: 10px; padding: 0.8rem 1rem;
                   margin-bottom: 0.5rem; border-left: 4px solid #1a3a5c; }
    .disclaimer  { background: #fff3cd; padding: 0.7rem 1rem; border-radius: 8px;
                   font-size: 0.82rem; color: #856404; margin-top: 1rem; }
    .user-bubble { background: #1a3a5c; color: white; padding: 0.7rem 1rem;
                   border-radius: 12px 12px 2px 12px; margin-bottom: 0.3rem; }
    .bot-bubble  { background: #f0f4f8; padding: 0.7rem 1rem;
                   border-radius: 12px 12px 12px 2px; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ─── Login Screen ─────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-title">⚖️ LexAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Your AI-powered Legal Research Assistant for Indian Law</div>',
                    unsafe_allow_html=True)
        st.divider()
        name = st.text_input("Your Name", placeholder="e.g. Rahul Sharma")
        role = st.selectbox("Your Role", ["Paralegal", "Junior Lawyer", "Law Student", "Individual / Citizen"])
        if st.button("Enter LexAI →", type="primary", use_container_width=True):
            if name.strip():
                st.session_state.logged_in = True
                st.session_state.user_name = name.strip().capitalize()
                st.session_state.user_id   = name.strip().lower().replace(" ", "_") + "_" + role[:3].lower()
                st.session_state.messages  = []
                st.rerun()
            else:
                st.error("Please enter your name to continue.")
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> LexAI provides general legal information from its knowledge base only.
        It is <b>not</b> a substitute for advice from a licensed advocate. Always consult a qualified
        legal professional for matters specific to your situation.
        </div>""", unsafe_allow_html=True)
    st.stop()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### ⚖️ LexAI")
    st.markdown(f"👤 **{st.session_state.user_name}**")
    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.session_num += 1
        st.session_state.thread_id  = f"session-{st.session_state.session_num}"
        st.session_state.messages   = []
        st.rerun()

    st.divider()
    st.markdown("**📚 Knowledge Base Topics**")
    for doc in DOCUMENTS:
        st.markdown(f"- {doc['topic']}")

    st.divider()
    st.markdown("**📊 Session Stats**")
    profile = load_profile(st.session_state.user_id)
    st.markdown(f"""
    <div class="stat-card">Total Queries: <b>{profile.get('queries', 0)}</b></div>
    <div class="stat-card">Topics Explored: <b>{len(profile.get('topics_asked', []))}</b></div>
    """, unsafe_allow_html=True)

    st.divider()
    admin_pw = st.text_input("🔐 Admin Password", type="password")
    if st.button("Admin Dashboard"):
        if admin_pw == "admin123":
            st.session_state.admin_mode = True
        else:
            st.error("Incorrect password")

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#888;'>
    🆘 <b>NALSA Helpline:</b> 15100<br>
    For emergency legal aid (arrest, custody)
    </div>""", unsafe_allow_html=True)


# ─── Admin Dashboard ──────────────────────────────────────────────────────────
if st.session_state.admin_mode:
    st.markdown("## 🔐 Admin Dashboard")
    profile_dir = Path("user_profiles")
    if profile_dir.exists():
        profiles = list(profile_dir.glob("*.json"))
        if profiles:
            st.markdown(f"**Registered Users: {len(profiles)}**")
            for p in profiles:
                data = json.loads(p.read_text())
                with st.expander(f"👤 {data.get('name', p.stem)} — {data.get('queries', 0)} queries"):
                    st.json(data)
        else:
            st.info("No user profiles yet.")
    if st.button("← Back to Chat"):
        st.session_state.admin_mode = False
        st.rerun()
    st.stop()


# ─── Main Chat UI ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚖️ LexAI — Legal Assistant</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Hello, {st.session_state.user_name}! Ask me anything about Indian law.</div>',
            unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(msg["content"])
            if msg.get("meta"):
                meta = msg["meta"]
                cols = st.columns(3)
                cols[0].caption(f"🔀 Route: `{meta.get('route', '—')}`")
                cols[1].caption(f"🎯 Faithfulness: `{meta.get('faithfulness', '—')}`")
                cols[2].caption(f"📂 Sources: {', '.join(meta.get('sources', [])) or '—'}")

# Input
if prompt := st.chat_input("Ask a legal question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # Generate answer
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Researching Indian law..."):
            result = ask(
                prompt,
                thread_id=st.session_state.thread_id,
                user_id=st.session_state.user_id,
            )
        answer = result.get("answer", "I'm sorry, I could not generate a response. Please try again.")
        st.markdown(answer)

        meta = {
            "route": result.get("route", "—"),
            "faithfulness": round(result.get("faithfulness", 1.0), 2),
            "sources": result.get("sources", []),
        }
        cols = st.columns(3)
        cols[0].caption(f"🔀 Route: `{meta['route']}`")
        cols[1].caption(f"🎯 Faithfulness: `{meta['faithfulness']}`")
        cols[2].caption(f"📂 Sources: {', '.join(meta['sources']) or '—'}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": meta,
    })
