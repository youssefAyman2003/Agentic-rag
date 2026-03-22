import os
import streamlit as st
from dotenv import load_dotenv
 
from vectorstore import ingest_pdfs
from graders import build_all_graders
from graph import build_graph, run_graph
 
# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-RAG · Neural Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');
 
/* ── CSS Variables ── */
:root {
    --bg-deep:       #0a0c14;
    --bg-card:       #111422;
    --bg-glass:      rgba(20, 24, 42, 0.80);
    --border:        rgba(99, 102, 241, 0.18);
    --border-bright: rgba(99, 102, 241, 0.55);
    --accent-violet: #6366f1;
    --accent-cyan:   #22d3ee;
    --accent-rose:   #fb7185;
    --accent-amber:  #fbbf24;
    --accent-emerald:#34d399;
    --text-primary:  #f1f5f9;
    --text-muted:    #94a3b8;
    --text-dim:      #475569;
    --glow-violet:   0 0 24px rgba(99,102,241,0.35);
    --glow-cyan:     0 0 24px rgba(34,211,238,0.35);
}
 
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}
 
/* Animated mesh background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(34,211,238,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 60% 30%, rgba(251,113,133,0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}
 
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f1e 0%, #10132a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }
 
/* Sidebar brand header */
.sidebar-brand {
    text-align: center;
    padding: 1.4rem 0 1rem;
}
.sidebar-brand .logo-circle {
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-violet), var(--accent-cyan));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    margin: 0 auto 0.7rem;
    box-shadow: var(--glow-violet);
}
.sidebar-brand h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    background: linear-gradient(90deg, var(--accent-violet), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}
.sidebar-brand p {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin: 0.2rem 0 0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
 
/* Stat badge */
.stat-badge {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    background: rgba(99,102,241,0.10);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    margin: 0.4rem 0;
}
.stat-badge .icon { font-size: 1.1rem; }
.stat-badge .label {
    font-size: 0.75rem;
    color: var(--text-dim);
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.stat-badge .value {
    margin-left: auto;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--accent-cyan);
    font-family: 'Syne', sans-serif;
}
 
/* Section label */
.section-label {
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-family: 'Space Mono', monospace;
    margin: 1.1rem 0 0.4rem;
}
 
/* Upload zone */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.06) !important;
    border: 1.5px dashed var(--border-bright) !important;
    border-radius: 12px !important;
    transition: border-color 0.25s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-violet) !important;
}
 
/* API key input */
[data-testid="stTextInput"] input {
    background: rgba(15,18,35,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-violet) !important;
    box-shadow: var(--glow-violet) !important;
}
 
/* Buttons */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(34,211,238,0.10)) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.30), rgba(34,211,238,0.20)) !important;
    border-color: var(--accent-violet) !important;
    box-shadow: var(--glow-violet) !important;
    transform: translateY(-1px) !important;
}
 
/* Divider */
hr { border-color: var(--border) !important; }
 
/* ── Trace steps ── */
.trace-step {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.45rem 0.7rem;
    margin: 0.25rem 0;
    background: rgba(99,102,241,0.07);
    border-left: 3px solid var(--accent-violet);
    border-radius: 0 8px 8px 0;
    font-size: 0.8rem;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    line-height: 1.45;
}
.trace-step::before { content: '▸'; color: var(--accent-violet); }
 
/* ── Main area ── */
[data-testid="block-container"] {
    padding-top: 1.5rem !important;
    position: relative;
    z-index: 1;
}
 
/* Hero header */
.hero-header {
    text-align: center;
    padding: 1.2rem 0 0.6rem;
    margin-bottom: 0.5rem;
}
.hero-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    background: linear-gradient(135deg, #f1f5f9 0%, var(--accent-violet) 45%, var(--accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.hero-header .subtitle {
    color: var(--text-muted);
    font-size: 0.88rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.04em;
}
 
/* Pipeline badges */
.pipeline-badges {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.55rem;
    margin: 0.8rem auto 1.4rem;
}
.pipe-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid;
}
.pipe-badge.violet {
    background: rgba(99,102,241,0.12);
    border-color: rgba(99,102,241,0.4);
    color: #a5b4fc;
}
.pipe-badge.cyan {
    background: rgba(34,211,238,0.10);
    border-color: rgba(34,211,238,0.35);
    color: #67e8f9;
}
.pipe-badge.rose {
    background: rgba(251,113,133,0.10);
    border-color: rgba(251,113,133,0.35);
    color: #fda4af;
}
.pipe-badge.amber {
    background: rgba(251,191,36,0.10);
    border-color: rgba(251,191,36,0.35);
    color: #fcd34d;
}
 
/* Info card */
.info-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(34,211,238,0.05));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    max-width: 520px;
    margin: 1.5rem auto;
}
.info-card .big-icon { font-size: 3rem; margin-bottom: 1rem; }
.info-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
}
.info-card p {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.6;
}
 
/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
 
/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has(.user) {
    background: transparent !important;
}
 
/* Message content wrappers */
.user-bubble {
    background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(99,102,241,0.12));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    margin: 0.2rem 0 0.2rem 15%;
    font-size: 0.92rem;
    line-height: 1.65;
    color: var(--text-primary);
    box-shadow: 0 4px 20px rgba(99,102,241,0.15);
}
.assistant-bubble {
    background: linear-gradient(135deg, rgba(20,24,45,0.95), rgba(15,18,38,0.90));
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.2rem 15% 0.2rem 0;
    font-size: 0.92rem;
    line-height: 1.65;
    color: var(--text-primary);
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
 
/* ── Source expander ── */
[data-testid="stExpander"] {
    background: rgba(15,18,35,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--accent-cyan) !important;
}
 
/* Source chunk card */
.chunk-card {
    background: rgba(99,102,241,0.05);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
.chunk-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-violet);
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.chunk-number {
    background: var(--accent-violet);
    color: white;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 700;
    flex-shrink: 0;
}
.chunk-text {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.6;
    font-family: 'DM Sans', sans-serif;
}
 
/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(15,18,35,0.9) !important;
    border: 1.5px solid var(--border-bright) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-violet) !important;
    box-shadow: var(--glow-violet) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
}
 
/* ── Alert/success/error ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
}
 
/* Spinner */
[data-testid="stSpinner"] {
    color: var(--accent-violet) !important;
}
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--accent-violet), var(--accent-cyan));
    border-radius: 99px;
}
 
/* ── Spinner overlay card ── */
.thinking-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(99,102,241,0.08);
    border: 1px solid var(--border-bright);
    border-radius: 12px;
    padding: 0.75rem 1.1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent-violet);
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)
 
 
# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "vectorstore":   None,
    "retriever":     None,
    "graders":       None,
    "graph":         None,
    "messages":      [],
    "last_steps":    [],
    "num_chunks":    0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo-circle">🧠</div>
        <h1>Self-RAG</h1>
        <p>Neural Document Chat</p>
    </div>
    """, unsafe_allow_html=True)
 
    st.divider()
 
    # API Key
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        st.markdown('<div class="section-label">🔑 API Configuration</div>', unsafe_allow_html=True)
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
            label_visibility="collapsed",
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
            st.success("✓ API key saved")
 
    st.divider()
 
    # PDF Uploader
    st.markdown('<div class="section-label">📄 Knowledge Base</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
 
    if uploaded_files and st.session_state.retriever is None:
        with st.spinner("⚙️  Indexing documents..."):
            try:
                retriever, num_chunks = ingest_pdfs(uploaded_files)
                st.session_state.retriever  = retriever
                st.session_state.num_chunks = num_chunks
 
                if st.session_state.graders is None:
                    st.session_state.graders = build_all_graders()
 
                st.session_state.graph = build_graph(
                    retriever=st.session_state.retriever,
                    graders=st.session_state.graders,
                )
                st.success(f"✅  Indexed {len(uploaded_files)} file(s)")
            except Exception as e:
                st.error(f"Error: {e}")
 
    # Stats
    if st.session_state.retriever is not None:
        st.markdown('<div class="section-label">📊 Index Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-badge">
            <span class="icon">📁</span>
            <span class="label">Files</span>
            <span class="value">{len(uploaded_files) if uploaded_files else '–'}</span>
        </div>
        <div class="stat-badge">
            <span class="icon">🧩</span>
            <span class="label">Chunks</span>
            <span class="value">{st.session_state.num_chunks}</span>
        </div>
        <div class="stat-badge">
            <span class="icon">💬</span>
            <span class="label">Turns</span>
            <span class="value">{len([m for m in st.session_state.messages if m['role']=='user'])}</span>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️  Clear & Reset", use_container_width=True):
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()
 
    st.divider()
 
    # Trace
    if st.session_state.last_steps:
        st.markdown('<div class="section-label">🔎 Last Query Trace</div>', unsafe_allow_html=True)
        with st.expander("View Self-RAG Steps", expanded=True):
            for step in st.session_state.last_steps:
                st.markdown(f'<div class="trace-step">{step}</div>', unsafe_allow_html=True)
 
    # Footer
    st.markdown("""
    <div style="position:absolute;bottom:1.2rem;left:0;right:0;text-align:center;">
        <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#334155;letter-spacing:0.1em;">
            POWERED BY GROQ · LANGGRAPH
        </span>
    </div>
    """, unsafe_allow_html=True)
 
 
# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>Chat with your Documents</h1>
    <p class="subtitle">Self-Reflective Retrieval Augmented Generation</p>
</div>
<div class="pipeline-badges">
    <span class="pipe-badge violet">🔍 Retrieval Grading</span>
    <span class="pipe-badge cyan">🛡️ Hallucination Check</span>
    <span class="pipe-badge rose">✅ Answer Grading</span>
    <span class="pipe-badge amber">✏️ Query Rewriting</span>
</div>
""", unsafe_allow_html=True)
 
# Welcome state
if st.session_state.retriever is None:
    st.markdown("""
    <div class="info-card">
        <div class="big-icon">📂</div>
        <h3>Upload your PDFs to begin</h3>
        <p>
            Drop one or more PDF files in the sidebar. The system will chunk, embed,
            and index them locally — then use Groq + Self-RAG quality gates to answer
            your questions with cited sources.
        </p>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "🔬", "Retrieval Grading", "Each chunk is scored for relevance before use"),
        (col2, "🧬", "Hallucination Guard", "Answers are checked against source docs"),
        (col3, "🔄", "Auto Query Rewrite", "Poor answers trigger smarter re-queries"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.2);
                        border-radius:14px;padding:1.2rem;text-align:center;height:100%;">
                <div style="font-size:2rem;margin-bottom:0.6rem;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;
                            color:#f1f5f9;margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.78rem;color:#94a3b8;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()
 
# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
 
# Chat input
if prompt := st.chat_input("Ask something about your documents…"):
 
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()
 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)
 
    with st.chat_message("assistant"):
        st.markdown("""
        <div class="thinking-card">
            ⚡ Running Self-RAG pipeline — grading, generating, verifying…
        </div>
        """, unsafe_allow_html=True)
 
        with st.spinner(""):
            try:
                result = run_graph(st.session_state.graph, prompt)
 
                answer = result.get("generation", "").strip()
                steps  = result.get("steps", [])
 
                if not answer:
                    answer = (
                        "I couldn't find a satisfactory answer in the uploaded documents. "
                        "Try rephrasing your question or uploading more relevant PDFs."
                    )
 
                st.markdown(f'<div class="assistant-bubble">{answer}</div>', unsafe_allow_html=True)
 
                # Source chunks
                docs = result.get("documents", [])
                if docs:
                    pages_by_file = {}
                    for doc in docs:
                        src  = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "?")
                        pages_by_file.setdefault(src, set()).add(page)
 
                    pages_summary = " · ".join(
                        f"{src} p.{', '.join(str(p) for p in sorted(pages, key=lambda x: (isinstance(x, str), x)))}"
                        for src, pages in pages_by_file.items()
                    )
 
                    with st.expander(f"📎 {len(docs)} source chunk(s)  ·  {pages_summary}"):
                        for i, doc in enumerate(docs, 1):
                            src  = doc.metadata.get("source", "unknown")
                            page = doc.metadata.get("page", "?")
                            preview = doc.page_content[:400]
                            ellipsis = "…" if len(doc.page_content) > 400 else ""
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div class="chunk-header">
                                    <span class="chunk-number">{i}</span>
                                    📄 {src} — Page {page}
                                </div>
                                <div class="chunk-text">{preview}{ellipsis}</div>
                            </div>
                            """, unsafe_allow_html=True)
 
            except Exception as e:
                answer = f"An error occurred: {e}"
                steps  = []
                st.error(answer)
 
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_steps = steps
    st.rerun()