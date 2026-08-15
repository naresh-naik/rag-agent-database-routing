"""
RAG Agent with Database Routing - Streamlit Production AI System UI.
UI/UX Pro Max Theme with Glassmorphism, Sidebar Controls & Custom Bottom Layout.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag_agent import ConversationMemory, DocumentParser, PipelineResult, build_pipeline, run_pipeline
from rag_agent.databases import add_documents, doc_count

load_dotenv()

# -- Constants & Database Config -----------------------------------------------

DB_LABELS = {
    "products": ("🛍️", "Products DB", "#0ea5e9"),
    "support": ("🎧", "Support DB", "#10b981"),
    "financial": ("💰", "Financial DB", "#f59e0b"),
}

EXAMPLE_QUERIES = [
    "What are the specs and price of the TechPro X1 laptop?",
    "How do I reset my password?",
    "What pricing plans are available?",
    "How do I set up two-factor authentication?",
    "What were the Q1 2025 revenue figures?",
    "How do I invite team members to my workspace?",
    "What payment methods do you accept?",
    "What is the return policy for physical products?",
    "Tell me about the DataFlow Analytics Suite features.",
    "How much does the Enterprise plan cost?",
]

# -- Page Config & UI/UX Pro Max Theme -----------------------------------------

st.set_page_config(
    page_title="Production AI System - RAG Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottom"] {
    background-color: #080c14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide MainMenu and Footer only, preserve Header for Sidebar Toggle */
#MainMenu, footer { visibility: hidden !important; }

header[data-testid="stHeader"] {
    background-color: #080c14 !important;
    z-index: 100 !important;
}

/* Sidebar Open/Close Toggle Button Styling */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarToggle"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
[data-testid="stHeader"] button {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(14, 165, 233, 0.4) !important;
    color: #38bdf8 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease-in-out !important;
}

[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="stSidebarToggle"]:hover,
button[aria-label="Close sidebar"]:hover,
button[aria-label="Open sidebar"]:hover,
[data-testid="stHeader"] button:hover {
    background: rgba(14, 165, 233, 0.25) !important;
    border-color: #0ea5e9 !important;
    color: #7dd3fc !important;
    box-shadow: 0 0 10px rgba(14, 165, 233, 0.3) !important;
}

[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarToggle"] svg,
button[aria-label="Close sidebar"] svg,
button[aria-label="Open sidebar"] svg,
[data-testid="stHeader"] svg {
    fill: #38bdf8 !important;
    color: #38bdf8 !important;
}

/* Sidebar Container Styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] span { color: #cbd5e1 !important; }

/* Section Title Headers */
.sidebar-section-header {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #0ea5e9 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-top: 14px !important;
    margin-bottom: 8px !important;
}

/* Card Container */
.glass-card {
    background: rgba(17, 24, 39, 0.70) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    margin-bottom: 16px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25) !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}
[data-testid="stChatMessage"] p { color: #f1f5f9 !important; font-size: 14.5px; line-height: 1.6; }
[data-testid="stChatMessage"] code {
    background-color: #090d16 !important;
    color: #38bdf8 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

/* Streamlit Expander Styling Overrides (Fix White Clashing Box) */
[data-testid="stExpander"] {
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    background: #1e293b !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 12px 16px !important;
    border-radius: 10px !important;
    transition: all 0.2s ease-in-out !important;
}
[data-testid="stExpander"] summary * {
    color: #38bdf8 !important;
    fill: #38bdf8 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover {
    background: #334155 !important;
    color: #7dd3fc !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #0b0f19 !important;
    color: #cbd5e1 !important;
    padding: 16px !important;
    border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] li,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] div,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] span {
    color: #cbd5e1 !important;
    font-size: 13.5px !important;
    line-height: 1.6 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] code {
    background: #090d16 !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}

/* BaseWeb Input Overrides (Fix White Sidebar API Key & Input Boxes) */
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"],
div[data-baseweb="input"],
div[data-baseweb="base-input"],
.stTextInput input,
.stTextInput > div,
.stTextInput > div > div {
    background-color: #0f172a !important;
    background: #0f172a !important;
    color: #f8fafc !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

[data-testid="stTextInput"] input {
    background-color: transparent !important;
    color: #f8fafc !important;
}

/* File Uploader Dropzone Styling (Pure Dark Slate Matching Theme) */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"],
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"] {
    background-color: #0f172a !important;
    background: #0f172a !important;
    border: 2px dashed #0ea5e9 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploadDropzone"] *,
section[data-testid="stFileUploaderDropzone"] *,
div[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] svg {
    color: #f8fafc !important;
    fill: #38bdf8 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {
    background-color: #1e293b !important;
    background: #1e293b !important;
    color: #38bdf8 !important;
    border: 1px solid #0ea5e9 !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploader"] button *,
[data-testid="stFileUploaderDropzone"] button * {
    color: #38bdf8 !important;
}

/* Sidebar Text Area Styling & White Placeholder Text Overrides */
[data-testid="stTextArea"],
[data-testid="stTextArea"] textarea,
.stTextArea textarea {
    background-color: #0f172a !important;
    background: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
}

[data-testid="stTextArea"] textarea::placeholder,
.stTextArea textarea::placeholder {
    color: #ffffff !important;
    opacity: 0.95 !important;
    font-weight: 500 !important;
}




/* Single Layer Chat Input Bar Override & White Corner Artifact Removal */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background-color: #080c14 !important;
    background: #080c14 !important;
    border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInputContainer"] {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding-bottom: 12px !important;
}

/* Remove White Background & Corner Artifacts from ALL Inner BaseWeb Elements */
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] div,
[data-testid="stChatInput"] div[data-baseweb="input"],
[data-testid="stChatInput"] div[data-baseweb="base-input"] {
    background-color: transparent !important;
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] *::before,
[data-testid="stChatInput"] *::after {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
}

/* Single Unified Layer Box Container */
[data-testid="stChatInput"] > div,
.stChatInput > div {
    background-color: #0f172a !important;
    background: #0f172a !important;
    border: 1px solid rgba(14, 165, 233, 0.4) !important;
    border-radius: 24px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    outline: none !important;
    transition: all 0.2s ease-in-out !important;
}

[data-testid="stChatInput"] > div:focus-within,
.stChatInput > div:focus-within {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.25), 0 4px 20px rgba(0, 0, 0, 0.5) !important;
}

/* Textarea inside single layer container */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #f8fafc !important;
    font-size: 14.5px !important;
    padding: 10px 16px !important;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputTextArea"]:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputTextArea"]::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Submit Button inside single layer */
[data-testid="stChatInputSubmitButton"] {
    background: transparent !important;
    border: none !important;
    color: #38bdf8 !important;
}

[data-testid="stChatInputSubmitButton"] svg {
    fill: #38bdf8 !important;
    color: #38bdf8 !important;
}




/* Custom Buttons */
.stButton button {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton button:hover {
    background: rgba(51, 65, 85, 0.9) !important;
    border-color: #0ea5e9 !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 10px;
}
.badge-route {
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid #0ea5e9;
    color: #38bdf8;
}
.badge-support {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid #10b981;
    color: #34d399;
}
.badge-financial {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid #f59e0b;
    color: #fbbf24;
}
.badge-fallback {
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid #a855f7;
    color: #c084fc;
}
.badge-eval-high {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid #22c55e;
    color: #4ade80;
}
.badge-eval-warn {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid #f59e0b;
    color: #fbbf24;
}
.status-pill-ok {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid #22c55e;
    color: #4ade80;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11.5px;
    font-weight: 600;
    display: inline-block;
}
.status-pill-missing {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11.5px;
    font-weight: 600;
    display: inline-block;
}

/* Hero Status Cards Grid */
.hero-card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
    margin-bottom: 20px;
}
.hero-card-item {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 14px;
}
.hero-card-title {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.hero-card-value {
    font-size: 13.5px;
    font-weight: 600;
    color: #f8fafc;
}

/* Headers */
.hero-title {
    font-size: 26px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #94a3b8;
    font-size: 14.5px;
    margin-bottom: 16px;
    line-height: 1.5;
}
.empty-card {
    background: rgba(15, 23, 42, 0.5);
    border: 1px dashed rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    text-align: center;
    padding: 40px 20px;
    margin-top: 20px;
}
.empty-card h4 { color: #cbd5e1; font-size: 18px; margin-bottom: 6px; }
.empty-card p { color: #64748b; font-size: 13.5px; }

/* Progress & Metrics */
.metric-box {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.metric-val { font-size: 20px; font-weight: 700; color: #f8fafc; }
.metric-lbl { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# -- Session State Initialization ----------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata" not in st.session_state:
    st.session_state.metadata: dict[int, PipelineResult] = {}
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "fallback_count" not in st.session_state:
    st.session_state.fallback_count = 0
if "doc_counts" not in st.session_state:
    st.session_state.doc_counts: dict[str, int] = {"products": 0, "support": 0, "financial": 0}


@st.cache_resource(show_spinner=False)
def _shared_pipeline(api_key: str):
    """One pipeline (Qdrant client) shared by every browser session.
    Required when QDRANT_PATH persists an embedded store: opening the same
    directory from multiple clients concurrently can lose writes."""
    return build_pipeline(api_key)


def _get_pipeline(api_key: str):
    if st.session_state.pipeline is None:
        st.session_state.pipeline = _shared_pipeline(api_key)
        # Seed sidebar counts from the store so a persisted database does
        # not show 0 docs (which would invite duplicate re-uploads).
        client = st.session_state.pipeline[0]
        for name in st.session_state.doc_counts:
            st.session_state.doc_counts[name] = doc_count(client, name)
    return st.session_state.pipeline
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()


# -- Sidebar Section Cards -----------------------------------------------------

with st.sidebar:
    # Brand Card
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <div style="background:linear-gradient(135deg, #0ea5e9, #6366f1);padding:8px 12px;border-radius:10px;font-size:20px;">⚡</div>
        <div>
            <div style="font-weight:700;font-size:16px;color:#f8fafc;">Production AI System</div>
            <div style="font-size:11.5px;color:#94a3b8;">LiteParse + Guardrails + Telemetry</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Section 1: Configuration
    st.markdown('<div class="sidebar-section-header">1. SYSTEM CONFIGURATION</div>', unsafe_allow_html=True)
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        label_visibility="collapsed",
    )

    api_key = groq_api_key
    if groq_api_key:
        st.markdown('<div class="status-pill-ok">● LLM Engine Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill-missing">○ API Key Required</div>', unsafe_allow_html=True)

    st.divider()

    # Section 2: Knowledge Base Drawers
    st.markdown('<div class="sidebar-section-header">2. KNOWLEDGE BASE (LITEPARSE PARSER)</div>', unsafe_allow_html=True)

    if api_key:
        for db_key, (icon, label, color) in DB_LABELS.items():
            count = st.session_state.doc_counts[db_key]
            with st.expander(f"{icon} {label} ({count} chunks)", expanded=False):
                uploaded = st.file_uploader(
                    f"Upload {label}",
                    type=["pdf", "docx", "xlsx", "pptx", "png", "jpg", "jpeg", "webp", "txt", "md"],
                    key=f"upload_{db_key}",
                    label_visibility="collapsed",
                )
                pasted = st.text_area(
                    "Paste Text",
                    key=f"paste_{db_key}",
                    height=90,
                    placeholder="Paste document text here...",
                    label_visibility="collapsed",
                )
                if st.button(f"Ingest into {label}", key=f"add_{db_key}", use_container_width=True):
                    if st.session_state.pipeline is None:
                        with st.spinner("Initializing Vector Engine..."):
                            _get_pipeline(groq_api_key)

                    client, embeddings, groq_client = st.session_state.pipeline

                    chunks: list[str] = []
                    engine_used = "Text Chunker"
                    if uploaded is not None:
                        parsed_chunks, engine_used = DocumentParser.parse_file(
                            uploaded, uploaded.name, groq_client=groq_client
                        )
                        chunks.extend(parsed_chunks)
                    if pasted.strip():
                        pasted_chunks, _ = DocumentParser.parse_file(
                            pasted.encode("utf-8"), f"{db_key}_pasted.txt", groq_client=groq_client
                        )
                        chunks.extend(pasted_chunks)

                    if chunks:
                        with st.spinner(f"Embedding {len(chunks)} chunk(s) via {engine_used}..."):
                            added = add_documents(client, embeddings, db_key, chunks)
                        st.session_state.doc_counts[db_key] += added
                        st.success(f"Added {added} chunk(s) via {engine_used}.")
                        st.rerun()
                    else:
                        st.warning("No document content found.")
        with st.expander("📷 Direct Image Vision Inspector", expanded=False):
            test_img_file = st.file_uploader(
                "Upload Image to Inspect Text",
                type=["png", "jpg", "jpeg", "webp"],
                key="direct_vision_inspect",
            )
            if test_img_file:
                groq_cl = st.session_state.pipeline[2] if st.session_state.pipeline else None
                with st.spinner("Extracting text via Groq Vision LLM & OCR..."):
                    img_chunks, engine_lbl = DocumentParser.parse_file(
                        test_img_file, test_img_file.name, groq_client=groq_cl
                    )
                st.info(f"Engine Used: {engine_lbl}")
                st.text_area("Extracted Image Text", value="\n\n".join(img_chunks), height=140)

    st.divider()

    # Section 3: Telemetry Counters
    st.markdown('<div class="sidebar-section-header">3. TELEMETRY COUNTERS</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{st.session_state.total_queries}</div><div class="metric-lbl">Queries</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{st.session_state.fallback_count}</div><div class="metric-lbl">Fallbacks</div></div>', unsafe_allow_html=True)

    st.divider()

    # Section 4: Session Memory Controls
    st.markdown('<div class="sidebar-section-header">4. SESSION MEMORY</div>', unsafe_allow_html=True)
    turns_count = len(st.session_state.memory.messages) // 2
    st.markdown(f"<p style='font-size:12px;color:#94a3b8;'>Active Memory Turns: <strong>{turns_count} / {st.session_state.memory.max_turns}</strong></p>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Session Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.metadata = {}
        st.session_state.memory.clear()
        st.session_state.total_queries = 0
        st.session_state.fallback_count = 0
        st.rerun()

    st.divider()

    # Section 5: Quick Samples
    st.markdown('<div class="sidebar-section-header">5. QUICK DEMO QUERIES</div>', unsafe_allow_html=True)
    for q in EXAMPLE_QUERIES:
        if st.button(q[:50] + ("..." if len(q) > 50 else ""), use_container_width=True, key=f"ex_{q}"):
            st.session_state.pending_query = q


# -- Main Workspace Header -----------------------------------------------------

st.markdown("""
<div>
    <div class="hero-title">Production AI System Architecture</div>
    <div class="hero-sub">
        Enterprise RAG Pipeline featuring <strong style="color:#0ea5e9;">LiteParse Layout Engine</strong>,
        <strong style="color:#10b981;">Session Memory</strong>,
        <strong style="color:#f59e0b;">Observability Tracing</strong>, and
        <strong style="color:#c084fc;">Faithfulness Evaluation</strong>.
    </div>
    <div class="hero-card-grid">
        <div class="hero-card-item">
            <div class="hero-card-title">Document Parser</div>
            <div class="hero-card-value" style="color:#38bdf8;">⚡ LiteParse Multi-Format Engine (PDF, DOCX, XLSX, PPTX, Images)</div>
        </div>
        <div class="hero-card-item">
            <div class="hero-card-title">Safety Guardrails</div>
            <div class="hero-card-value" style="color:#34d399;">🛡️ Enforced (Prompt Injection Protection)</div>
        </div>
        <div class="hero-card-item">
            <div class="hero-card-title">Faithfulness Evaluator</div>
            <div class="hero-card-value" style="color:#fbbf24;">🎯 Active (Deterministic Confidence Scoring)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -- Chat Stream & Metadata Cards ----------------------------------------------

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and i in st.session_state.metadata:
            result = st.session_state.metadata[i]
            icon, label, color = DB_LABELS.get(result.routing.database, ("🔀", "Unknown", "#64748b"))

            # Metadata Badges Row
            if result.used_fallback:
                st.markdown('<div class="badge badge-fallback">🌐 Web Fallback Agent</div>', unsafe_allow_html=True)
            else:
                badge_class = "badge-route" if result.routing.database == "products" else "badge-support" if result.routing.database == "support" else "badge-financial"
                st.markdown(f'<div class="badge {badge_class}">{icon} Routed to {label}</div>', unsafe_allow_html=True)

            eval_class = "badge-eval-high" if result.evaluation.is_faithful else "badge-eval-warn"
            st.markdown(
                f'<div class="badge {eval_class}">🎯 Score: {result.evaluation.groundedness_score:.2f} ({result.evaluation.status_label})</div>',
                unsafe_allow_html=True,
            )

        st.markdown(message["content"])

        if message["role"] == "assistant" and i in st.session_state.metadata:
            result = st.session_state.metadata[i]
            if result.docs:
                with st.expander(f"📑 Grounded Sources ({len(result.docs)} documents)", expanded=False):
                    for j, doc in enumerate(result.docs):
                        st.markdown(f"**[{j+1}]** *(Similarity Score: `{doc.score:.2f}`)*\n\n{doc.text}")

            with st.expander("⚡ System Telemetry & Execution Trace", expanded=False):
                st.markdown(f"**Routing Logic:** *{result.routing.reasoning}*")
                st.markdown(f"**Total Pipeline Latency:** `{result.trace.total_latency_ms:.1f} ms`")
                st.markdown("**Phase Latency Breakdown:**")
                for s in result.trace.steps:
                    st.markdown(f"- `{s.step_name}`: **{s.latency_ms:.1f} ms** *(Details: {s.details})*")

total_docs = sum(st.session_state.doc_counts.values())

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-card">
        <h4>System Ready for Queries</h4>
        <p>Your prompt will be processed through Input Guardrails, routed to the target Qdrant Vector Store,<br>
        and evaluated for faithfulness with millisecond-level telemetry tracing.</p>
    </div>
    """, unsafe_allow_html=True)

# -- Input Handling ------------------------------------------------------------

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    prompt = st.chat_input(
        "Ask a question about your knowledge base...",
        disabled=(not api_key),
    )

# -- Execution Stream ----------------------------------------------------------

if prompt:
    if not groq_api_key:
        st.error("Enter your Groq API Key in the sidebar to proceed.")
        st.stop()

    if st.session_state.pipeline is None:
        with st.spinner("Initializing Vector Stores & Agents..."):
            _get_pipeline(groq_api_key)

    client, embeddings, groq_client = st.session_state.pipeline

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing (Guardrails -> Route -> Retrieve -> Eval)..."):
            try:
                result = run_pipeline(
                    query=prompt,
                    client=client,
                    embeddings=embeddings,
                    groq_client=groq_client,
                    memory=st.session_state.memory,
                )

                icon, label, color = DB_LABELS.get(result.routing.database, ("🔀", "Unknown", "#64748b"))

                if result.used_fallback:
                    st.markdown('<div class="badge badge-fallback">🌐 Web Fallback Agent</div>', unsafe_allow_html=True)
                    st.session_state.fallback_count += 1
                else:
                    badge_class = "badge-route" if result.routing.database == "products" else "badge-support" if result.routing.database == "support" else "badge-financial"
                    st.markdown(f'<div class="badge {badge_class}">{icon} Routed to {label}</div>', unsafe_allow_html=True)

                eval_class = "badge-eval-high" if result.evaluation.is_faithful else "badge-eval-warn"
                st.markdown(
                    f'<div class="badge {eval_class}">🎯 Score: {result.evaluation.groundedness_score:.2f} ({result.evaluation.status_label})</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(result.answer)

                if result.docs:
                    with st.expander(f"📑 Grounded Sources ({len(result.docs)} documents)", expanded=False):
                        for j, doc in enumerate(result.docs):
                            st.markdown(f"**[{j+1}]** *(Similarity Score: `{doc.score:.2f}`)*\n\n{doc.text}")

                with st.expander("⚡ System Telemetry & Execution Trace", expanded=False):
                    st.markdown(f"**Routing Logic:** *{result.routing.reasoning}*")
                    st.markdown(f"**Total Pipeline Latency:** `{result.trace.total_latency_ms:.1f} ms`")
                    st.markdown("**Phase Latency Breakdown:**")
                    for s in result.trace.steps:
                        st.markdown(f"- `{s.step_name}`: **{s.latency_ms:.1f} ms** *(Details: {s.details})*")

                msg_index = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
                st.session_state.metadata[msg_index] = result
                st.session_state.total_queries += 1

            except Exception as e:
                error_msg = f"**Error:** {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
