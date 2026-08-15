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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---------- Design tokens ---------- */
:root {
    --bg: #070b12;
    --surface: #0d1424;
    --surface-2: #111a2e;
    --line: rgba(148, 163, 184, 0.14);
    --text: #e6edf7;
    --muted: #8b98ad;
    --accent: #38bdf8;
    --accent-2: #818cf8;
    --grad: linear-gradient(135deg, #38bdf8, #818cf8);
}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottom"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Ambient aurora glow behind the workspace */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(55% 32% at 16% -4%, rgba(56, 189, 248, 0.13), transparent 60%),
        radial-gradient(45% 28% at 88% 0%, rgba(129, 140, 248, 0.11), transparent 55%);
}
[data-testid="stAppViewContainer"] > div { position: relative; z-index: 1; }

/* Hide MainMenu and Footer only, preserve Header for Sidebar Toggle */
#MainMenu, footer { visibility: hidden !important; }

/* Slimmer, softer dividers */
hr { border-color: var(--line) !important; opacity: 0.7; margin: 1rem 0 !important; }

header[data-testid="stHeader"] {
    background-color: transparent !important;
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
    background-color: #0b1120 !important;
    border-right: 1px solid var(--line) !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] > div {
    background: transparent !important;
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
    display: flex;
    align-items: center;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    letter-spacing: 1.4px !important;
    text-transform: uppercase !important;
    margin-top: 16px !important;
    margin-bottom: 10px !important;
}
.sidebar-section-header::before {
    content: "";
    width: 14px;
    height: 2px;
    border-radius: 2px;
    background: var(--grad);
    margin-right: 8px;
    flex-shrink: 0;
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
    background: linear-gradient(180deg, rgba(17, 26, 46, 0.72), rgba(13, 20, 36, 0.72)) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 6px 18px rgba(2, 6, 16, 0.35) !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stUserMessage"] {
    border-left: 3px solid var(--accent) !important;
}
[data-testid="stAssistantMessage"] {
    border-left: 3px solid var(--accent-2) !important;
}
[data-testid="stChatMessage"] p { color: #f1f5f9 !important; font-size: 14.5px; line-height: 1.65; }
[data-testid="stChatMessage"] code {
    background-color: #090d16 !important;
    color: #7dd3fc !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    border: 1px solid rgba(56, 189, 248, 0.18) !important;
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
    background: rgba(30, 41, 59, 0.65) !important;
    border: 1px solid var(--line) !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    transition: all 0.18s ease-in-out !important;
}
.stButton button:hover {
    background: rgba(51, 65, 85, 0.85) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(56, 189, 248, 0.18) !important;
}

/* Quick demo query chips: compact, left-aligned, no wasted height.
   The .demo-query-btn div renders as the sibling directly before the
   button's block (Streamlit auto-closes the div), hence the + selector. */
.demo-query-btn + div button {
    width: 100%;
    text-align: left !important;
    padding: 8px 12px !important;
    border-radius: 9px !important;
    white-space: normal !important;
    line-height: 1.35 !important;
}

/* Danger-style reset button (same sibling trick) */
.reset-btn + div button {
    border: 1px solid rgba(248, 113, 113, 0.35) !important;
    color: #fca5a5 !important;
}
.reset-btn + div button:hover {
    background: rgba(239, 68, 68, 0.12) !important;
    border-color: #ef4444 !important;
    color: #fecaca !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.15) !important;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 10px;
    backdrop-filter: blur(6px);
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
    gap: 14px;
    margin-top: 16px;
    margin-bottom: 22px;
}
@media (max-width: 900px) {
    .hero-card-grid { grid-template-columns: 1fr; }
}
.hero-card-item {
    position: relative;
    background: linear-gradient(180deg, rgba(17, 26, 46, 0.85), rgba(13, 20, 36, 0.85));
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 18px;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.hero-card-item:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 8px 24px rgba(2, 6, 16, 0.45), 0 0 0 1px rgba(56, 189, 248, 0.12);
}
.hero-card-item::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--hero-accent, var(--grad));
    opacity: 0.9;
}
.hero-card-title {
    font-size: 10.5px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.hero-card-value {
    font-size: 13.5px;
    font-weight: 600;
    color: #f8fafc;
    line-height: 1.45;
}

/* Headers */
.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 8px;
    letter-spacing: -0.6px;
}
.hero-title .grad {
    background: var(--grad);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--muted);
    font-size: 14.5px;
    margin-bottom: 16px;
    line-height: 1.55;
    max-width: 760px;
}
.empty-card {
    background: linear-gradient(180deg, rgba(17, 26, 46, 0.6), rgba(13, 20, 36, 0.6));
    border: 1px dashed rgba(148, 163, 184, 0.28);
    border-radius: 16px;
    text-align: center;
    padding: 48px 24px;
    margin-top: 24px;
}
.empty-card .empty-icon {
    font-size: 34px;
    margin-bottom: 12px;
}
.empty-card h4 { color: #e2e8f0; font-size: 19px; font-weight: 700; margin-bottom: 8px; }
.empty-card p { color: var(--muted); font-size: 13.5px; line-height: 1.6; }

/* Live Stats Strip */
.stats-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 22px;
}
@media (max-width: 900px) {
    .stats-strip { grid-template-columns: repeat(2, 1fr); }
}
.stat-card {
    background: linear-gradient(180deg, rgba(17, 26, 46, 0.8), rgba(13, 20, 36, 0.8));
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.stat-icon {
    width: 38px; height: 38px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 10px;
    font-size: 17px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.22);
    flex-shrink: 0;
}
.stat-value {
    font-size: 19px;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.1;
    letter-spacing: -0.3px;
}
.stat-label {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Progress & Metrics */
.metric-box {
    background: linear-gradient(180deg, rgba(17, 26, 46, 0.85), rgba(13, 20, 36, 0.85));
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.metric-box:hover {
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-1px);
}
.metric-val {
    font-size: 22px;
    font-weight: 800;
    background: var(--grad);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-top: 2px; }

/* Sidebar status pills */
.status-pill-ok, .status-pill-missing {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 4px;
}
.status-pill-ok { animation: pulse-ok 2.4s ease-in-out infinite; }
@keyframes pulse-ok {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.25); }
    50% { box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.06); }
}

/* Sidebar brand card */
.brand-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(129, 140, 248, 0.1));
    border: 1px solid rgba(56, 189, 248, 0.22);
    margin-bottom: 14px;
}
.brand-logo {
    background: var(--grad);
    padding: 9px 13px;
    border-radius: 11px;
    font-size: 21px;
    box-shadow: 0 4px 14px rgba(56, 189, 248, 0.35);
}
.brand-name { font-weight: 800; font-size: 16px; color: #f8fafc; letter-spacing: -0.2px; }
.brand-tag { font-size: 11px; color: var(--muted); margin-top: 2px; }

/* DB count chip in knowledge base headers */
.count-chip {
    display: inline-block;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #7dd3fc;
    border-radius: 999px;
    padding: 1px 9px;
    font-size: 11px;
    font-weight: 700;
    margin-left: 6px;
}
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
    <div class="brand-card">
        <div class="brand-logo">⚡</div>
        <div>
            <div class="brand-name">Production AI System</div>
            <div class="brand-tag">LiteParse + Guardrails + Telemetry</div>
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
            with st.expander(f"{icon} {label} — {count} chunk{'s' if count != 1 else ''}", expanded=False):
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
    else:
        st.markdown("""
        <div style="border:1px dashed rgba(148,163,184,0.25);border-radius:12px;padding:16px 14px;
                    color:#8b98ad;font-size:12.5px;line-height:1.55;text-align:center;">
            🔒 Connect your Groq API key above<br>to manage the knowledge bases.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Section 3: Database Distribution (session totals live in the header stats strip)
    st.markdown('<div class="sidebar-section-header">3. DATABASE DISTRIBUTION</div>', unsafe_allow_html=True)
    total_chunks = max(sum(st.session_state.doc_counts.values()), 1)
    for db_key, (icon, label, color) in DB_LABELS.items():
        cnt = st.session_state.doc_counts[db_key]
        pct = cnt * 100 // total_chunks if cnt else 0
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
                <span style="color:#cbd5e1;font-weight:600;">{icon}&nbsp;&nbsp;{label}</span>
                <span style="color:#8b98ad;font-weight:700;">{cnt}</span>
            </div>
            <div style="height:6px;border-radius:999px;background:rgba(148,163,184,0.12);overflow:hidden;">
                <div style="height:100%;width:{max(pct, 2 if cnt else 0)}%;background:{color};border-radius:999px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Section 4: Session Memory Controls
    st.markdown('<div class="sidebar-section-header">4. SESSION MEMORY</div>', unsafe_allow_html=True)
    turns_count = len(st.session_state.memory.messages) // 2
    st.markdown(f"<p style='font-size:12px;color:#94a3b8;'>Active Memory Turns: <strong>{turns_count} / {st.session_state.memory.max_turns}</strong></p>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Reset Session Memory", use_container_width=True):
            st.session_state.messages = []
            st.session_state.metadata = {}
            st.session_state.memory.clear()
            st.session_state.total_queries = 0
            st.session_state.fallback_count = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Section 5: Quick Samples
    st.markdown('<div class="sidebar-section-header">5. QUICK DEMO QUERIES</div>', unsafe_allow_html=True)
    for q in EXAMPLE_QUERIES:
        st.markdown('<div class="demo-query-btn">', unsafe_allow_html=True)
        if st.button(q[:50] + ("..." if len(q) > 50 else ""), use_container_width=True, key=f"ex_{q}"):
            st.session_state.pending_query = q
        st.markdown('</div>', unsafe_allow_html=True)


# -- Main Workspace Header -----------------------------------------------------

total_docs = sum(st.session_state.doc_counts.values())
turns_now = len(st.session_state.memory.messages) // 2

st.markdown(f"""
<div>
    <div class="hero-title">Production AI System <span class="grad">Architecture</span></div>
    <div class="hero-sub">
        Enterprise RAG Pipeline featuring <strong style="color:#38bdf8;">LiteParse Layout Engine</strong>,
        <strong style="color:#34d399;">Session Memory</strong>,
        <strong style="color:#fbbf24;">Observability Tracing</strong>, and
        <strong style="color:#c084fc;">Faithfulness Evaluation</strong>.
    </div>
    <div class="stats-strip">
        <div class="stat-card">
            <div class="stat-icon" style="background:rgba(56,189,248,0.1);border-color:rgba(56,189,248,0.22);">📚</div>
            <div><div class="stat-value">{total_docs}</div><div class="stat-label">Chunks Indexed</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon" style="background:rgba(52,211,153,0.1);border-color:rgba(52,211,153,0.22);">💬</div>
            <div><div class="stat-value">{st.session_state.total_queries}</div><div class="stat-label">Queries Served</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon" style="background:rgba(192,132,252,0.1);border-color:rgba(192,132,252,0.22);">🌐</div>
            <div><div class="stat-value">{st.session_state.fallback_count}</div><div class="stat-label">Web Fallbacks</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon" style="background:rgba(251,191,36,0.1);border-color:rgba(251,191,36,0.22);">🧠</div>
            <div><div class="stat-value">{turns_now}/{st.session_state.memory.max_turns}</div><div class="stat-label">Memory Turns</div></div>
        </div>
    </div>
    <div class="hero-card-grid">
        <div class="hero-card-item" style="--hero-accent: linear-gradient(90deg, #38bdf8, #0ea5e9);">
            <div class="hero-card-title">Document Parser</div>
            <div class="hero-card-value" style="color:#7dd3fc;">⚡ LiteParse Multi-Format Engine<br><span style="color:#8b98ad;font-weight:500;font-size:12px;">PDF · DOCX · XLSX · PPTX · Images</span></div>
        </div>
        <div class="hero-card-item" style="--hero-accent: linear-gradient(90deg, #34d399, #10b981);">
            <div class="hero-card-title">Safety Guardrails</div>
            <div class="hero-card-value" style="color:#6ee7b7;">🛡️ Enforced on Every Request<br><span style="color:#8b98ad;font-weight:500;font-size:12px;">Prompt Injection Protection</span></div>
        </div>
        <div class="hero-card-item" style="--hero-accent: linear-gradient(90deg, #fbbf24, #f59e0b);">
            <div class="hero-card-title">Faithfulness Evaluator</div>
            <div class="hero-card-value" style="color:#fcd34d;">🎯 Active on Every Answer<br><span style="color:#8b98ad;font-weight:500;font-size:12px;">Deterministic Confidence Scoring</span></div>
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

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-card">
        <div class="empty-icon">🚀</div>
        <h4>System Ready for Queries</h4>
        <p>Your prompt will be processed through Input Guardrails, routed to the target Qdrant Vector Store,<br>
        and evaluated for faithfulness with millisecond-level telemetry tracing.<br>
        <span style="opacity:0.75;">Try a quick demo query from the sidebar, or type below.</span></p>
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
