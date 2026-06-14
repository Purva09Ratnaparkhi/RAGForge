"""
Streamlit Frontend — Interactive Research Paper Q&A.
=====================================================

A clean, modern Streamlit interface for RAGForge featuring:

**Sidebar**
* PDF upload widget (multiple files)
* "Process Documents" button → calls ``POST /ingest``
* List of ingested documents
* Clear all button
* RAGAS evaluation on demand

**Main panel**
* Chat interface with ``st.chat_message``
* Streaming responses via SSE
* Expandable "Sources" section per response
* Confidence score badge (green > 0.8, yellow 0.6–0.8, red < 0.6)
* Warning banner for guardrail flags

**Session state**
* Conversation history persisted across turns
* Clear chat resets memory

Connects to
-----------
* ``src.serving.api`` — all actions go through the FastAPI backend.
"""

from __future__ import annotations

import json

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAGForge — Research Paper Q&A",
    page_icon="RF",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Base ────────────────────────────────── */
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    /* Apply Inter to text elements only — do NOT override icon fonts */
    .stMarkdown, .stText, .stCaption, .stChatMessage,
    p, h1, h2, h3, h4, h5, h6, span, li, td, th, label, input, textarea, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    /* Preserve Streamlit's Material Symbols icon font */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    span[class*="Icon"] {
        font-family: 'Material Symbols Rounded' !important;
    }
    .stApp {
        background-color: #0e1117;
    }

    /* ── Sidebar ─────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    /* ── Chat messages ───────────────────────── */
    .stChatMessage {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
    }

    /* ── Confidence badges ───────────────────── */
    .conf-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .conf-high {
        background-color: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .conf-mid {
        background-color: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.3);
    }
    .conf-low {
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
    }

    /* ── App header ──────────────────────────── */
    .app-title {
        font-size: 1.5em;
        font-weight: 700;
        color: #e6edf3;
        letter-spacing: -0.02em;
        margin: 0;
        padding: 0;
    }
    .app-subtitle {
        font-size: 0.85em;
        color: #7d8590;
        margin-top: 2px;
    }

    /* ── Sidebar header ──────────────────────── */
    .sidebar-brand {
        font-size: 1.25em;
        font-weight: 700;
        color: #e6edf3;
        letter-spacing: -0.01em;
    }
    .sidebar-section {
        font-size: 0.8em;
        font-weight: 600;
        color: #7d8590;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }

    /* ── Status indicator ────────────────────── */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.85em;
        color: #3fb950;
    }
    .status-online::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #3fb950;
        display: inline-block;
        flex-shrink: 0;
    }
    .status-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.85em;
        color: #f85149;
    }
    .status-offline::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #f85149;
        display: inline-block;
        flex-shrink: 0;
    }

    /* ── Source cards ─────────────────────────── */
    .source-card {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 3px 0;
        font-size: 0.85em;
        color: #c9d1d9;
    }
    .source-card strong {
        color: #e6edf3;
    }

    /* ── Document list item ──────────────────── */
    .doc-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 8px;
        border-radius: 6px;
        font-size: 0.88em;
        color: #c9d1d9;
        background-color: rgba(110, 118, 129, 0.04);
        margin-bottom: 4px;
    }
    .doc-icon {
        width: 14px;
        height: 16px;
        flex-shrink: 0;
    }

    /* ── Upload area ─────────────────────────── */
    .stFileUploader {
        border: 1px dashed #30363d !important;
        border-radius: 8px !important;
    }

    /* ── Buttons ─────────────────────────────── */
    .stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.88em;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #484f58;
        color: #e6edf3;
    }

    /* Primary action button override */
    .stButton > button[kind="primary"],
    div[data-testid="stButton"] > button:first-child {
        background-color: #238636;
        border-color: rgba(35, 134, 54, 0.4);
        color: #ffffff;
    }
    div[data-testid="stButton"] > button:first-child:hover {
        background-color: #2ea043;
        border-color: rgba(46, 160, 67, 0.4);
    }

    /* ── Divider ─────────────────────────────── */
    hr {
        border-color: #21262d !important;
    }

    /* ── Hide Streamlit branding ─────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []


# ── Helper functions ──────────────────────────────────────────────────────


def get_confidence_badge(score: float) -> str:
    """Return an HTML badge for the confidence score.

    Parameters
    ----------
    score : float
        Confidence score (0.0–1.0).

    Returns
    -------
    str
        HTML span element.
    """
    if score >= 0.8:
        css = "conf-badge conf-high"
        label = f"High confidence · {score:.0%}"
    elif score >= 0.6:
        css = "conf-badge conf-mid"
        label = f"Medium confidence · {score:.0%}"
    else:
        css = "conf-badge conf-low"
        label = f"Low confidence · {score:.0%}"
    return f'<span class="{css}">{label}</span>'


def check_api_health() -> dict | None:
    """Check if the API is running.

    Returns
    -------
    dict | None
    """
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        return resp.json()
    except Exception:
        return None


def query_api_streaming(query: str, history: list[str]):
    """Send a query to the SSE endpoint and yield chunks.

    Parameters
    ----------
    query : str
    history : list[str]

    Yields
    ------
    dict
        Events with ``event`` and ``data`` keys.
    """
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"query": query, "history": history},
            stream=True,
            timeout=120,
        )
        event_type = ""
        data_buffer = ""
        has_data = False

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                if event_type or has_data:
                    yield {"event": event_type, "data": data_buffer}
                    event_type = ""
                    data_buffer = ""
                    has_data = False
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                val = line[6:] if line.startswith("data: ") else line[5:]
                if has_data:
                    data_buffer += "\n" + val
                else:
                    data_buffer = val
                    has_data = True
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}


def query_api_sync(query: str, history: list[str]) -> dict:
    """Send a synchronous query to the API.

    Parameters
    ----------
    query : str
    history : list[str]

    Returns
    -------
    dict
    """
    try:
        resp = requests.post(
            f"{API_BASE}/query/sync",
            json={"query": query, "history": history},
            timeout=120,
        )
        return resp.json()
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": [], "guardrail_result": {}}


# ── SVG Icons ────────────────────────────────────────────────────────────

_ICON_FILE = (
    '<svg class="doc-icon" viewBox="0 0 16 16" fill="currentColor">'
    '<path d="M3.75 1.5a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h8.5'
    "a.25.25 0 0 0 .25-.25V4.664a.25.25 0 0 0-.073-.177l-2.914-2.914"
    "a.25.25 0 0 0-.177-.073ZM3.75 0h5.336c.464 0 .909.184 1.237.513"
    "l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 "
    '12.25 16h-8.5A1.75 1.75 0 0 1 2 14.25V1.75C2 .784 2.784 0 3.75 0Z"/>'
    "</svg>"
)


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-brand">RAGForge</p>', unsafe_allow_html=True)
    st.caption("Research Paper Q&A System")

    st.divider()

    # API Status
    health = check_api_health()
    if health:
        doc_count = health.get("documents_count", 0)
        st.markdown(
            f'<span class="status-online">Online — {doc_count} doc{"s" if doc_count != 1 else ""} indexed</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-offline">Offline — run: python main.py</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # PDF Upload
    st.markdown('<p class="sidebar-section">Upload Papers</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("Process Documents", use_container_width=True):
        with st.spinner("Processing PDFs..."):
            files = [
                ("files", (f.name, f.getvalue(), "application/pdf"))
                for f in uploaded_files
            ]
            try:
                resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Processed {data['files_processed']} files — "
                        f"{data['total_chunks']} chunks"
                    )
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Document list
    st.markdown(
        '<p class="sidebar-section">Indexed Documents</p>',
        unsafe_allow_html=True,
    )
    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=3)
        if docs_resp.status_code == 200:
            docs_data = docs_resp.json()
            if docs_data["count"] > 0:
                for doc in docs_data["documents"]:
                    st.markdown(
                        f'<div class="doc-item">{_ICON_FILE} {doc["filename"]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No documents indexed yet.")
        else:
            st.caption("Could not fetch documents.")
    except Exception:
        st.caption("API not available.")

    st.divider()

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Docs", use_container_width=True):
            try:
                requests.delete(f"{API_BASE}/documents", timeout=10)
                st.success("Documents cleared")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()


# ── Main panel ───────────────────────────────────────────────────────────

st.markdown('<p class="app-title">Research Paper Q&A</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">'
    "Ask questions about your uploaded research papers. Powered by RAGForge."
    "</p>",
    unsafe_allow_html=True,
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]

            # Confidence badge
            if "confidence_score" in meta:
                st.markdown(
                    get_confidence_badge(meta["confidence_score"]),
                    unsafe_allow_html=True,
                )

            # Warning
            if meta.get("warning"):
                st.warning(meta["warning"])

            # Sources
            if meta.get("sources"):
                with st.expander(f"Sources ({len(meta['sources'])} chunks)"):
                    for src in meta["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f"<strong>{src.get('source', 'unknown')}</strong>"
                            f" — Page {src.get('page', '?')}"
                            f" — Section: {src.get('section', 'unknown')}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

# Chat input
if prompt := st.chat_input("Ask a question about your papers..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        if not health:
            st.error(
                "The API is not running. Start it with: `python main.py`"
            )
        else:
            # Try streaming first, fall back to sync
            answer_parts = []
            sources = []
            guardrails = {}
            placeholder = st.empty()

            try:
                full_answer = ""
                for event in query_api_streaming(prompt, st.session_state.history):
                    if event["event"] == "token":
                        full_answer += event["data"]
                        placeholder.markdown(full_answer + "▌")
                    elif event["event"] == "sources":
                        sources = json.loads(event["data"])
                    elif event["event"] == "guardrails":
                        guardrails = json.loads(event["data"])
                    elif event["event"] == "error":
                        error = json.loads(event["data"])
                        st.error(f"Error: {error.get('error', 'Unknown')}")
                    elif event["event"] == "done":
                        break

                # Final display
                placeholder.markdown(full_answer)

            except Exception:
                # Fallback to sync
                result = query_api_sync(prompt, st.session_state.history)
                full_answer = result.get("answer", "Error getting response")
                sources = result.get("sources", [])
                guardrails = result.get("guardrail_result", {})
                placeholder.markdown(full_answer)

            # Display metadata
            conf_score = guardrails.get("confidence_score", 0.7)
            st.markdown(
                get_confidence_badge(conf_score),
                unsafe_allow_html=True,
            )

            if guardrails.get("warning"):
                st.warning(guardrails["warning"])

            if sources:
                with st.expander(f"Sources ({len(sources)} chunks)"):
                    for src in sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f"<strong>{src.get('source', 'unknown')}</strong>"
                            f" — Page {src.get('page', '?')}"
                            f" — Section: {src.get('section', 'unknown')}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "metadata": {
                    "sources": sources,
                    "confidence_score": conf_score,
                    "warning": guardrails.get("warning"),
                },
            })
            st.session_state.history.append(f"Human: {prompt}")
            st.session_state.history.append(f"AI: {full_answer}")
