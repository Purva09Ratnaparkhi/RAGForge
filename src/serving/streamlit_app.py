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
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0d0d2b 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Chat messages */
    .stChatMessage {
        background: rgba(30, 30, 60, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    /* Badges */
    .confidence-high {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        display: inline-block;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        display: inline-block;
    }
    .confidence-low {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        display: inline-block;
    }

    /* Title gradient */
    .title-gradient {
        background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    /* Source cards */
    .source-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.9em;
    }

    /* Upload area */
    .stFileUploader {
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
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
        css_class = "confidence-high"
        label = f"✓ High Confidence ({score:.0%})"
    elif score >= 0.6:
        css_class = "confidence-medium"
        label = f"⚠ Medium Confidence ({score:.0%})"
    else:
        css_class = "confidence-low"
        label = f"✗ Low Confidence ({score:.0%})"
    return f'<span class="{css_class}">{label}</span>'


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

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                if event_type and data_buffer:
                    yield {"event": event_type, "data": data_buffer}
                    event_type = ""
                    data_buffer = ""
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_buffer = line[5:].strip()
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


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="title-gradient">🔬 RAGForge</p>', unsafe_allow_html=True)
    st.caption("Research Paper Q&A System")

    st.divider()

    # API Status
    health = check_api_health()
    if health:
        st.success(f"🟢 API Online — {health.get('documents_count', 0)} docs indexed")
    else:
        st.error("🔴 API Offline — Start with: `python main.py`")

    st.divider()

    # PDF Upload
    st.subheader("📄 Upload Papers")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("⚡ Process Documents", use_container_width=True):
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
                        f"✓ Processed {data['files_processed']} files → "
                        f"{data['total_chunks']} chunks"
                    )
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Document list
    st.subheader("📚 Indexed Documents")
    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=3)
        if docs_resp.status_code == 200:
            docs_data = docs_resp.json()
            if docs_data["count"] > 0:
                for doc in docs_data["documents"]:
                    st.markdown(f"• {doc['filename']}")
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
        if st.button("🗑️ Clear Docs", use_container_width=True):
            try:
                requests.delete(f"{API_BASE}/documents", timeout=10)
                st.success("Documents cleared")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()


# ── Main panel ───────────────────────────────────────────────────────────

st.markdown('<p class="title-gradient">Research Paper Q&A</p>', unsafe_allow_html=True)
st.caption("Ask questions about your uploaded research papers. Powered by RAGForge.")

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
                with st.expander(f"📖 Sources ({len(meta['sources'])} chunks)"):
                    for src in meta["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f"📄 **{src.get('source', 'unknown')}** "
                            f"— Page {src.get('page', '?')} "
                            f"— Section: {src.get('section', 'unknown')}"
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
                with st.expander(f"📖 Sources ({len(sources)} chunks)"):
                    for src in sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f"📄 **{src.get('source', 'unknown')}** "
                            f"— Page {src.get('page', '?')} "
                            f"— Section: {src.get('section', 'unknown')}"
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
