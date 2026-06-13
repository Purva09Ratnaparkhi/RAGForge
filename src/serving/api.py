"""
FastAPI Backend — Component 12: REST API with SSE streaming.
=============================================================

Provides the HTTP interface for the RAGForge system:

* ``POST /ingest``    — upload PDF files, trigger the extraction pipeline
* ``POST /query``     — SSE streaming query endpoint
* ``GET  /health``    — health check
* ``GET  /documents`` — list ingested documents
* ``DELETE /documents`` — clear the index

Technology choices
------------------
* **FastAPI** — best-in-class Python async web framework with automatic
  OpenAPI docs, Pydantic validation, and native ``asyncio`` support.
* **SSE (Server-Sent Events)** via ``sse-starlette`` — lighter than
  WebSockets for one-way streaming; works with Streamlit's SSE consumer.
* **CORS** enabled for ``localhost:8501`` so the Streamlit frontend can
  call the API from a different port.

Connects to
-----------
* ``src.agents.rag_agent``        — invokes the compiled graph.
* ``src.extraction.pipeline``     — processes uploaded PDFs.
* ``src.chunking.smart_chunker``  — chunks extracted documents.
* ``src.vector_store.faiss_store`` — builds/loads the FAISS index.
* ``src.memory.conversation_memory`` — manages chat history per session.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config.settings import (
    CORS_ORIGINS,
    FAISS_STORE_DIR,
    PROCESSED_DIR,
    RAW_PDF_DIR,
)

# ── Pydantic models ──────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for the ``/query`` endpoint.

    Attributes
    ----------
    query : str
        User question.
    history : list[str]
        Optional conversation history for reformulation.
    """

    query: str
    history: list[str] = []


class IngestResponse(BaseModel):
    """Response from the ``/ingest`` endpoint."""

    message: str
    files_processed: int
    total_chunks: int


class QueryResponse(BaseModel):
    """Non-streaming query response (fallback)."""

    answer: str
    sources: list[dict]
    guardrail_result: dict


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    index_exists: bool
    documents_count: int


# ── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(
    title="RAGForge API",
    description="Research Paper Q&A with Retrieval-Augmented Generation",
    version="1.0.0",
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ─────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and index are healthy.

    Returns
    -------
    HealthResponse
    """
    index_exists = (FAISS_STORE_DIR / "index").exists()
    doc_count = len(list(PROCESSED_DIR.glob("*.pkl")))
    return HealthResponse(
        status="healthy",
        index_exists=index_exists,
        documents_count=doc_count,
    )


# ── Document listing ─────────────────────────────────────────────────────


@app.get("/documents")
async def list_documents():
    """List all ingested documents.

    Returns
    -------
    dict
        Document filenames and metadata.
    """
    processed = list(PROCESSED_DIR.glob("*.pkl"))
    documents = [{"filename": p.stem + ".pdf", "processed_file": p.name} for p in processed]
    return {"documents": documents, "count": len(documents)}


@app.delete("/documents")
async def clear_documents():
    """Clear all ingested documents and indexes.

    Returns
    -------
    dict
    """
    logger.info("[API] Clearing all documents and indexes")

    # Clear processed
    for f in PROCESSED_DIR.glob("*.pkl"):
        f.unlink()

    # Clear FAISS
    index_dir = FAISS_STORE_DIR / "index"
    if index_dir.exists():
        shutil.rmtree(index_dir)
    bm25_path = FAISS_STORE_DIR / "bm25.pkl"
    if bm25_path.exists():
        bm25_path.unlink()
    summary_dir = FAISS_STORE_DIR / "summary_index"
    if summary_dir.exists():
        shutil.rmtree(summary_dir)

    return {"message": "All documents and indexes cleared"}


# ── PDF ingestion ────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    """Upload and process PDF files.

    Saves uploaded files to ``data/raw/``, runs the extraction pipeline,
    chunks the output, and builds/updates the FAISS index.

    Parameters
    ----------
    files : list[UploadFile]
        PDF files to ingest.

    Returns
    -------
    IngestResponse
    """
    logger.info("[API] Ingesting {} files", len(files))

    # Save uploaded files
    saved_paths: list[Path] = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted. Got: {file.filename}",
            )
        dest = RAW_PDF_DIR / file.filename
        with open(dest, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_paths.append(dest)
        logger.info("[API] Saved {} ({} bytes)", file.filename, len(content))

    # Run extraction pipeline (in thread to not block event loop)
    try:
        from src.extraction.pipeline import ExtractionPipeline
        from src.chunking.smart_chunker import chunk_documents
        from src.vector_store.faiss_store import FAISSVectorStore

        pipeline = ExtractionPipeline()
        all_chunks = []

        for pdf_path in saved_paths:
            docs = await asyncio.to_thread(pipeline.run, pdf_path)
            chunks = await asyncio.to_thread(chunk_documents, docs)
            all_chunks.extend(chunks)

        # Build / update FAISS index
        store = FAISSVectorStore()
        if (FAISS_STORE_DIR / "index").exists():
            await asyncio.to_thread(store.add_documents, all_chunks)
        else:
            await asyncio.to_thread(store.build, all_chunks)

        logger.info(
            "[API] Ingestion complete: {} files → {} chunks",
            len(saved_paths), len(all_chunks),
        )
        return IngestResponse(
            message="Ingestion successful",
            files_processed=len(saved_paths),
            total_chunks=len(all_chunks),
        )

    except Exception as exc:
        logger.error("[API] Ingestion failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Query endpoint (SSE streaming) ───────────────────────────────────────


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Process a query with SSE streaming response.

    The response is a Server-Sent Events stream where each event
    contains a chunk of the answer or metadata.

    Parameters
    ----------
    request : QueryRequest
        Query and optional conversation history.

    Returns
    -------
    EventSourceResponse
        SSE stream with answer tokens and metadata.
    """
    logger.info("[API] Query received: {}", request.query[:80])

    async def token_stream():
        try:
            from src.agents.rag_agent import build_rag_graph

            graph = build_rag_graph()

            # Run the graph
            initial_state = {
                "query": request.query,
                "query_type": "",
                "rewritten_queries": [],
                "retrieved_docs": [],
                "reranked_docs": [],
                "compressed_docs": [],
                "answer": "",
                "guardrail_result": {},
                "context_str": "",
                "history_str": "\n".join(request.history),
                "messages": [],
            }

            # Run in thread pool (graph is sync)
            result = await asyncio.to_thread(graph.invoke, initial_state)

            answer = result.get("answer", "")
            guardrails = result.get("guardrail_result", {})

            # Stream answer in chunks (simulate token streaming)
            words = answer.split(" ")
            chunk_size = 3  # stream 3 words at a time
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i > 0:
                    chunk = " " + chunk
                yield {"event": "token", "data": chunk}
                await asyncio.sleep(0.02)

            # Send sources
            sources = []
            for doc in result.get("compressed_docs", result.get("reranked_docs", [])):
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0),
                    "section": doc.metadata.get("section", "unknown"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                })

            yield {
                "event": "sources",
                "data": json.dumps(sources),
            }

            # Send guardrail results
            yield {
                "event": "guardrails",
                "data": json.dumps(guardrails),
            }

            # Send completion
            yield {"event": "done", "data": ""}

        except Exception as exc:
            logger.error("[API] Query processing failed: {}", exc)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}),
            }

    return EventSourceResponse(token_stream())


# ── Non-streaming query (fallback) ───────────────────────────────────────


@app.post("/query/sync", response_model=QueryResponse)
async def query_sync(request: QueryRequest):
    """Process a query synchronously (non-streaming fallback).

    Parameters
    ----------
    request : QueryRequest

    Returns
    -------
    QueryResponse
    """
    logger.info("[API] Sync query: {}", request.query[:80])

    try:
        from src.agents.rag_agent import build_rag_graph

        graph = build_rag_graph()
        initial_state = {
            "query": request.query,
            "query_type": "",
            "rewritten_queries": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "compressed_docs": [],
            "answer": "",
            "guardrail_result": {},
            "context_str": "",
            "history_str": "\n".join(request.history),
            "messages": [],
        }

        result = await asyncio.to_thread(graph.invoke, initial_state)

        sources = []
        for doc in result.get("compressed_docs", result.get("reranked_docs", [])):
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "section": doc.metadata.get("section", "unknown"),
            })

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            guardrail_result=result.get("guardrail_result", {}),
        )

    except Exception as exc:
        logger.error("[API] Sync query failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))
