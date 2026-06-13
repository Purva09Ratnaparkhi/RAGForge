"""
Agentic RAG — Component 13: LangGraph-based routing agent.
============================================================

Implements a ``StateGraph`` that routes user queries through the full
RAG pipeline with conditional branching based on query classification.

Graph structure
---------------
::

    START → classify_query
    classify_query → rewrite_query       (if complex or conversational)
    classify_query → retrieve            (if simple)
    classify_query → handle_out_of_scope (if out_of_scope)
    rewrite_query  → retrieve
    retrieve       → rerank
    rerank         → compress
    compress       → generate
    generate       → check_guardrails
    check_guardrails → END
    handle_out_of_scope → END

Each node is a Python function that reads from and writes to the shared
``RAGState`` TypedDict.  This makes the pipeline fully traceable —
every intermediate result (rewritten queries, retrieved docs, reranked
docs, guardrail scores) is captured in the state.

Technology choices
------------------
* **LangGraph** ``StateGraph`` — provides explicit, debuggable control
  flow compared to implicit LangChain chains.  The graph is compiled
  into a runnable that supports both sync and async execution.

Connects to
-----------
* Every retrieval, generation, and guardrail component.
* ``src.serving.api`` — invokes the compiled graph from the API layer.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from src.generation.llm_factory import get_llm_with_fallback
from src.generation.prompts import (
    CLASSIFY_QUERY_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
    QA_PROMPT,
)
from src.guardrails.guardrails import check_guardrails
from src.retrieval.compressor import compress
from src.retrieval.query_rewriter import rewrite
from src.retrieval.reranker import CrossEncoderReranker


# ── State definition ──────────────────────────────────────────────────────


class RAGState(TypedDict):
    """Shared state flowing through the LangGraph pipeline.

    Attributes
    ----------
    query : str
        Original user query.
    query_type : str
        Classification result: "simple" | "complex" | "conversational" | "out_of_scope".
    rewritten_queries : list[str]
        Query variants from the rewriter.
    retrieved_docs : list[Document]
        Documents from hybrid retrieval.
    reranked_docs : list[Document]
        Top-k documents after cross-encoder reranking.
    compressed_docs : list[Document]
        Documents after context compression.
    answer : str
        Generated answer.
    guardrail_result : dict
        Output from the guardrails checker.
    context_str : str
        Concatenated context string for generation.
    history_str : str
        Formatted conversation history.
    messages : Annotated[list, add_messages]
        LangGraph message accumulator.
    """

    query: str
    query_type: str
    rewritten_queries: list[str]
    retrieved_docs: list[Document]
    reranked_docs: list[Document]
    compressed_docs: list[Document]
    answer: str
    guardrail_result: dict
    context_str: str
    history_str: str
    messages: Annotated[list, add_messages]


# ── Lazy-loaded shared resources ──────────────────────────────────────────

_llm = None
_reranker = None
_hybrid_retriever = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm_with_fallback()
    return _llm


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_hybrid_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from src.retrieval.hybrid_retriever import HybridRetriever
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


# ── Node functions ────────────────────────────────────────────────────────


def classify_query(state: RAGState) -> dict:
    """Classify the query into one of four types.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    dict
        Updated state with ``query_type``.
    """
    query = state["query"]
    logger.info("[Agent] Classifying query: {}", query[:80])

    llm = _get_llm()
    chain = CLASSIFY_QUERY_PROMPT | llm

    try:
        result = chain.invoke({"query": query})
        text = result.content if hasattr(result, "content") else str(result)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            query_type = parsed.get("type", "simple")
        else:
            query_type = "simple"
    except Exception as exc:
        logger.warning("[Agent] Classification failed: {}. Defaulting to 'simple'", exc)
        query_type = "simple"

    # Validate
    valid_types = {"simple", "complex", "conversational", "out_of_scope"}
    if query_type not in valid_types:
        query_type = "simple"

    logger.info("[Agent] Query classified as: {}", query_type)
    return {"query_type": query_type}


def rewrite_query(state: RAGState) -> dict:
    """Rewrite complex/conversational queries into multiple variants.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    dict
        Updated state with ``rewritten_queries``.
    """
    query = state["query"]
    history_str = state.get("history_str", "")
    logger.info("[Agent] Rewriting query")

    llm = _get_llm()
    history = history_str.split("\n") if history_str else None
    variants = rewrite(query, llm, history=history)

    return {"rewritten_queries": variants}


def retrieve(state: RAGState) -> dict:
    """Retrieve documents using hybrid search.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    dict
        Updated state with ``retrieved_docs``.
    """
    queries = state.get("rewritten_queries", [state["query"]])
    if not queries:
        queries = [state["query"]]

    logger.info("[Agent] Retrieving with {} query variants", len(queries))

    try:
        retriever = _get_hybrid_retriever()
        if len(queries) == 1:
            docs = retriever.retrieve(queries[0])
        else:
            docs = retriever.retrieve_multi_query(queries)
    except FileNotFoundError:
        logger.warning(
            "[Agent] No FAISS index found — no documents have been ingested. "
            "Please upload and process PDFs first."
        )
        docs = []
    except Exception as exc:
        logger.error("[Agent] Retrieval failed: {}", exc)
        docs = []

    logger.info("[Agent] Retrieved {} documents", len(docs))
    return {"retrieved_docs": docs}


def rerank(state: RAGState) -> dict:
    """Rerank retrieved documents with cross-encoder.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    dict
        Updated state with ``reranked_docs``.
    """
    docs = state.get("retrieved_docs", [])
    query = state["query"]
    logger.info("[Agent] Reranking {} documents", len(docs))

    reranker = _get_reranker()
    reranked = reranker.rerank(query, docs)

    return {"reranked_docs": reranked}


def compress_docs(state: RAGState) -> dict:
    """Compress reranked documents to relevant content only.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    dict
        Updated state with ``compressed_docs`` and ``context_str``.
    """
    docs = state.get("reranked_docs", [])
    query = state["query"]
    logger.info("[Agent] Compressing {} documents", len(docs))

    llm = _get_llm()

    try:
        compressed = compress(query, docs, llm)
    except Exception as exc:
        logger.warning("[Agent] Compression failed: {}. Using reranked docs.", exc)
        compressed = docs

    context_str = "\n\n---\n\n".join(
        d.page_content for d in compressed if d.page_content
    )

    return {"compressed_docs": compressed, "context_str": context_str}


def generate(state: RAGState) -> dict:
    """Generate an answer using the LLM with compressed context.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    dict
        Updated state with ``answer`` and ``messages``.
    """
    context = state.get("context_str", "")
    question = state["query"]
    history = state.get("history_str", "")
    logger.info("[Agent] Generating answer")

    llm = _get_llm()
    chain = QA_PROMPT | llm

    try:
        result = chain.invoke({
            "context": context,
            "question": question,
            "history": history,
        })
        answer = result.content if hasattr(result, "content") else str(result)
    except Exception as exc:
        logger.error("[Agent] Generation failed: {}", exc)
        answer = (
            "I apologise, but I encountered an error generating an answer. "
            "Please try again."
        )

    messages = [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]

    return {"answer": answer, "messages": messages}


def check_guardrails_node(state: RAGState) -> dict:
    """Run guardrail checks on the generated answer.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    dict
        Updated state with ``guardrail_result``.
    """
    context = state.get("context_str", "")
    question = state["query"]
    answer = state.get("answer", "")
    logger.info("[Agent] Checking guardrails")

    llm = _get_llm()
    result = check_guardrails(context, question, answer, llm)

    return {"guardrail_result": result}


def handle_out_of_scope(state: RAGState) -> dict:
    """Return a polite out-of-scope response.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    dict
        Updated state with ``answer``, ``guardrail_result``, ``messages``.
    """
    logger.info("[Agent] Handling out-of-scope query")
    answer = OUT_OF_SCOPE_RESPONSE
    messages = [
        HumanMessage(content=state["query"]),
        AIMessage(content=answer),
    ]
    return {
        "answer": answer,
        "guardrail_result": {
            "grounded": True,
            "confidence_score": 1.0,
            "hallucination_flags": [],
            "unsupported_claims": [],
            "warning": None,
        },
        "messages": messages,
    }


# ── Routing function ─────────────────────────────────────────────────────


def route_after_classification(state: RAGState) -> str:
    """Determine next node based on query type.

    Parameters
    ----------
    state : RAGState

    Returns
    -------
    str
        Name of the next node.
    """
    query_type = state.get("query_type", "simple")
    if query_type in ("complex", "conversational"):
        return "rewrite_query"
    elif query_type == "out_of_scope":
        return "handle_out_of_scope"
    else:
        return "retrieve"


# ── Graph construction ────────────────────────────────────────────────────


def build_rag_graph() -> StateGraph:
    """Build and compile the RAG agent graph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph runnable.

    Example
    -------
    >>> graph = build_rag_graph()
    >>> result = graph.invoke({"query": "What is attention?"})
    >>> result["answer"]
    'Attention is a mechanism...'
    """
    logger.info("[Agent] Building RAG graph")

    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("classify_query", classify_query)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("compress", compress_docs)
    graph.add_node("generate", generate)
    graph.add_node("check_guardrails", check_guardrails_node)
    graph.add_node("handle_out_of_scope", handle_out_of_scope)

    # Set entry point
    graph.set_entry_point("classify_query")

    # Conditional edge from classification
    graph.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "rewrite_query": "rewrite_query",
            "retrieve": "retrieve",
            "handle_out_of_scope": "handle_out_of_scope",
        },
    )

    # Linear edges
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", "check_guardrails")
    graph.add_edge("check_guardrails", END)
    graph.add_edge("handle_out_of_scope", END)

    compiled = graph.compile()
    logger.info("[Agent] RAG graph compiled")
    return compiled
