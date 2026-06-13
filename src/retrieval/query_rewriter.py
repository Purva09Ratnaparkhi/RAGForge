"""
Query Rewriter — Component 5: Multi-technique query enhancement.
================================================================

Applies three query-rewriting techniques in sequence to improve
retrieval recall and precision:

1. **Expansion** — generates 3 alternative phrasings via LLM to
   capture different vocabulary for the same intent.
2. **Reformulation** — rewrites conversational queries (with pronouns
   and references) into standalone questions using chat history.
3. **HyDE** (Hypothetical Document Embeddings) — generates a fake
   "ideal answer" paragraph, then uses its embedding as the retrieval
   vector.  This bridges the vocabulary gap between short queries and
   long academic passages.

Technology choices
------------------
* LangChain ``ChatPromptTemplate`` + LLM chain for all three techniques.
* ``HypotheticalDocumentEmbedder`` concept implemented manually so we
  have full control over the prompt and embedding step.

Connects to
-----------
* ``src.generation.llm_factory`` — provides the LLM for rewriting.
* ``src.generation.prompts``     — prompt templates live there.
* ``src.agents.rag_agent``       — calls ``rewrite`` for complex queries.
"""

from __future__ import annotations

import json
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger


# ── Prompt templates (inline — also mirrored in src.generation.prompts) ───

_EXPANSION_PROMPT = ChatPromptTemplate.from_template(
    "Given the query: \"{query}\"\n"
    "Generate 3 alternative phrasings that mean the same thing but use "
    "different vocabulary. Return them as a numbered list.\n"
    "1."
)

_REFORMULATION_PROMPT = ChatPromptTemplate.from_template(
    "Given conversation history:\n{history}\n\n"
    "And the latest query: \"{query}\"\n\n"
    "Rewrite the query as a complete standalone question with no pronouns "
    "or references to previous messages. Return ONLY the rewritten question."
)

_HYDE_PROMPT = ChatPromptTemplate.from_template(
    "Write a detailed paragraph that would be the ideal answer to: \"{query}\"\n"
    "Base it on academic research paper content. Be specific and technical."
)


# ── Internal helpers ──────────────────────────────────────────────────────


def _parse_numbered_list(text: str) -> list[str]:
    """Extract items from a numbered list in LLM output.

    Parameters
    ----------
    text : str
        Raw LLM output containing numbered items.

    Returns
    -------
    list[str]
        Extracted items with numbering stripped.
    """
    items: list[str] = []
    for line in text.strip().splitlines():
        # Match "1. ...", "1) ...", "- ..." patterns
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned).strip()
        if cleaned:
            items.append(cleaned)
    return items


# ── Public API ────────────────────────────────────────────────────────────


def expand_query(query: str, llm) -> list[str]:
    """Generate alternative phrasings of the query.

    Parameters
    ----------
    query : str
        Original user query.
    llm
        LangChain LLM object (from ``llm_factory``).

    Returns
    -------
    list[str]
        The original query plus up to 3 alternatives.

    Example
    -------
    >>> alternatives = expand_query("What is attention?", llm)
    >>> len(alternatives)
    4
    """
    logger.debug("[QueryRewriter] Expanding query: {}", query)
    try:
        chain = _EXPANSION_PROMPT | llm
        result = chain.invoke({"query": query})
        text = result.content if hasattr(result, "content") else str(result)
        alternatives = _parse_numbered_list(text)
        logger.info(
            "[QueryRewriter] Expansion produced {} alternatives",
            len(alternatives),
        )
        return [query] + alternatives[:3]
    except Exception as exc:
        logger.warning("[QueryRewriter] Expansion failed: {}. Using original.", exc)
        return [query]


def reformulate_query(query: str, history: list[str], llm) -> str:
    """Rewrite a conversational query as a standalone question.

    Parameters
    ----------
    query : str
        Latest user query (may contain pronouns / references).
    history : list[str]
        Previous conversation turns as strings.
    llm
        LangChain LLM object.

    Returns
    -------
    str
        Rewritten standalone query.

    Example
    -------
    >>> reformulate_query("What about its limitations?",
    ...                   ["Q: What is BERT?", "A: BERT is..."], llm)
    'What are the limitations of BERT?'
    """
    if not history:
        return query

    logger.debug("[QueryRewriter] Reformulating conversational query")
    try:
        history_str = "\n".join(history[-6:])  # last 3 turns max
        chain = _REFORMULATION_PROMPT | llm
        result = chain.invoke({"query": query, "history": history_str})
        rewritten = result.content if hasattr(result, "content") else str(result)
        rewritten = rewritten.strip().strip('"')
        logger.info("[QueryRewriter] Reformulated: {}", rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("[QueryRewriter] Reformulation failed: {}", exc)
        return query


def generate_hypothetical_document(query: str, llm) -> str:
    """Generate a hypothetical ideal answer for HyDE retrieval.

    Parameters
    ----------
    query : str
        User query.
    llm
        LangChain LLM object.

    Returns
    -------
    str
        A synthetic paragraph to be used as the retrieval vector.

    Example
    -------
    >>> hypo = generate_hypothetical_document("What is attention?", llm)
    >>> len(hypo) > 100
    True
    """
    logger.debug("[QueryRewriter] Generating HyDE document")
    try:
        chain = _HYDE_PROMPT | llm
        result = chain.invoke({"query": query})
        hypo = result.content if hasattr(result, "content") else str(result)
        logger.info(
            "[QueryRewriter] HyDE document generated ({} chars)", len(hypo)
        )
        return hypo.strip()
    except Exception as exc:
        logger.warning("[QueryRewriter] HyDE generation failed: {}", exc)
        return query


def rewrite(
    query: str,
    llm,
    history: list[str] | None = None,
) -> list[str]:
    """Run the full query rewriting pipeline.

    Applies expansion, reformulation (if history present), and HyDE in
    sequence.  Returns all query variants for downstream retrieval.

    Parameters
    ----------
    query : str
        Original user query.
    llm
        LangChain LLM object.
    history : list[str] | None
        Conversation history (if conversational query).

    Returns
    -------
    list[str]
        All query variants: original + expanded + reformulated + HyDE.

    Example
    -------
    >>> variants = rewrite("What is attention?", llm)
    >>> len(variants) >= 2
    True
    """
    logger.info("[QueryRewriter] Rewriting query: {}", query)
    variants: list[str] = []

    # 1. Reformulation (only if conversational)
    working_query = query
    if history:
        working_query = reformulate_query(query, history, llm)
        if working_query != query:
            variants.append(working_query)

    # 2. Expansion
    expanded = expand_query(working_query, llm)
    variants.extend(expanded)

    # 3. HyDE
    hyde_doc = generate_hypothetical_document(working_query, llm)
    if hyde_doc != working_query:
        variants.append(hyde_doc)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    logger.info(
        "[QueryRewriter] {} unique query variants generated", len(unique)
    )
    return unique
