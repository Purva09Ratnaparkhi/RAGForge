"""
Context Compressor — Component 8: Two-stage context compression.
=================================================================

After reranking, the context compressor reduces each chunk to only
the sentences relevant to the query (Step 1), then removes near-
duplicate chunks that survived the pipeline (Step 2).

This is critical for fitting more useful information into the LLM's
limited context window and reducing distraction from irrelevant
passages that happened to be in a relevant chunk.

Pipeline
--------
1. **LLMChainExtractor** — for each chunk, the LLM extracts only the
   sentences that directly answer or relate to the query.
2. **EmbeddingsRedundantFilter** — removes chunks whose embedding
   cosine similarity exceeds 0.95 (near-duplicates from overlapping
   windows or multi-query retrieval).

Both are chained via ``DocumentCompressorPipeline`` and wrapped in
``ContextualCompressionRetriever``.

Connects to
-----------
* ``src.retrieval.reranker``  — provides reranked documents as input.
* ``src.agents.rag_agent``    — calls ``compress`` before generation.
* ``src.generation.llm_factory`` — provides the LLM for extraction.
* ``src.embeddings.bge_embedder`` — provides embeddings for dedup.
"""

from __future__ import annotations

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
)
from langchain_core.documents import Document
from loguru import logger

from src.embeddings.bge_embedder import BGEEmbedder


def build_compressor(llm, embedder: BGEEmbedder | None = None):
    """Build the two-stage compression pipeline.

    Parameters
    ----------
    llm
        LangChain LLM for extracting relevant sentences.
    embedder : BGEEmbedder | None
        Embedding model for redundancy filtering.  If ``None``, a new
        instance is created.

    Returns
    -------
    DocumentCompressorPipeline
        Ready-to-use compressor pipeline.

    Example
    -------
    >>> pipeline = build_compressor(llm)
    """
    if embedder is None:
        embedder = BGEEmbedder()

    # Step 1: LLM-based relevant sentence extraction
    extractor = LLMChainExtractor.from_llm(llm)

    # Step 2: Embedding-based redundancy filter
    redundant_filter = EmbeddingsFilter(
        embeddings=embedder.get_langchain_embeddings(),
        similarity_threshold=0.95,
    )

    pipeline = DocumentCompressorPipeline(
        transformers=[extractor, redundant_filter]
    )
    logger.info("[Compressor] Pipeline built (LLMExtractor + RedundancyFilter)")
    return pipeline


def compress(
    query: str,
    documents: list[Document],
    llm,
    embedder: BGEEmbedder | None = None,
) -> list[Document]:
    """Compress documents to only query-relevant content.

    Parameters
    ----------
    query : str
        User query.
    documents : list[Document]
        Reranked candidate documents.
    llm
        LangChain LLM for extraction.
    embedder : BGEEmbedder | None
        Embedding model for redundancy check.

    Returns
    -------
    list[Document]
        Compressed documents with only relevant sentences.

    Example
    -------
    >>> compressed = compress("What is attention?", docs, llm)
    >>> len(compressed) <= len(docs)
    True
    """
    if not documents:
        logger.warning("[Compressor] No documents to compress")
        return []

    logger.info(
        "[Compressor] Compressing {} documents for: {}",
        len(documents), query[:80],
    )

    from langchain_core.prompts import ChatPromptTemplate

    extract_prompt = ChatPromptTemplate.from_template(
        "Given the following text, extract ONLY the sentences that are "
        "directly relevant to answering the question. If no sentences "
        "are relevant, respond with 'NO_RELEVANT_CONTENT'.\n\n"
        "Question: {question}\n\n"
        "Text:\n{text}\n\n"
        "Relevant sentences:"
    )

    chain = extract_prompt | llm
    compressed: list[Document] = []

    for doc in documents:
        try:
            result = chain.invoke({
                "question": query,
                "text": doc.page_content[:2000],
            })
            text = result.content if hasattr(result, "content") else str(result)

            # Skip if LLM found nothing relevant
            if not text.strip() or "NO_RELEVANT_CONTENT" in text:
                logger.debug("[Compressor] Chunk had no relevant content, skipping")
                continue

            compressed.append(
                Document(
                    page_content=text.strip(),
                    metadata={**doc.metadata, "compressed": True},
                )
            )
        except Exception as exc:
            logger.warning(
                "[Compressor] Failed to compress chunk: {}. Keeping original.",
                exc,
            )
            compressed.append(doc)

    # If compression removed everything, fall back to originals
    if not compressed:
        logger.warning("[Compressor] All chunks filtered out. Keeping originals.")
        compressed = list(documents)

    # Final dedup pass
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in compressed:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    logger.info(
        "[Compressor] Compressed {} → {} documents",
        len(documents), len(unique),
    )
    return unique


def build_compression_retriever(base_retriever, llm, embedder=None):
    """Wrap a base retriever with contextual compression.

    Parameters
    ----------
    base_retriever
        Any LangChain retriever (e.g. hybrid retriever).
    llm
        LangChain LLM.
    embedder
        Optional BGE embedder.

    Returns
    -------
    ContextualCompressionRetriever
        A retriever that automatically compresses results.

    Example
    -------
    >>> comp_retriever = build_compression_retriever(hybrid, llm)
    >>> docs = comp_retriever.invoke("What is attention?")
    """
    pipeline = build_compressor(llm, embedder)
    return ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=base_retriever,
    )
