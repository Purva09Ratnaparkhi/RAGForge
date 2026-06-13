"""
Smart Chunker — Component 1: Three-strategy document chunking.
===============================================================

Splits clean LangChain ``Document`` objects into smaller, semantically
coherent chunks optimised for embedding and retrieval.

Strategy pipeline (applied in order)
-------------------------------------
A. **Recursive Chunking** — structurally splits text at paragraph,
   sentence, and word boundaries using LangChain's
   ``RecursiveCharacterTextSplitter``.
B. **Semantic Chunking** — merges adjacent chunks whose embeddings are
   highly similar (above ``SEMANTIC_THRESHOLD``), indicating they belong
   to the same topic and should not have been split.
C. **Context-Aware Overlap** — prepends the last 1-2 sentences of the
   previous chunk to each chunk so no sentence loses its surrounding
   context at a chunk boundary.

Technology choices
------------------
* ``RecursiveCharacterTextSplitter`` — gold-standard structural splitter
  in LangChain; respects natural text boundaries.
* ``SemanticChunker`` from ``langchain_experimental`` — uses embedding
  cosine similarity to detect topic shifts.
* Custom overlap logic for fine-grained context preservation.

Connects to
-----------
* ``src.embeddings.bge_embedder`` — provides embeddings for semantic
  chunking.
* ``config.settings`` — ``CHUNK_SIZE``, ``CHUNK_OVERLAP``,
  ``SEMANTIC_THRESHOLD``, ``EMBEDDING_MODEL``.
"""

from __future__ import annotations

import re
from copy import deepcopy

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    SEMANTIC_THRESHOLD,
)

# ── Strategy A: Recursive Chunking ────────────────────────────────────────


def _recursive_chunk(documents: list[Document]) -> list[Document]:
    """Split documents using ``RecursiveCharacterTextSplitter``.

    Parameters
    ----------
    documents : list[Document]
        Clean page-level Document objects.

    Returns
    -------
    list[Document]
        Smaller chunk-level Documents.  All original metadata is preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.debug("[SmartChunker] Recursive split → {} chunks", len(chunks))
    return chunks


# ── Strategy B: Semantic Chunking (merge pass) ────────────────────────────


def _semantic_merge(chunks: list[Document]) -> list[Document]:
    """Merge adjacent chunks whose cosine similarity exceeds the threshold.

    Uses ``SemanticChunker`` from ``langchain_experimental`` to identify
    topic boundaries, then merges chunks that are topically continuous.

    Parameters
    ----------
    chunks : list[Document]
        Chunks from the recursive splitting step.

    Returns
    -------
    list[Document]
        Potentially fewer, larger chunks after merging.
    """
    if len(chunks) <= 1:
        return chunks

    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )

        # Group chunks by source document to maintain coherence
        merged: list[Document] = []
        source_groups: dict[str, list[Document]] = {}
        for chunk in chunks:
            key = chunk.metadata.get("source", "unknown")
            source_groups.setdefault(key, []).append(chunk)

        for source, group in source_groups.items():
            combined_text = "\n\n".join(c.page_content for c in group)
            try:
                semantic_docs = semantic_chunker.create_documents([combined_text])
                # Transfer metadata from the first chunk of the group
                base_meta = deepcopy(group[0].metadata)
                for sd in semantic_docs:
                    sd.metadata = deepcopy(base_meta)
                merged.extend(semantic_docs)
            except Exception as exc:
                logger.warning(
                    "[SmartChunker] Semantic merge failed for {}: {}. "
                    "Keeping recursive chunks.",
                    source, exc,
                )
                merged.extend(group)

        logger.debug(
            "[SmartChunker] Semantic merge: {} → {} chunks",
            len(chunks), len(merged),
        )
        return merged

    except ImportError:
        logger.warning(
            "[SmartChunker] langchain_experimental not available; "
            "skipping semantic merge"
        )
        return chunks


# ── Strategy C: Context-Aware Overlap ─────────────────────────────────────


def _extract_last_sentences(text: str, n: int = 2) -> str:
    """Extract the last ``n`` sentences from text.

    Parameters
    ----------
    text : str
    n : int

    Returns
    -------
    str
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    last = sentences[-n:] if len(sentences) >= n else sentences
    return " ".join(last)


def _add_context_overlap(chunks: list[Document]) -> list[Document]:
    """Prepend trailing sentences from the previous chunk.

    For each chunk (after the first), prepend the last 1-2 sentences of
    the preceding chunk.  This ensures context continuity at boundaries.

    Parameters
    ----------
    chunks : list[Document]

    Returns
    -------
    list[Document]
    """
    if len(chunks) <= 1:
        return chunks

    result: list[Document] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = _extract_last_sentences(chunks[i - 1].page_content)
        current = chunks[i]

        # Only prepend if from the same source document
        same_source = (
            current.metadata.get("source")
            == chunks[i - 1].metadata.get("source")
        )
        if same_source and prev_tail:
            new_content = f"[...] {prev_tail}\n\n{current.page_content}"
        else:
            new_content = current.page_content

        new_doc = Document(
            page_content=new_content,
            metadata=deepcopy(current.metadata),
        )
        result.append(new_doc)

    logger.debug("[SmartChunker] Context overlap applied to {} chunks", len(result))
    return result


# ── Public API ────────────────────────────────────────────────────────────


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Run the full 3-strategy chunking pipeline.

    Parameters
    ----------
    documents : list[Document]
        Clean page-level Documents from the extraction pipeline.

    Returns
    -------
    list[Document]
        Chunk-level Documents with ``chunk_index`` and ``chunk_total``
        added to each chunk's metadata.

    Example
    -------
    >>> chunks = chunk_documents(page_docs)
    >>> chunks[0].metadata["chunk_index"]
    0
    """
    logger.info(
        "[SmartChunker] Chunking {} documents", len(documents)
    )

    # Strategy A: Recursive chunking
    chunks = _recursive_chunk(documents)

    # Strategy B: Semantic merge
    chunks = _semantic_merge(chunks)

    # Strategy C: Context-aware overlap
    chunks = _add_context_overlap(chunks)

    # Add chunk indexing metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    logger.info(
        "[SmartChunker] Final output: {} chunks from {} documents",
        len(chunks), len(documents),
    )
    return chunks
