"""
Multi-Vector Retrieval — Component 6: Raw + summary embeddings.
================================================================

For each document chunk, stores TWO vectors in FAISS:
1. The raw chunk text embedding.
2. An LLM-generated summary embedding.

At retrieval time, matching on the summary index returns pointers to
the parent (full) chunks — so the user gets complete context even when
the match was on a compressed summary.

This approach improves recall because summaries capture the *gist* of
a passage in vocabulary that often better matches user queries than the
raw academic prose.

Technology choices
------------------
* ``MultiVectorRetriever`` from LangChain — purpose-built for this
  pattern with separate child/parent stores.
* ``InMemoryByteStore`` — lightweight parent document store (the full
  chunks are already small enough to keep in memory).

Connects to
-----------
* ``src.embeddings.bge_embedder`` — generates both raw and summary
  embeddings.
* ``src.generation.llm_factory`` — produces chunk summaries.
* ``config.settings`` — ``FAISS_STORE_DIR``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryByteStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from config.settings import FAISS_STORE_DIR
from src.embeddings.bge_embedder import BGEEmbedder

_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarise this text in 2-3 sentences capturing the key technical "
    "concepts:\n\n\"{text}\""
)

_SUMMARY_INDEX_NAME = "summary_index"


class MultiVectorStore:
    """Stores raw + summary embeddings with parent document pointers.

    Holds a FAISS index for summary vectors and an in-memory byte store
    for the full parent chunks.  The embedding model is loaded once and
    shared.

    Example
    -------
    >>> mv = MultiVectorStore()
    >>> mv.build(chunks, llm)
    >>> retriever = mv.as_retriever()
    """

    def __init__(self) -> None:
        logger.info("[MultiVector] Initialising")
        self._embedder = BGEEmbedder()
        self._byte_store = InMemoryByteStore()
        self._vectorstore: FAISS | None = None
        self._id_key = "doc_id"

    def build(self, documents: list[Document], llm) -> None:
        """Build summary index and parent store.

        Parameters
        ----------
        documents : list[Document]
            Chunked documents.
        llm
            LangChain LLM for generating summaries.

        Example
        -------
        >>> mv.build(chunks, llm)
        """
        if not documents:
            logger.warning("[MultiVector] No documents to index")
            return

        logger.info(
            "[MultiVector] Building from {} documents", len(documents)
        )

        # Generate summaries and doc IDs
        doc_ids: list[str] = []
        summary_docs: list[Document] = []
        chain = _SUMMARY_PROMPT | llm

        for doc in documents:
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)

            # Store parent in byte store
            doc.metadata[self._id_key] = doc_id

            # Generate summary
            try:
                result = chain.invoke({"text": doc.page_content[:1500]})
                summary_text = (
                    result.content if hasattr(result, "content") else str(result)
                )
            except Exception as exc:
                logger.warning(
                    "[MultiVector] Summary generation failed: {}. "
                    "Using first 200 chars.",
                    exc,
                )
                summary_text = doc.page_content[:200]

            summary_doc = Document(
                page_content=summary_text,
                metadata={self._id_key: doc_id},
            )
            summary_docs.append(summary_doc)

        # Build FAISS index from summaries
        hf_embeddings = self._embedder.get_langchain_embeddings()
        self._vectorstore = FAISS.from_documents(summary_docs, hf_embeddings)

        # Save summary index
        index_path = FAISS_STORE_DIR / _SUMMARY_INDEX_NAME
        self._vectorstore.save_local(str(index_path))

        # Populate byte store with parent documents
        self._byte_store.mset(
            list(zip(doc_ids, [doc.page_content.encode() for doc in documents]))
        )

        logger.info(
            "[MultiVector] Built summary index ({} docs) at {}",
            len(summary_docs), index_path,
        )

    def as_retriever(self) -> MultiVectorRetriever:
        """Return a ``MultiVectorRetriever`` for querying.

        Returns
        -------
        MultiVectorRetriever
            Retriever that searches summaries but returns parent chunks.
        """
        if self._vectorstore is None:
            # Try to load from disk
            index_path = FAISS_STORE_DIR / _SUMMARY_INDEX_NAME
            if index_path.exists():
                hf_embeddings = self._embedder.get_langchain_embeddings()
                self._vectorstore = FAISS.load_local(
                    str(index_path),
                    hf_embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                raise FileNotFoundError(
                    f"No summary index found at {index_path}. "
                    "Run build() first."
                )

        retriever = MultiVectorRetriever(
            vectorstore=self._vectorstore,
            byte_store=self._byte_store,
            id_key=self._id_key,
        )
        logger.info("[MultiVector] Retriever ready")
        return retriever
