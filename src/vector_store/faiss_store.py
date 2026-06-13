"""
FAISS Vector Store — Component 3: Persisted similarity search.
===============================================================

Manages a FAISS index for dense vector similarity search and a pickled
BM25 retriever for keyword-based search.  Both are persisted to disk
so that the index survives server restarts without re-embedding.

Why FAISS?
----------
FAISS (Facebook AI Similarity Search) is the most widely deployed
open-source vector similarity library.  It runs on CPU, supports
incremental additions, and integrates natively with LangChain.

Why persist BM25 here?
-----------------------
The BM25 retriever is built from the same document corpus.  Storing it
alongside the FAISS index keeps the ingestion artefacts co-located and
ensures both are in sync.

Connects to
-----------
* ``src.embeddings.bge_embedder`` — provides the embedding function.
* ``src.retrieval.hybrid_retriever`` — loads both indexes at query time.
* ``config.settings`` — ``FAISS_STORE_DIR``, ``RETRIEVAL_TOP_K``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from config.settings import FAISS_STORE_DIR, RETRIEVAL_TOP_K
from src.embeddings.bge_embedder import BGEEmbedder

# Sub-directory names within FAISS_STORE_DIR
_INDEX_NAME = "index"
_BM25_FILE = "bm25.pkl"


class FAISSVectorStore:
    """Manages the FAISS index and co-located BM25 retriever.

    The BGE embedding model is loaded once and shared between build,
    load, and add operations.

    Example
    -------
    >>> store = FAISSVectorStore()
    >>> store.build(documents)
    >>> retriever = store.as_retriever(k=10)
    """

    def __init__(self) -> None:
        logger.info("[FAISSStore] Initialising")
        self._embedder = BGEEmbedder()
        self._store: FAISS | None = None

    # ── Build / Persist ───────────────────────────────────────────────

    def build(self, documents: list[Document]) -> None:
        """Build a new FAISS index + BM25 retriever and save to disk.

        Parameters
        ----------
        documents : list[Document]
            Chunked documents to index.

        Example
        -------
        >>> store.build(chunked_docs)
        """
        if not documents:
            logger.warning("[FAISSStore] No documents to index")
            return

        logger.info(
            "[FAISSStore] Building index from {} documents", len(documents)
        )

        hf_embeddings = self._embedder.get_langchain_embeddings()

        # Build FAISS
        self._store = FAISS.from_documents(documents, hf_embeddings)
        index_path = FAISS_STORE_DIR / _INDEX_NAME
        self._store.save_local(str(index_path))
        logger.info("[FAISSStore] FAISS index saved to {}", index_path)

        # Build and pickle BM25
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = RETRIEVAL_TOP_K
        bm25_path = FAISS_STORE_DIR / _BM25_FILE
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        logger.info("[FAISSStore] BM25 retriever saved to {}", bm25_path)

    # ── Load ──────────────────────────────────────────────────────────

    def load(self) -> FAISS:
        """Load a previously-persisted FAISS index from disk.

        Returns
        -------
        FAISS
            The loaded vector store.

        Raises
        ------
        FileNotFoundError
            If the index directory does not exist.

        Example
        -------
        >>> faiss_store = store.load()
        """
        index_path = FAISS_STORE_DIR / _INDEX_NAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found at {index_path}. Run build() first."
            )

        logger.info("[FAISSStore] Loading index from {}", index_path)
        hf_embeddings = self._embedder.get_langchain_embeddings()
        self._store = FAISS.load_local(
            str(index_path),
            hf_embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("[FAISSStore] Index loaded")
        return self._store

    def load_bm25(self) -> BM25Retriever:
        """Load the pickled BM25 retriever from disk.

        Returns
        -------
        BM25Retriever

        Raises
        ------
        FileNotFoundError
            If the BM25 pickle does not exist.
        """
        bm25_path = FAISS_STORE_DIR / _BM25_FILE
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"No BM25 retriever found at {bm25_path}. Run build() first."
            )

        logger.info("[FAISSStore] Loading BM25 from {}", bm25_path)
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)  # noqa: S301
        return bm25

    # ── Retriever interface ───────────────────────────────────────────

    def as_retriever(self, k: int = RETRIEVAL_TOP_K):
        """Return a LangChain retriever backed by the FAISS index.

        Parameters
        ----------
        k : int
            Number of documents to retrieve.

        Returns
        -------
        VectorStoreRetriever
        """
        if self._store is None:
            self.load()
        return self._store.as_retriever(search_kwargs={"k": k})

    # ── Incremental ingestion ─────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to an existing FAISS index.

        Parameters
        ----------
        documents : list[Document]
            New documents to add.

        Example
        -------
        >>> store.add_documents(new_chunks)
        """
        if self._store is None:
            self.load()

        logger.info(
            "[FAISSStore] Adding {} documents to existing index",
            len(documents),
        )
        self._store.add_documents(documents)

        # Re-save
        index_path = FAISS_STORE_DIR / _INDEX_NAME
        self._store.save_local(str(index_path))
        logger.info("[FAISSStore] Updated index saved")
