"""
Hybrid Retriever — Component 4: FAISS + BM25 with Reciprocal Rank Fusion.
==========================================================================

Combines dense vector retrieval (FAISS) with sparse keyword retrieval
(BM25) via LangChain's ``EnsembleRetriever``.  The ensemble applies
Reciprocal Rank Fusion (RRF) internally:

    score = Σ  1 / (rank_i + k)    for each retriever i

where k = 60 (``RRF_K`` from settings).  This balances semantic
understanding (FAISS) with exact keyword matching (BM25), which is
critical for academic queries containing specific technical terms,
acronyms, or author names.

Why hybrid?
-----------
Pure vector search misses exact keyword matches (e.g. "BERT" vs
semantically similar "transformer model").  Pure BM25 misses
paraphrased concepts.  Combining them with RRF gives the best of both.

Connects to
-----------
* ``src.vector_store.faiss_store`` — provides FAISS + BM25 indexes.
* ``src.agents.rag_agent``        — calls ``retrieve`` in the graph.
* ``config.settings``             — ``BM25_WEIGHT``, ``VECTOR_WEIGHT``,
  ``RETRIEVAL_TOP_K``.
"""

from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from loguru import logger

from config.settings import BM25_WEIGHT, RETRIEVAL_TOP_K, VECTOR_WEIGHT
from src.vector_store.faiss_store import FAISSVectorStore


class HybridRetriever:
    """Combines FAISS and BM25 retrievers with RRF fusion.

    Loads both indexes from disk on initialisation.  This class is
    justified because it holds loaded indexes that should persist
    across queries.

    Example
    -------
    >>> retriever = HybridRetriever()
    >>> docs = retriever.retrieve("What is attention?")
    >>> len(docs)
    20
    """

    def __init__(self) -> None:
        logger.info("[HybridRetriever] Initialising")
        self._store = FAISSVectorStore()

        # Load FAISS retriever
        self._faiss_retriever = self._store.as_retriever(k=RETRIEVAL_TOP_K)

        # Load BM25 retriever
        self._bm25_retriever = self._store.load_bm25()
        self._bm25_retriever.k = RETRIEVAL_TOP_K

        # Build ensemble
        self._ensemble = EnsembleRetriever(
            retrievers=[self._bm25_retriever, self._faiss_retriever],
            weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        )
        logger.info(
            "[HybridRetriever] Ready (BM25 weight={}, FAISS weight={})",
            BM25_WEIGHT, VECTOR_WEIGHT,
        )

    def retrieve(
        self, query: str, k: int = RETRIEVAL_TOP_K
    ) -> list[Document]:
        """Retrieve documents using hybrid FAISS + BM25 search.

        Parameters
        ----------
        query : str
            User query string.
        k : int
            Maximum number of documents to return.

        Returns
        -------
        list[Document]
            Top-k documents ranked by RRF score.

        Example
        -------
        >>> docs = retriever.retrieve("transformer architecture")
        """
        logger.info("[HybridRetriever] Retrieving for: {}", query[:80])
        results = self._ensemble.invoke(query)

        # EnsembleRetriever may return more than k; trim
        results = results[:k]
        logger.info(
            "[HybridRetriever] Retrieved {} documents", len(results)
        )
        return results

    def retrieve_multi_query(
        self, queries: list[str], k: int = RETRIEVAL_TOP_K
    ) -> list[Document]:
        """Retrieve using multiple query variants and merge via dedup.

        Parameters
        ----------
        queries : list[str]
            Query variants (from query rewriter).
        k : int
            Maximum number of unique documents to return.

        Returns
        -------
        list[Document]
            Deduplicated top-k documents.
        """
        logger.info(
            "[HybridRetriever] Multi-query retrieval ({} variants)",
            len(queries),
        )
        seen_contents: set[str] = set()
        merged: list[Document] = []

        for query in queries:
            docs = self.retrieve(query, k=k)
            for doc in docs:
                content_hash = doc.page_content[:200]
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    merged.append(doc)

        # Return top-k unique documents
        merged = merged[:k]
        logger.info(
            "[HybridRetriever] Multi-query merged → {} unique docs",
            len(merged),
        )
        return merged
