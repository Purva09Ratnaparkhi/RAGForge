"""
Cross-Encoder Reranker — Component 7: Neural relevance reranking.
==================================================================

After initial retrieval (which is fast but coarse), the cross-encoder
evaluates each (query, document) pair jointly to produce a fine-grained
relevance score.  The top-k by this score are kept for generation.

Why cross-encoder?
------------------
Bi-encoder embeddings (used in FAISS) encode query and document
independently, so they cannot model word-level interactions.  A
cross-encoder processes both together through a full transformer,
enabling much more accurate relevance judgements — at the cost of
being too slow for the initial retrieval stage (hence the two-stage
approach).

Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — a compact (22 M
parameter) model fine-tuned on MS MARCO passage ranking, striking
a good balance between accuracy and CPU inference speed.

Connects to
-----------
* ``src.agents.rag_agent`` — calls ``rerank`` after retrieval.
* ``config.settings``      — ``RERANKER_MODEL``, ``RERANK_TOP_K``.
"""

from __future__ import annotations

from langchain_core.documents import Document
from loguru import logger
from sentence_transformers import CrossEncoder

from config.settings import RERANK_TOP_K, RERANKER_MODEL


class CrossEncoderReranker:
    """Reranks retrieved documents using a cross-encoder model.

    The cross-encoder is loaded once in ``__init__`` and reused.
    This class is justified because the model (~80 MB) should be loaded
    once and kept in memory.

    Example
    -------
    >>> reranker = CrossEncoderReranker()
    >>> top = reranker.rerank("What is attention?", candidates)
    >>> len(top)
    5
    """

    def __init__(self) -> None:
        logger.info("[Reranker] Loading {}", RERANKER_MODEL)
        self._model = CrossEncoder(RERANKER_MODEL)
        logger.info("[Reranker] Model loaded")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = RERANK_TOP_K,
    ) -> list[Document]:
        """Score and rerank documents by relevance to the query.

        Parameters
        ----------
        query : str
            User query.
        documents : list[Document]
            Candidate documents from the retrieval stage.
        top_k : int
            Number of top-ranked documents to return.

        Returns
        -------
        list[Document]
            Top-k documents sorted by cross-encoder relevance score
            (descending).  Each document gets a ``rerank_score`` field
            added to its metadata.

        Example
        -------
        >>> reranked = reranker.rerank("What is BERT?", docs, top_k=5)
        >>> reranked[0].metadata["rerank_score"]
        0.987
        """
        if not documents:
            logger.warning("[Reranker] No documents to rerank")
            return []

        logger.info(
            "[Reranker] Reranking {} candidates for: {}",
            len(documents), query[:80],
        )

        # Build (query, document) pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored = list(zip(scores, documents))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_docs: list[Document] = []
        for score, doc in scored[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            top_docs.append(doc)

        logger.info(
            "[Reranker] Top-{} scores: {}",
            top_k,
            [f"{s:.3f}" for s, _ in scored[:top_k]],
        )
        return top_docs
