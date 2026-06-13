"""
BGE Embedder — Component 2: Dense vector embeddings.
=====================================================

Wraps the ``BAAI/bge-small-en-v1.5`` model from HuggingFace for
generating 384-dimensional normalised embeddings.

Why BGE?
--------
BGE (BAAI General Embedding) models are among the highest-scoring
open-source embedding models on the MTEB leaderboard.  The ``small``
variant provides an excellent trade-off between quality and speed for
CPU-based deployment.

Important BGE detail
--------------------
BGE queries must be prefixed with:
    ``"Represent this sentence for searching relevant passages: "``
This prefix was part of the model's training objective and significantly
improves retrieval accuracy.  The ``embed_query`` method applies it
automatically.

Connects to
-----------
* ``src.chunking.smart_chunker`` — semantic chunking uses these embeddings.
* ``src.vector_store.faiss_store`` — builds the FAISS index from embeddings.
* ``config.settings`` — ``EMBEDDING_MODEL``, ``EMBEDDING_DEVICE``.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from config.settings import EMBEDDING_DEVICE, EMBEDDING_MODEL

# BGE query prefix — required for asymmetric retrieval
_BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


class BGEEmbedder:
    """Wrapper around ``HuggingFaceEmbeddings`` for BGE models.

    The model is loaded once in ``__init__`` and reused.  This class is
    justified because the underlying HuggingFace model is expensive to
    load (~100 MB) and should not be re-initialised per call.

    Parameters
    ----------
    (none — all configuration comes from ``config.settings``)

    Example
    -------
    >>> embedder = BGEEmbedder()
    >>> vec = embedder.embed_query("What is attention?")
    >>> len(vec)
    384
    """

    def __init__(self) -> None:
        logger.info(
            "[BGEEmbedder] Loading {} on {}",
            EMBEDDING_MODEL, EMBEDDING_DEVICE,
        )
        self._hf = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("[BGEEmbedder] Model loaded")

    # ── Public API ────────────────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        No query prefix is applied — documents are embedded as-is.

        Parameters
        ----------
        texts : list[str]
            Document texts to embed.

        Returns
        -------
        list[list[float]]
            One 384-d vector per text.

        Example
        -------
        >>> vecs = embedder.embed_documents(["Hello", "World"])
        >>> len(vecs)
        2
        """
        return self._hf.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query with the BGE retrieval prefix.

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        list[float]
            384-dimensional normalised vector.

        Example
        -------
        >>> vec = embedder.embed_query("What is attention?")
        >>> len(vec)
        384
        """
        prefixed = f"{_BGE_QUERY_PREFIX}{query}"
        return self._hf.embed_query(prefixed)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the raw HuggingFaceEmbeddings object.

        Useful for passing directly to LangChain components that accept
        an ``Embeddings`` instance (e.g. ``FAISS.from_documents``).

        Returns
        -------
        HuggingFaceEmbeddings
        """
        return self._hf
