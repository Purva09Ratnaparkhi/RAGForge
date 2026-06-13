"""
Tests for the hybrid retrieval pipeline.
==========================================

Tests cover query rewriting, result deduplication, and reranking logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document
from src.retrieval.query_rewriter import _parse_numbered_list


# ── Query rewriter tests ─────────────────────────────────────────────────


class TestQueryRewriter:
    """Tests for query rewriting utilities."""

    def test_parse_numbered_list_standard(self):
        """Standard numbered list is parsed correctly."""
        text = "1. First item\n2. Second item\n3. Third item"
        result = _parse_numbered_list(text)
        assert len(result) == 3
        assert result[0] == "First item"

    def test_parse_numbered_list_with_parens(self):
        """Parenthesis-style numbering is handled."""
        text = "1) Alpha\n2) Beta\n3) Gamma"
        result = _parse_numbered_list(text)
        assert len(result) == 3
        assert result[1] == "Beta"

    def test_parse_numbered_list_with_bullets(self):
        """Bullet-style lists are handled."""
        text = "- First\n- Second\n- Third"
        result = _parse_numbered_list(text)
        assert len(result) == 3

    def test_parse_numbered_list_empty(self):
        """Empty input returns empty list."""
        assert _parse_numbered_list("") == []


# ── Reranker tests (mocked) ──────────────────────────────────────────────


class TestReranker:
    """Tests for the cross-encoder reranker (mocked model)."""

    def test_rerank_empty_documents(self):
        """Empty document list returns empty result."""
        from src.retrieval.reranker import CrossEncoderReranker

        with patch.object(CrossEncoderReranker, "__init__", lambda self: None):
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker._model = MagicMock()
            result = reranker.rerank("query", [])
            assert result == []

    def test_rerank_adds_score_metadata(self):
        """Reranked documents get rerank_score in metadata."""
        from src.retrieval.reranker import CrossEncoderReranker

        with patch.object(CrossEncoderReranker, "__init__", lambda self: None):
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.3, 0.7]
            reranker._model = mock_model

            docs = [
                Document(page_content="doc1", metadata={}),
                Document(page_content="doc2", metadata={}),
                Document(page_content="doc3", metadata={}),
            ]
            result = reranker.rerank("query", docs, top_k=2)
            assert len(result) == 2
            assert result[0].metadata["rerank_score"] == 0.9
            assert result[1].metadata["rerank_score"] == 0.7


# ── Deduplication tests ──────────────────────────────────────────────────


class TestDeduplication:
    """Tests for result deduplication logic."""

    def test_dedup_by_content_prefix(self):
        """Documents with identical first 200 chars are deduped."""
        content = "A" * 300
        docs = [
            Document(page_content=content, metadata={"id": 1}),
            Document(page_content=content, metadata={"id": 2}),
        ]
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        assert len(unique) == 1
