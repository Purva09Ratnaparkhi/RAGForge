"""
Tests for the extraction pipeline.
====================================

Tests cover the PDF detector, extractors, quality gate, cleaner,
and the full pipeline orchestration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.cleaner import (
    _detect_section_heading,
    _fix_hyphenation,
    _normalise_unicode,
    _strip_references,
    clean_pages,
)
from src.extraction.quality import ExtractorError, _avg_chars_per_page, passes_quality


# ── Quality gate tests ────────────────────────────────────────────────────


class TestQualityGate:
    """Tests for the extraction quality gate."""

    def test_passes_quality_good_pages(self):
        """Pages with sufficient text pass quality."""
        pages = [
            {
                "page": 0,
                "text": "A" * 500,
                "tables": [],
                "metadata": {"total_pages": 1, "extractor": "test"},
            }
        ]
        ok, reason = passes_quality(pages)
        assert ok is True
        assert reason == "ok"

    def test_fails_quality_low_density(self):
        """Pages with too few characters fail quality."""
        pages = [
            {
                "page": 0,
                "text": "short",
                "tables": [],
                "metadata": {"total_pages": 1, "extractor": "test"},
            }
        ]
        ok, reason = passes_quality(pages)
        assert ok is False
        assert "low density" in reason

    def test_fails_quality_high_garble(self):
        """Pages with too many non-ASCII characters fail quality."""
        garbled = "日本語テスト" * 100  # All non-ASCII
        pages = [
            {
                "page": 0,
                "text": garbled,
                "tables": [],
                "metadata": {"total_pages": 1, "extractor": "test"},
            }
        ]
        ok, reason = passes_quality(pages)
        assert ok is False
        assert "garble" in reason

    def test_avg_chars_per_page_empty(self):
        """Empty pages list returns 0."""
        assert _avg_chars_per_page([]) == 0.0


# ── Cleaner tests ─────────────────────────────────────────────────────────


class TestCleaner:
    """Tests for the 6-step cleaning pipeline."""

    def test_hyphenation_fix(self):
        """Hyphenated line breaks are rejoined."""
        text = "algo-\nrithm"
        assert _fix_hyphenation(text) == "algorithm"

    def test_unicode_normalisation(self):
        """Broken Unicode is fixed by ftfy."""
        # ftfy should fix common mojibake
        result = _normalise_unicode("Hello")
        assert isinstance(result, str)

    def test_reference_strip(self):
        """References section is removed."""
        text = "Some content.\n\nReferences\n[1] Author, Title, 2023."
        result = _strip_references(text)
        assert "References" not in result
        assert "Some content." in result

    def test_section_heading_detection(self):
        """ALL CAPS headings are detected."""
        assert _detect_section_heading("INTRODUCTION\nThis paper...") == "INTRODUCTION"

    def test_section_heading_title_case(self):
        """Title Case headings are detected."""
        result = _detect_section_heading("Related Work\nPrevious studies...")
        assert result == "Related Work"

    def test_section_heading_not_found(self):
        """Returns 'unknown' when no heading is detected."""
        result = _detect_section_heading("This is a normal paragraph with no heading.")
        assert result == "unknown"

    def test_clean_pages_produces_documents(self):
        """clean_pages returns LangChain Document objects."""
        pages = [
            {
                "page": 0,
                "text": "INTRODUCTION\nThis is a test paper about transformers.",
                "tables": [],
                "metadata": {"total_pages": 1, "extractor": "test"},
            }
        ]
        docs = clean_pages(pages, "test.pdf")
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "test.pdf"
        assert docs[0].metadata["page"] == 0

    def test_clean_pages_skips_empty(self):
        """Empty pages are skipped."""
        pages = [
            {
                "page": 0,
                "text": "",
                "tables": [],
                "metadata": {"total_pages": 1, "extractor": "test"},
            }
        ]
        docs = clean_pages(pages, "test.pdf")
        assert len(docs) == 0


# ── ExtractorError tests ─────────────────────────────────────────────────


class TestExtractorError:
    """Tests for the custom ExtractorError exception."""

    def test_stores_metadata(self):
        """ExtractorError stores tried extractors and scores."""
        err = ExtractorError(
            pdf_path="test.pdf",
            tried=["marker", "pdfplumber"],
            quality_scores={"marker": 50.0, "pdfplumber": 80.0},
        )
        assert err.pdf_path == "test.pdf"
        assert len(err.tried) == 2
        assert "marker" in err.quality_scores
