"""
Tests for guardrails and generation components.
=================================================

Tests cover guardrail result structure, JSON parsing, hallucination
detection, and confidence scoring logic.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.guardrails.guardrails import (
    _parse_json_response,
    check_groundedness,
    detect_hallucinations,
    score_confidence,
)


# ── JSON parsing tests ───────────────────────────────────────────────────


class TestJSONParsing:
    """Tests for best-effort JSON extraction from LLM output."""

    def test_parse_clean_json(self):
        """Clean JSON string is parsed correctly."""
        text = '{"grounded": true, "unsupported_claims": []}'
        result = _parse_json_response(text)
        assert result["grounded"] is True
        assert result["unsupported_claims"] == []

    def test_parse_json_in_code_block(self):
        """JSON inside markdown code block is extracted."""
        text = 'Here is the analysis:\n```json\n{"grounded": false}\n```'
        result = _parse_json_response(text)
        assert result["grounded"] is False

    def test_parse_json_with_preamble(self):
        """JSON with preceding text is extracted."""
        text = 'The answer appears grounded. {"confidence": 8}'
        result = _parse_json_response(text)
        assert result["confidence"] == 8

    def test_parse_invalid_json(self):
        """Invalid JSON returns empty dict."""
        result = _parse_json_response("not json at all")
        assert result == {}


# ── Hallucination detection tests ────────────────────────────────────────


class TestHallucinationDetection:
    """Tests for the spaCy-based hallucination detector."""

    def test_detect_number_not_in_context(self):
        """Numbers in answer but not context are flagged."""
        context = "The model achieves 85% accuracy on the test set."
        answer = "The model achieves 95% accuracy."
        flags = detect_hallucinations(context, answer)
        # Should flag 95% since it's not in context
        has_number_flag = any("95" in f for f in flags)
        assert has_number_flag

    def test_no_flags_when_matching(self):
        """No flags when all entities are in context."""
        context = "BERT was proposed by Google in 2018."
        answer = "BERT was proposed by Google in 2018."
        flags = detect_hallucinations(context, answer)
        assert len(flags) == 0


# ── Groundedness check tests (mocked LLM) ────────────────────────────────


class TestGroundedness:
    """Tests for LLM-based groundedness checking."""

    def test_grounded_response(self):
        """Grounded answer returns True."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"grounded": true, "unsupported_claims": []}'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with patch("src.guardrails.guardrails.GROUNDEDNESS_PROMPT.__or__",
                   return_value=mock_chain):
            result = check_groundedness("context", "answer", mock_llm)
            assert result["grounded"] is True

    def test_ungrounded_response(self):
        """Ungrounded answer returns False with claims."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"grounded": false, "unsupported_claims": ["claim X"]}'
        )
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with patch("src.guardrails.guardrails.GROUNDEDNESS_PROMPT.__or__",
                   return_value=mock_chain):
            result = check_groundedness("context", "answer", mock_llm)
            assert result["grounded"] is False
            assert len(result["unsupported_claims"]) == 1


# ── Confidence scoring tests (mocked LLM) ────────────────────────────────


class TestConfidenceScoring:
    """Tests for LLM-based confidence scoring."""

    def test_high_confidence(self):
        """High confidence score normalises correctly."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"confidence": 9, "reasoning": "Well supported"}'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with patch("src.guardrails.guardrails.CONFIDENCE_PROMPT.__or__",
                   return_value=mock_chain):
            score, reason = score_confidence("ctx", "q", "a", mock_llm)
            assert score == 0.9
            assert "Well supported" in reason
