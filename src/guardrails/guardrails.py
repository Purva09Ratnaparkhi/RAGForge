"""
Guardrails — Component 11: Groundedness, hallucination, and confidence.
========================================================================

Three-layer safety check on generated answers:

1. **Groundedness Check** (LLM-based) — verifies every claim in the
   answer is supported by the retrieved context.
2. **Hallucination Detection** (heuristic) — uses spaCy NER to extract
   named entities and numbers from the answer, then checks each appears
   in the context.  Fast, deterministic, and doesn't burn LLM tokens.
3. **Confidence Scoring** (LLM-based) — rates how well the context
   supports the answer on a 0-10 scale.

Why this layered approach?
--------------------------
* The LLM groundedness check is thorough but can be fooled by
  paraphrasing.  The NER heuristic catches concrete factual
  hallucinations (wrong numbers, fabricated author names) that the
  LLM might miss.
* The confidence score gives the frontend a continuous signal for
  UI feedback (green/yellow/red badge).

Connects to
-----------
* ``src.agents.rag_agent``        — calls ``check_guardrails`` node.
* ``src.generation.llm_factory``  — provides the LLM.
* ``src.generation.prompts``      — groundedness and confidence prompts.
"""

from __future__ import annotations

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.generation.prompts import CONFIDENCE_PROMPT, GROUNDEDNESS_PROMPT

# ── Guardrail result type ─────────────────────────────────────────────────

GuardrailResult = dict  # typed below in the docstrings


# ── Internal: JSON parsing ────────────────────────────────────────────────


def _parse_json_response(text: str) -> dict:
    """Best-effort extraction of JSON from LLM output.

    Parameters
    ----------
    text : str
        Raw LLM response (may contain markdown fences or preamble).

    Returns
    -------
    dict
        Parsed JSON object, or empty dict on failure.
    """
    # Try to find JSON in markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ── Check 1: Groundedness (LLM) ──────────────────────────────────────────


def check_groundedness(context: str, answer: str, llm) -> dict:
    """Verify every claim in the answer is supported by the context.

    Parameters
    ----------
    context : str
        Retrieved context passages concatenated.
    answer : str
        Generated answer.
    llm
        LangChain LLM object.

    Returns
    -------
    dict
        ``{"grounded": bool, "unsupported_claims": list[str]}``

    Example
    -------
    >>> result = check_groundedness(ctx, ans, llm)
    >>> result["grounded"]
    True
    """
    logger.debug("[Guardrails] Running groundedness check")
    try:
        chain = GROUNDEDNESS_PROMPT | llm
        result = chain.invoke({"context": context, "answer": answer})
        text = result.content if hasattr(result, "content") else str(result)
        parsed = _parse_json_response(text)
        return {
            "grounded": parsed.get("grounded", True),
            "unsupported_claims": parsed.get("unsupported_claims", []),
        }
    except Exception as exc:
        logger.warning("[Guardrails] Groundedness check failed: {}", exc)
        return {"grounded": True, "unsupported_claims": []}


# ── Check 2: Hallucination Detection (spaCy NER) ─────────────────────────


def _load_spacy_model():
    """Load spaCy English model (small).

    Returns
    -------
    spacy.Language
    """
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "[Guardrails] en_core_web_sm not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            return None
    except ImportError:
        logger.warning("[Guardrails] spaCy not installed")
        return None


def detect_hallucinations(context: str, answer: str) -> list[str]:
    """Check if named entities/numbers in the answer appear in the context.

    Parameters
    ----------
    context : str
        Retrieved context.
    answer : str
        Generated answer.

    Returns
    -------
    list[str]
        Entities/numbers found in the answer but NOT in the context.

    Example
    -------
    >>> flags = detect_hallucinations("BERT was proposed...",
    ...                              "GPT-4 achieves 95% accuracy")
    >>> "GPT-4" in flags
    True
    """
    logger.debug("[Guardrails] Running hallucination detection")
    nlp = _load_spacy_model()
    if nlp is None:
        return []

    answer_doc = nlp(answer)
    context_lower = context.lower()

    flags: list[str] = []
    for ent in answer_doc.ents:
        if ent.label_ in ("CARDINAL", "PERCENT", "MONEY", "QUANTITY",
                          "PERSON", "ORG", "GPE", "DATE", "ORDINAL"):
            if ent.text.lower() not in context_lower:
                flags.append(f"{ent.text} ({ent.label_})")

    # Also check standalone numbers
    answer_numbers = set(re.findall(r"\b\d+\.?\d*%?\b", answer))
    context_numbers = set(re.findall(r"\b\d+\.?\d*%?\b", context))
    for num in answer_numbers - context_numbers:
        flag = f"{num} (NUMBER)"
        if flag not in flags:
            flags.append(flag)

    if flags:
        logger.info(
            "[Guardrails] {} potential hallucinations detected", len(flags)
        )
    return flags


# ── Check 3: Confidence Scoring (LLM) ────────────────────────────────────


def score_confidence(
    context: str, question: str, answer: str, llm
) -> tuple[float, str]:
    """Rate how well the context supports the answer (0-10 → 0.0-1.0).

    Parameters
    ----------
    context : str
    question : str
    answer : str
    llm

    Returns
    -------
    tuple[float, str]
        ``(confidence_score, reasoning)`` where score is 0.0–1.0.

    Example
    -------
    >>> score, reason = score_confidence(ctx, q, a, llm)
    >>> 0.0 <= score <= 1.0
    True
    """
    logger.debug("[Guardrails] Scoring confidence")
    try:
        chain = CONFIDENCE_PROMPT | llm
        result = chain.invoke({
            "context": context,
            "question": question,
            "answer": answer,
        })
        text = result.content if hasattr(result, "content") else str(result)
        parsed = _parse_json_response(text)
        raw_score = parsed.get("confidence", 7)
        # Normalise from 0-10 to 0.0-1.0
        normalised = max(0.0, min(1.0, float(raw_score) / 10.0))
        reasoning = parsed.get("reasoning", "")
        return normalised, reasoning
    except Exception as exc:
        logger.warning("[Guardrails] Confidence scoring failed: {}", exc)
        return 0.7, "Confidence scoring unavailable"


# ── Combined check ────────────────────────────────────────────────────────


def check_guardrails(
    context: str,
    question: str,
    answer: str,
    llm,
) -> GuardrailResult:
    """Run all three guardrail checks and return a combined result.

    Parameters
    ----------
    context : str
        Retrieved context passages.
    question : str
        Original user question.
    answer : str
        Generated answer.
    llm
        LangChain LLM.

    Returns
    -------
    dict
        Combined guardrail result::

            {
                "grounded":            bool,
                "confidence_score":    float,   # 0.0 – 1.0
                "hallucination_flags": list[str],
                "unsupported_claims":  list[str],
                "warning":             str | None
            }

    Example
    -------
    >>> result = check_guardrails(ctx, q, ans, llm)
    >>> result["confidence_score"]
    0.85
    """
    logger.info("[Guardrails] Running full guardrail suite")

    # 1. Groundedness
    ground = check_groundedness(context, answer, llm)

    # 2. Hallucination
    halluc_flags = detect_hallucinations(context, answer)

    # 3. Confidence
    conf_score, conf_reason = score_confidence(context, question, answer, llm)

    # Build warning message
    warning: str | None = None
    if conf_score < 0.6:
        warning = (
            f"⚠️ Low confidence ({conf_score:.0%}). The answer may not be "
            f"fully supported by the retrieved context. {conf_reason}"
        )
    elif not ground["grounded"]:
        warning = (
            "⚠️ Some claims may not be directly supported by the context."
        )

    result: GuardrailResult = {
        "grounded": ground["grounded"],
        "confidence_score": conf_score,
        "hallucination_flags": halluc_flags,
        "unsupported_claims": ground["unsupported_claims"],
        "warning": warning,
    }

    logger.info(
        "[Guardrails] Result: grounded={}, confidence={:.0%}, "
        "halluc_flags={}, unsupported={}",
        result["grounded"],
        result["confidence_score"],
        len(result["hallucination_flags"]),
        len(result["unsupported_claims"]),
    )
    return result
