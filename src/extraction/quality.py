"""
Extraction Quality Gate & Fallback Chain
=========================================

Decides whether the output from an extractor is "good enough" by checking
character density and garble ratio (non-ASCII noise).  When quality fails,
``run_fallback_chain`` tries the next extractor in priority order.

Why two checks?
---------------
* **Low density** catches scanned pages that an extractor read as near-empty.
* **High garble** catches font-encoding failures where the extractor
  produces gibberish Unicode (common with some LaTeX-generated PDFs).

Connects to
-----------
* ``src.extraction.extractors`` — the four extractor classes.
* ``src.extraction.pipeline``  — orchestrates extraction + quality checks.
* ``config.settings``          — reads thresholds.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import (
    GARBLE_RATIO_THRESHOLD,
    MIN_CHARS_PER_PAGE,
    OCR_DPI_RETRY,
)

if TYPE_CHECKING:
    from src.extraction.extractors import PageDict


# ── Domain exception ──────────────────────────────────────────────────────


class ExtractorError(Exception):
    """Raised when all extraction attempts fail quality checks.

    Attributes
    ----------
    tried : list[str]
        Names of extractors that were attempted.
    quality_scores : dict[str, float]
        Mapping of extractor name → average chars/page score.
    pdf_path : str
        Path to the problematic PDF file.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        tried: list[str] | None = None,
        quality_scores: dict[str, float] | None = None,
    ) -> None:
        self.pdf_path = str(pdf_path)
        self.tried = tried or []
        self.quality_scores = quality_scores or {}
        super().__init__(
            f"All extractors failed for {self.pdf_path}. "
            f"Tried: {self.tried}, scores: {self.quality_scores}"
        )


# ── Quality check functions ───────────────────────────────────────────────


def _avg_chars_per_page(pages: list[PageDict]) -> float:
    """Compute mean character count across pages.

    Parameters
    ----------
    pages : list[PageDict]

    Returns
    -------
    float
        Mean character count (0.0 if no pages).
    """
    if not pages:
        return 0.0
    return sum(len(p["text"]) for p in pages) / len(pages)


def _garble_ratio(pages: list[PageDict]) -> float:
    """Compute fraction of non-ASCII characters across all pages.

    Parameters
    ----------
    pages : list[PageDict]

    Returns
    -------
    float
        Ratio in [0, 1].  Higher = more garbled text.
    """
    all_text = "".join(p["text"] for p in pages)
    if not all_text:
        return 0.0
    non_ascii = len(re.findall(r"[^\x00-\x7F]", all_text))
    return non_ascii / len(all_text)


def passes_quality(pages: list[PageDict]) -> tuple[bool, str]:
    """Check whether extracted pages meet quality thresholds.

    Parameters
    ----------
    pages : list[PageDict]
        Extracted page dicts from any extractor.

    Returns
    -------
    tuple[bool, str]
        ``(True, "ok")`` if quality is acceptable, otherwise
        ``(False, reason_string)``.

    Example
    -------
    >>> passes_quality([{"page": 0, "text": "x" * 200, "tables": [],
    ...                  "metadata": {"total_pages": 1, "extractor": "test"}}])
    (True, 'ok')
    """
    avg = _avg_chars_per_page(pages)
    if avg < MIN_CHARS_PER_PAGE:
        reason = f"low density (avg {avg:.0f} chars/page < {MIN_CHARS_PER_PAGE})"
        logger.warning("[Quality] FAIL — {}", reason)
        return False, reason

    garble = _garble_ratio(pages)
    if garble > GARBLE_RATIO_THRESHOLD:
        reason = f"high garble ({garble:.2%} non-ASCII > {GARBLE_RATIO_THRESHOLD:.0%})"
        logger.warning("[Quality] FAIL — {}", reason)
        return False, reason

    logger.debug(
        "[Quality] PASS — avg {:.0f} chars/page, garble {:.2%}",
        avg, garble,
    )
    return True, "ok"


# ── Fallback chain ────────────────────────────────────────────────────────


def run_fallback_chain(
    pdf_path: str | Path,
    pdf_type: str,
    extractor_map: dict,
    primary_extractor_name: str,
    primary_pages: list[PageDict],
) -> list[PageDict]:
    """Try fallback extractors until one passes quality.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF being processed.
    pdf_type : str
        ``"native"``, ``"scanned"``, or ``"mixed"``.
    extractor_map : dict
        Mapping of extractor name → extractor instance.
    primary_extractor_name : str
        Name of the extractor that already failed.
    primary_pages : list[PageDict]
        Pages from the primary extractor (kept for scoring).

    Returns
    -------
    list[PageDict]
        Pages from the first fallback extractor that passes quality.

    Raises
    ------
    ExtractorError
        If all fallback extractors also fail.
    """
    pdf_path = Path(pdf_path)
    tried = [primary_extractor_name]
    scores: dict[str, float] = {
        primary_extractor_name: _avg_chars_per_page(primary_pages)
    }

    if pdf_type == "scanned":
        # Only fallback is Tesseract at higher DPI
        logger.info(
            "[Fallback] Scanned PDF — retrying Tesseract at {} DPI",
            OCR_DPI_RETRY,
        )
        from src.extraction.extractors import TesseractExtractor

        retry_ext = TesseractExtractor(dpi=OCR_DPI_RETRY)
        retry_pages = retry_ext.extract(pdf_path)
        tried.append(f"tesseract_{OCR_DPI_RETRY}")
        scores[f"tesseract_{OCR_DPI_RETRY}"] = _avg_chars_per_page(retry_pages)

        ok, reason = passes_quality(retry_pages)
        if ok:
            logger.info("[Fallback] Tesseract retry PASSED")
            return retry_pages

        logger.error(
            "[Fallback] All extractors failed for {} — {}", pdf_path.name, reason
        )
        raise ExtractorError(pdf_path, tried=tried, quality_scores=scores)

    # Native / mixed: try pdfplumber → PyMuPDF
    fallback_order = ["pdfplumber", "pymupdf"]
    for name in fallback_order:
        if name == primary_extractor_name:
            continue  # skip the one that already failed
        if name not in extractor_map:
            continue

        logger.info("[Fallback] Trying {} for {}", name, pdf_path.name)
        try:
            pages = extractor_map[name].extract(pdf_path)
        except Exception as exc:
            logger.warning("[Fallback] {} failed with error: {}", name, exc)
            tried.append(name)
            scores[name] = 0.0
            continue

        tried.append(name)
        scores[name] = _avg_chars_per_page(pages)

        ok, reason = passes_quality(pages)
        if ok:
            logger.info("[Fallback] {} PASSED for {}", name, pdf_path.name)
            return pages
        logger.warning("[Fallback] {} FAILED — {}", name, reason)

    logger.error("[Fallback] All extractors failed for {}", pdf_path.name)
    raise ExtractorError(pdf_path, tried=tried, quality_scores=scores)
