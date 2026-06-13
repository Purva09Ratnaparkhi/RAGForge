"""
PDF Type Detector — Step 0 of the extraction pipeline.
======================================================

Analyses a PDF file to determine whether it is **native** (digitally-created
text), **scanned** (image-based requiring OCR), or **mixed** (some pages
have text, others are scans).  The classification drives which extractor
is selected by the pipeline.

Technology choice
-----------------
PyMuPDF (``fitz``) is used because it exposes low-level page metrics — text
block bounding boxes, character counts, column clustering — without needing
to render the PDF first.  This makes detection fast (< 1 s for most files).

Connects to
-----------
* ``src.extraction.pipeline`` — calls ``detect_pdf_type`` before choosing
  an extractor.
* ``config.settings`` — reads ``MIN_CHARS_PER_PAGE`` threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
from loguru import logger

from config.settings import MIN_CHARS_PER_PAGE

# ── Public types ───────────────────────────────────────────────────────────

PDFType = Literal["native", "scanned", "mixed"]


@dataclass
class PDFDetectionResult:
    """Structured result from ``detect_pdf_type``.

    Attributes
    ----------
    pdf_type : PDFType
        One of ``"native"``, ``"scanned"``, or ``"mixed"``.
    is_complex_layout : bool
        ``True`` when the PDF has multi-column or highly fragmented layouts
        and marker-pdf should be preferred.
    avg_chars_per_page : float
        Mean character count per page — useful for logging / diagnostics.
    total_pages : int
        Number of pages in the PDF.
    complex_page_ratio : float
        Fraction of pages flagged as layout-complex.
    """

    pdf_type: PDFType
    is_complex_layout: bool = False
    avg_chars_per_page: float = 0.0
    total_pages: int = 0
    complex_page_ratio: float = 0.0


# ── Internal helpers ───────────────────────────────────────────────────────


def _count_column_clusters(blocks: list[dict], tolerance: float = 40.0) -> int:
    """Group text-block left-edge x coordinates into clusters.

    Parameters
    ----------
    blocks : list[dict]
        Block dicts from ``page.get_text("dict")["blocks"]``.
    tolerance : float
        Maximum pixel distance between x coordinates to consider them
        part of the same column cluster.

    Returns
    -------
    int
        Number of distinct column clusters detected.

    Example
    -------
    >>> _count_column_clusters([{"bbox": (72, 0, 300, 50)},
    ...                         {"bbox": (310, 0, 550, 50)}])
    2
    """
    # Only consider text blocks (type 0), not image blocks (type 1).
    x_coords = sorted(
        b["bbox"][0] for b in blocks if b.get("type", -1) == 0
    )
    if not x_coords:
        return 0

    clusters: list[float] = [x_coords[0]]
    for x in x_coords[1:]:
        if abs(x - clusters[-1]) > tolerance:
            clusters.append(x)
    return len(clusters)


def _is_page_complex(page: fitz.Page, block_threshold: int = 20) -> bool:
    """Decide whether a single page has a complex layout.

    A page is complex if:
    * It has > 2 distinct column clusters, **or**
    * It has more than ``block_threshold`` text blocks.

    Parameters
    ----------
    page : fitz.Page
        A PyMuPDF page object.
    block_threshold : int
        Maximum number of text blocks before a page is flagged.

    Returns
    -------
    bool
    """
    page_dict = page.get_text("dict")
    blocks = page_dict.get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type", -1) == 0]

    if len(text_blocks) > block_threshold:
        return True
    if _count_column_clusters(blocks) > 2:
        return True
    return False


# ── Public API ─────────────────────────────────────────────────────────────


def detect_pdf_type(pdf_path: str | Path) -> PDFDetectionResult:
    """Classify a PDF as native, scanned, or mixed.

    Parameters
    ----------
    pdf_path : str | Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    PDFDetectionResult
        Structured detection result including layout complexity flag.

    Example
    -------
    >>> result = detect_pdf_type("paper.pdf")
    >>> result.pdf_type
    'native'
    """
    pdf_path = Path(pdf_path)
    logger.info("[PDFDetector] Analysing {}", pdf_path.name)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if total_pages == 0:
        logger.warning("[PDFDetector] {} has 0 pages", pdf_path.name)
        doc.close()
        return PDFDetectionResult(
            pdf_type="scanned", total_pages=0, avg_chars_per_page=0.0
        )

    page_char_counts: list[int] = []
    complex_pages = 0

    for page in doc:
        text = page.get_text("text")
        page_char_counts.append(len(text))
        if _is_page_complex(page):
            complex_pages += 1

    doc.close()

    avg_chars = sum(page_char_counts) / total_pages
    complex_ratio = complex_pages / total_pages

    # ── Classification rules ──────────────────────────────────────────
    # Rule 1: very few characters → scanned
    if avg_chars < MIN_CHARS_PER_PAGE:
        logger.info(
            "[PDFDetector] {} → scanned  (avg {:.0f} chars/page < {})",
            pdf_path.name, avg_chars, MIN_CHARS_PER_PAGE,
        )
        return PDFDetectionResult(
            pdf_type="scanned",
            avg_chars_per_page=avg_chars,
            total_pages=total_pages,
            complex_page_ratio=complex_ratio,
        )

    # Rule 2: some pages blank, some with text → mixed
    low_pages = sum(1 for c in page_char_counts if c < MIN_CHARS_PER_PAGE)
    good_pages = total_pages - low_pages
    if low_pages > 0 and good_pages > 0:
        logger.info(
            "[PDFDetector] {} → mixed  ({} low-text pages, {} good pages)",
            pdf_path.name, low_pages, good_pages,
        )
        return PDFDetectionResult(
            pdf_type="mixed",
            is_complex_layout=(complex_ratio > 0.3),
            avg_chars_per_page=avg_chars,
            total_pages=total_pages,
            complex_page_ratio=complex_ratio,
        )

    # Rule 3: native (complex flag set if > 30 % pages are complex)
    is_complex = complex_ratio > 0.3
    logger.info(
        "[PDFDetector] {} → native  (avg {:.0f} chars/page, complex={}, ratio={:.0%})",
        pdf_path.name, avg_chars, is_complex, complex_ratio,
    )
    return PDFDetectionResult(
        pdf_type="native",
        is_complex_layout=is_complex,
        avg_chars_per_page=avg_chars,
        total_pages=total_pages,
        complex_page_ratio=complex_ratio,
    )


def detect_batch(pdf_paths: list[str | Path]) -> dict[str, PDFDetectionResult]:
    """Classify multiple PDFs at once.

    Parameters
    ----------
    pdf_paths : list[str | Path]
        List of PDF file paths.

    Returns
    -------
    dict[str, PDFDetectionResult]
        Mapping from filename to its detection result.

    Example
    -------
    >>> results = detect_batch(["a.pdf", "b.pdf"])
    >>> results["a.pdf"].pdf_type
    'native'
    """
    results: dict[str, PDFDetectionResult] = {}
    for p in pdf_paths:
        path = Path(p)
        try:
            results[path.name] = detect_pdf_type(path)
        except Exception as exc:
            logger.error("[PDFDetector] Failed on {}: {}", path.name, exc)
            results[path.name] = PDFDetectionResult(
                pdf_type="scanned", total_pages=0, avg_chars_per_page=0.0
            )
    return results
