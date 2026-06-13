"""
Extraction Pipeline — orchestrates the full PDF → Document flow.
================================================================

The ``ExtractionPipeline`` class is the single entry point for turning
raw PDF files into clean LangChain ``Document`` objects.  It coordinates:

1. **Detection** — classifies the PDF type (native / scanned / mixed).
2. **Primary extraction** — picks the best extractor for the detected type.
3. **Quality gate** — checks character density and garble ratio.
4. **Fallback chain** — tries alternative extractors on failure.
5. **Cleaning** — runs the 6-step cleaning pipeline.

A class is used here (rather than bare functions) because it holds
four heavy extractor instances that should be loaded once and reused
across multiple PDFs during batch ingestion.

Connects to
-----------
* ``src.extraction.pdf_detector`` — PDF type classification.
* ``src.extraction.extractors``   — the four extractor back-ends.
* ``src.extraction.quality``      — quality gate + fallback.
* ``src.extraction.cleaner``      — 6-step cleaning.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from config.settings import PROCESSED_DIR
from src.extraction.cleaner import clean_pages
from src.extraction.extractors import (
    MarkerPDFExtractor,
    PDFPlumberExtractor,
    PyMuPDFExtractor,
    TesseractExtractor,
)
from src.extraction.pdf_detector import detect_pdf_type
from src.extraction.quality import passes_quality, run_fallback_chain


class ExtractionPipeline:
    """End-to-end PDF extraction pipeline.

    Loads all four extractor instances once and reuses them across calls.
    Use ``run`` for a single PDF or ``run_batch`` for a directory of PDFs.

    Example
    -------
    >>> pipeline = ExtractionPipeline()
    >>> docs = pipeline.run("data/raw/paper.pdf")
    >>> len(docs)
    12
    """

    def __init__(self) -> None:
        logger.info("[Pipeline] Initialising extraction pipeline")
        self.extractor_map = {
            "marker": MarkerPDFExtractor(),
            "pdfplumber": PDFPlumberExtractor(),
            "pymupdf": PyMuPDFExtractor(),
            "tesseract": TesseractExtractor(),
        }

    # ── Single PDF ────────────────────────────────────────────────────

    def run(self, pdf_path: str | Path) -> list[Document]:
        """Extract, quality-check, and clean a single PDF.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        list[Document]
            Clean LangChain Document objects (one per non-empty page).

        Raises
        ------
        ExtractorError
            If all extractors fail quality checks.
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        logger.info("[Pipeline] ── Processing {} ──", filename)

        # 1. Detect PDF type
        detection = detect_pdf_type(pdf_path)
        logger.info(
            "[Pipeline] {} → type={}, complex={}",
            filename, detection.pdf_type, detection.is_complex_layout,
        )

        # 2. Pick primary extractor
        primary_name = self._select_extractor(
            detection.pdf_type, detection.is_complex_layout
        )
        logger.info("[Pipeline] Primary extractor: {}", primary_name)

        # 3. Run primary extractor
        extractor = self.extractor_map[primary_name]
        if primary_name == "tesseract":
            pages = extractor.extract(pdf_path)
        else:
            try:
                pages = extractor.extract(pdf_path)
            except Exception as exc:
                logger.warning(
                    "[Pipeline] {} failed on {}: {}",
                    primary_name, filename, exc,
                )
                pages = []

        # 4. Quality check
        if pages:
            ok, reason = passes_quality(pages)
        else:
            ok, reason = False, "empty extraction"

        if not ok:
            logger.warning(
                "[Pipeline] {} quality fail ({}). Running fallback chain…",
                primary_name, reason,
            )
            pages = run_fallback_chain(
                pdf_path=pdf_path,
                pdf_type=detection.pdf_type,
                extractor_map=self.extractor_map,
                primary_extractor_name=primary_name,
                primary_pages=pages,
            )

        # 5. Clean
        documents = clean_pages(pages, source_filename=filename)

        # 6. Persist processed documents
        self._save_processed(filename, documents)

        logger.info(
            "[Pipeline] ✓ {} → {} documents (extractor: {})",
            filename, len(documents),
            pages[0]["metadata"].get("extractor", "unknown") if pages else "none",
        )
        return documents

    # ── Batch processing ──────────────────────────────────────────────

    def run_batch(self, pdf_dir: str | Path) -> dict[str, list[Document]]:
        """Process all PDFs in a directory.

        Parameters
        ----------
        pdf_dir : str | Path
            Directory containing PDF files.

        Returns
        -------
        dict[str, list[Document]]
            Mapping from filename to its list of Documents.
        """
        pdf_dir = Path(pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        logger.info(
            "[Pipeline] Batch processing {} PDFs from {}",
            len(pdf_files), pdf_dir,
        )

        results: dict[str, list[Document]] = {}
        for pdf_path in pdf_files:
            try:
                results[pdf_path.name] = self.run(pdf_path)
            except Exception as exc:
                logger.error(
                    "[Pipeline] FAILED on {}: {}", pdf_path.name, exc
                )
                results[pdf_path.name] = []

        total_docs = sum(len(v) for v in results.values())
        logger.info(
            "[Pipeline] Batch complete: {} PDFs → {} total documents",
            len(results), total_docs,
        )
        return results

    # ── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _select_extractor(pdf_type: str, is_complex: bool) -> str:
        """Choose the primary extractor based on PDF type and complexity.

        Parameters
        ----------
        pdf_type : str
        is_complex : bool

        Returns
        -------
        str
            Extractor key for ``self.extractor_map``.
        """
        if pdf_type == "scanned":
            return "tesseract"
        # For native/mixed, marker-pdf is always primary (best quality)
        return "marker"

    @staticmethod
    def _save_processed(filename: str, documents: list[Document]) -> None:
        """Pickle processed documents for later use.

        Parameters
        ----------
        filename : str
        documents : list[Document]
        """
        out_path = PROCESSED_DIR / f"{Path(filename).stem}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(documents, f)
        logger.debug("[Pipeline] Saved processed docs to {}", out_path)
