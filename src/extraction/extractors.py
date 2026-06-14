"""
PDF Text Extractors — six extraction back-ends.
=================================================

Each extractor converts a PDF file into a list of ``PageDict`` objects —
a standardised intermediate representation that the downstream cleaner
and quality gate understand.

Extractors (in priority order for native PDFs)
----------------------------------------------
1. **PyMuPDF4LLMExtractor** — fast, LLM-optimised Markdown output using
   ``pymupdf4llm``.  Built on top of PyMuPDF for speed.
2. **DoclingExtractor** — IBM's document understanding library with
   layout analysis, table extraction, and Markdown output.
3. **PyMuPDFExtractor** — fast and reliable plain-text baseline using
   ``fitz`` with block-level bounding box metadata.
4. **TesseractExtractor** — OCR-only; used when the PDF is scanned.

Legacy extractors (kept for backward compatibility)
---------------------------------------------------
5. **MarkerPDFExtractor** — uses ``marker-pdf`` (slow model loading).
6. **PDFPlumberExtractor** — uses ``pdfplumber`` for font-encoded PDFs.

Why multiple extractors?
------------------------
Academic PDFs vary wildly in encoding, layout, and scan quality.  A single
extractor cannot handle every edge case.  The quality gate
(``src.extraction.quality``) decides whether an extractor's output is good
enough, and if not, the pipeline falls through to the next one.

Data contract — ``PageDict``
-----------------------------
Every extractor returns ``list[PageDict]`` where each dict has the keys:
``page``, ``text``, ``tables``, ``metadata``.  See type alias below.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict

from loguru import logger

from config.settings import OCR_DPI

# ── Shared data types ──────────────────────────────────────────────────────


class PageMetadata(TypedDict, total=False):
    """Metadata attached to each extracted page."""

    total_pages: int
    extractor: str
    format: str
    blocks: list[dict[str, Any]]
    width: float
    height: float


class PageDict(TypedDict):
    """Standardised per-page extraction result.

    Keys
    ----
    page : int
        Zero-based page index.
    text : str
        Extracted plain text (or Markdown for marker-pdf).
    tables : list[str]
        Markdown-formatted table strings found on this page.
    metadata : PageMetadata
        Extractor-specific metadata.
    """

    page: int
    text: str
    tables: list[str]
    metadata: PageMetadata


# ── Helper: table → Markdown ──────────────────────────────────────────────


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber-style table (list of rows) to Markdown.

    Parameters
    ----------
    table : list[list[str | None]]
        Each inner list is a row; first row is treated as the header.

    Returns
    -------
    str
        Markdown table string.

    Example
    -------
    >>> _table_to_markdown([["A", "B"], ["1", "2"]])
    '| A | B |\\n| --- | --- |\\n| 1 | 2 |'
    """
    if not table or not table[0]:
        return ""

    def _cell(v: str | None) -> str:
        return str(v).replace("|", "\\|").strip() if v else ""

    header = table[0]
    md_lines = [
        "| " + " | ".join(_cell(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in table[1:]:
        # Pad row if it has fewer cells than the header
        padded = list(row) + [None] * (len(header) - len(row))
        md_lines.append("| " + " | ".join(_cell(c) for c in padded[:len(header)]) + " |")
    return "\n".join(md_lines)


# ── Extractor 1: pymupdf4llm ─────────────────────────────────────────────


class PyMuPDF4LLMExtractor:
    """Primary extractor using ``pymupdf4llm`` for LLM-optimised Markdown.

    This is fast because it leverages PyMuPDF under the hood — no heavy
    neural-network models to load.  Produces clean Markdown with headings,
    lists, and table formatting preserved.

    Example
    -------
    >>> ext = PyMuPDF4LLMExtractor()
    >>> pages = ext.extract("paper.pdf")
    >>> pages[0]["text"][:50]
    '# Introduction\\nRecent advances in ...'
    """

    def extract(self, pdf_path: str | Path) -> list[PageDict]:
        """Extract text from a PDF using pymupdf4llm.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        list[PageDict]
            One entry per page.
        """
        import pymupdf4llm

        pdf_path = Path(pdf_path)
        logger.info("[PyMuPDF4LLM] Extracting {}", pdf_path.name)

        # page_chunks=True returns a list of dicts, one per page,
        # each with keys: "metadata", "text", "tables", "images"
        md_chunks = pymupdf4llm.to_markdown(
            str(pdf_path), page_chunks=True
        )

        total = len(md_chunks)
        pages: list[PageDict] = []
        for i, chunk in enumerate(md_chunks):
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            text = text.strip()
            if not text:
                # keep empty pages for page-index alignment
                pass
            pages.append(
                PageDict(
                    page=i,
                    text=text,
                    tables=[],  # tables are embedded in the markdown
                    metadata=PageMetadata(
                        total_pages=total,
                        extractor="pymupdf4llm",
                        format="markdown",
                    ),
                )
            )

        logger.info(
            "[PyMuPDF4LLM] {} → {} pages extracted", pdf_path.name, len(pages)
        )
        with open("extracted_pages.txt", "w", encoding="utf-8") as f:
            for page in pages:
                f.write(f"Page {page['page']}\n")
                f.write(page['text'])
                f.write("\n\n")
        return pages


# ── Extractor 2: Docling ─────────────────────────────────────────────────


class DoclingExtractor:
    """Extractor using IBM's ``docling`` library.

    Docling provides layout analysis, table extraction, and outputs
    clean Markdown.  Heavier than pymupdf4llm but more capable on
    complex academic layouts.

    Example
    -------
    >>> ext = DoclingExtractor()
    >>> pages = ext.extract("paper.pdf")
    """

    def extract(self, pdf_path: str | Path) -> list[PageDict]:
        """Extract text from a PDF using Docling.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        list[PageDict]
            One entry per page.
        """
        from docling.document_converter import DocumentConverter

        pdf_path = Path(pdf_path)
        logger.info("[Docling] Extracting {}", pdf_path.name)

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown_text = result.document.export_to_markdown()

        # Docling does not natively split output per page.
        # We use page-break markers ("---" / form-feed) when present,
        # otherwise treat the whole document as a single logical page.
        pages = self._split_to_pages(markdown_text, pdf_path.name)

        logger.info(
            "[Docling] {} → {} pages extracted", pdf_path.name, len(pages)
        )
        return pages

    @staticmethod
    def _split_to_pages(
        markdown: str, filename: str
    ) -> list[PageDict]:
        """Split Docling Markdown output into per-page dicts."""
        # Try splitting on horizontal rules or form-feed characters
        page_pattern = re.compile(r"\n-{3,}\n|\f")
        raw_pages = page_pattern.split(markdown)

        # If no splits found, treat entire output as page 0
        if len(raw_pages) <= 1:
            raw_pages = [markdown]

        total = len(raw_pages)
        pages: list[PageDict] = []
        for i, text in enumerate(raw_pages):
            text = text.strip()
            pages.append(
                PageDict(
                    page=i,
                    text=text,
                    tables=[],  # tables are embedded in the markdown
                    metadata=PageMetadata(
                        total_pages=total,
                        extractor="docling",
                        format="markdown",
                    ),
                )
            )

        return pages


# ── Extractor 3 (legacy): marker-pdf ─────────────────────────────────────


class MarkerPDFExtractor:
    """Highest-quality extractor using the ``marker-pdf`` library.

    Marker runs layout detection, OCR when needed, and table structure
    recognition.  It outputs clean Markdown which preserves heading
    hierarchy, lists, and table formatting.

    The model ensemble is loaded once in ``__init__`` and reused across
    calls (these are large neural-network models).

    Parameters
    ----------
    (none — configuration comes from settings)

    Example
    -------
    >>> ext = MarkerPDFExtractor()
    >>> pages = ext.extract("paper.pdf")
    >>> pages[0]["text"][:50]
    '# Introduction\\nRecent advances in ...'
    """

    def __init__(self) -> None:
        self._models = None  # lazy-loaded

    def _ensure_models(self) -> None:
        """Lazy-load marker models on first use."""
        if self._models is not None:
            return
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            logger.info("[MarkerPDF] Loading models (first call)…")
            self._model_dict = create_model_dict()
            self._converter = PdfConverter(artifact_dict=self._model_dict)
            self._models = True
            logger.info("[MarkerPDF] Models loaded successfully")
        except ImportError:
            # Fallback to older marker-pdf API
            try:
                from marker.convert import convert_single_pdf  # noqa: F401
                from marker.models import load_all_models
                logger.info("[MarkerPDF] Loading models (legacy API)…")
                self._legacy_models = load_all_models()
                self._models = "legacy"
                logger.info("[MarkerPDF] Legacy models loaded successfully")
            except ImportError as e:
                logger.error("[MarkerPDF] marker-pdf not installed: {}", e)
                raise

    def extract(self, pdf_path: str | Path) -> list[PageDict]:
        """Extract text from a PDF using marker-pdf.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        list[PageDict]
            One entry per detected page.
        """
        self._ensure_models()
        pdf_path = Path(pdf_path)
        logger.info("[MarkerPDF] Extracting {}", pdf_path.name)

        if self._models == "legacy":
            return self._extract_legacy(pdf_path)

        # New API (marker >= 1.0)
        rendered = self._converter(str(pdf_path))
        markdown_text = rendered.markdown if hasattr(rendered, 'markdown') else str(rendered)
        metadata_out = getattr(rendered, 'metadata', {})

        return self._split_markdown_to_pages(
            markdown_text, pdf_path.name, metadata_out
        )

    def _extract_legacy(self, pdf_path: Path) -> list[PageDict]:
        """Extract using the older marker-pdf API."""
        from marker.convert import convert_single_pdf
        markdown_text, images, meta = convert_single_pdf(
            str(pdf_path), self._legacy_models
        )
        return self._split_markdown_to_pages(markdown_text, pdf_path.name, meta)

    @staticmethod
    def _split_markdown_to_pages(
        markdown: str, filename: str, meta: Any
    ) -> list[PageDict]:
        """Split marker-pdf Markdown output into per-page dicts.

        Marker inserts horizontal rules (``---``) or ``{page_N}`` tokens
        between pages.  We split on those patterns.
        """
        # Split on explicit page markers or horizontal rules
        page_pattern = re.compile(r"\n-{3,}\n|\{page_\d+\}")
        raw_pages = page_pattern.split(markdown)

        total = len(raw_pages)
        pages: list[PageDict] = []
        for i, text in enumerate(raw_pages):
            text = text.strip()
            if not text:
                continue
            pages.append(
                PageDict(
                    page=i,
                    text=text,
                    tables=[],  # marker embeds tables directly in markdown
                    metadata=PageMetadata(
                        total_pages=total,
                        extractor="marker-pdf",
                        format="markdown",
                    ),
                )
            )

        logger.info(
            "[MarkerPDF] {} → {} pages extracted", filename, len(pages)
        )
        return pages


# ── Extractor 4 (legacy): pdfplumber ─────────────────────────────────────


class PDFPlumberExtractor:
    """Extracts text and tables using ``pdfplumber``.

    Good for PDFs with standard font encodings and embedded tables.
    Tables are extracted as structured data and converted to Markdown
    strings for downstream consumption.

    Example
    -------
    >>> ext = PDFPlumberExtractor()
    >>> pages = ext.extract("paper.pdf")
    >>> len(pages[0]["tables"])
    1
    """

    def extract(self, pdf_path: str | Path) -> list[PageDict]:
        """Extract text and tables from a PDF.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        list[PageDict]
        """
        import pdfplumber

        pdf_path = Path(pdf_path)
        logger.info("[PDFPlumber] Extracting {}", pdf_path.name)

        pages: list[PageDict] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""

                # Extract tables
                raw_tables = page.extract_tables() or []
                md_tables = [
                    _table_to_markdown(t) for t in raw_tables if t
                ]

                pages.append(
                    PageDict(
                        page=i,
                        text=text,
                        tables=md_tables,
                        metadata=PageMetadata(
                            total_pages=total,
                            extractor="pdfplumber",
                        ),
                    )
                )

        logger.info(
            "[PDFPlumber] {} → {} pages, {} tables total",
            pdf_path.name,
            len(pages),
            sum(len(p["tables"]) for p in pages),
        )
        return pages


# ── Extractor 3: PyMuPDF ─────────────────────────────────────────────────


class PyMuPDFExtractor:
    """Fast, reliable baseline extractor using PyMuPDF (``fitz``).

    Extracts plain text and stores block-level bounding box information
    in metadata.  The bounding boxes are used by the header/footer removal
    step in the cleaner.

    Example
    -------
    >>> ext = PyMuPDFExtractor()
    >>> pages = ext.extract("paper.pdf")
    >>> "blocks" in pages[0]["metadata"]
    True
    """

    def extract(self, pdf_path: str | Path) -> list[PageDict]:
        """Extract text and block metadata from a PDF.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        list[PageDict]
        """
        import fitz

        pdf_path = Path(pdf_path)
        logger.info("[PyMuPDF] Extracting {}", pdf_path.name)

        pages: list[PageDict] = []
        doc = fitz.open(str(pdf_path))
        total = len(doc)

        for i, page in enumerate(doc):
            text = page.get_text("text")
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])
            rect = page.rect

            pages.append(
                PageDict(
                    page=i,
                    text=text,
                    tables=[],
                    metadata=PageMetadata(
                        total_pages=total,
                        extractor="pymupdf",
                        blocks=blocks,
                        width=rect.width,
                        height=rect.height,
                    ),
                )
            )

        doc.close()
        logger.info("[PyMuPDF] {} → {} pages extracted", pdf_path.name, len(pages))
        return pages


# ── Extractor 4: Tesseract OCR ───────────────────────────────────────────


class TesseractExtractor:
    """OCR extractor for scanned PDFs using Tesseract.

    Converts each PDF page to a PIL image via ``pdf2image``, then runs
    Tesseract OCR.  Accepts a ``dpi`` parameter (default 300, retry at 400
    for better quality on difficult scans).

    Parameters
    ----------
    dpi : int
        Resolution for rendering PDF pages to images.

    Example
    -------
    >>> ext = TesseractExtractor(dpi=300)
    >>> pages = ext.extract("scanned_paper.pdf")
    """

    def __init__(self, dpi: int = OCR_DPI) -> None:
        self.dpi = dpi

    def extract(self, pdf_path: str | Path) -> list[PageDict]:
        """Extract text from a scanned PDF via OCR.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.

        Returns
        -------
        list[PageDict]
        """
        from pdf2image import convert_from_path
        import pytesseract

        pdf_path = Path(pdf_path)
        logger.info(
            "[Tesseract] Extracting {} at {} DPI", pdf_path.name, self.dpi
        )

        images = convert_from_path(
            str(pdf_path), dpi=self.dpi, thread_count=4
        )
        total = len(images)
        pages: list[PageDict] = []

        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang="eng")
            pages.append(
                PageDict(
                    page=i,
                    text=text,
                    tables=[],
                    metadata=PageMetadata(
                        total_pages=total,
                        extractor=f"tesseract_{self.dpi}",
                    ),
                )
            )

        logger.info(
            "[Tesseract] {} → {} pages extracted (DPI={})",
            pdf_path.name, len(pages), self.dpi,
        )
        return pages
