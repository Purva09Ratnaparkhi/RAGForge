"""
Document Cleaner — 6-step cleaning pipeline.
=============================================

Transforms raw ``PageDict`` objects (from any extractor) into clean
LangChain ``Document`` objects with structured metadata.  Every step
is a pure function; the pipeline calls them in a fixed order.

Cleaning steps (applied IN ORDER)
---------------------------------
1. **Header/Footer Removal** — detects repeating top/bottom lines
   across pages and strips them.
2. **Hyphenation Fix** — rejoins words split across line breaks
   (e.g. "algo-\\nrithm" → "algorithm").
3. **Unicode Normalisation** — uses ``ftfy`` to fix mojibake, curly
   quotes, and other encoding artefacts.
4. **Reference Section Strip** — removes the References / Bibliography
   section at the end of the paper (noisy for retrieval).
5. **Table Injection** — appends Markdown tables back into the page
   text so they appear in the correct reading position.
6. **Metadata Attachment** — wraps each cleaned page in a LangChain
   ``Document`` with rich metadata.

Technology choices
------------------
* ``ftfy`` — purpose-built for fixing broken Unicode; more reliable
  than ad-hoc regex for academic PDF text.
* ``re`` — standard-library regex for header/footer and hyphen patterns.

Connects to
-----------
* ``src.extraction.pipeline`` — calls ``clean_pages`` after extraction.
* ``config.settings`` — reads ``HEADER_FOOTER_REPEAT_RATIO``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING

import ftfy
from langchain_core.documents import Document
from loguru import logger

from config.settings import HEADER_FOOTER_REPEAT_RATIO

if TYPE_CHECKING:
    from src.extraction.extractors import PageDict


# ── Step 1: Header / Footer Removal ───────────────────────────────────────


def _detect_repeating_lines(
    pages: list[PageDict], n_lines: int = 3
) -> set[str]:
    """Find lines that repeat across > HEADER_FOOTER_REPEAT_RATIO of pages.

    We check the first and last ``n_lines`` lines of each page.

    Parameters
    ----------
    pages : list[PageDict]
    n_lines : int
        Number of lines from top/bottom to consider.

    Returns
    -------
    set[str]
        Normalised line strings to remove.
    """
    if len(pages) < 3:
        return set()

    line_counter: Counter[str] = Counter()
    for p in pages:
        lines = p["text"].splitlines()
        candidates = lines[:n_lines] + lines[-n_lines:]
        # Deduplicate within a single page
        for line in set(candidates):
            stripped = line.strip()
            if stripped:
                line_counter[stripped] += 1

    threshold = len(pages) * HEADER_FOOTER_REPEAT_RATIO
    repeating = {
        line for line, count in line_counter.items() if count >= threshold
    }
    if repeating:
        logger.debug(
            "[Cleaner] Found {} repeating header/footer lines", len(repeating)
        )
    return repeating


def _remove_headers_footers(text: str, repeating: set[str]) -> str:
    """Strip lines identified as repeating headers/footers.

    Parameters
    ----------
    text : str
    repeating : set[str]

    Returns
    -------
    str
    """
    if not repeating:
        return text
    return "\n".join(
        line
        for line in text.splitlines()
        if line.strip() not in repeating
    )


# ── Step 2: Hyphenation Fix ──────────────────────────────────────────────


def _fix_hyphenation(text: str) -> str:
    """Rejoin words split across line breaks with a hyphen.

    Matches patterns like ``algo-\\nrithm`` and joins to ``algorithm``.
    Only lowercase-to-lowercase transitions are fixed to avoid breaking
    compound proper nouns.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


# ── Step 3: Unicode Normalisation ────────────────────────────────────────


def _normalise_unicode(text: str) -> str:
    """Fix broken Unicode using ftfy, then apply NFC normalisation.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    return text


# ── Step 4: Reference Section Strip ──────────────────────────────────────


_REF_PATTERNS = re.compile(
    r"^\s*(References|Bibliography|Works Cited|REFERENCES|BIBLIOGRAPHY)\s*$",
    re.MULTILINE,
)


def _strip_references(text: str) -> str:
    """Remove everything after a References / Bibliography heading.

    The reference section is usually noisy for retrieval (lots of names,
    years, and titles that confuse similarity search).

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    match = _REF_PATTERNS.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


# ── Step 5: Table Injection ──────────────────────────────────────────────


def _inject_tables(text: str, tables: list[str]) -> str:
    """Append Markdown tables to the end of the page text.

    Parameters
    ----------
    text : str
    tables : list[str]
        Markdown-formatted table strings.

    Returns
    -------
    str
    """
    if not tables:
        return text
    return text + "\n\n" + "\n\n".join(tables)


# ── Step 6: Metadata / Section Detection ─────────────────────────────────


def _detect_section_heading(text: str) -> str:
    """Detect the section heading from the first ~200 characters.

    A section heading is a short line (< 80 chars), with no trailing
    period, that is ALL CAPS or Title Case.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Detected heading or ``"unknown"``.

    Example
    -------
    >>> _detect_section_heading("INTRODUCTION\\nThis paper ...")
    'INTRODUCTION'
    """
    preview = text[:200]
    for line in preview.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) >= 80:
            continue
        if stripped.endswith("."):
            continue
        # Check ALL CAPS or Title Case
        if stripped.isupper() or stripped.istitle():
            # Filter out very short lines (page numbers, etc.)
            if len(stripped) > 2:
                return stripped
    return "unknown"


# ── Public API ────────────────────────────────────────────────────────────


def clean_pages(
    pages: list[PageDict], source_filename: str
) -> list[Document]:
    """Run the full 6-step cleaning pipeline.

    Parameters
    ----------
    pages : list[PageDict]
        Raw extracted page dicts from any extractor.
    source_filename : str
        Original PDF filename (stored in metadata).

    Returns
    -------
    list[Document]
        Clean LangChain Document objects, one per page.

    Example
    -------
    >>> docs = clean_pages(pages, "paper.pdf")
    >>> docs[0].metadata["source"]
    'paper.pdf'
    """
    logger.info(
        "[Cleaner] Cleaning {} pages from {}", len(pages), source_filename
    )

    # Pre-compute repeating lines across all pages (Step 1)
    repeating_lines = _detect_repeating_lines(pages)

    documents: list[Document] = []
    for page in pages:
        text = page["text"]

        # Step 1: Header/Footer Removal
        text = _remove_headers_footers(text, repeating_lines)

        # Step 2: Hyphenation Fix
        text = _fix_hyphenation(text)

        # Step 3: Unicode Normalisation
        text = _normalise_unicode(text)

        # Step 4: Reference Section Strip
        text = _strip_references(text)

        # Step 5: Table Injection
        text = _inject_tables(text, page["tables"])

        # Step 6: Metadata Attachment
        cleaned_text = text.strip()
        if not cleaned_text:
            logger.debug(
                "[Cleaner] Skipping empty page {} from {}",
                page["page"], source_filename,
            )
            continue

        doc = Document(
            page_content=cleaned_text,
            metadata={
                "source": source_filename,
                "page": page["page"],
                "total_pages": page["metadata"].get("total_pages", 0),
                "extractor": page["metadata"].get("extractor", "unknown"),
                "char_count": len(cleaned_text),
                "section": _detect_section_heading(cleaned_text),
                "has_tables": len(page["tables"]) > 0,
            },
        )
        documents.append(doc)

    logger.info(
        "[Cleaner] Produced {} clean Documents from {}",
        len(documents), source_filename,
    )
    return documents
