"""
Ingest CLI — batch process PDFs from a directory.
===================================================

Usage::

    python scripts/ingest.py --input data/raw --verbose

Processes all PDF files in the input directory through the full
extraction → chunking → indexing pipeline and prints per-file stats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config.settings import RAW_PDF_DIR
from src.chunking.smart_chunker import chunk_documents
from src.extraction.pipeline import ExtractionPipeline
from src.vector_store.faiss_store import FAISSVectorStore


def main() -> None:
    """CLI entry point for batch PDF ingestion."""
    parser = argparse.ArgumentParser(
        description="RAGForge — Batch PDF Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(RAW_PDF_DIR),
        help="Directory containing PDF files (default: data/raw)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error("Input directory does not exist: {}", input_dir)
        sys.exit(1)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in {}", input_dir)
        sys.exit(0)

    logger.info("Found {} PDF files in {}", len(pdf_files), input_dir)

    # Initialise pipeline
    pipeline = ExtractionPipeline()
    all_chunks = []
    results_table = []

    for pdf_path in pdf_files:
        try:
            # Extract
            docs = pipeline.run(pdf_path)

            # Chunk
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)

            # Determine extractor used
            extractor = "unknown"
            if docs:
                extractor = docs[0].metadata.get("extractor", "unknown")

            results_table.append({
                "filename": pdf_path.name,
                "type": "detected",
                "extractor": extractor,
                "pages": len(docs),
                "chunks": len(chunks),
            })

        except Exception as exc:
            logger.error("Failed to process {}: {}", pdf_path.name, exc)
            results_table.append({
                "filename": pdf_path.name,
                "type": "ERROR",
                "extractor": "N/A",
                "pages": 0,
                "chunks": 0,
            })

    # Build FAISS index
    if all_chunks:
        logger.info("Building FAISS index from {} chunks", len(all_chunks))
        store = FAISSVectorStore()
        store.build(all_chunks)
        logger.info("FAISS index built and saved")
    else:
        logger.warning("No chunks produced — skipping index build")

    # Print results table
    print("\n" + "=" * 75)
    print(f"{'Filename':<30} {'Type':<10} {'Extractor':<15} {'Pages':>6} {'Chunks':>7}")
    print("=" * 75)
    for row in results_table:
        print(
            f"{row['filename']:<30} {row['type']:<10} {row['extractor']:<15} "
            f"{row['pages']:>6} {row['chunks']:>7}"
        )
    print("=" * 75)
    print(f"Total: {len(results_table)} files, {len(all_chunks)} chunks\n")


if __name__ == "__main__":
    main()
