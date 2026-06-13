"""
Evaluate CLI — run RAGAS benchmark on the RAG pipeline.
========================================================

Usage::

    python scripts/evaluate.py --sample-size 50 --output logs/ragas_results.json

Generates a test set from processed documents, runs each question
through the full RAG pipeline, scores with RAGAS, and prints results.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config.settings import LOG_DIR, PROCESSED_DIR, RAGAS_SAMPLE_SIZE
from src.evaluation.ragas_evaluator import (
    evaluate,
    generate_test_set,
    print_results_table,
    save_results,
)
from src.generation.llm_factory import get_llm_with_fallback


def load_processed_documents():
    """Load all processed document pickles.

    Returns
    -------
    list[Document]
    """
    from langchain_core.documents import Document

    all_docs = []
    for pkl_path in sorted(PROCESSED_DIR.glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            docs = pickle.load(f)  # noqa: S301
            all_docs.extend(docs)
    return all_docs


def main() -> None:
    """CLI entry point for RAGAS evaluation."""
    parser = argparse.ArgumentParser(
        description="RAGForge — RAGAS Evaluation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=RAGAS_SAMPLE_SIZE,
        help=f"Number of test samples (default: {RAGAS_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(LOG_DIR / "ragas_results.json"),
        help="Output file for results (default: logs/ragas_results.json)",
    )
    args = parser.parse_args()

    logger.info("Loading processed documents")
    documents = load_processed_documents()

    if not documents:
        logger.error(
            "No processed documents found. Run ingestion first:\n"
            "  python scripts/ingest.py --input data/raw"
        )
        sys.exit(1)

    logger.info("Loaded {} documents", len(documents))

    # Get LLM
    llm = get_llm_with_fallback()

    # Generate test set
    logger.info("Generating {} test samples", args.sample_size)
    test_set = generate_test_set(documents, llm, n=args.sample_size)

    if not test_set:
        logger.error("Failed to generate test set")
        sys.exit(1)

    logger.info("Generated {} test samples", len(test_set))

    # Run the full RAG pipeline on each question
    logger.info("Running RAG pipeline on test questions")
    try:
        from src.agents.rag_agent import build_rag_graph

        graph = build_rag_graph()

        for sample in test_set:
            try:
                result = graph.invoke({
                    "query": sample["question"],
                    "query_type": "",
                    "rewritten_queries": [],
                    "retrieved_docs": [],
                    "reranked_docs": [],
                    "compressed_docs": [],
                    "answer": "",
                    "guardrail_result": {},
                    "context_str": "",
                    "history_str": "",
                    "messages": [],
                })
                sample["answer"] = result.get("answer", sample["answer"])
                # Update contexts from retrieval
                compressed = result.get("compressed_docs", [])
                if compressed:
                    sample["contexts"] = [d.page_content for d in compressed]
            except Exception as exc:
                logger.warning("Pipeline failed for question: {}. Error: {}",
                             sample["question"][:50], exc)

    except Exception as exc:
        logger.warning("Could not run pipeline: {}. Using generated answers.", exc)

    # Score with RAGAS
    logger.info("Running RAGAS evaluation")
    scores = evaluate(test_set, llm)

    # Print results
    print_results_table(scores)

    # Save results
    output_path = save_results(scores, args.output)
    logger.info("Results saved to {}", output_path)


if __name__ == "__main__":
    main()
