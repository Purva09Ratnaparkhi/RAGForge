"""
RAGAS Evaluator — Component 9: Automated RAG pipeline evaluation.
==================================================================

Uses the RAGAS framework to score the RAG pipeline across four
complementary metrics:

* **Faithfulness** — does the answer only use information from the
  retrieved context?  (Penalises hallucination.)
* **Answer Relevancy** — does the answer actually address the question?
  (Penalises off-topic responses.)
* **Context Precision** — were the retrieved chunks actually relevant?
  (Penalises noisy retrieval.)
* **Context Recall** — was all relevant information retrieved?
  (Penalises incomplete retrieval.)

Together these four metrics give a holistic picture of the pipeline's
quality and highlight which component needs improvement.

Connects to
-----------
* ``scripts/evaluate.py`` — CLI entry point for benchmarking.
* ``src.generation.llm_factory`` — provides LLM for test-set generation
  and RAGAS internal LLM calls.
* ``config.settings`` — ``RAGAS_SAMPLE_SIZE``.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from config.settings import LOG_DIR, RAGAS_SAMPLE_SIZE


def evaluate(dataset: list[dict], llm=None) -> dict[str, float]:
    """Score the RAG pipeline using RAGAS metrics.

    Parameters
    ----------
    dataset : list[dict]
        Evaluation samples, each with keys::

            {
                "question":     str,
                "answer":       str,
                "contexts":     list[str],
                "ground_truth": str
            }

    llm : optional
        LangChain LLM for RAGAS internals.  If ``None``, RAGAS uses
        its default.

    Returns
    -------
    dict[str, float]
        Mapping of metric name → score (0.0–1.0).

    Example
    -------
    >>> scores = evaluate(test_data)
    >>> scores["faithfulness"]
    0.87
    """
    logger.info("[RAGAS] Evaluating {} samples", len(dataset))

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # Convert to HuggingFace Dataset format
        hf_dataset = Dataset.from_dict({
            "question": [d["question"] for d in dataset],
            "answer": [d["answer"] for d in dataset],
            "contexts": [d["contexts"] for d in dataset],
            "ground_truth": [d["ground_truth"] for d in dataset],
        })

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        eval_kwargs = {"dataset": hf_dataset, "metrics": metrics}
        if llm is not None:
            eval_kwargs["llm"] = llm

        result = ragas_evaluate(**eval_kwargs)

        scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
        }

        # Compute overall average
        scores["overall"] = sum(scores.values()) / len(scores)

        logger.info("[RAGAS] Scores: {}", scores)
        return scores

    except ImportError as exc:
        logger.error("[RAGAS] Required packages not installed: {}", exc)
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "overall": 0.0,
            "error": str(exc),
        }
    except Exception as exc:
        logger.error("[RAGAS] Evaluation failed: {}", exc)
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "overall": 0.0,
            "error": str(exc),
        }


def generate_test_set(
    documents: list[Document],
    llm,
    n: int = RAGAS_SAMPLE_SIZE,
) -> list[dict]:
    """Generate Q&A pairs from documents for evaluation.

    Uses the LLM to create questions and ground-truth answers from
    document chunks.

    Parameters
    ----------
    documents : list[Document]
        Processed document chunks.
    llm
        LangChain LLM for generating Q&A pairs.
    n : int
        Number of samples to generate.

    Returns
    -------
    list[dict]
        Evaluation samples ready for ``evaluate()``.

    Example
    -------
    >>> test_data = generate_test_set(docs, llm, n=20)
    >>> len(test_data)
    20
    """
    from langchain_core.prompts import ChatPromptTemplate

    logger.info("[RAGAS] Generating test set ({} samples from {} docs)", n, len(documents))

    prompt = ChatPromptTemplate.from_template(
        "Based on the following text, generate a specific factual question "
        "and its answer.\n\n"
        "Text:\n{text}\n\n"
        "Respond with JSON:\n"
        '{{"question": "...", "answer": "..."}}'
    )

    chain = prompt | llm
    test_set: list[dict] = []

    # Sample documents (cycle if n > len(documents))
    import itertools
    sampled = list(itertools.islice(itertools.cycle(documents), n))

    for i, doc in enumerate(sampled):
        try:
            result = chain.invoke({"text": doc.page_content[:1500]})
            text = result.content if hasattr(result, "content") else str(result)

            # Parse JSON from response
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                test_set.append({
                    "question": parsed.get("question", ""),
                    "answer": parsed.get("answer", ""),
                    "contexts": [doc.page_content],
                    "ground_truth": parsed.get("answer", ""),
                })
        except Exception as exc:
            logger.warning("[RAGAS] Failed to generate sample {}: {}", i, exc)
            continue

    logger.info("[RAGAS] Generated {} test samples", len(test_set))
    return test_set


def save_results(scores: dict, output_path: str | Path | None = None) -> Path:
    """Save evaluation results to a JSON file.

    Parameters
    ----------
    scores : dict
        Metric scores from ``evaluate()``.
    output_path : str | Path | None
        Output file path.  Defaults to ``logs/ragas_results.json``.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if output_path is None:
        output_path = LOG_DIR / "ragas_results.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    logger.info("[RAGAS] Results saved to {}", output_path)
    return output_path


def print_results_table(scores: dict) -> None:
    """Print a formatted table of RAGAS scores.

    Parameters
    ----------
    scores : dict
        Metric scores.
    """
    print("\n" + "─" * 40)
    print(f"{'Metric':<25} {'Score':>8}")
    print("─" * 40)

    metric_order = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    for metric in metric_order:
        if metric in scores:
            print(f"{metric.replace('_', ' ').title():<25} {scores[metric]:>8.2f}")

    print("─" * 40)
    if "overall" in scores:
        print(f"{'Overall':<25} {scores['overall']:>8.2f}")
    print("─" * 40 + "\n")
