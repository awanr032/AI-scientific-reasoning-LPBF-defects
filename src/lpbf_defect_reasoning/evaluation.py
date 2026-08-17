"""Benchmark evaluation of retrieval quality and answer latency."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import pandas as pd

from .config import MIN_EVIDENCE_SUPPORT

# A small set of domain-specific questions with expected defect/parameter
# labels, used to sanity-check retrieval quality after any change to the
# graph, index, or pinned models.
BENCHMARK: list[dict[str, Any]] = [
    {
        "question": "Why does high laser power lead to keyhole porosity in LPBF?",
        "expected_defects": ["porosity", "keyhole"],
        "expected_parameters": ["laser power"],
    },
    {
        "question": "How does high scan speed contribute to lack of fusion defects in LPBF?",
        "expected_defects": ["lack_of_fusion"],
        "expected_parameters": ["scan speed"],
    },
    {
        "question": "Which process parameters influence porosity formation in LPBF?",
        "expected_defects": ["porosity"],
        "expected_parameters": ["laser power", "scan speed", "energy density", "hatch spacing"],
    },
    {
        "question": "How does hatch spacing affect lack of fusion defects?",
        "expected_defects": ["lack_of_fusion"],
        "expected_parameters": ["hatch spacing"],
    },
    {
        "question": "Why does insufficient laser power lead to lack of fusion in LPBF?",
        "expected_defects": ["lack_of_fusion"],
        "expected_parameters": ["laser power"],
    },
    {
        "question": "What role does energy density play in LPBF defect formation?",
        "expected_defects": ["porosity", "lack_of_fusion", "keyhole"],
        "expected_parameters": ["energy density"],
    },
    {
        "question": "How can layer thickness contribute to lack of fusion in LPBF?",
        "expected_defects": ["lack_of_fusion"],
        "expected_parameters": ["layer thickness"],
    },
    {
        "question": "What causes residual stress formation during LPBF?",
        "expected_defects": ["residual_stress"],
        "expected_parameters": ["scan strategy", "layer thickness"],
    },
    {
        "question": "Why does unstable melt pool behaviour lead to porosity?",
        "expected_defects": ["porosity"],
        "expected_parameters": [],
    },
    {
        "question": "How do process parameters contribute to cracking in LPBF?",
        "expected_defects": ["crack"],
        "expected_parameters": ["laser power", "energy density", "hatch spacing"],
    },
]


def evaluate_graph_rag(
    model_function: Callable[[str, int], tuple[str, list[dict[str, Any]]]],
    benchmark: list[dict[str, Any]] = BENCHMARK,
    k: int = 5,
) -> pd.DataFrame:
    """Score a QA function against the benchmark and return a results DataFrame.

    Args:
        model_function: Callable ``(question, k) -> (answer, evidence_chunks)``,
            e.g. ``GraphRagPipeline.qa``.
        benchmark: List of benchmark items (see ``BENCHMARK`` for the shape).
        k: Number of evidence chunks to retrieve per question.

    Returns:
        A DataFrame with one row per benchmark question, including
        retrieval_accuracy, defect_label_precision/recall,
        parameter_accuracy, latency (seconds), and the generated answer.
    """
    rows = []

    for item in benchmark:
        question = item["question"]

        start = time.time()
        response, evidence = model_function(question, k)
        latency = time.time() - start

        retrieved_defects_all: list[str] = []
        retrieved_parameters_all: list[str] = []
        for ev in evidence:
            retrieved_defects_all += ev.get("possible_defects", [])
            retrieved_parameters_all += ev.get("possible_parameters", [])

        defect_counts = Counter(retrieved_defects_all)
        parameter_counts = Counter(retrieved_parameters_all)

        retrieved_defects = [d for d, c in defect_counts.items() if c >= MIN_EVIDENCE_SUPPORT]
        retrieved_parameters = [p for p, c in parameter_counts.items() if c >= MIN_EVIDENCE_SUPPORT]

        expected_defects = item.get("expected_defects", [])
        expected_parameters = item.get("expected_parameters", [])

        correct_defects = set(expected_defects).intersection(retrieved_defects)
        correct_params = set(expected_parameters).intersection(retrieved_parameters)

        retrieval_accuracy = len(correct_defects) / max(len(expected_defects), 1)
        defect_label_precision = len(correct_defects) / max(len(retrieved_defects), 1)
        defect_label_recall = len(correct_defects) / max(len(expected_defects), 1)

        parameter_accuracy = (
            None if not expected_parameters else len(correct_params) / len(expected_parameters)
        )

        rows.append(
            {
                "question": question,
                "retrieved_defects": retrieved_defects,
                "retrieved_defect_counts": dict(defect_counts),
                "retrieved_parameters": retrieved_parameters,
                "retrieved_parameter_counts": dict(parameter_counts),
                "retrieval_accuracy": retrieval_accuracy,
                "defect_label_precision": defect_label_precision,
                "defect_label_recall": defect_label_recall,
                "parameter_accuracy": parameter_accuracy,
                "latency": latency,
                "answer": response,
            }
        )

    return pd.DataFrame(rows)
