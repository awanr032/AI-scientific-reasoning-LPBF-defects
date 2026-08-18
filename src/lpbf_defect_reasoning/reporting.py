"""Aggregating evidence and formatting structured agent output."""

from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_evidence(evidence_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate defect/parameter/mechanism labels across a set of evidence chunks."""
    defects: list[str] = []
    parameters: list[str] = []
    mechanisms: list[str] = []
    chunk_ids: list[str] = []

    for c in evidence_chunks:
        defects += c.get("possible_defects", [])
        parameters += c.get("possible_parameters", [])
        mechanisms += c.get("possible_mechanisms", [])
        chunk_ids.append(c.get("chunk_id", ""))

    return {
        "chunk_ids": chunk_ids,
        "defects": dict(Counter(defects)),
        "parameters": dict(Counter(parameters)),
        "mechanisms": dict(Counter(mechanisms)),
    }


def build_structured_report(
    question: str,
    question_type: str,
    summary: dict[str, Any],
    answer: str | None,
    paths: list[list[str]],
) -> dict[str, Any]:
    """Assemble the final structured report returned by the reasoning agent."""
    return {
        "question": question,
        "question_type": question_type,
        "top_chunk_ids": summary["chunk_ids"],
        "defects": summary["defects"],
        "parameters": summary["parameters"],
        "mechanisms": summary["mechanisms"],
        "graph_paths": paths,
        "answer": answer,
    }


def format_report(report: dict[str, Any]) -> str:
    """Render a structured report as human-readable text."""
    lines = [
        f"Question: {report['question']}",
        f"Question type: {report['question_type']}",
        "",
        "Top chunk IDs:",
    ]
    lines += [f"- {cid}" for cid in report["top_chunk_ids"]] or ["- none"]

    sections = (("Defects", "defects"), ("Parameters", "parameters"), ("Mechanisms", "mechanisms"))
    for label, key in sections:
        lines += ["", f"{label}:"]
        values = report[key]
        lines += [f"- {k}: {v}" for k, v in values.items()] if values else ["- none"]

    lines += ["", "Graph paths:"]
    if report["graph_paths"]:
        lines += [" -> ".join(p) for p in report["graph_paths"]]
    else:
        lines += ["- no short path found"]

    if report["answer"] is not None:
        lines += ["", "Answer:", "", report["answer"]]

    return "\n".join(lines)


def pretty_print_report(report: dict[str, Any]) -> None:
    """Print a structured report to stdout."""
    print(format_report(report))
