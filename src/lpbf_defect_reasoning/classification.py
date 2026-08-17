"""Lightweight rule-based question-type classification."""

from __future__ import annotations


def classify_question(question: str) -> str:
    """Classify a question as one of comparison/lookup/explanation/general."""
    q = question.lower()

    if "compare" in q:
        return "comparison"

    if q.startswith("which") or "what parameters" in q or "associated with" in q:
        return "lookup"

    if "why" in q or "how does" in q or "how can" in q:
        return "explanation"

    return "general"
