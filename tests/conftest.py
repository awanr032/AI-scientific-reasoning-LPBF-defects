"""Shared fixtures for the test suite.

These fixtures build a small, fully synthetic chunk set so tests never need
network access or the real (multi-GB) pretrained models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "c1",
            "text": "High laser power increases the risk of keyhole porosity.",
            "relations": [
                ["laser power", "increases", "keyhole"],
                ["keyhole", "causes", "porosity"],
            ],
            "possible_defects": ["porosity", "keyhole"],
            "possible_parameters": ["laser power"],
            "possible_mechanisms": ["keyhole instability"],
        },
        {
            "chunk_id": "c2",
            "text": "Keyhole collapse traps gas and forms pores in the melt track.",
            "relations": [["keyhole", "causes", "porosity"]],
            "possible_defects": ["porosity", "keyhole"],
            "possible_parameters": [],
            "possible_mechanisms": ["keyhole instability"],
        },
        {
            "chunk_id": "c3",
            "text": "High scan speed reduces energy input and can cause lack of fusion.",
            "relations": [
                ["scan speed", "reduces", "energy density"],
                ["energy density", "causes", "lack_of_fusion"],
            ],
            "possible_defects": ["lack_of_fusion"],
            "possible_parameters": ["scan speed", "energy density"],
            "possible_mechanisms": ["insufficient melting"],
        },
        {
            "chunk_id": "c4",
            "text": "Lack of fusion defects arise from insufficient melt pool overlap.",
            "relations": [["energy density", "causes", "lack_of_fusion"]],
            "possible_defects": ["lack_of_fusion"],
            "possible_parameters": ["energy density"],
            "possible_mechanisms": ["insufficient melting"],
        },
    ]


class FakeEmbedder:
    """Deterministic bag-of-words embedder standing in for a real sentence encoder."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in text.lower().split():
                out[i, hash(token) % self.dim] += 1.0
        return out


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


class FakeAnswerer:
    """Answerer stand-in that just echoes the evidence chunk ids, no LLM needed."""

    def generate(self, question: str, evidence_chunks: list[dict[str, Any]]) -> str:
        ids = ", ".join(c["chunk_id"] for c in evidence_chunks)
        return f"[fake answer for: {question}] evidence={ids}"


@pytest.fixture
def fake_answerer() -> FakeAnswerer:
    return FakeAnswerer()
