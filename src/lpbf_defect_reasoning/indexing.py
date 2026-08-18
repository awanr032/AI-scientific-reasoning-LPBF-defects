"""Semantic (FAISS) indexing of literature chunks.

The embedding model is injected rather than constructed here, so tests and
non-GPU environments can supply a lightweight stand-in instead of downloading
``sentence-transformers/all-MiniLM-L6-v2``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    """Minimal interface required of an embedding model."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        """Encode a list of texts into an array-like of shape (len(texts), dim)."""
        ...


def load_default_embedder() -> Embedder:
    """Construct the pinned sentence-transformers embedding model.

    Imports ``sentence_transformers`` lazily so the rest of the package can
    be imported (and unit tested) without that heavy dependency installed.
    """
    from sentence_transformers import SentenceTransformer

    from .config import EMBEDDING_MODEL

    return SentenceTransformer(EMBEDDING_MODEL.repo_id, revision=EMBEDDING_MODEL.revision)


class SemanticIndex:
    """A FAISS flat-L2 index over chunk embeddings, with the source chunks."""

    def __init__(self, chunks: list[dict[str, Any]], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder

        texts = [c["text"] for c in chunks]
        self.embeddings = np.asarray(
            embedder.encode(texts, show_progress_bar=False), dtype="float32"
        )

        import faiss  # imported lazily; heavy optional dependency

        self._faiss = faiss
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def __len__(self) -> int:
        return self.index.ntotal

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self.embedder.encode([query]), dtype="float32")

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return the top_k chunks by embedding similarity to the query."""
        qv = self.encode_query(query)
        _, indices = self.index.search(qv, top_k)
        return [self.chunks[i] for i in indices[0]]
