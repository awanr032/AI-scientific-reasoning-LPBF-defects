"""AI system for defect reasoning in Laser Powder Bed Fusion (LPBF, Ti-6Al-4V).

This package implements a Graph-RAG pipeline: it builds a knowledge graph
and a semantic (FAISS) index from pre-extracted literature chunks, retrieves
evidence for a question using both, and generates a grounded answer with an
instruction-tuned LLM.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lpbf-defect-reasoning")
except PackageNotFoundError:  # pragma: no cover - local/editable checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
