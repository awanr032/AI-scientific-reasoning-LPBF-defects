"""Loading pre-extracted literature chunks from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    """Load the JSON list of literature chunks produced by the extraction step.

    Each chunk is expected to look roughly like::

        {
            "chunk_id": "...",
            "text": "...",
            "relations": [["subject", "relation", "object"], ...],
            "possible_defects": [...],
            "possible_parameters": [...],
            "possible_mechanisms": [...],
        }

    Args:
        path: Path to a JSON file containing a list of chunk objects.

    Returns:
        The parsed list of chunk dictionaries.

    Raises:
        ValueError: If the file does not contain a JSON list.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError(f"Expected a JSON list of chunks in {path}, got {type(chunks)!r}")

    return chunks
