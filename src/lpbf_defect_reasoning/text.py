"""Query normalization used before matching questions against graph nodes."""

from __future__ import annotations

# Multi-word phrases mapped onto the single-token node names used in the graph.
QUERY_NORMALIZE: dict[str, str] = {
    "lack of fusion": "lack_of_fusion",
    "residual stress": "residual_stress",
    "surface roughness": "surface_defect",
    "keyhole porosity": "keyhole",
    "unstable melt pool": "melt_pool_instability",
    "temperature gradient": "thermal_gradient",
}

# Graph nodes that are too generic to usefully anchor retrieval/path-finding.
GENERIC_NODES: frozenset[str] = frozenset({"lpbf", "slm", "am"})


def normalize_query(query: str) -> str:
    """Lowercase a query and rewrite known multi-word phrases to graph node ids."""
    q = query.lower()
    for old, new in QUERY_NORMALIZE.items():
        q = q.replace(old, new)
    return q
