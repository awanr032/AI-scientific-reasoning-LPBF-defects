"""Graph-guided semantic retrieval of evidence chunks."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .indexing import SemanticIndex
from .text import GENERIC_NODES, normalize_query


def match_graph_nodes(query: str, graph: nx.DiGraph) -> list[str]:
    """Return graph nodes whose name appears in the (normalized) query text."""
    q = normalize_query(query)
    return [n for n in graph.nodes if n not in GENERIC_NODES and n.lower() in q]


def graph_rag_retrieve(
    query: str,
    graph: nx.DiGraph,
    index: SemanticIndex,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve evidence chunks for a query using graph structure + embeddings.

    Matches query terms to graph nodes, collects chunks cited as evidence on
    edges touching those nodes, then ranks that candidate set by embedding
    distance to the query. Falls back to a plain semantic search over the
    full index when no graph nodes match.

    Args:
        query: The natural-language question.
        graph: Knowledge graph from ``lpbf_defect_reasoning.graph.build_graph``.
        index: Semantic index over the same chunk set as the graph's evidence.
        top_k: Number of chunks to return.

    Returns:
        Up to ``top_k`` chunk dictionaries, most relevant first.
    """
    matched_nodes = match_graph_nodes(query, graph)

    candidate_chunk_ids: set[str] = set()
    for n in matched_nodes:
        for nbr in graph.successors(n):
            candidate_chunk_ids.update(graph[n][nbr]["evidence"])
        for pred in graph.predecessors(n):
            candidate_chunk_ids.update(graph[pred][n]["evidence"])

    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(index.chunks)}
    cand_idx = [id_to_idx[cid] for cid in candidate_chunk_ids if cid in id_to_idx]

    if not cand_idx:
        return index.search(query, top_k)

    qv = index.encode_query(query)
    sub_emb = index.embeddings[cand_idx]
    dists = ((sub_emb - qv) ** 2).sum(axis=1)
    best = np.argsort(dists)[:top_k]

    return [index.chunks[cand_idx[i]] for i in best]
