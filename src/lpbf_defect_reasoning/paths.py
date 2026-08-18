"""Shortest-path discovery between graph nodes mentioned in a question."""

from __future__ import annotations

import networkx as nx

from .retrieval import match_graph_nodes


def find_graph_paths(question: str, graph: nx.DiGraph, max_paths: int = 3) -> list[list[str]]:
    """Find shortest paths between graph nodes mentioned in the question.

    Args:
        question: The natural-language question.
        graph: Knowledge graph from ``lpbf_defect_reasoning.graph.build_graph``.
        max_paths: Maximum number of unique paths to return.

    Returns:
        Up to ``max_paths`` unique node-id paths (each a list of node names).
    """
    matched_nodes = match_graph_nodes(question, graph)

    paths: list[list[str]] = []
    for src in matched_nodes:
        for dst in matched_nodes:
            if src == dst:
                continue
            try:
                paths.append(nx.shortest_path(graph, source=src, target=dst))
            except nx.NetworkXNoPath:
                pass
            except nx.NodeNotFound:
                pass

    seen: set[tuple[str, ...]] = set()
    unique_paths: list[list[str]] = []
    for p in paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_paths.append(p)

    return unique_paths[:max_paths]
