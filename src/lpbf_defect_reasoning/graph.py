"""Building the defect/parameter/mechanism knowledge graph from chunks."""

from __future__ import annotations

from typing import Any

import networkx as nx


def build_graph(chunks: list[dict[str, Any]]) -> nx.DiGraph:
    """Build a directed knowledge graph from (subject, relation, object) triples.

    Each chunk may contain a ``relations`` list of ``[subject, relation,
    object]`` triples. Edges are deduplicated by (subject, object) pair; the
    relation labels and supporting chunk ids observed for that pair are
    accumulated on the edge, along with an occurrence ``weight``.

    Args:
        chunks: Literature chunks as returned by ``lpbf_defect_reasoning.io.load_chunks``.

    Returns:
        A ``networkx.DiGraph`` with edge attributes ``weight`` (int),
        ``evidence`` (list of chunk ids), and ``relations`` (list of relation
        labels).
    """
    graph = nx.DiGraph()

    for chunk in chunks:
        for s, r, o in chunk.get("relations", []):
            if not s or not o:
                continue

            if graph.has_edge(s, o):
                graph[s][o]["weight"] += 1
                graph[s][o]["evidence"].append(chunk["chunk_id"])
                graph[s][o]["relations"].append(r)
            else:
                graph.add_edge(s, o, weight=1, evidence=[chunk["chunk_id"]], relations=[r])

    # Deduplicate evidence/relations accumulated per edge.
    for u, v in graph.edges():
        graph[u][v]["evidence"] = list(set(graph[u][v]["evidence"]))
        graph[u][v]["relations"] = list(set(graph[u][v]["relations"]))

    return graph
