from lpbf_defect_reasoning.graph import build_graph
from lpbf_defect_reasoning.indexing import SemanticIndex
from lpbf_defect_reasoning.retrieval import graph_rag_retrieve, match_graph_nodes


def test_match_graph_nodes_finds_mentioned_nodes(sample_chunks):
    graph = build_graph(sample_chunks)
    matched = match_graph_nodes("Why does high laser power lead to keyhole porosity?", graph)

    assert "laser power" in matched
    assert "keyhole" in matched


def test_match_graph_nodes_excludes_generic_nodes(sample_chunks):
    graph = build_graph(sample_chunks)
    graph.add_node("lpbf")
    matched = match_graph_nodes("defects in lpbf", graph)
    assert "lpbf" not in matched


def test_graph_rag_retrieve_uses_graph_evidence(sample_chunks, fake_embedder):
    graph = build_graph(sample_chunks)
    index = SemanticIndex(sample_chunks, fake_embedder)

    results = graph_rag_retrieve(
        "Why does high laser power lead to keyhole porosity?", graph, index, top_k=5
    )

    result_ids = {r["chunk_id"] for r in results}
    # c1/c2 are evidence on edges touching "laser power"/"keyhole"; c3/c4 are not.
    assert result_ids <= {"c1", "c2"}
    assert result_ids  # non-empty


def test_graph_rag_retrieve_falls_back_to_semantic_search_without_graph_matches(
    sample_chunks, fake_embedder
):
    graph = build_graph(sample_chunks)
    index = SemanticIndex(sample_chunks, fake_embedder)

    results = graph_rag_retrieve("completely unrelated query text", graph, index, top_k=2)
    assert len(results) == 2
