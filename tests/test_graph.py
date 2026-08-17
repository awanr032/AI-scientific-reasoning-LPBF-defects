from lpbf_defect_reasoning.graph import build_graph


def test_build_graph_node_and_edge_counts(sample_chunks):
    graph = build_graph(sample_chunks)

    assert graph.number_of_nodes() > 0
    assert graph.has_edge("keyhole", "porosity")
    assert graph.has_edge("laser power", "keyhole")


def test_build_graph_deduplicates_repeated_edges(sample_chunks):
    # "keyhole" -> "porosity" appears in both c1 and c2
    graph = build_graph(sample_chunks)

    edge = graph["keyhole"]["porosity"]
    assert edge["weight"] == 2
    assert set(edge["evidence"]) == {"c1", "c2"}
    assert edge["relations"] == ["causes"]


def test_build_graph_ignores_relations_missing_subject_or_object():
    chunks = [
        {
            "chunk_id": "c1",
            "relations": [
                ["", "causes", "porosity"],
                [None, "causes", "keyhole"],
                ["laser power", "increases", "keyhole"],
            ],
        }
    ]
    graph = build_graph(chunks)

    assert graph.number_of_edges() == 1
    assert graph.has_edge("laser power", "keyhole")
