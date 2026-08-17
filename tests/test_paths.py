from lpbf_defect_reasoning.graph import build_graph
from lpbf_defect_reasoning.paths import find_graph_paths


def test_find_graph_paths_returns_path_between_mentioned_nodes(sample_chunks):
    graph = build_graph(sample_chunks)

    paths = find_graph_paths("Why does laser power cause porosity?", graph)

    assert any(p == ["laser power", "keyhole", "porosity"] for p in paths)


def test_find_graph_paths_returns_empty_when_no_nodes_mentioned(sample_chunks):
    graph = build_graph(sample_chunks)
    assert find_graph_paths("totally unrelated question", graph) == []


def test_find_graph_paths_respects_max_paths(sample_chunks):
    graph = build_graph(sample_chunks)
    paths = find_graph_paths(
        "laser power keyhole porosity scan speed energy density lack_of_fusion",
        graph,
        max_paths=1,
    )
    assert len(paths) <= 1
