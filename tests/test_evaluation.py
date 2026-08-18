from lpbf_defect_reasoning.evaluation import evaluate_graph_rag
from lpbf_defect_reasoning.pipeline import GraphRagPipeline


def test_evaluate_graph_rag_scores_a_small_custom_benchmark(
    sample_chunks, fake_embedder, fake_answerer
):
    pipeline = GraphRagPipeline(sample_chunks, embedder=fake_embedder, answerer=fake_answerer)

    benchmark = [
        {
            "question": "Why does high laser power lead to keyhole porosity?",
            "expected_defects": ["porosity", "keyhole"],
            "expected_parameters": ["laser power"],
        }
    ]

    df = evaluate_graph_rag(pipeline.qa, benchmark, k=5)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["retrieval_accuracy"] == 1.0
    assert row["latency"] >= 0
    assert "porosity" in row["retrieved_defects"]


def test_evaluate_graph_rag_handles_no_expected_parameters(
    sample_chunks, fake_embedder, fake_answerer
):
    pipeline = GraphRagPipeline(sample_chunks, embedder=fake_embedder, answerer=fake_answerer)

    benchmark = [
        {
            "question": "Why does high laser power lead to keyhole porosity?",
            "expected_defects": ["porosity"],
            "expected_parameters": [],
        }
    ]

    df = evaluate_graph_rag(pipeline.qa, benchmark, k=5)
    assert df.iloc[0]["parameter_accuracy"] is None
