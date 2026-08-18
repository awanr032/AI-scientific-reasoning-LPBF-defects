import pytest

from lpbf_defect_reasoning.pipeline import GraphRagPipeline


def test_pipeline_qa_returns_answer_and_evidence(sample_chunks, fake_embedder, fake_answerer):
    pipeline = GraphRagPipeline(sample_chunks, embedder=fake_embedder, answerer=fake_answerer)

    answer, evidence = pipeline.qa("Why does high laser power lead to keyhole porosity?")

    assert "fake answer" in answer
    assert evidence


def test_pipeline_qa_without_answerer_raises(sample_chunks, fake_embedder):
    pipeline = GraphRagPipeline(sample_chunks, embedder=fake_embedder, answerer=None)

    with pytest.raises(RuntimeError):
        pipeline.qa("Why does high laser power lead to keyhole porosity?")


def test_pipeline_agent_skips_generation_for_lookup_questions(sample_chunks, fake_embedder):
    pipeline = GraphRagPipeline(sample_chunks, embedder=fake_embedder, answerer=None)

    report = pipeline.agent("Which process parameters influence porosity formation?")

    assert report["question_type"] == "lookup"
    assert report["answer"] is None


def test_pipeline_agent_builds_full_report(sample_chunks, fake_embedder, fake_answerer):
    pipeline = GraphRagPipeline(sample_chunks, embedder=fake_embedder, answerer=fake_answerer)

    report = pipeline.agent("Why does high laser power lead to keyhole porosity?")

    assert report["question_type"] == "explanation"
    assert report["answer"] is not None
    assert report["graph_paths"]
    assert "porosity" in report["defects"]
