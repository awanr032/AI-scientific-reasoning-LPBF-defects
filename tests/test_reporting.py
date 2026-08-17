from lpbf_defect_reasoning.reporting import (
    build_structured_report,
    format_report,
    summarize_evidence,
)


def test_summarize_evidence_aggregates_and_counts(sample_chunks):
    summary = summarize_evidence(sample_chunks[:2])  # c1, c2 both mention porosity/keyhole

    assert summary["chunk_ids"] == ["c1", "c2"]
    assert summary["defects"]["porosity"] == 2
    assert summary["defects"]["keyhole"] == 2
    assert summary["parameters"] == {"laser power": 1}


def test_summarize_evidence_handles_missing_keys():
    summary = summarize_evidence([{"chunk_id": "x"}])
    assert summary == {"chunk_ids": ["x"], "defects": {}, "parameters": {}, "mechanisms": {}}


def test_build_structured_report_shape(sample_chunks):
    summary = summarize_evidence(sample_chunks[:1])
    report = build_structured_report(
        question="Why?",
        question_type="explanation",
        summary=summary,
        answer="because reasons",
        paths=[["laser power", "keyhole"]],
    )

    assert report["question"] == "Why?"
    assert report["question_type"] == "explanation"
    assert report["answer"] == "because reasons"
    assert report["graph_paths"] == [["laser power", "keyhole"]]


def test_format_report_includes_answer_only_when_present(sample_chunks):
    summary = summarize_evidence(sample_chunks[:1])

    with_answer = build_structured_report("Q", "explanation", summary, "A1", [])
    assert "Answer:" in format_report(with_answer)

    without_answer = build_structured_report("Q", "lookup", summary, None, [])
    assert "Answer:" not in format_report(without_answer)
