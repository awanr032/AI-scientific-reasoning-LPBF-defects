from lpbf_defect_reasoning.generation import build_prompt


def test_build_prompt_includes_question_and_evidence():
    evidence = [{"chunk_id": "c1", "text": "Some evidence text."}]
    prompt = build_prompt("Why?", evidence)

    assert "Why?" in prompt
    assert "[c1] Some evidence text." in prompt
    assert prompt.strip().endswith("Answer:")
