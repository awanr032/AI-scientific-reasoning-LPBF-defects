import pytest

from lpbf_defect_reasoning.classification import classify_question


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Compare porosity and lack of fusion in LPBF.", "comparison"),
        ("Which process parameters influence porosity formation?", "lookup"),
        ("What parameters are associated with cracking?", "lookup"),
        ("Why does high laser power lead to keyhole porosity?", "explanation"),
        ("How does hatch spacing affect lack of fusion?", "explanation"),
        ("Tell me about Ti-6Al-4V microstructure.", "general"),
    ],
)
def test_classify_question(question, expected):
    assert classify_question(question) == expected
