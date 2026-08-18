import json

import pytest

from lpbf_defect_reasoning.io import load_chunks


def test_load_chunks_reads_json_list(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps([{"chunk_id": "c1", "text": "hello"}]), encoding="utf-8")

    chunks = load_chunks(path)
    assert chunks == [{"chunk_id": "c1", "text": "hello"}]


def test_load_chunks_rejects_non_list(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_chunks(path)
