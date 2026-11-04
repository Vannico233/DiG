import json

import pytest

from dig_rl.io import read_rows, sha256, verify_data, write_json, write_jsonl


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_prepared_jsonl_preserves_unicode_separators(tmp_path, separator, newline):
    rows = [dict(id="one", question="before" + separator + "after", answer="1", demo="work"),
            dict(id="two", question="question", answer="2", demo="step" + separator + "end")]
    path = tmp_path / "train.jsonl"
    # Include blank physical lines and no final newline as accepted JSONL input.
    path.write_bytes((newline + (newline * 2).join(json.dumps(row, ensure_ascii=False)
                                                for row in rows)).encode("utf-8"))
    assert read_rows(path) == rows
    write_json(tmp_path / "manifest.json", {"files": {"train": {"rows": 2, "sha256": sha256(path)}}})
    assert verify_data(tmp_path)["files"]["train"]["rows"] == 2
    write_jsonl(path, rows)
    assert read_rows(path) == rows
