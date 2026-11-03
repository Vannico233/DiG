from dataclasses import asdict, replace
import pytest
from dig_rl.config import Config
from dig_rl.data import normalize
from dig_rl.io import write_jsonl, write_json, sha256, verify_data


def fixture_data(tmp_path):
    cfg = Config()
    row = normalize(dict(question="What is 1+1?", answer="One plus one is two.\n#### 2"), "gsm8k")
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [row])
    manifest = dict(config=asdict(cfg), model_id="model-revision",
                    files={"train": dict(rows=1, sha256=sha256(path))})
    write_json(tmp_path / "manifest.json", manifest)
    return cfg, path


def test_changed_data_rejected_even_if_manifest_unchanged(tmp_path):
    cfg, path = fixture_data(tmp_path)
    verify_data(tmp_path, cfg, "model-revision")
    path.write_text(path.read_text().replace("What is", "Compute"))
    with pytest.raises(ValueError, match="checksum"):
        verify_data(tmp_path, cfg)


def test_prepared_horizon_and_model_must_match(tmp_path):
    cfg, _ = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="protocol"):
        verify_data(tmp_path, replace(cfg, horizon=4096))
    with pytest.raises(ValueError, match="identity"):
        verify_data(tmp_path, cfg, "other-model")
