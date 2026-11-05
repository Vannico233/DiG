import importlib.util
import json
from pathlib import Path

import pytest

from dig_rl.config import Config
from test_exploratory import EvalEngine

spec = importlib.util.spec_from_file_location("finish_base", Path(__file__).resolve().parents[1] / "scripts/finish_base_eval.py")
finish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finish)


def setup_case(tmp_path):
    rows = [dict(id=str(i)) for i in range(9)]
    cfg = Config()
    job = dict(name="math_numeric_rloo", gpu="2")
    model = dict(initial_parameters_sha256="initial", core={})
    with pytest.raises(RuntimeError, match="interrupt"):
        finish.runner.evaluate_job(EvalEngine("2"), rows, cfg, tmp_path, job,
            dict(eval_samples=4), "math_numeric_base", model)
    path = tmp_path / "eval/math_numeric_base/predictions.jsonl"
    protocol = finish.runner.read_json(path.parent / "protocol.json")
    return rows, cfg, path, protocol, job, model


def test_four_gpu_partition_matches_serial_seeds_and_records(tmp_path):
    rows, cfg, path, protocol, job, model = setup_case(tmp_path)
    directory = path.parent / "parallel"
    before, existing = finish.read_prefix(path, rows, 4)
    assert len(existing) == 2
    calls = []
    for slot in range(4):
        engine = EvalEngine()
        finish.evaluate_indices(engine, rows, cfg, list(range(2, len(rows)))[slot::4], directory, protocol, slot)
        calls += engine.calls
    assert sorted(calls) == [(str(j), cfg.seed + 200000000 + 10007*j) for j in range(2, len(rows))]
    finish.merge_parts(path, before, rows, cfg, directory, protocol)
    assert path.read_bytes().startswith(before)
    assert len(list(directory.glob("prefix-*.jsonl"))) == 1
    assert next(directory.glob("prefix-*.jsonl")).read_bytes() == before
    reference = tmp_path / "serial"
    finish.runner.evaluate_job(EvalEngine(), rows, cfg, reference, job,
        dict(eval_samples=4), "math_numeric_base", model)
    assert finish.runner.read_lines(path) == finish.runner.read_lines(reference / "eval/math_numeric_base/predictions.jsonl")
    result = finish.runner.evaluate_job(None, rows, cfg, tmp_path, job,
        dict(eval_samples=4), "math_numeric_base", model)
    assert result["mean_success"] == .5 and result["prompts"] == len(rows)


def test_resume_does_not_regenerate_completed_parts(tmp_path):
    rows, cfg, path, protocol, _, _ = setup_case(tmp_path)
    directory = path.parent / "parallel"
    engine = EvalEngine("5")
    with pytest.raises(RuntimeError):
        finish.evaluate_indices(engine, rows, cfg, list(range(2, len(rows))), directory, protocol, 0)
    repeated = EvalEngine()
    finish.evaluate_indices(repeated, rows, cfg, list(range(2, len(rows))), directory, protocol, 0)
    assert [x[0] for x in repeated.calls] == ["5", "6", "7", "8"]


@pytest.mark.parametrize("key,value", [("seed", 0), ("index", 3),
    ("protocol_sha256", "different"), ("helper_sha256", "different")])
def test_mismatched_part_cannot_be_reused(tmp_path, key, value):
    rows, cfg, path, protocol, _, _ = setup_case(tmp_path)
    directory = path.parent / "parallel"
    finish.evaluate_indices(EvalEngine(), rows, cfg, [2], directory, protocol, 0)
    part_path = finish.part_path(directory, 2)
    part = finish.runner.read_json(part_path)
    part[key] = value
    finish.runner.write_json(part_path, part)
    with pytest.raises(ValueError, match="protocol changed"):
        finish.read_part(directory, 2, rows, cfg, protocol)


def test_merge_missing_parts_keeps_original_file(tmp_path):
    rows, cfg, path, protocol, _, _ = setup_case(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Missing parallel"):
        finish.merge_parts(path, before, rows, cfg, path.parent / "parallel", protocol)
    assert path.read_bytes() == before


def test_concurrent_original_writer_is_not_overwritten(tmp_path):
    rows, cfg, path, protocol, _, _ = setup_case(tmp_path)
    before = path.read_bytes()
    path.write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="changed during"):
        finish.merge_parts(path, before, rows, cfg, path.parent / "parallel", protocol)
    assert path.read_bytes() == before + b"\n"


def test_incomplete_prefix_is_not_silently_discarded(tmp_path):
    rows, _, path, _, _, _ = setup_case(tmp_path)
    before = path.read_bytes() + b'{"id":'
    path.write_bytes(before)
    with pytest.raises(ValueError, match="Incomplete final"):
        finish.read_prefix(path, rows, 4)
    assert path.read_bytes() == before


def test_wrong_question_and_scores_rejected(tmp_path):
    rows, _, path, _, _, _ = setup_case(tmp_path)
    records = finish.runner.read_lines(path)
    records[0]["id"] = "1"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="question/order"):
        finish.read_prefix(path, rows, 4)
    records[0]["id"] = "0"
    records[0]["score"] = .75
    with pytest.raises(ValueError, match="scores"):
        finish.validate_record(records[0], "0", 4)
