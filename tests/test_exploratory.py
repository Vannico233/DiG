import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from dig_rl.config import Config
from test_training import FakeEngine, ROWS

spec = importlib.util.spec_from_file_location("exploratory", Path(__file__).resolve().parents[1] / "scripts/run_exploratory.py")
explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explore)


def reports():
    return {"gsm8k": dict(a=.7, enabled=False, audit_seconds=100., audit_tokens=80, audit_responses=16),
            "math_numeric": dict(a=0., enabled=False, audit_seconds=21000., audit_tokens=160, audit_responses=32)}


def test_four_card_pairing_retains_failed_gate_and_zero_predictor():
    jobs = explore.make_jobs(reports(), ["0", "1", "2", "3"], .5)
    assert [(j["name"], j["gpu"]) for j in jobs] == [
        ("gsm8k_rloo", "0"), ("gsm8k_dig", "1"), ("math_numeric_rloo", "2"), ("math_numeric_dig", "3")]
    assert not any(j["gate_passed"] for j in jobs)
    assert jobs[1]["a"] == .7 and jobs[1]["predictor_active"]
    assert jobs[3]["a"] == 0 and not jobs[3]["predictor_active"] and jobs[3]["rho"] == .5
    assert jobs[3]["effective_arm"] == "dig_zero_predictor_exploratory"
    assert jobs[2]["audit_seconds"] == 0 and jobs[3]["audit_seconds"] == 21000


@pytest.mark.parametrize("gpus,rho", [(["0", "1"], .5), (["0", "1", "2", "2"], .5),
                                      (["0", "1", "2", "3"], 0), (["0", "1", "2", "3"], 1),
                                      (["0", "1", "2", "3"], float("nan"))])
def test_invalid_experiment_rejected(gpus, rho):
    with pytest.raises(ValueError):
        explore.make_jobs(reports(), gpus, rho)


def test_source_audit_must_match_core_and_calibration():
    cfg = replace(Config(), calibration_size=2, audit_size=4)
    ident = dict(code="same-estimator", data="same-data")
    report = dict(identity=ident, a=0., audit_seconds=21000.)
    cal = [dict(uu=2., ug=-.1), dict(uu=3., ug=.01)]
    explore.validate_audit(report, ident, cal, [{}] * 4, cfg)
    with pytest.raises(ValueError, match="identity"):
        explore.validate_audit(report, dict(ident, code="different"), cal, [{}] * 4, cfg)
    with pytest.raises(ValueError, match="scale"):
        explore.validate_audit(dict(report, a=.5), ident, cal, [{}] * 4, cfg)
    with pytest.raises(ValueError, match="Incomplete"):
        explore.validate_audit(report, ident, cal[:1], [{}] * 4, cfg)


def test_expensive_audit_does_not_consume_exploratory_training_budget(tmp_path):
    cfg = replace(Config(), group_size=2, rollout_batch_size=2, candidates=3)
    job = explore.make_jobs(reports(), ["0", "1", "2", "3"], .5)[3]
    controls = dict(max_steps=2, train_seconds=1000., save_every=1)
    e = FakeEngine()
    initial = e.parameter_state()
    result = explore.train_job(e, ROWS, cfg, tmp_path, job, controls, {"test": "one"}, initial)
    assert result["steps"] == 2
    assert result["end_to_end_seconds"] == result["training_seconds"] + 21000
    assert result["charged_generated_tokens"] == result["training_generated_tokens"] + 160
    assert result["parameter_delta_l2"] > 0
    before = result["checkpoint_sha256"]
    fresh = FakeEngine()
    repeated = explore.train_job(fresh, ROWS, cfg, tmp_path, job, controls, {"test": "one"}, initial)
    assert repeated == result and not fresh.calls
    assert explore.file_hash(tmp_path / "train/math_numeric_dig/checkpoint.pt") == before
    torch.testing.assert_close(fresh.p, e.p, rtol=0, atol=0)


def test_exploratory_resume_preserves_estimator_rng(tmp_path, monkeypatch):
    import dig_rl.train as training
    cfg = replace(Config(), group_size=2, rollout_batch_size=2, candidates=3)
    job = explore.make_jobs(reports(), ["0", "1", "2", "3"], .5)[1]
    controls = dict(max_steps=4, train_seconds=1000., save_every=1)
    full, interrupted = FakeEngine(), FakeEngine()
    initial = full.parameter_state()
    explore.train_job(full, ROWS, cfg, tmp_path / "full", job, controls, {}, initial)
    real = training.update
    def stop(engine, optimizer, rows, sampler, cfg, step, a, rho):
        if step == 2:
            raise RuntimeError("interrupt")
        return real(engine, optimizer, rows, sampler, cfg, step, a, rho)
    monkeypatch.setattr(training, "update", stop)
    with pytest.raises(RuntimeError, match="interrupt"):
        explore.train_job(interrupted, ROWS, cfg, tmp_path / "resume", job, controls, {}, initial)
    monkeypatch.setattr(training, "update", real)
    fresh = FakeEngine()
    explore.train_job(fresh, ROWS, cfg, tmp_path / "resume", job, controls, {}, initial)
    torch.testing.assert_close(fresh.p, full.p, rtol=0, atol=0)
    records = explore.read_lines(tmp_path / "resume/train/gsm8k_dig/metrics.jsonl")
    assert [r["step"] for r in records] == [1, 2, 3, 4]


class EvalEngine:
    tokenizer = type("Tokenizer", (), {"decode": lambda self, *a, **k: "answer"})()
    def __init__(self, interrupt=None):
        self.calls, self.interrupt = [], interrupt
    def rollout(self, row, seed, count):
        if row["id"] == self.interrupt:
            raise RuntimeError("interrupt")
        self.calls.append((row["id"], seed))
        return [torch.tensor([[0, 1]])] * count, torch.tensor([0., 1., 0., 1.]), dict(
            seconds=1., generated_tokens=count, responses=count, parsed=count, truncated=0, rewards=[0, 1, 0, 1])
    def tensors(self, row):
        return torch.tensor([[0]]), None


def test_evaluation_resume_uses_same_prompts_and_seeds(tmp_path):
    rows = [dict(id=str(i)) for i in range(3)]
    cfg, controls = Config(), dict(eval_samples=4)
    job = explore.make_jobs(reports(), ["0", "1", "2", "3"], .5)[0]
    with pytest.raises(RuntimeError):
        explore.evaluate_job(EvalEngine("1"), rows, cfg, tmp_path, job, controls, job["name"], {"checkpoint": "same"})
    engine = EvalEngine()
    result = explore.evaluate_job(engine, rows, cfg, tmp_path, job, controls, job["name"], {"checkpoint": "same"})
    assert engine.calls == [("1", cfg.seed + 200000000 + 10007), ("2", cfg.seed + 200000000 + 20014)]
    assert result["mean_success"] == .5 and result["prompts"] == 3
    with pytest.raises(ValueError, match="protocol"):
        explore.evaluate_job(engine, rows, cfg, tmp_path, job, controls, job["name"], {"checkpoint": "changed"})


def test_paired_comparison_requires_matching_unique_questions():
    left = [dict(id="a", score=.25), dict(id="b", score=.5)]
    right = [dict(id="b", score=.75), dict(id="a", score=.5)]
    assert explore.paired_difference(left, right)["difference"] == .25
    with pytest.raises(ValueError):
        explore.paired_difference(left, [dict(id="other", score=1.)])


def test_mismatched_initialization_is_rejected_before_final_results(tmp_path):
    manifest = {"jobs": explore.make_jobs(reports(), ["0", "1", "2", "3"], .5)}
    for arm, digest in (("rloo", "original"), ("dig", "different")):
        explore.write_json(tmp_path / "train" / f"gsm8k_{arm}" / "protocol.json",
                           dict(initial_parameters_sha256=digest))
    with pytest.raises(ValueError, match="different parameters"):
        explore.check_initialization(tmp_path, manifest)
