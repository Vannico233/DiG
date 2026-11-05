import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dig_rl.config import Config
from dig_rl.objective import Sampler
from dig_rl.train import update
from test_training import FakeEngine, ROWS
from test_exploratory import EvalEngine

spec = importlib.util.spec_from_file_location("signal_diagnostic", Path(__file__).resolve().parents[1] / "scripts/diagnose_signal.py")
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


def test_profiling_preserves_existing_rloo_update_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = replace(Config(), group_size=2, rollout_batch_size=2, candidates=3)
    plain, observed = FakeEngine(), FakeEngine()
    profile = diagnose.ProfiledEngine(observed, tmp_path, cfg)
    results = []
    for engine in (plain, profile):
        results.append(update(engine, torch.optim.SGD([engine.p], lr=.01), ROWS, Sampler(13), cfg, 0, 0., 1.))
    torch.testing.assert_close(plain.p, observed.p, rtol=0, atol=0)
    assert plain.calls == observed.calls
    assert results[0] == results[1]
    groups = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.json"))]
    assert len(groups) == 3
    assert [g["seed"] for g in groups] == [seed for _, seed in plain.calls]
    assert sum(g["responses"] for g in groups) == results[0]["responses"]
    assert all(g["constant_reward_group"] == (len(set(g["rewards"])) == 1) for g in groups)


def test_lr_probe_changes_only_sgd_step_scale():
    low, high = FakeEngine(), FakeEngine()
    initial = low.p.detach().clone()
    records = []
    for engine, lr in ((low, .01), (high, 1.)):
        cfg = replace(Config(), group_size=2, rollout_batch_size=2, candidates=8, learning_rate=lr)
        records.append(update(engine, torch.optim.SGD([engine.p], lr=lr), ROWS, Sampler(13), cfg, 0, 0., 1.))
    assert low.calls == high.calls
    assert records[0]["gradient_norm"] == records[1]["gradient_norm"]
    torch.testing.assert_close(high.p-initial, 100*(low.p-initial), rtol=1e-12, atol=1e-12)


def test_resume_overwrites_only_uncommitted_group_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = replace(Config(), group_size=2, rollout_batch_size=2, candidates=3)
    controls = dict(max_steps=4, train_seconds=1e6, save_every=1)
    job = diagnose.make_jobs(["0", "1", "2", "3"], [.01, 1.])[0]
    full = FakeEngine()
    initial = full.parameter_state()
    diagnose.runner.train_job(diagnose.ProfiledEngine(full, tmp_path / "full/profile", cfg), ROWS,
        cfg, tmp_path / "full", job, controls, {}, initial)
    class Interrupted(FakeEngine):
        def rollout(self, row, seed):
            if len(self.calls) == 5:
                raise RuntimeError("interrupted mid-update")
            return super().rollout(row, seed)
    with pytest.raises(RuntimeError):
        diagnose.runner.train_job(diagnose.ProfiledEngine(Interrupted(), tmp_path / "resume/profile", cfg),
            ROWS, cfg, tmp_path / "resume", job, controls, {}, initial)
    checkpoint = torch.load(tmp_path / "resume/train" / job["name"] / "checkpoint.pt", weights_only=True)
    assert checkpoint["step"] == 1
    restored = FakeEngine()
    diagnose.runner.train_job(diagnose.ProfiledEngine(restored, tmp_path / "resume/profile", cfg, 1),
        ROWS, cfg, tmp_path / "resume", job, controls, {}, initial)
    torch.testing.assert_close(restored.p, full.p, rtol=0, atol=0)
    def groups(directory):
        records = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
        return [{k:v for k,v in r.items() if k != "group_wall_seconds"} for r in records]
    assert groups(tmp_path / "full/profile") == groups(tmp_path / "resume/profile")


def test_evaluation_is_not_counted_as_training_profile(tmp_path):
    profile = diagnose.ProfiledEngine(EvalEngine(), tmp_path, Config())
    profile.rollout(dict(id="a"), 123, 4)
    assert profile.pending is None and not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("rates", [[.01, .01], [.01, 0.], [.01, float("nan")], [.01, float("inf")]])
def test_invalid_lr_screen_rejected(rates):
    with pytest.raises(ValueError):
        diagnose.make_jobs(["0", "1", "2", "3"], rates)


def test_four_jobs_cover_both_datasets_and_learning_rates():
    jobs = diagnose.make_jobs(["0", "1", "2", "3"], [.01, 1.])
    assert [(j["dataset"], j["learning_rate"], j["gpu"]) for j in jobs] == [
        ("gsm8k", .01, "0"), ("gsm8k", 1., "1"), ("math_numeric", .01, "2"), ("math_numeric", 1., "3")]
    assert all(j["a"] == 0 and j["rho"] == 1 for j in jobs)


def test_reference_requires_identical_question_order_and_samples(tmp_path):
    path = tmp_path / "predictions.jsonl"
    record = dict(id="a", responses=4, rewards=[0, 1, 0, 1], score=.5)
    path.write_text(json.dumps(record) + "\n" + '{"incomplete":')
    assert diagnose.read_reference(path, ["a"], 4) == [record]
    with pytest.raises(ValueError):
        diagnose.read_reference(path, ["other"], 4)
    with pytest.raises(ValueError):
        diagnose.read_reference(path, ["a"], 2)


def test_unfinished_source_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    diagnose.runner.write_json(source / "experiment.json", dict(test="source"))
    manifest = dict(source_run=str(source), source_manifest_sha256=diagnose.runner.file_hash(source / "experiment.json"))
    diagnose.runner.write_json(source / "status.json", dict(state="running"))
    with pytest.raises(RuntimeError, match="not completed"):
        diagnose.ensure_source_complete(manifest)
    diagnose.runner.write_json(source / "status.json", dict(state="completed"))
    diagnose.ensure_source_complete(manifest)
    diagnose.runner.write_json(source / "experiment.json", dict(test="changed"))
    with pytest.raises(ValueError, match="changed"):
        diagnose.ensure_source_complete(manifest)


def test_compute_process_on_selected_gpu_blocks_launch(monkeypatch):
    def query(command, **kwargs):
        return SimpleNamespace(stdout="0, GPU-zero\n1, GPU-one\n" if "--query-gpu=index,uuid" in command else "GPU-one, 12345\n")
    monkeypatch.setattr(diagnose.subprocess, "run", query)
    diagnose.ensure_gpus_idle(["0"])
    with pytest.raises(RuntimeError, match="compute processes"):
        diagnose.ensure_gpus_idle(["0", "1"])
    with pytest.raises(ValueError, match="unavailable"):
        diagnose.ensure_gpus_idle(["2"])


def test_utilization_sample_includes_job_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="0, 35, 9344, 118.2\n7, 90, 50000, 500\n"))
    jobs = diagnose.make_jobs(["0", "1", "2", "3"], [.01, 1.])
    diagnose.runner.write_json(tmp_path / "progress_gsm8k_lr0.json", dict(phase="training"))
    diagnose.sample_gpus(tmp_path, jobs)
    record = json.loads((tmp_path / "gpu_samples.jsonl").read_text())
    assert len(record["gpus"]) == 1
    assert record["gpus"][0]["phase"] == "training"
    assert record["gpus"][0]["utilization_percent"] == "35"
