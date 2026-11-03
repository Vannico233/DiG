from dataclasses import replace
import json
import random
import pytest
import torch
from dig_rl.config import Config
from dig_rl.objective import Sampler, coefficients
from dig_rl.train import update, train, resolve_decision


class FakeEngine:
    def __init__(self):
        self.p = torch.nn.Parameter(torch.tensor([.2, -.1], dtype=torch.float64))
        self.params = [("adapter", self.p)]
        self.calls = []
    def sync(self):
        pass
    def demo_backward(self, row, weight):
        loss = -(self.p * torch.tensor(row["u"])).sum()
        (weight * loss).backward()
        return dict(loss=float(loss.detach()), seconds=0., scored_tokens=2)
    def rollout(self, row, seed):
        self.calls.append((row["id"], seed))
        rewards = torch.tensor([random.Random(seed).randrange(2), 1.])
        return [], rewards, dict(seconds=0., responses=2, generated_tokens=4, parsed=2, truncated=0)
    def rl_backward(self, row, sequences, rewards, weight):
        g = torch.tensor(row["g"]) * (1 + rewards[0])
        loss = -(self.p * g).sum()
        (weight * loss).backward()
        return dict(loss=float(loss.detach()), seconds=0., scored_tokens=4)
    def parameter_state(self):
        return {"adapter": self.p.detach().clone()}
    def load_parameters(self, state):
        with torch.no_grad():
            self.p.copy_(state["adapter"])


ROWS = [dict(id="a", u=[1., 2.], g=[3., -1.]), dict(id="b", u=[-1., .5], g=[-2., 2.])]


def test_update_matches_explicit_batch_formula():
    cfg = replace(Config(), group_size=2, rollout_batch_size=2, candidates=5)
    e, sampler = FakeEngine(), Sampler(12)
    expected_draws = Sampler(12).draw(2, 5, .5)
    initial = e.p.detach().clone()
    optimizer = torch.optim.SGD([e.p], lr=cfg.learning_rate)
    result = update(e, optimizer, ROWS, sampler, cfg, 0, .7, .5)
    direction = torch.zeros(2, dtype=torch.float64)
    for i, z, seed in expected_draws:
        u = torch.tensor(ROWS[i]["u"], dtype=torch.float64)
        g = torch.tensor(ROWS[i]["g"], dtype=torch.float64) * (1 + random.Random(seed).randrange(2))
        direction += (.7 * u + int(z) / .5 * (g - .7*u)) / 5
    torch.testing.assert_close(e.p, initial + cfg.learning_rate * direction)
    assert len(e.calls) == result["selected"] == sum(z for _, z, _ in expected_draws)


def test_empty_check_batch_updates_prediction_only():
    cfg = replace(Config(), candidates=3)
    e = FakeEngine()
    initial = e.p.detach().clone()
    result = update(e, torch.optim.SGD([e.p], lr=.01), ROWS[:1], Sampler(1), cfg, 0, 1., 1e-12)
    assert result["selected"] == 0 and not e.calls
    torch.testing.assert_close(e.p, initial + cfg.learning_rate * torch.tensor(ROWS[0]["u"]))


def test_rho_one_skips_demo():
    cfg = replace(Config(), candidates=2)
    e = FakeEngine()
    def forbidden(*args, **kwargs):
        raise AssertionError("rho=1 must not compute demo gradient")
    e.demo_backward = forbidden
    update(e, torch.optim.SGD([e.p], lr=.01), ROWS, Sampler(2), cfg, 0, 9., 1.)


def test_resume_is_identical_to_uninterrupted(tmp_path, monkeypatch):
    import dig_rl.train as module
    cfg = replace(Config(), candidates=3, max_steps=4, save_every=1, budget_seconds=1e6)
    decision = dict(a=.7, rho=.5, overhead=0., effective_arm="dig")
    expected, interrupted = FakeEngine(), FakeEngine()
    train(expected, ROWS, cfg, tmp_path / "full", {"test": "identity"}, decision)
    real_update = module.update
    def stop(engine, optimizer, rows, sampler, cfg, step, a, rho):
        if step == 2:
            raise RuntimeError("simulated interruption")
        return real_update(engine, optimizer, rows, sampler, cfg, step, a, rho)
    monkeypatch.setattr(module, "update", stop)
    with pytest.raises(RuntimeError):
        train(interrupted, ROWS, cfg, tmp_path / "resume", {"test": "identity"}, decision)
    monkeypatch.setattr(module, "update", real_update)
    restored = FakeEngine()
    train(restored, ROWS, cfg, tmp_path / "resume", {"test": "identity"}, decision, resume=True)
    torch.testing.assert_close(restored.p, expected.p, rtol=0, atol=0)
    records = [json.loads(x) for x in (tmp_path / "resume" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2, 3, 4]


def test_decision_requires_matching_audit_and_falls_back():
    ident = {"model": "x"}
    with pytest.raises(ValueError):
        resolve_decision("dig", None, ident)
    report = dict(identity=ident, enabled=False, audit_seconds=100.)
    result = resolve_decision("dig", report, ident)
    assert result["rho"] == 1 and result["a"] == 0 and result["overhead"] == 100.


def test_audit_work_is_charged_before_any_training(tmp_path):
    cfg = replace(Config(), max_steps=1, budget_seconds=10.)
    e = FakeEngine()
    decision = dict(a=.7, rho=.5, effective_arm="dig", overhead=100.,
                    setup_responses=16, setup_generated_tokens=1000, setup_scored_tokens=2000)
    train(e, ROWS, cfg, tmp_path / "exhausted", {"test": "identity"}, decision)
    summary = json.loads((tmp_path / "exhausted" / "summary.json").read_text())
    assert summary["steps"] == 0 and not e.calls
    assert summary["charged_responses"] == 16
    assert summary["charged_generated_tokens"] == 1000
