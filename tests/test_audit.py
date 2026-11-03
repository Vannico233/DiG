from dataclasses import replace
import pytest
import torch
from dig_rl.audit import fit_scale, paired_statistics, between_variance, decide
from dig_rl.config import Config


def test_calibration_scale_and_residual_identity():
    u = torch.tensor([1., 2.])
    mu = torch.tensor([2., 4.])
    noise = torch.tensor([.3, -.8])
    # Independently average all four combinations of zero-mean replicate noise.
    records = [paired_statistics(mu + s * noise, mu + t * noise, u, 2)
               for s in (-1, 1) for t in (-1, 1)]
    assert fit_scale(records) == pytest.approx(2)
    assert sum(r["epsilon"] for r in records) / 4 == pytest.approx(0, abs=1e-12)
    assert any(r["epsilon"] < 0 for r in records)  # Never silently clamp noisy estimates.


def test_between_variance_and_delete_one():
    x = torch.tensor([1., 3., 5., 7.], dtype=torch.float64)
    tau = between_variance(float(x.square().sum()), float(x.sum().square()), len(x))
    assert tau == pytest.approx(float(x.var(unbiased=True)))
    for i in range(4):
        left = torch.cat((x[:i], x[i + 1:]))
        loo = between_variance(float(x.square().sum() - x[i]**2),
                               float(x.sum()**2 - 2*x[i]*x.sum() + x[i]**2), 3)
        assert loo == pytest.approx(float(left.var(unbiased=True)))


def test_gate_pass_fail_and_negative_noise_estimates():
    cfg = replace(Config(), gate_relative_margin=.01)
    rows = [dict(nu=1., epsilon=.25, demo_seconds=.1, rl_seconds=1.) for _ in range(96)]
    assert decide(rows, 4., [4.] * 96, 0., cfg)["enabled"]
    assert not decide(rows, .1, [.1] * 96, 0., cfg)["enabled"]
    for row in rows:
        row["epsilon"] = -.01
    assert not decide(rows, 4., [4.] * 96, 0., cfg)["enabled"]


def test_audit_overhead_can_remove_advantage():
    cfg = Config()
    rows = [dict(nu=1., epsilon=.25, demo_seconds=.1, rl_seconds=1.) for _ in range(96)]
    assert not decide(rows, 4., [4.] * 96, 1e9, cfg)["enabled"]
