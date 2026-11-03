import itertools
import random
import pytest
import torch
from dig_rl.objective import candidate_loss, coefficients, cost_coefficients, rloo_advantages, Sampler


def test_negative_demo_coefficient_and_autograd():
    theta = torch.tensor([0.3, -0.2], requires_grad=True)
    u, g = torch.tensor([2., -1.]), torch.tensor([1., 3.])
    loss = candidate_loss(-theta @ u, -theta @ g, a=.7, rho=.25, selected=True)
    loss.backward()
    assert coefficients(.7, .25, True) == pytest.approx((-2.1, 4))
    torch.testing.assert_close(-theta.grad, .7 * u + 4 * (g - .7 * u))


@pytest.mark.parametrize("rho", [1., .5, .125])
def test_exact_rloo_mean_and_total_variance(rho):
    # Exhaust all responses for G=2, two prompts, and both Bernoulli outcomes.
    ps = [torch.tensor([.8, .01, .19], dtype=torch.float64),
          torch.tensor([.2, .3, .5], dtype=torch.float64)]
    reward = torch.tensor([1., 1., 0.], dtype=torch.float64)
    mus, sigmas, errors, outcomes = [], [], [], []
    for i, p in enumerate(ps):
        mu = p * (reward - p @ reward)
        h = .7 * (torch.eye(3, dtype=torch.float64)[1] - p)
        mus.append(mu); errors.append((mu - h).square().sum())
        sigma = 0.
        mean = torch.zeros(3, dtype=torch.float64)
        for a, b in itertools.product(range(3), repeat=2):
            prob = float(p[a] * p[b])
            adv = rloo_advantages(reward[[a, b]]).double()
            scores = torch.eye(3, dtype=torch.float64)[[a, b]] - p
            g = (adv[:, None] * scores).mean(0)
            mean += prob * g
            sigma += prob * (g - mu).square().sum()
            for z, zp in ((0, 1 - rho), (1, rho)):
                outcomes.append((prob * zp / 2, h + z / rho * (g - h)))
        torch.testing.assert_close(mean, mu, atol=1e-12, rtol=1e-12)
        sigmas.append(sigma)
    overall = torch.stack(mus).mean(0)
    empirical = sum(prob * g for prob, g in outcomes)
    variance = sum(prob * (g - overall).square().sum() for prob, g in outcomes)
    tau = torch.stack([(m - overall).square().sum() for m in mus]).mean()
    theory = tau + sum(sigmas) / (2 * rho) + (1 / rho - 1) * sum(errors) / 2
    torch.testing.assert_close(empirical, overall, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(variance, theory, atol=1e-12, rtol=1e-12)


def test_sampler_restores_and_keeps_duplicate_positions():
    sampler = Sampler(12)
    sampler.draw(4, 10, .5)
    state = sampler.state_dict()
    expected = sampler.draw(1, 8, .2)
    restored = Sampler(999)
    restored.load_state_dict(state)
    assert restored.draw(1, 8, .2) == expected
    assert len(set(seed for _, _, seed in expected)) == 8


def test_zero_selected_is_not_forced():
    sampler = Sampler(1)
    assert not any(z for _, z, _ in sampler.draw(2, 4, 1e-12))


def test_cost_identity_and_perfect_prediction_can_fail():
    for rho in (.1, .5, 1.):
        tau, nu, eps, d, c = .1, 4., 0., .1, 1.
        f, full = cost_coefficients(tau, nu, eps, d, c, rho)
        v = tau + nu / rho + (1 / rho - 1) * eps
        assert f - full == pytest.approx(d * v - c * (1 - rho) * (tau - eps))
        assert f > full


@pytest.mark.parametrize("a,rho,z", [(-1,.5,1), (1,0,1), (1,1.1,1), (float('nan'),.5,1), (1,.5,2)])
def test_invalid_coefficients(a,rho,z):
    with pytest.raises(ValueError):
        coefficients(a,rho,z)
