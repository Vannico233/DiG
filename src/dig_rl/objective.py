"""Pure estimator algebra. No clipping, group standardization, or extra SFT."""
import math
import random
import torch


def rloo_advantages(rewards):
    r = rewards.detach().float()
    if r.ndim != 1 or r.numel() < 2 or not torch.isfinite(r).all():
        raise ValueError("Expected a finite 1-D group with G >= 2")
    return r - (r.sum() - r) / (r.numel() - 1)


def coefficients(a, rho, selected):
    if not math.isfinite(a) or a < 0 or not math.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("a >= 0 and 0 < rho <= 1 required")
    if selected not in (0, 1, False, True):
        raise ValueError("selected must be Bernoulli")
    rl = int(selected) / rho
    return a * (1 - rl), rl


def candidate_loss(demo_loss, rl_loss, *, a, rho, selected):
    d, r = coefficients(a, rho, selected)
    if selected and rl_loss is None:
        raise ValueError("Selected candidates require an RL estimate")
    return d * demo_loss + (r * rl_loss if selected else 0)


class Sampler:
    """Separate RNG streams; selection happens before any current rollout."""
    def __init__(self, seed):
        self.questions = random.Random(seed)
        self.checks = random.Random(seed + 9173)
        self.responses = random.Random(seed + 27183)

    def draw(self, size, batch, rho):
        if size < 1 or batch < 1 or not 0 < rho <= 1:
            raise ValueError("Invalid sampling dimensions/probability")
        return [(self.questions.randrange(size), self.checks.random() < rho,
                 self.responses.randrange(2**63 - 1)) for _ in range(batch)]

    def state_dict(self):
        return {k: getattr(self, k).getstate() for k in ("questions", "checks", "responses")}

    def load_state_dict(self, state):
        for k, v in state.items():
            getattr(self, k).setstate(v)


def cost_coefficients(tau, nu, epsilon, d, c, rho):
    if not 0 < rho <= 1 or c <= 0 or d < 0:
        raise ValueError("Invalid costs or probability")
    variance = tau + nu / rho + (1 / rho - 1) * epsilon
    return (d + rho * c) * variance, c * (tau + nu)
