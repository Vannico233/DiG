"""Math-only generation with the same raw-softmax, per-response random streams."""
import random
import torch
from .round2_engine import BatchedEngine


class BufferedSampler:
    """Move uniforms once; check finite distributions once per completed batch.

    Invalid distributions are remembered on device and fail before any response
    is scored or used for training. Index clamping prevents invalid scatter on
    that error path; it does not change a valid inverse-CDF sample.
    """
    def __init__(self, seeds, cap, device):
        streams = [random.Random(int(s)) for s in seeds]
        self.uniforms = torch.tensor([[[s.random()] for s in streams] for _ in range(cap)],
                                     dtype=torch.float32, device=device).clamp_max_(1.-2**-24)
        self.invalid = torch.zeros((), dtype=torch.bool, device=device)
        self.step = 0

    def __call__(self, input_ids, scores):
        probabilities = scores.float().softmax(-1)
        self.invalid.logical_or_(~torch.isfinite(probabilities).all())
        cumulative = probabilities.cumsum(-1)
        cumulative[:, -1] = 1.
        choices = torch.searchsorted(cumulative.contiguous(), self.uniforms[self.step], right=True)
        choices.clamp_max_(scores.shape[-1]-1)
        self.step += 1
        return torch.full_like(scores, -torch.inf).scatter_(1, choices, 0.)

    def validate(self):
        if bool(self.invalid):
            raise FloatingPointError("Invalid sampling distribution; batch discarded")


class MathEngine(BatchedEngine):
    def sampling_processor(self, seeds, cap):
        return BufferedSampler(seeds, cap, self.device)

    def encoded(self, row):
        if row["images"]:
            raise ValueError("Math experiment requires text-only inputs")
        return super().encoded(row)
