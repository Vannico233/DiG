from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    horizon: int = 2048
    max_prompt_tokens: int = 2048
    group_size: int = 4
    rollout_batch_size: int = 4
    candidates: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 0.01
    max_steps: int = 200
    budget_seconds: float = 14400.0
    save_every: int = 25
    seed: int = 13
    calibration_size: int = 32
    audit_size: int = 96
    validation_size: int = 256
    split_seed: int = 20261103
    rho_grid: tuple = (0.125, 0.25, 0.5, 0.75)
    gate_relative_margin: float = 0.05
    gate_alpha: float = 0.05
    require_fast_kernels: bool = True

    def __post_init__(self):
        for key in ("horizon", "max_prompt_tokens", "group_size", "rollout_batch_size", "candidates",
                    "lora_rank", "lora_alpha", "max_steps", "save_every", "calibration_size", "audit_size", "validation_size"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.group_size < 2 or self.audit_size < 4 or self.rollout_batch_size > self.group_size:
            raise ValueError("G >= 2, audit_size >= 4, rollout_batch_size <= G required")
        for key in ("learning_rate", "budget_seconds"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")
        if not 0 < self.gate_alpha < 0.5 or not 0 <= self.gate_relative_margin < 1:
            raise ValueError("Invalid gate settings")
        if not self.rho_grid or any(not 0 < x < 1 for x in self.rho_grid):
            raise ValueError("rho_grid must contain probabilities strictly between 0 and 1")
        if len(set(self.rho_grid)) != len(self.rho_grid):
            raise ValueError("Repeated rho")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "rho_grid" in data:
            data["rho_grid"] = tuple(data["rho_grid"])
        return cls(**data)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
