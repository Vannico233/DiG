"""Round-two protocol, independent of the archived pilot's config and identities."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from pathlib import Path

DATASETS = ("gsm8k", "math", "geo3k", "scienceqa_image")


@dataclass(frozen=True)
class Protocol:
    dataset: str
    seed: int = 13
    max_new_tokens: int = 4096
    max_input_tokens: int = 8192
    loss_normalizer: int = 2048
    group_size: int = 8
    candidates: int = 32
    active_sequences: int = 16
    max_pixels: int = 1048576
    optimizer: str = "adamw"
    learning_rate: float = 1e-6
    max_steps: int = 32
    train_seconds: float = 7200
    calibration_size: int = 32
    validation_size: int = 64
    rho: float = .5
    thinking: bool = False

    def __post_init__(self):
        if self.dataset not in DATASETS or self.optimizer not in ("adamw", "sgd"):
            raise ValueError("Unknown dataset/optimizer")
        for key in ("seed", "max_new_tokens", "max_input_tokens", "loss_normalizer", "group_size",
                    "candidates", "active_sequences", "max_pixels", "max_steps", "calibration_size", "validation_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("Expected positive integer: " + key)
        if self.group_size < 2 or self.max_new_tokens < self.loss_normalizer:
            raise ValueError("G >= 2 and output limit >= fixed loss normalizer required")
        if not 0 < self.rho <= 1 or any(not math.isfinite(x) or x <= 0 for x in (self.learning_rate, self.train_seconds)):
            raise ValueError("Invalid probability, learning rate or budget")

    def fingerprint(self):
        return object_hash(asdict(self))


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def boxed_text(text):
    if text.rfind("<think>") > text.rfind("</think>"):
        return None
    text = text.rsplit("</think>", 1)[-1]
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, end = 1, start
    while end < len(text) and depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end - 1].strip() if depth == 0 else None


def score_answer(text, row):
    answer = boxed_text(text)
    if answer is None:
        return 0., False
    target = row["answer"].strip()
    if row["kind"] == "choice":
        parsed = re.fullmatch(r"[A-Z]", answer.upper()) is not None
        return float(parsed and answer.upper() == target), parsed
    if re.sub(r"\s+", "", answer) == re.sub(r"\s+", "", target):
        return 1., True
    if row["kind"] == "numeric":
        from .data import reward
        return reward(text, target), True
    from math_verify import parse, verify
    # Only the final box is graded; earlier correct guesses do not earn reward.
    gold = parse("$" + target + "$")
    predicted = parse("$" + answer + "$")
    return float(bool(gold and predicted and verify(gold, predicted))), bool(predicted)


def image_path(root, relative):
    path, root = (Path(root) / relative).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Invalid local image path")
    return path


def jobs(protocols, gpus=("0", "1", "2", "3"), slots=2):
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(not str(g).isdigit() for g in gpus):
        raise ValueError("Four distinct GPU indices required")
    if slots not in (1, 2):
        raise ValueError("One or two workers per GPU supported")
    result = []
    for dataset, gpu in zip(DATASETS, gpus):
        cfg = protocols[dataset]
        for arm in ("rloo", "dig"):
            result.append(dict(name=f"{dataset}_{arm}", dataset=dataset, arm=arm, gpu=str(gpu),
                               config=asdict(cfg), rho=1. if arm == "rloo" else cfg.rho))
    return result
