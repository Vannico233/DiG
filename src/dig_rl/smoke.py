"""32-prompt plumbing check with an explicitly UNCALIBRATED predictor; not an experiment."""
import argparse
from pathlib import Path
import torch
from .config import Config
from .io import read_rows, verify_data, append_json, write_json
from .objective import Sampler
from .runtime import Engine
from .train import update


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--config", default="configs/qwen35_4b.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh smoke output")
    cfg = Config.load(args.config)
    verify_data(args.data_dir, cfg)
    rows = read_rows(args.data_dir / "calibration.jsonl")[:32]
    if len(rows) < 32:
        raise ValueError("The smoke protocol requires 32 eligible calibration prompts")
    engine = Engine(args.model_dir, cfg)
    verify_data(args.data_dir, cfg, engine.model_id)
    optimizer = torch.optim.SGD([p for _, p in engine.params], lr=cfg.learning_rate)
    sampler = Sampler(cfg.seed)
    args.output.mkdir(parents=True)
    for step in range(max(1, (32 + cfg.candidates - 1) // cfg.candidates)):
        record = update(engine, optimizer, rows, sampler, cfg, step, a=1.0, rho=0.5)
        append_json(args.output / "metrics.jsonl", record)
    write_json(args.output / "summary.json", dict(ok=True, eligible_prompts=len(rows),
        purpose="Plumbing only: a=1 and rho=.5 deliberately uncalibrated; no checkpoint is reused",
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30))


if __name__ == "__main__":
    main()
