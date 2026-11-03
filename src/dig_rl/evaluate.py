"""Fixed-checkpoint, prompt-level stochastic pass@1 evaluation (not best-of-G)."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .audit import load_checkpoint
from .config import Config
from .io import append_json, read_rows, sha256, verify_data, write_json
from .runtime import Engine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--config", default="configs/qwen35_4b.json")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--max-problems", type=int, default=256)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.samples < 1 or args.max_problems < 1:
        raise ValueError("Fresh output and positive evaluation sizes required")
    cfg = Config.load(args.config)
    verify_data(args.data_dir, cfg)
    rows = read_rows(args.data_dir / f"{args.split}.jsonl")[:args.max_problems]
    engine = Engine(args.model_dir, cfg)
    verify_data(args.data_dir, cfg, engine.model_id)
    if args.checkpoint:
        load_checkpoint(engine, args.checkpoint)
    args.output.mkdir(parents=True)
    scores, generated, truncated = [], 0, 0
    start = time.monotonic()
    for j, row in enumerate(rows):
        sequences, rewards, stats = engine.rollout(row, cfg.seed + 200000000 + 10007 * j, args.samples)
        score = float(rewards.mean())
        scores.append(score)
        generated += stats["generated_tokens"]
        truncated += stats["truncated"]
        prompt_len = engine.tensors(row)[0].shape[1]
        texts = [engine.tokenizer.decode(s[0, prompt_len:], skip_special_tokens=True) for s in sequences]
        append_json(args.output / "predictions.jsonl", dict(id=row["id"], score=score, texts=texts, **stats))
        print(f"evaluation {j + 1}/{len(rows)} score={score:.3f}", flush=True)
    rng = np.random.default_rng(20261103)
    means = np.mean(rng.choice(scores, size=(2000, len(scores)), replace=True), axis=1)
    write_json(args.output / "summary.json", dict(
        mean_success=float(np.mean(scores)), prompt_bootstrap_95=np.quantile(means, [0.025, 0.975]).tolist(),
        prompts=len(rows), samples_per_prompt=args.samples, protocol="raw-policy stochastic pass@1, thinking off",
        generated_tokens=generated, truncated_fraction=truncated / (len(rows) * args.samples),
        elapsed_seconds=time.monotonic() - start,
        data_sha256=sha256(args.data_dir / f"{args.split}.jsonl"),
        checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
        note="Fixed subset, prompt-cluster bootstrap; not a proof of generalization or paired improvement"))


if __name__ == "__main__":
    main()
