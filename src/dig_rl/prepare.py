"""Resolve immutable HF revisions, download once, prepare disjoint token-eligible splits."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from .config import Config
from .data import normalize, split_rows
from .io import sha256, write_json, write_jsonl
from .hf_downloads import hf_retry, resolve_revisions
from .model_downloads import download_model_snapshot
from .runtime import prompt_ids, demo_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=Path("assets"))
    p.add_argument("--config", default="configs/qwen35_4b.json")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--gsm8k-revision", default="main")
    p.add_argument("--math-revision", default="main")
    p.add_argument("--network-attempts", type=int, default=4)
    p.add_argument("--model-transport", choices=("curl", "hub"),
                   default=os.environ.get("DIG_MODEL_TRANSPORT", "curl"))
    p.add_argument("--metadata-only", action="store_true", help="Resolve/save repository revisions without downloading model weights")
    args = p.parse_args()
    if args.network_attempts < 1:
        p.error("--network-attempts must be positive")
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from datasets import load_dataset
    cfg, api = Config.load(args.config), HfApi()
    args.assets.mkdir(parents=True, exist_ok=True)
    lock_path = args.assets / "download_lock.json"
    requested = {"model": ("Qwen/Qwen3.5-4B", args.model_revision),
                 "gsm8k": ("openai/gsm8k", args.gsm8k_revision),
                 "math_numeric": ("jeggers/competition_math", args.math_revision)}
    lock = resolve_revisions(api, requested, lock_path, attempts=args.network_attempts)
    if args.metadata_only:
        print("Repository revisions saved. Model weights were not downloaded.", flush=True)
        return
    model_dir = args.assets / "model"
    download_model_snapshot(api, requested["model"][0], lock["resolved"]["model"],
                            model_dir, args.network_attempts, args.model_transport)
    write_json(model_dir / "dig_model.json", {"repo": requested["model"][0], "revision": lock["resolved"]["model"]})
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    for name, subset in (("gsm8k", "main"), ("math_numeric", "numeric")):
        dest = args.assets / "data" / name
        if (dest / "manifest.json").exists():
            manifest = json.loads((dest / "manifest.json").read_text())
            keys = ("horizon", "max_prompt_tokens", "calibration_size", "audit_size", "validation_size", "split_seed")
            if any(manifest["config"][k] != asdict(cfg)[k] for k in keys):
                raise ValueError("Prepared split settings changed; use a new assets directory")
            for split, entry in manifest["files"].items():
                if sha256(dest / f"{split}.jsonl") != entry["sha256"]:
                    raise ValueError("Prepared data checksum mismatch")
            print(f"Verified existing {name}", flush=True)
            continue
        source = hf_retry("Loading pinned dataset " + name, lambda: load_dataset(
            requested[name][0], subset, revision=lock["resolved"][name]), args.network_attempts)
        eligible, excluded = {}, []
        for split in ("train", "test"):
            eligible[split] = []
            for raw in source[split]:
                row = normalize(raw, name)
                if row is None:
                    continue
                prompt_len = prompt_ids(processor, row["question"]).shape[1]
                demo_len = demo_ids(processor.tokenizer, row["demo"]).shape[1]
                reason = "prompt_over_limit" if prompt_len > cfg.max_prompt_tokens else (
                    "demo_over_H" if split == "train" and demo_len > cfg.horizon else None)
                if reason:
                    excluded.append(dict(id=row["id"], split=split, reason=reason))
                    continue
                row.update(prompt_tokens=prompt_len, demo_tokens=demo_len)
                eligible[split].append(row)
        splits = split_rows(eligible["train"], eligible["test"], cfg)
        dest.mkdir(parents=True, exist_ok=True)
        files = {}
        for split, rows in splits.items():
            out = dest / f"{split}.jsonl"
            write_jsonl(out, rows)
            files[split] = dict(rows=len(rows), sha256=sha256(out))
        write_json(dest / "excluded.json", excluded)
        write_json(dest / "manifest.json", dict(source=requested[name][0], revision=lock["resolved"][name],
            subset=subset, model_id=sha256(model_dir / "dig_model.json"), config=asdict(cfg), files=files,
            eligible_before_dedup={k: len(v) for k, v in eligible.items()}, excluded=len(excluded)))
        print(f"Prepared {name}: { {k: v['rows'] for k, v in files.items()} }", flush=True)


if __name__ == "__main__":
    main()
