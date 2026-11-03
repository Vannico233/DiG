"""Combine real audit/train/evaluation outputs; missing or skipped arms stay explicit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .io import write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    rows, comparisons = [], {}
    for dataset in ("gsm8k", "math_numeric"):
        audit = read(args.run / "audit" / dataset / "decision.json")
        for arm in ("rloo", "dig"):
            train = read(args.run / "train" / f"{dataset}_{arm}" / "summary.json")
            ev = read(args.run / "eval" / f"{dataset}_{arm}" / "summary.json")
            rows.append(dict(dataset=dataset, arm=arm, audit_enabled=audit.get("enabled"),
                a=audit.get("a") if arm == "dig" else 0, rho=audit.get("rho") if arm == "dig" else 1,
                effective_arm=train.get("decision", {}).get("effective_arm"), steps=train.get("steps"),
                charged_seconds=train.get("elapsed_seconds"), stop_reason=train.get("stop_reason"),
                responses=train.get("charged_responses"),
                generated_tokens=train.get("charged_generated_tokens"),
                success=ev.get("mean_success"), truncation=ev.get("truncated_fraction"),
                status="evaluated" if ev else ("trained" if train else "missing_or_skipped")))
        paths = [args.run / "eval" / f"{dataset}_{arm}" / "predictions.jsonl" for arm in ("rloo", "dig")]
        if all(p.exists() for p in paths):
            pair = [{r["id"]: r["score"] for r in (json.loads(x) for x in p.read_text().splitlines())} for p in paths]
            if pair[0].keys() != pair[1].keys():
                raise ValueError("Cannot pair different evaluation prompt sets")
            diffs = [pair[1][k] - pair[0][k] for k in pair[0]]
            rng = np.random.default_rng(20261103)
            bs = rng.choice(diffs, size=(2000, len(diffs)), replace=True).mean(axis=1)
            comparisons[dataset] = dict(dig_minus_rloo=float(np.mean(diffs)),
                paired_prompt_bootstrap_95=np.quantile(bs, [.025, .975]).tolist(),
                warning="Single seed, possibly different charged compute; inspect budgets before interpreting")
    with (args.run / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(args.run / "comparisons.json", comparisons)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
