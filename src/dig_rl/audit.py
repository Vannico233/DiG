"""Frozen-policy calibration and independent repeated-RL gradient audit."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import NormalDist
import time
import numpy as np
import torch
from .config import Config
from .io import atomic_save, read_rows, sha256, write_json, append_json, verify_data, experiment_identity
from .objective import cost_coefficients
from .runtime import Engine


def dot(x, y):
    return float(torch.dot(x.double(), y.double()))


def fit_scale(records):
    u = sum(r["uu"] for r in records)
    chi = sum(r["ug"] for r in records)
    return max(0.0, chi / u) if u > 0 else 0.0


def paired_statistics(g1, g2, u, a):
    return dict(uu=dot(u, u), ug=0.5 * dot(u, g1 + g2),
                cross=dot(g1, g2), nu=0.5 * dot(g1 - g2, g1 - g2),
                epsilon=dot(g1 - a * u, g2 - a * u))


def between_variance(cross_sum, sum_dot, n):
    if n < 2:
        raise ValueError("At least two independent questions required")
    return cross_sum / n - (sum_dot - cross_sum) / (n * (n - 1))


def decide(records, tau, tau_loo, overhead, cfg):
    """Jackknife normal intervals are screening estimates, NOT finite-sample certificates."""
    n = len(records)
    keys = ("nu", "epsilon", "demo_seconds", "rl_seconds")
    values = np.array([[r[k] for k in keys] for r in records], dtype=float)
    mean = values.mean(0)
    loo = (values.sum(0) - values) / (n - 1)
    charge = overhead / (cfg.max_steps * cfg.candidates)
    nu, eps, demo, c = mean
    d = demo + charge
    z = NormalDist().inv_cdf(1 - cfg.gate_alpha / len(cfg.rho_grid))
    candidates = []
    for rho in cfg.rho_grid:
        f, base = cost_coefficients(tau, nu, eps, d, c, rho)
        gain = base - f
        leave_gains = []
        for t, (v, e, dd, cc) in zip(tau_loo, loo):
            ff, bb = cost_coefficients(t, v, e, dd + charge, cc, rho)
            leave_gains.append(bb - ff)
        pseudo = n * gain - (n - 1) * np.array(leave_gains)
        se = float(np.std(pseudo, ddof=1) / math.sqrt(n))
        lower = gain - z * se
        passed = bool(tau > 0 and nu >= 0 and eps >= 0 and eps < tau and base > 0
                      and lower > cfg.gate_relative_margin * base)
        candidates.append(dict(rho=rho, F=f, F_RL=base, gain=gain, gain_lower_approx=lower,
                               jackknife_se=se, ratio=f / base if base > 0 else None, passed=passed))
    passing = [r for r in candidates if r["passed"]]
    best = min(passing, key=lambda r: r["F"]) if passing else None
    return dict(enabled=best is not None, rho=best["rho"] if best else 1.0,
                tau2=float(tau), nu2=float(nu), epsilon2=float(eps),
                d_seconds=float(d), c_seconds=float(c), audit_amortization_seconds=charge,
                candidates=candidates,
                uncertainty="Bonferroni-adjusted normal jackknife SCREENING interval; not a rigorous certificate",
                reason="positive held-out cost margin" if best else "insufficient held-out evidence; use full RLOO")


def measured(engine, row, seed, a, progress=None):
    report = progress or (lambda stage: None)
    report("demo_backward")
    engine.zero_grad()
    demo = engine.demo_backward(row)
    u = engine.gradient()
    gs, times, responses, tokens = [], [], 0, 0
    scored_tokens = demo["scored_tokens"]
    for j in range(2):
        report("rollout_"+str(j+1))
        sequences, rewards, gen = engine.rollout(row, seed + j * 1000003)
        engine.zero_grad()
        report("rl_backward_"+str(j+1))
        score = engine.rl_backward(row, sequences, rewards)
        gs.append(engine.gradient())
        scored_tokens += score["scored_tokens"]
        times.append(gen["seconds"] + score["seconds"])
        responses += gen["responses"]
        tokens += gen["generated_tokens"]
    result = paired_statistics(gs[0], gs[1], u, a)
    result.update(id=row["id"], demo_seconds=demo["seconds"], rl_seconds=float(np.mean(times)),
                  responses=responses, generated_tokens=tokens, scored_tokens=scored_tokens)
    return result, gs


def load_checkpoint(engine, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"]["model"] != engine.model_id:
        raise ValueError("Checkpoint base model identity differs")
    engine.load_parameters(state["parameters"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--config", default="configs/qwen35_4b.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial-checkpoint", type=Path)
    args = p.parse_args()
    cfg = Config.load(args.config)
    verify_data(args.data_dir, cfg)
    if args.output.exists():
        raise ValueError("Use a fresh audit output directory")
    calibration = read_rows(args.data_dir / "calibration.jsonl")
    evaluation = read_rows(args.data_dir / "audit.jsonl")
    if len(calibration) != cfg.calibration_size or len(evaluation) != cfg.audit_size:
        raise ValueError("Prepared split sizes differ from audit config")
    if {r["id"] for r in calibration} & {r["id"] for r in evaluation}:
        raise ValueError("Calibration/evaluation overlap")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    engine = Engine(args.model_dir, cfg)
    verify_data(args.data_dir, cfg, engine.model_id)
    if args.initial_checkpoint:
        load_checkpoint(engine, args.initial_checkpoint)
    records = []
    for j, row in enumerate(calibration):
        record, _ = measured(engine, row, cfg.seed + 10007 * j, 0)
        records.append(record)
        append_json(args.output / "calibration.jsonl", record)
        print(f"calibration {j + 1}/{len(calibration)}", flush=True)
    a = fit_scale(records)
    evaluated, sum1, sum2 = [], None, None
    vectors = args.output / "vectors"
    vectors.mkdir()
    for j, row in enumerate(evaluation):
        record, (g1, g2) = measured(engine, row, cfg.seed + 100000000 + 10007 * j, a)
        evaluated.append(record)
        sum1 = g1.double() if sum1 is None else sum1 + g1.double()
        sum2 = g2.double() if sum2 is None else sum2 + g2.double()
        atomic_save(vectors / f"{j}.pt", (g1, g2))
        append_json(args.output / "evaluation.jsonl", record)
        print(f"audit {j + 1}/{len(evaluation)}", flush=True)
    n = len(evaluated)
    cross_sum = sum(r["cross"] for r in evaluated)
    sum_dot = dot(sum1, sum2)
    tau = between_variance(cross_sum, sum_dot, n)
    tau_loo = []
    for j, r in enumerate(evaluated):
        path = vectors / f"{j}.pt"
        g1, g2 = torch.load(path, weights_only=True, map_location="cpu")
        leave_dot = sum_dot - dot(g1, sum2) - dot(sum1, g2) + r["cross"]
        tau_loo.append(between_variance(cross_sum - r["cross"], leave_dot, n - 1))
        path.unlink()  # Only scratch files produced by this run; raw scalar audit records remain.
    vectors.rmdir()
    overhead = time.monotonic() - start
    decision = decide(evaluated, tau, tau_loo, overhead, cfg)
    decision.update(a=a, audit_seconds=overhead, config=asdict(cfg),
        identity=experiment_identity(cfg, engine.model_id, args.data_dir, args.initial_checkpoint),
        versions=engine.runtime, trainable_parameters=sum(p.numel() for _, p in engine.params),
        audit_responses=sum(r["responses"] for r in records + evaluated),
        audit_tokens=sum(r["generated_tokens"] for r in records + evaluated),
        audit_scored_tokens=sum(r["scored_tokens"] for r in records + evaluated))
    write_json(args.output / "decision.json", decision)
    print(json.dumps({k: decision[k] for k in ("enabled", "a", "rho", "reason")}), flush=True)


if __name__ == "__main__":
    main()
