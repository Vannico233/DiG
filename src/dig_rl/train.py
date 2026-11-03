"""On-policy DiG / full-RLOO training, matched plain SGD, resumable full state."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import torch
from .audit import load_checkpoint
from .config import Config
from .io import append_json, atomic_save, read_rows, write_json, verify_data, experiment_identity
from .objective import Sampler, coefficients
from .runtime import Engine


def identity(engine, cfg, data_dir, initial_checkpoint=None):
    return experiment_identity(cfg, engine.model_id, data_dir, initial_checkpoint)


def resolve_decision(arm, report, expected):
    setup = dict(setup_responses=0, setup_generated_tokens=0, setup_scored_tokens=0)
    if arm == "rloo":
        return dict(effective_arm="rloo", a=0.0, rho=1.0, overhead=0.0, **setup)
    if report is None or report["identity"] != expected:
        raise ValueError("DiG requires an audit at the SAME model, seed, config, code and dataset state")
    setup = dict(setup_responses=report.get("audit_responses", 0),
                 setup_generated_tokens=report.get("audit_tokens", 0),
                 setup_scored_tokens=report.get("audit_scored_tokens", 0))
    if not report["enabled"]:
        return dict(effective_arm="rloo_fallback", a=0.0, rho=1.0, overhead=report["audit_seconds"], **setup)
    a, rho = float(report["a"]), float(report["rho"])
    coefficients(a, rho, True)
    return dict(effective_arm="dig", a=a, rho=rho, overhead=report["audit_seconds"], **setup)


def update(engine, optimizer, rows, sampler, cfg, step, a, rho):
    # Complete candidate/mask/seed list exists BEFORE the first generation call.
    draws = sampler.draw(len(rows), cfg.candidates, rho)
    optimizer.zero_grad(set_to_none=True)
    lr = cfg.learning_rate / math.sqrt(step + 1)
    for group in optimizer.param_groups:
        group["lr"] = lr
    totals = dict(selected=0, responses=0, generated_tokens=0, scored_tokens=0,
                  demo_seconds=0.0, rl_seconds=0.0, parsed=0, truncated=0, reward_sum=0.0,
                  objective=0.0)
    for index, selected, seed in draws:
        row = rows[index]
        demo_weight, rl_weight = coefficients(a, rho, selected)
        if demo_weight != 0:
            measured = engine.demo_backward(row, demo_weight / cfg.candidates)
            totals["objective"] += demo_weight * measured["loss"] / cfg.candidates
            totals["demo_seconds"] += measured["seconds"]
            totals["scored_tokens"] += measured["scored_tokens"]
        if selected:
            sequences, rewards, gen = engine.rollout(row, seed)
            measured = engine.rl_backward(row, sequences, rewards, rl_weight / cfg.candidates)
            totals["objective"] += rl_weight * measured["loss"] / cfg.candidates
            totals["rl_seconds"] += gen["seconds"] + measured["seconds"]
            totals["scored_tokens"] += measured["scored_tokens"]
            for key in ("responses", "generated_tokens", "parsed", "truncated"):
                totals[key] += gen[key]
            totals["reward_sum"] += float(rewards.sum())
            totals["selected"] += 1
    # Check without transferring the entire gradient to CPU during every update.
    norm2 = 0.0
    for _, param in engine.params:
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError("Nonfinite gradient; model was NOT updated")
            norm2 += float(param.grad.detach().double().square().sum())
    optimizer.step()  # No momentum, weight decay, PPO clipping, or gradient clipping.
    totals.update(step=step + 1, learning_rate=lr, gradient_norm=math.sqrt(norm2),
                  candidate_ids=[rows[i]["id"] for i, _, _ in draws],
                  selected_mask=[bool(z) for _, z, _ in draws],
                  candidates=cfg.candidates,
                  sampled_reward=totals["reward_sum"] / totals["responses"] if totals["responses"] else None)
    return totals


def train(engine, rows, cfg, output, ident, decision, resume=False):
    output = Path(output)
    optimizer = torch.optim.SGD([p for _, p in engine.params], lr=cfg.learning_rate,
                                momentum=0.0, weight_decay=0.0)
    sampler = Sampler(cfg.seed)
    step, elapsed = 0, float(decision["overhead"])
    cumulative = dict(responses=0, generated_tokens=0, scored_tokens=0, candidates=0)
    if resume:
        state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state["identity"] != ident or state["decision"] != decision:
            raise ValueError("Resume config, dataset, code, or decision mismatch")
        engine.load_parameters(state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, elapsed, cumulative = state["step"], state["elapsed"], state["cumulative"]
        timing_path = output / "checkpoint_timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            if timing["step"] == step:
                elapsed = max(elapsed, timing["elapsed"])
        # Discard only log records newer than the last durable checkpoint.
        metrics = output / "metrics.jsonl"
        if metrics.exists():
            kept = [json.loads(line) for line in metrics.read_text().splitlines() if line]
            from .io import write_jsonl
            write_jsonl(metrics, [r for r in kept if r["step"] <= step])
    elif output.exists():
        raise ValueError("Output already exists; use --resume for the exact same run")
    else:
        output.mkdir(parents=True)
    write_json(output / "protocol.json", dict(config=asdict(cfg), identity=ident, decision=decision,
        optimizer="SGD; no clipping/momentum/weight decay", schedule="eta0/sqrt(step+1)",
        sampling="temperature=1, top_p=1, top_k=0; no additional generation processors",
        thinking=False, reward="final boxed numeric answer, tolerance=1e-6",
        budget="training wall time plus DiG audit setup; model load/evaluation excluded; batch overshoot logged",
        checkpoint_selection="final checkpoint; no test-driven selection"))
    elapsed_at_start, segment_start = elapsed, time.monotonic()
    def save():
        atomic_save(output / "checkpoint.pt", dict(identity=ident, decision=decision,
            parameters=engine.parameter_state(), optimizer=optimizer.state_dict(), sampler=sampler.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            step=step, elapsed=elapsed, cumulative=cumulative))
        write_json(output / "checkpoint_timing.json", dict(step=step,
                   elapsed=elapsed_at_start + time.monotonic() - segment_start))
    while step < cfg.max_steps and elapsed < cfg.budget_seconds:
        engine.sync(); started = time.monotonic()
        record = update(engine, optimizer, rows, sampler, cfg, step, decision["a"], decision["rho"])
        engine.sync()
        elapsed = elapsed_at_start + time.monotonic() - segment_start
        step += 1
        for k in cumulative:
            cumulative[k] += record[k]
        record.update(elapsed_seconds=elapsed, cumulative=cumulative.copy(), effective_arm=decision["effective_arm"])
        append_json(output / "metrics.jsonl", record)
        print(json.dumps({k: record[k] for k in ("step", "selected", "sampled_reward", "gradient_norm", "elapsed_seconds")}), flush=True)
        if step % cfg.save_every == 0:
            save()
        elapsed = elapsed_at_start + time.monotonic() - segment_start
    save()
    elapsed = elapsed_at_start + time.monotonic() - segment_start
    write_json(output / "summary.json", dict(steps=step, elapsed_seconds=elapsed, cumulative=cumulative,
        charged_responses=cumulative["responses"] + decision.get("setup_responses", 0),
        charged_generated_tokens=cumulative["generated_tokens"] + decision.get("setup_generated_tokens", 0),
        charged_scored_tokens=cumulative["scored_tokens"] + decision.get("setup_scored_tokens", 0),
        decision=decision, budget_overshoot_seconds=max(0, elapsed - cfg.budget_seconds),
        stop_reason="budget" if elapsed >= cfg.budget_seconds else "max_steps",
        note="Equal step caps do not imply equal compute. Compare charged seconds and evaluation results."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--config", default="configs/qwen35_4b.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=["rloo", "dig"], required=True)
    p.add_argument("--audit-file", type=Path)
    p.add_argument("--initial-checkpoint", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = Config.load(args.config)
    verify_data(args.data_dir, cfg)
    engine = Engine(args.model_dir, cfg)
    verify_data(args.data_dir, cfg, engine.model_id)
    ident = identity(engine, cfg, args.data_dir, args.initial_checkpoint)
    report = json.loads(args.audit_file.read_text()) if args.audit_file else None
    decision = resolve_decision(args.arm, report, ident)
    if args.initial_checkpoint and not args.resume:
        load_checkpoint(engine, args.initial_checkpoint)
    train(engine, read_rows(args.data_dir / "train.jsonl"), cfg, args.output, ident, decision, args.resume)


if __name__ == "__main__":
    main()
