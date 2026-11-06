"""Bounded math ablations with durable optimizer, sampler AND predictor state."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import torch
from .io import atomic_save, append_json, write_json, write_jsonl, sha256
from .objective import Sampler
from .online_predictor import OnlinePredictor, update_online
from .round2_protocol import object_hash
from .round2_train import records, sha256_tensor


def train_math(engine, rows, cfg, directory, identity, arm, progress):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    optimizer = torch.optim.AdamW([p for _,p in engine.params], lr=cfg.learning_rate, weight_decay=0.)
    sampler = Sampler(cfg.seed)
    predictor = OnlinePredictor([n for n,_ in engine.params])
    initial = engine.parameter_state()
    binding = dict(identity=identity, config=asdict(cfg), arm=arm,
        initial=object_hash({n:sha256_tensor(p) for n,p in initial.items()}),
        predictor=dict(decay=predictor.decay, ridge=predictor.ridge, cap=predictor.cap))
    step, prior = 0, 0.
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["binding"] != binding:
            raise ValueError("Checkpoint protocol changed")
        engine.load_parameters(state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        predictor.load_state_dict(state["predictor"])
        torch.set_rng_state(state["rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, prior = state["step"], state["training_seconds"]
        old = records(directory / "metrics.jsonl", repair=True)
        if any(r["step"] > step for r in old):
            (directory / "metrics.jsonl").replace(directory / f"metrics.interrupted-{time.time_ns()}.jsonl")
            write_jsonl(directory / "metrics.jsonl", [r for r in old if r["step"] <= step])
        if (directory / "summary.json").exists():
            summary = json.loads((directory / "summary.json").read_text())
            if summary["checkpoint_sha256"] != sha256(checkpoint):
                raise ValueError("Completed checkpoint changed")
            return summary
    elif (directory / "metrics.jsonl").exists():
        raise ValueError("Metrics without checkpoint")
    write_json(directory / "protocol.json", dict(binding=binding,
        selection="Final checkpoint; no validation selection or early stopping",
        comparison="Matched candidate steps; separately report sampled tokens and total training wall time",
        caveat="Exploratory AdamW experiment; conditional gradient identity is not an accuracy/efficiency guarantee"))
    started = time.monotonic()
    def elapsed():
        return prior + time.monotonic()-started
    def save():
        atomic_save(checkpoint, dict(binding=binding, parameters=engine.parameter_state(),
            optimizer=optimizer.state_dict(), sampler=sampler.state_dict(), predictor=predictor.state_dict(),
            step=step, training_seconds=elapsed(), rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    if not checkpoint.exists():
        save()
    while step < cfg.max_steps and elapsed() < cfg.train_seconds:
        progress("training", step=step, total=cfg.max_steps, seconds=elapsed())
        record = update_online(engine, optimizer, rows, sampler, cfg, step, predictor, arm)
        step += 1
        record["training_seconds"] = elapsed()
        append_json(directory / "metrics.jsonl", record)
        save()
        print(f"step={step}/{cfg.max_steps} tokens={record['generated_tokens']} mixed={record['mixed_groups']}/{record['selected']}", flush=True)
    save()
    delta = sum(float((p.detach().cpu().double()-initial[n].double()).square().sum()) for n,p in engine.params)
    summary = dict(arm=arm, steps=step, training_seconds=elapsed(), parameter_delta_l2=math.sqrt(delta),
        checkpoint_sha256=sha256(checkpoint), stop_reason="max_steps" if step == cfg.max_steps else "budget",
        matched_steps_complete=step==cfg.max_steps, budget_overshoot_seconds=max(0., elapsed()-cfg.train_seconds),
        final_scales=predictor.scales(), predictor_updates=predictor.updates)
    write_json(directory / "summary.json", summary)
    return summary
