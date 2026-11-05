"""Batched DiG estimator with independent setup and resumable paired pilots."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import torch
from .objective import Sampler, coefficients
from .io import atomic_save, append_json, write_json, read_rows, sha256
from .round2_protocol import object_hash


def update(engine, optimizer, rows, sampler, cfg, step, a, rho):
    draws = sampler.draw(len(rows), cfg.candidates, rho)
    selected = [(rows[index], seed) for index, yes, seed in draws if yes]
    optimizer.zero_grad(set_to_none=True)
    lr = cfg.learning_rate / math.sqrt(step+1) if cfg.optimizer == "sgd" else cfg.learning_rate
    for group in optimizer.param_groups:
        group["lr"] = lr
    started = time.monotonic()
    batches = iter(engine.rollout_many(selected))
    generation = time.monotonic() - started
    totals = dict(generation_seconds=generation, backward_seconds=0., generated_tokens=0,
                  responses=0, scored_tokens=0, parsed=0, truncated=0, reward_sum=0.,
                  all_correct_groups=0, all_wrong_groups=0, mixed_groups=0)
    for index, yes, seed in draws:
        row = rows[index]
        demo_weight, rl_weight = coefficients(a, rho, yes)
        if demo_weight:
            record = engine.demo_backward(row, demo_weight / cfg.candidates)
            totals["backward_seconds"] += record["seconds"]
            totals["scored_tokens"] += record["scored_tokens"]
        if yes:
            group = next(batches)
            rewards = torch.tensor(group["rewards"])
            record = engine.rl_backward(row, group["sequences"], rewards, rl_weight / cfg.candidates)
            totals["backward_seconds"] += record["seconds"]
            totals["scored_tokens"] += record["scored_tokens"]
            for key in ("generated_tokens", "responses", "parsed", "truncated"):
                totals[key] += group[key]
            totals["reward_sum"] += sum(group["rewards"])
            key = "all_correct_groups" if bool((rewards == 1).all()) else (
                "all_wrong_groups" if bool((rewards == 0).all()) else "mixed_groups")
            totals[key] += 1
    norm2 = 0.
    for _, p in engine.params:
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError("Nonfinite gradient; no update committed")
            norm2 += float(p.grad.detach().double().square().sum())
    optimizer.step()
    totals.update(step=step+1, gradient_norm=math.sqrt(norm2), learning_rate=lr,
        candidate_ids=[rows[i]["id"] for i, _, _ in draws], selected_mask=[bool(z) for _, z, _ in draws],
        candidates=cfg.candidates, selected=len(selected))
    return totals


def records(path, repair=False):
    if not Path(path).exists():
        return []
    path = Path(path)
    data = path.read_bytes()
    lines = data.split(b"\n")
    if lines[-1]:
        # A killed writer may leave half a UTF-8/JSON record. Only a worker that
        # owns the file may repair it; a live status/summary reader never writes.
        tail = lines.pop()
        if repair:
            path.with_name(path.name+f".interrupted-{time.time_ns()}").write_bytes(data)
            path.write_bytes(b"\n".join(lines)+b"\n")
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(engine, rows, cfg, directory, model_identity, progress):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=model_identity, config=cfg.fingerprint(), ids=[r["id"] for r in rows],
                    decoding="greedy", samples=1)
    meta = directory / "protocol.json"
    if meta.exists() and json.loads(meta.read_text()) != protocol:
        raise ValueError("Evaluation protocol changed")
    write_json(meta, protocol)
    path = directory / "predictions.jsonl"
    done = records(path, repair=True)
    if [r["id"] for r in done] != [r["id"] for r in rows[:len(done)]] or len(done) > len(rows):
        raise ValueError("Evaluation is not an exact prefix")
    for start in range(len(done), len(rows), cfg.active_sequences):
        batch = rows[start:start+cfg.active_sequences]
        progress("evaluating", completed=start, total=len(rows))
        groups = engine.rollout_many([(r, cfg.seed+100000000+j) for j,r in enumerate(batch, start)], count=1, greedy=True)
        for row, group in zip(batch, groups):
            item = {k:v for k,v in group.items() if k != "sequences"}
            item.update(id=row["id"], score=group["rewards"][0])
            append_json(path, item)
            done.append(item)
    summary = dict(accuracy=sum(r["score"] for r in done)/len(done), prompts=len(done),
        truncated_fraction=sum(r["truncated"] for r in done)/len(done),
        parsed_fraction=sum(r["parsed"] for r in done)/len(done),
        generated_tokens=sum(r["generated_tokens"] for r in done),
        generation_seconds=sum(r["seconds"] for r in done), protocol=protocol)
    write_json(directory / "summary.json", summary)
    return summary


def train(engine, rows, cfg, directory, identity, a, rho, progress):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    kwargs = dict(lr=cfg.learning_rate, weight_decay=0.)
    optimizer = (torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.SGD)(
        [p for _,p in engine.params], **kwargs)
    sampler, step, prior = Sampler(cfg.seed), 0, 0.
    initial = engine.parameter_state()
    binding = dict(identity=identity, a=a, rho=rho, initial=object_hash(
        {n:sha256_tensor(p) for n,p in initial.items()}))
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["binding"] != binding:
            raise ValueError("Resume code/data/config/setup/initialization differ")
        engine.load_parameters(state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        step, prior = state["step"], state["training_seconds"]
        torch.set_rng_state(state["rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        old = records(directory / "metrics.jsonl", repair=True)
        if any(r["step"] > step for r in old):
            from .io import write_jsonl
            (directory / "metrics.jsonl").replace(directory / f"metrics.interrupted-{time.time_ns()}.jsonl")
            write_jsonl(directory / "metrics.jsonl", [r for r in old if r["step"] <= step])
        if (directory / "summary.json").exists():
            summary = json.loads((directory / "summary.json").read_text())
            if summary["checkpoint_sha256"] != sha256(checkpoint):
                raise ValueError("Completed checkpoint changed")
            return summary
    elif (directory / "metrics.jsonl").exists():
        raise ValueError("Metrics without checkpoint")
    write_json(directory / "protocol.json", dict(binding=binding, config=asdict(cfg),
        selection="Final checkpoint, no validation selection", optimizer=cfg.optimizer,
        budget="Per-worker training wall time; co-resident time is not isolated GPU time",
        gate="Exploratory fixed rho; no passed audit or AdamW efficiency certificate claimed"))
    started = time.monotonic()
    def elapsed():
        return prior + time.monotonic() - started
    def save():
        atomic_save(checkpoint, dict(binding=binding, parameters=engine.parameter_state(),
            optimizer=optimizer.state_dict(), sampler=sampler.state_dict(), step=step,
            training_seconds=elapsed(), rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    if not checkpoint.exists():
        save()
    while step < cfg.max_steps and elapsed() < cfg.train_seconds:
        progress("training", step=step, total=cfg.max_steps, seconds=elapsed())
        record = update(engine, optimizer, rows, sampler, cfg, step, a, rho)
        step += 1
        record["training_seconds"] = elapsed()
        append_json(directory / "metrics.jsonl", record)
        save()  # Every completed update is durable; partial updates are replayed.
        print(json.dumps(record), flush=True)
    save()
    delta = sum(float((p.detach().cpu().double()-initial[n].double()).square().sum()) for n,p in engine.params)
    summary = dict(steps=step, training_seconds=elapsed(), parameter_delta_l2=math.sqrt(delta),
        checkpoint_sha256=sha256(checkpoint), stop_reason="max_steps" if step == cfg.max_steps else "budget",
        a=a, rho=rho, budget_overshoot_seconds=max(0., elapsed()-cfg.train_seconds))
    write_json(directory / "summary.json", summary)
    return summary


def sha256_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
