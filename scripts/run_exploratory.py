"""Four independent GPU pilots. Keep failed audits intact and report both cost ledgers.

This runner imports the existing estimator unchanged. Its own hash is recorded
separately, so calibration at an identical core/model/data state remains reusable.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("gsm8k", "math_numeric")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runner_hash():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_audit(report, expected, calibration, evaluation, cfg):
    if report["identity"] != expected:
        raise ValueError("Audit core code/model/config/libraries/data identity differs; do not reuse calibration")
    if len(calibration) != cfg.calibration_size or len(evaluation) != cfg.audit_size:
        raise ValueError("Incomplete calibration/audit records")
    uu, ug = sum(x["uu"] for x in calibration), sum(x["ug"] for x in calibration)
    fitted = max(0.0, ug / uu) if uu > 0 else 0.0
    if not all(math.isfinite(x) for x in (uu, ug, fitted, report["a"], report["audit_seconds"])):
        raise ValueError("Nonfinite audit statistics")
    if report["audit_seconds"] < 0 or not math.isclose(fitted, report["a"], rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError("Saved calibration scale/cost is inconsistent")


def make_jobs(reports, gpus, rho):
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("Four distinct GPUs are required")
    if not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("Exploratory rho must be strictly between zero and one")
    jobs = []
    for dataset in DATASETS:
        report = reports[dataset]
        for arm in ("rloo", "dig"):
            a = float(report["a"]) if arm == "dig" else 0.0
            if not math.isfinite(a) or a < 0:
                raise ValueError("Invalid calibrated scale")
            jobs.append(dict(name=f"{dataset}_{arm}", dataset=dataset, arm=arm,
                gpu=str(gpus[len(jobs)]), a=a, rho=rho if arm == "dig" else 1.0,
                gate_passed=bool(report["enabled"]), predictor_active=arm == "dig" and a > 0,
                effective_arm=("dig_zero_predictor_exploratory" if a == 0 else "dig_exploratory")
                              if arm == "dig" else "rloo",
                audit_seconds=float(report["audit_seconds"]) if arm == "dig" else 0.0,
                setup_responses=report.get("audit_responses", 0) if arm == "dig" else 0,
                setup_tokens=report.get("audit_tokens", 0) if arm == "dig" else 0))
    return jobs


def progress(output, job, phase, **details):
    value = dict(job=job["name"], gpu=job["gpu"], phase=phase, pid=os.getpid(),
                 updated_utc=now(), **details)
    write_json(output / f"progress_{job['name']}.json", value)


def validate_manifest(output):
    manifest = read_json(output / "experiment.json")
    if manifest["runner_sha256"] != runner_hash():
        raise ValueError("Experiment runner changed; restore its recorded version before resuming")
    if manifest["root"] != str(ROOT) or manifest["output"] != str(output):
        raise ValueError("Experiment mount path changed")
    if file_hash(output / "config.json") != manifest["config_sha256"]:
        raise ValueError("Experiment config changed")
    for relative, digest in manifest["audit_files"].items():
        if file_hash(output / relative) != digest:
            raise ValueError("Copied source audit changed: " + relative)
    return manifest


def prepare(args, output):
    from dig_rl.config import Config
    from dig_rl.io import experiment_identity, sha256, verify_data
    cfg = Config.load(args.config)
    source = Path(args.audit_run).resolve()
    reports, copies = {}, {}
    model_id = sha256(ROOT / "assets/model/dig_model.json")
    for dataset in DATASETS:
        data = ROOT / "assets/data" / dataset
        verify_data(data, cfg, model_id)
        expected = experiment_identity(cfg, model_id, data)
        report = read_json(source / "audit" / dataset / "decision.json")
        calibration = read_lines(source / "audit" / dataset / "calibration.jsonl")
        evaluation = read_lines(source / "audit" / dataset / "evaluation.jsonl")
        validate_audit(report, expected, calibration, evaluation, cfg)
        reports[dataset] = report
        for name in ("decision.json", "calibration.jsonl", "evaluation.jsonl"):
            copies[f"audit/{dataset}/{name}"] = source / "audit" / dataset / name
    controls = dict(train_seconds=args.train_seconds, max_steps=args.max_steps,
                    save_every=args.save_every, eval_problems=args.eval_problems,
                    eval_samples=args.eval_samples)
    if any(not math.isfinite(x) or x <= 0 for x in controls.values()):
        raise ValueError("Positive finite experiment limits required")
    if args.eval_problems > cfg.validation_size:
        raise ValueError("Evaluation exceeds the fixed validation split")
    jobs = make_jobs(reports, args.gpus, args.rho)
    # Refuse an existing directory before writing any experiment files.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, output / "config.json")
    for relative, source_file in copies.items():
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, dest)
    manifest = dict(schema=1, root=str(ROOT), output=str(output), created_utc=now(),
        mode="exploratory_fixed_rho", runner_sha256=runner_hash(),
        source_audit_run=str(source), config_sha256=file_hash(output / "config.json"),
        audit_files={p: file_hash(output / p) for p in copies}, jobs=jobs, controls=controls,
        design="Fixed rho chosen for exploration after observing failed gates; not gate-approved or confirmatory",
        budget="Equal training wall-time limits; full source audit charged separately to DiG end-to-end totals",
        evaluation="Fixed final checkpoint and original model; identical validation prompts and rollout seeds; no test access")
    write_json(output / "experiment.json", manifest)
    return manifest


def parameter_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def cost_ledger(training_seconds, job, cumulative):
    return dict(training_seconds=training_seconds, audit_seconds=job["audit_seconds"],
        end_to_end_seconds=training_seconds + job["audit_seconds"],
        training_responses=cumulative["responses"],
        charged_responses=cumulative["responses"] + job["setup_responses"],
        training_generated_tokens=cumulative["generated_tokens"],
        charged_generated_tokens=cumulative["generated_tokens"] + job["setup_tokens"])


def train_job(engine, rows, cfg, output, job, controls, ident, initial):
    import torch
    from dig_rl.io import atomic_save, append_json, write_jsonl
    from dig_rl.objective import Sampler
    from dig_rl.train import update
    directory = output / "train" / job["name"]
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    optimizer = torch.optim.SGD([p for _, p in engine.params], lr=cfg.learning_rate,
                                momentum=0.0, weight_decay=0.0)
    sampler = Sampler(cfg.seed)
    step, elapsed = 0, 0.0
    cumulative = dict(responses=0, generated_tokens=0, scored_tokens=0, candidates=0)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["identity"] != ident:
            raise ValueError("Resume identity changed")
        engine.load_parameters(state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, elapsed, cumulative = state["step"], state["training_seconds"], state["cumulative"]
        timing = directory / "checkpoint_timing.json"
        if timing.exists() and read_json(timing)["step"] == step:
            elapsed = max(elapsed, read_json(timing)["training_seconds"])
        metrics = directory / "metrics.jsonl"
        records = read_lines(metrics)
        if any(r["step"] > step for r in records):
            # Preserve the superseded tail when rolling back to a durable checkpoint.
            shutil.copyfile(metrics, directory / f"metrics.before-resume-{time.time_ns()}.jsonl")
            write_jsonl(metrics, [r for r in records if r["step"] <= step])
        summary_path = directory / "summary.json"
        if summary_path.exists():
            finished = read_json(summary_path)
            if (finished["identity"] != ident or finished["steps"] != step
                    or finished["checkpoint_sha256"] != file_hash(checkpoint)):
                raise ValueError("Completed training summary/checkpoint differs")
            # Evaluation may have been interrupted. Do not rewrite the checkpoint:
            # its exact hash is part of the resumable evaluation protocol.
            return finished
    elif (directory / "metrics.jsonl").exists():
        raise ValueError("Metrics exist without a checkpoint; refusing to restart over them")
    write_json(directory / "protocol.json", dict(identity=ident, job=job, controls=controls,
        initial_parameters_sha256=parameter_hash(initial),
        budget="training wall time only; source audit separately charged in end_to_end_seconds; batch overshoot recorded",
        estimator="Existing dig_rl.train.update unchanged; a from source calibration; fixed exploratory rho",
        optimizer="SGD, no momentum/weight decay/clipping; eta0/sqrt(step+1)",
        checkpoint_selection="final checkpoint, independent of validation scores"))
    prior, start = elapsed, time.monotonic()
    def spent():
        return prior + time.monotonic() - start
    def save():
        atomic_save(checkpoint, dict(identity=ident, parameters=engine.parameter_state(),
            optimizer=optimizer.state_dict(), sampler=sampler.state_dict(), step=step,
            training_seconds=spent(), cumulative=cumulative, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        write_json(directory / "checkpoint_timing.json", dict(step=step, training_seconds=spent()))
    if not checkpoint.exists():
        save()
    while step < controls["max_steps"] and spent() < controls["train_seconds"]:
        progress(output, job, "training", step=step, max_steps=controls["max_steps"],
                 training_seconds=spent(), limit_seconds=controls["train_seconds"], activity="computing next update")
        record = update(engine, optimizer, rows, sampler, cfg, step, job["a"], job["rho"])
        engine.sync()
        step += 1
        for key in cumulative:
            cumulative[key] += record[key]
        record.update(training_seconds=spent(), cumulative=cumulative.copy(), effective_arm=job["effective_arm"])
        append_json(directory / "metrics.jsonl", record)
        print(json.dumps({k: record[k] for k in ("step", "selected", "sampled_reward", "gradient_norm", "training_seconds")}), flush=True)
        if step == 1 or step % controls["save_every"] == 0:
            save()
    save()
    elapsed = spent()
    delta = sum(float((p.detach().cpu().double() - initial[n].double()).square().sum())
                for n, p in engine.params)
    result = dict(steps=step, **cost_ledger(elapsed, job, cumulative),
        cumulative=cumulative, job=job, identity=ident, parameter_delta_l2=math.sqrt(delta),
        checkpoint_sha256=file_hash(checkpoint),
        stop_reason="budget" if elapsed >= controls["train_seconds"] else "max_steps",
        budget_overshoot_seconds=max(0.0, elapsed - controls["train_seconds"]))
    write_json(directory / "summary.json", result)
    return result


def evaluate_job(engine, rows, cfg, output, job, controls, label, model_identity):
    import numpy as np
    from dig_rl.io import append_json
    directory = output / "eval" / label
    directory.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=model_identity, prompts=[r["id"] for r in rows],
                    samples=controls["eval_samples"], seed=cfg.seed, runner=runner_hash())
    meta = directory / "protocol.json"
    if meta.exists() and read_json(meta) != protocol:
        raise ValueError("Evaluation resume protocol changed")
    write_json(meta, protocol)
    predictions = directory / "predictions.jsonl"
    records = read_lines(predictions)
    if [r["id"] for r in records] != [r["id"] for r in rows[:len(records)]] or len(records) > len(rows):
        raise ValueError("Evaluation records are not an exact prefix of this validation split")
    for j in range(len(records), len(rows)):
        row = rows[j]
        progress(output, job, "evaluating_base" if label.endswith("_base") else "evaluating",
                 completed=j, total=len(rows))
        sequences, rewards, stats = engine.rollout(row, cfg.seed + 200000000 + 10007 * j, controls["eval_samples"])
        prompt_len = engine.tensors(row)[0].shape[1]
        texts = [engine.tokenizer.decode(s[0, prompt_len:], skip_special_tokens=True) for s in sequences]
        record = dict(id=row["id"], score=float(rewards.mean()), texts=texts, **stats)
        append_json(predictions, record)
        records.append(record)
        print(f"{label}: validation {j + 1}/{len(rows)} score={record['score']:.3f}", flush=True)
    scores = np.array([r["score"] for r in records])
    means = np.random.default_rng(20261103).choice(scores, size=(2000, len(scores)), replace=True).mean(axis=1)
    result = dict(mean_success=float(scores.mean()), prompt_bootstrap_95=np.quantile(means, [.025, .975]).tolist(),
        prompts=len(rows), samples_per_prompt=controls["eval_samples"],
        generated_tokens=sum(r["generated_tokens"] for r in records),
        truncated_fraction=sum(r["truncated"] for r in records) / (len(rows) * controls["eval_samples"]),
        parsed_fraction=sum(r["parsed"] for r in records) / (len(rows) * controls["eval_samples"]),
        generation_seconds=sum(r["seconds"] for r in records),
        protocol="Validation stochastic pass@1, thinking off; not best-of-four or held-out test accuracy",
        model_identity=model_identity)
    write_json(directory / "summary.json", result)
    return result


def worker(output, name):
    from dig_rl.config import Config
    from dig_rl.io import experiment_identity, read_rows, verify_data
    from dig_rl.runtime import Engine
    manifest = validate_manifest(output)
    job = next(j for j in manifest["jobs"] if j["name"] == name)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu"]:
        raise ValueError("Worker GPU assignment differs from experiment manifest")
    cfg, controls = Config.load(output / "config.json"), manifest["controls"]
    progress(output, job, "loading")
    try:
        engine = Engine(ROOT / "assets/model", cfg)
        data = ROOT / "assets/data" / job["dataset"]
        verify_data(data, cfg, engine.model_id)
        core = experiment_identity(cfg, engine.model_id, data)
        report = read_json(output / "audit" / job["dataset"] / "decision.json")
        if report["identity"] != core:
            raise ValueError("Runtime no longer matches the source audit")
        ident = dict(core=core, experiment=object_hash(manifest), job=job["name"], runner=runner_hash())
        initial = {n: t.detach().cpu().clone() for n, t in engine.parameter_state().items()}
        validation = read_rows(data / "validation.jsonl")[:controls["eval_problems"]]
        if len(validation) != controls["eval_problems"]:
            raise ValueError("Insufficient validation prompts")
        train_result = train_job(engine, read_rows(data / "train.jsonl"), cfg, output, job, controls, ident, initial)
        checkpoint = output / "train" / name / "checkpoint.pt"
        evaluate_job(engine, validation, cfg, output, job, controls, name, dict(checkpoint_sha256=file_hash(checkpoint)))
        if job["arm"] == "rloo":
            engine.load_parameters(initial)
            evaluate_job(engine, validation, cfg, output, job, controls, job["dataset"] + "_base",
                         dict(initial_parameters_sha256=parameter_hash(initial), core=core))
        progress(output, job, "completed", steps=train_result["steps"], training_seconds=train_result["training_seconds"])
    except BaseException as error:
        progress(output, job, "failed", error=f"{type(error).__name__}: {error}")
        raise


def paired_difference(left, right):
    import numpy as np
    a, b = {r["id"]: r["score"] for r in left}, {r["id"]: r["score"] for r in right}
    if a.keys() != b.keys() or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Cannot compare different/duplicate validation prompts")
    diffs = np.array([b[k] - a[k] for k in a])
    samples = np.random.default_rng(20261103).choice(diffs, size=(2000, len(diffs)), replace=True).mean(axis=1)
    return dict(difference=float(diffs.mean()), paired_prompt_bootstrap_95=np.quantile(samples, [.025, .975]).tolist())


def summarize(output):
    import csv
    manifest = validate_manifest(output)
    rows, comparisons = [], {}
    for job in manifest["jobs"]:
        name = job["name"]
        training = read_json(output / "train" / name / "summary.json")
        ev = read_json(output / "eval" / name / "summary.json")
        base = read_json(output / "eval" / f"{job['dataset']}_base" / "summary.json")
        rows.append(dict(dataset=job["dataset"], arm=job["arm"], effective_arm=job["effective_arm"],
            gate_passed=job["gate_passed"], predictor_active=job["predictor_active"], a=job["a"], rho=job["rho"],
            steps=training["steps"], training_seconds=training["training_seconds"],
            audit_seconds=training["audit_seconds"], end_to_end_seconds=training["end_to_end_seconds"],
            training_generated_tokens=training["training_generated_tokens"],
            charged_generated_tokens=training["charged_generated_tokens"],
            parameter_delta_l2=training["parameter_delta_l2"], validation_success=ev["mean_success"],
            initial_validation_success=base["mean_success"], gain_over_initial=ev["mean_success"] - base["mean_success"],
            truncation=ev["truncated_fraction"]))
    for dataset in DATASETS:
        protocols = [read_json(output / "train" / f"{dataset}_{arm}" / "protocol.json") for arm in ("rloo", "dig")]
        if protocols[0]["initial_parameters_sha256"] != protocols[1]["initial_parameters_sha256"]:
            raise ValueError("Paired arms started with different parameters")
        predictions = {arm: read_lines(output / "eval" / f"{dataset}_{arm}" / "predictions.jsonl")
                       for arm in ("base", "rloo", "dig")}
        comparisons[dataset] = {f"{b}_minus_{a}": paired_difference(predictions[a], predictions[b])
                               for a, b in (("base", "rloo"), ("base", "dig"), ("rloo", "dig"))}
    comparisons["interpretation"] = "Exploratory, single seed, fixed validation set; failed source gates retained. Equal training limits do not imply equal end-to-end cost. Zero predictor is sampled RLOO, not evidence of demonstration benefit."
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "comparisons.json", comparisons)
    print(json.dumps(rows, indent=2), flush=True)
    return rows


def status(output):
    manifest = read_json(output / "experiment.json")
    print(now(), "|", output.name, flush=True)
    for job in manifest["jobs"]:
        path = output / f"progress_{job['name']}.json"
        p = read_json(path) if path.exists() else dict(phase="pending")
        detail = (f"step {p.get('step', 0)}/{manifest['controls']['max_steps']} | {p.get('training_seconds', 0)/60:.1f} min"
                  if p["phase"] == "training" else
                  f"{p.get('completed', 0)}/{p.get('total', 0)} prompts" if p["phase"].startswith("evaluating") else p.get("error", ""))
        print(f"GPU {job['gpu']} {job['name']:<20} {p['phase']:<16} {detail}", flush=True)


def check_initialization(output, manifest):
    for dataset in DATASETS:
        paths = [output / "train" / f"{dataset}_{arm}" / "protocol.json" for arm in ("rloo", "dig")]
        if all(p.exists() for p in paths):
            pair = [read_json(p) for p in paths]
            if pair[0]["initial_parameters_sha256"] != pair[1]["initial_parameters_sha256"]:
                raise ValueError("Paired arms started with different parameters: " + dataset)


def launch_workers(output, manifest):
    import fcntl
    lock_path = output.parent / (output.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = []
        write_json(output / "status.json", dict(state="running", pid=os.getpid(), started_utc=now()))
        try:
            for job in manifest["jobs"]:
                done = output / f"progress_{job['name']}.json"
                if done.exists() and read_json(done)["phase"] == "completed":
                    continue
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=job["gpu"], PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
                log = (output / f"{job['name']}.log").open("a", encoding="utf-8")
                cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output), "--job", job["name"]]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active.append((job, proc, log))
                print(f"Started {job['name']} on GPU {job['gpu']} (PID {proc.pid})", flush=True)
            while active:
                check_initialization(output, manifest)
                for item in list(active):
                    job, proc, log = item
                    rc = proc.poll()
                    if rc is not None:
                        log.close()
                        active.remove(item)
                        if rc:
                            raise RuntimeError(f"{job['name']} failed (exit {rc}); see {job['name']}.log")
                status(output)
                if active:
                    time.sleep(30)
            summarize(output)
            write_json(output / "status.json", dict(state="completed", finished_utc=now()))
        except BaseException as error:
            write_json(output / "status.json", dict(state="failed", error=f"{type(error).__name__}: {error}", updated_utc=now()))
            raise
        finally:
            for job, proc, log in active:
                proc.terminate()
            for job, proc, log in active:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                log.close()


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "resume", "worker", "status", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name == "launch":
            p.add_argument("--audit-run", type=Path, required=True)
            p.add_argument("--config", type=Path, default=Path("configs/qwen35_4b.json"))
            p.add_argument("--gpus", nargs=4, default=["0", "1", "2", "3"])
            p.add_argument("--rho", type=float, default=.5)
            p.add_argument("--train-seconds", type=float, default=14400)
            p.add_argument("--max-steps", type=int, default=200)
            p.add_argument("--save-every", type=int, default=5)
            p.add_argument("--eval-problems", type=int, default=256)
            p.add_argument("--eval-samples", type=int, default=4)
        elif name == "worker":
            p.add_argument("--job", required=True)
        elif name == "status":
            p.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "runs" or not output.is_relative_to(ROOT / "runs"):
        parser.error("Experiment output must be a subdirectory of this repository's runs directory")
    if args.command in ("launch", "resume"):
        def stop(signum, frame):
            raise KeyboardInterrupt("Experiment launcher stopped")
        signal.signal(signal.SIGTERM, stop)
        manifest = prepare(args, output) if args.command == "launch" else validate_manifest(output)
        launch_workers(output, manifest)
    elif args.command == "worker":
        import fcntl
        if args.job not in {j["name"] for j in validate_manifest(output)["jobs"]}:
            parser.error("Unknown experiment job")
        with (output / ("worker-" + args.job + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(output, args.job)
    elif args.command == "summarize":
        summarize(output)
    else:
        while True:
            if args.watch:
                print("\033[2J\033[H", end="")
            status(output)
            if not args.watch:
                break
            print("Ctrl+C stops this monitor only.", flush=True)
            subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"], check=False)
            time.sleep(10)


if __name__ == "__main__":
    main()
