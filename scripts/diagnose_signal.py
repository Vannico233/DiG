"""Short matched RLOO learning-rate screen with per-group profiling.

This is a training-signal diagnostic, not a DiG comparison or a VLM experiment.
The existing estimator and source evaluation remain unchanged.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("signal_original_runner", ROOT / "scripts/run_exploratory.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def script_hash():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_reference(path, ids, samples):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
            if len(records) == len(ids):
                break
    if [r["id"] for r in records] != ids:
        raise ValueError("Reference predictions do not cover the fixed validation prefix")
    for record in records:
        if (record["responses"] != samples or len(record["rewards"]) != samples
                or record["score"] != sum(record["rewards"]) / samples):
            raise ValueError("Reference evaluation sample count/scores differ")
    return records


def make_jobs(gpus, learning_rates):
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(not g.isdigit() for g in gpus):
        raise ValueError("Four distinct numeric GPU IDs required")
    if (len(learning_rates) != 2 or len(set(learning_rates)) != 2
            or any(not math.isfinite(x) or x <= 0 for x in learning_rates)):
        raise ValueError("Two distinct positive finite learning rates required")
    jobs = []
    for dataset in runner.DATASETS:
        for index, lr in enumerate(learning_rates):
            jobs.append(dict(name=f"{dataset}_lr{index}", dataset=dataset, gpu=gpus[len(jobs)],
                learning_rate=lr, a=0.0, rho=1.0, arm="rloo", effective_arm="rloo_lr_screen",
                audit_seconds=0., setup_responses=0, setup_tokens=0))
    return jobs


def prepare(args, output):
    from dig_rl.config import Config
    from dig_rl.io import experiment_identity, read_rows, sha256, verify_data, write_jsonl
    source = args.source_run.resolve()
    source_manifest = runner.validate_manifest(source)
    cfg = Config.load(source / "config.json")
    if args.steps < 1 or not 1 <= args.eval_problems <= cfg.validation_size:
        raise ValueError("Positive step limit and valid validation prefix required")
    if not math.isfinite(args.train_seconds) or args.train_seconds <= 0:
        raise ValueError("Positive finite training time limit required")
    jobs = make_jobs(args.gpus, args.learning_rates)
    references = {}
    model_id = sha256(ROOT / "assets/model/dig_model.json")
    for dataset in runner.DATASETS:
        data = ROOT / "assets/data" / dataset
        verify_data(data, cfg, model_id)
        identity = experiment_identity(cfg, model_id, data)
        protocol = runner.read_json(source / "eval" / (dataset + "_base") / "protocol.json")
        training = runner.read_json(source / "train" / (dataset + "_rloo") / "protocol.json")
        if (protocol["model"]["core"] != identity or protocol["seed"] != cfg.seed
                or protocol["runner"] != runner.runner_hash()
                or protocol["model"]["initial_parameters_sha256"] != training["initial_parameters_sha256"]
                or protocol["samples"] != source_manifest["controls"]["eval_samples"]):
            raise ValueError("Reference model/config/data/libraries/protocol changed")
        validation = read_rows(data / "validation.jsonl")[:args.eval_problems]
        ids = [r["id"] for r in validation]
        if protocol["prompts"][:len(ids)] != ids:
            raise ValueError("Reference validation question order differs")
        records = read_reference(source / "eval" / (dataset + "_base") / "predictions.jsonl", ids, protocol["samples"])
        references[dataset] = dict(protocol=protocol, records=records)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes((source / "config.json").read_bytes())
    for dataset, reference in references.items():
        directory = output / "reference" / dataset
        directory.mkdir(parents=True)
        runner.write_json(directory / "protocol.json", reference["protocol"])
        write_jsonl(directory / "predictions.jsonl", reference["records"])
    manifest = dict(schema=1, kind="rloo_learning_rate_signal_screen", created_utc=runner.now(),
        root=str(ROOT), output=str(output), source_run=str(source),
        source_manifest_sha256=runner.file_hash(source / "experiment.json"),
        script_sha256=script_hash(), runner_sha256=runner.runner_hash(), jobs=jobs,
        controls=dict(max_steps=args.steps, train_seconds=args.train_seconds, save_every=1,
                      eval_problems=args.eval_problems, eval_samples=source_manifest["controls"]["eval_samples"]),
        design="Same initialization, question/rollout RNG streams, objective and SGD schedule; only initial learning rate differs. Final checkpoint on fixed validation prefix; no test access. Single seed diagnostic, not a tuned recommendation.")
    manifest["files"] = {str(p.relative_to(output)).replace("\\", "/"): runner.file_hash(p)
                         for p in [output / "config.json", *sorted((output / "reference").rglob("*.*"))]}
    runner.write_json(output / "experiment.json", manifest)
    return manifest


def validate(output):
    manifest = runner.read_json(output / "experiment.json")
    if (manifest["root"] != str(ROOT) or manifest["output"] != str(output)
            or manifest["script_sha256"] != script_hash() or manifest["runner_sha256"] != runner.runner_hash()):
        raise ValueError("Diagnostic mount path/code changed")
    for path, digest in manifest["files"].items():
        if runner.file_hash(output / path) != digest:
            raise ValueError("Diagnostic input changed: " + path)
    return manifest


class ProfiledEngine:
    """Observe the existing RLOO update without changing its gradients or RNG."""
    def __init__(self, engine, directory, cfg, step=0):
        self.engine, self.directory, self.cfg = engine, directory, cfg
        self.group_index, self.pending = step * cfg.candidates, None

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def rollout(self, row, seed, count=None):
        if count is not None:  # The evaluator supplies count explicitly.
            return self.engine.rollout(row, seed, count)
        import torch
        if self.pending is not None:
            raise RuntimeError("Previous training group has not been scored")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.engine.device)
        started = time.monotonic()
        sequences, rewards, stats = self.engine.rollout(row, seed)
        values = rewards.detach().cpu().tolist()
        self.pending = dict(step=self.group_index // self.cfg.candidates + 1,
            candidate=self.group_index % self.cfg.candidates, id=row["id"], seed=seed,
            rewards=values, constant_reward_group=len(set(values)) == 1,
            generation_seconds=stats["seconds"], generated_tokens=stats["generated_tokens"],
            responses=stats["responses"], parsed=stats["parsed"], truncated=stats["truncated"],
            started_monotonic=started)
        return sequences, rewards, stats

    def rl_backward(self, row, sequences, rewards, weight=1.0):
        import torch
        result = self.engine.rl_backward(row, sequences, rewards, weight)
        record = dict(self.pending)
        record.update(backward_seconds=result["seconds"], scored_tokens=result["scored_tokens"],
                      group_wall_seconds=time.monotonic() - record.pop("started_monotonic"),
                      peak_allocated_gib=(torch.cuda.max_memory_allocated(self.engine.device) / 2**30
                                          if torch.cuda.is_available() else None))
        runner.write_json(self.directory / f"step-{record['step']:04d}-group-{record['candidate']:02d}.json", record)
        print(f"train step {record['step']} group {record['candidate']+1}/{self.cfg.candidates}: "
              f"generate={record['generation_seconds']:.1f}s backward={record['backward_seconds']:.1f}s "
              f"constant_reward={record['constant_reward_group']}", flush=True)
        self.pending = None
        self.group_index += 1
        return result


def worker(output, name):
    import torch
    from dig_rl.config import Config
    from dig_rl.io import experiment_identity, read_rows, verify_data
    from dig_rl.runtime import Engine
    manifest = validate(output)
    job = next(j for j in manifest["jobs"] if j["name"] == name)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu"]:
        raise ValueError("GPU assignment differs")
    base_cfg = Config.load(output / "config.json")
    cfg = replace(base_cfg, learning_rate=job["learning_rate"])
    runner.progress(output, job, "loading")
    try:
        engine = Engine(ROOT / "assets/model", cfg)
        data = ROOT / "assets/data" / job["dataset"]
        verify_data(data, base_cfg, engine.model_id)
        reference = runner.read_json(output / "reference" / job["dataset"] / "protocol.json")
        if experiment_identity(base_cfg, engine.model_id, data) != reference["model"]["core"]:
            raise ValueError("Current runtime differs from the initial-model reference")
        initial = engine.parameter_state()
        if runner.parameter_hash(initial) != reference["model"]["initial_parameters_sha256"]:
            raise ValueError("Initial parameters differ from reference")
        identity = dict(core=experiment_identity(cfg, engine.model_id, data),
                        diagnostic=runner.object_hash(manifest), job=name)
        checkpoint = output / "train" / name / "checkpoint.pt"
        step = torch.load(checkpoint, map_location="cpu", weights_only=True)["step"] if checkpoint.exists() else 0
        profile = ProfiledEngine(engine, output / "profile" / name, cfg, step)
        trained = runner.train_job(profile, read_rows(data / "train.jsonl"), cfg, output, job,
                                   manifest["controls"], identity, initial)
        validation = read_rows(data / "validation.jsonl")[:manifest["controls"]["eval_problems"]]
        runner.evaluate_job(profile, validation, cfg, output, job, manifest["controls"], name,
                            dict(checkpoint_sha256=runner.file_hash(checkpoint)))
        runner.progress(output, job, "completed", steps=trained["steps"], training_seconds=trained["training_seconds"])
    except BaseException as error:
        runner.progress(output, job, "failed", error=f"{type(error).__name__}: {error}")
        raise


def summarize(output):
    manifest = validate(output)
    result = []
    for job in manifest["jobs"]:
        name = job["name"]
        trained = runner.read_json(output / "train" / name / "summary.json")
        evaluation = runner.read_json(output / "eval" / name / "summary.json")
        metrics = runner.read_lines(output / "train" / name / "metrics.jsonl")
        groups = [runner.read_json(p) for p in sorted((output / "profile" / name).glob("step-*.json"))]
        groups = [g for g in groups if g["step"] <= trained["steps"]]
        if len(groups) != trained["cumulative"]["candidates"]:
            raise ValueError("Incomplete completed-step profiling records")
        baseline = runner.read_lines(output / "reference" / job["dataset"] / "predictions.jsonl")
        predictions = runner.read_lines(output / "eval" / name / "predictions.jsonl")
        generation = sum(g["generation_seconds"] for g in groups)
        backward = sum(g["backward_seconds"] for g in groups)
        tokens = sum(g["generated_tokens"] for g in groups)
        result.append(dict(dataset=job["dataset"], learning_rate=job["learning_rate"], steps=trained["steps"],
            training_seconds=trained["training_seconds"], generation_seconds=generation, backward_seconds=backward,
            other_training_seconds=trained["training_seconds"]-generation-backward,
            generation_tokens_per_second=tokens/generation if generation else None,
            zero_gradient_steps=sum(m["gradient_norm"] == 0 for m in metrics),
            constant_reward_groups=sum(g["constant_reward_group"] for g in groups), groups=len(groups),
            parameter_delta_l2=trained["parameter_delta_l2"], validation_success=evaluation["mean_success"],
            initial_validation_success=sum(r["score"] for r in baseline)/len(baseline),
            gain_over_initial=runner.paired_difference(baseline, predictions),
            peak_allocated_gib=max((g["peak_allocated_gib"] or 0 for g in groups), default=0),
            evaluation_seconds=evaluation["generation_seconds"], truncated_fraction=evaluation["truncated_fraction"]))
    comparisons = {}
    for dataset in runner.DATASETS:
        jobs = [j for j in manifest["jobs"] if j["dataset"] == dataset]
        predictions = [runner.read_lines(output / "eval" / j["name"] / "predictions.jsonl") for j in jobs]
        comparisons[dataset] = dict(left_lr=jobs[0]["learning_rate"], right_lr=jobs[1]["learning_rate"],
                                   **runner.paired_difference(*predictions))
    report = dict(results=result, comparisons=comparisons,
                  note="Small validation-prefix, single-seed learning-rate diagnostic; not a DiG win or a tuned learning-rate recommendation.")
    runner.write_json(output / "signal_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def ensure_source_complete(manifest):
    source = Path(manifest["source_run"])
    if runner.file_hash(source / "experiment.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source experiment changed")
    if runner.read_json(source / "status.json")["state"] != "completed":
        raise RuntimeError("Source evaluation is not completed. Finish it before starting this diagnostic.")


def ensure_gpus_idle(gpus):
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                               check=True, capture_output=True, text=True).stdout
    mapping = dict(tuple(s.strip() for s in line.split(",", 1)) for line in inventory.splitlines() if line.strip())
    if not set(gpus).issubset(mapping):
        raise ValueError("Requested GPU index unavailable")
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True).stdout
    selected = {mapping[g] for g in gpus}
    if any(line.split(",", 1)[0].strip() in selected for line in active.splitlines()):
        raise RuntimeError("Selected GPUs have compute processes. No diagnostic workers started.")


def sample_gpus(output, jobs):
    sample = dict(utc=runner.now(), gpus=[])
    try:
        raw = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                              "--format=csv,noheader,nounits"], check=True, capture_output=True,
                             text=True, timeout=5).stdout
        mapping = {j["gpu"]: j for j in jobs}
        for line in raw.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) != 4 or fields[0] not in mapping:
                continue
            job = mapping[fields[0]]
            path = output / f"progress_{job['name']}.json"
            phase = runner.read_json(path)["phase"] if path.exists() else "pending"
            sample["gpus"].append(dict(index=fields[0], job=job["name"], phase=phase,
                utilization_percent=fields[1], memory_mib=fields[2], power_watts=fields[3]))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        sample["error"] = str(error)
    with (output / "gpu_samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample) + "\n")


def launch(output):
    import fcntl
    manifest = validate(output)
    with ExitStack() as stack:
        source = Path(manifest["source_run"])
        locks = []
        for path in (output.parent / (output.name + ".lock"), source.parent / (source.name + ".lock")):
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        ensure_source_complete(manifest)
        ensure_gpus_idle([j["gpu"] for j in manifest["jobs"]])
        active = []
        runner.write_json(output / "status.json", dict(state="running", started_utc=runner.now(), pid=os.getpid()))
        try:
            for job in manifest["jobs"]:
                path = output / f"progress_{job['name']}.json"
                if path.exists() and runner.read_json(path)["phase"] == "completed":
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpu"], PYTHONUNBUFFERED="1")
                log = stack.enter_context((output / (job["name"] + ".log")).open("a", encoding="utf-8"))
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                    "--job", job["name"]], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=tuple(lock.fileno() for lock in locks))
                active.append(proc)
            while active:
                for proc in list(active):
                    rc = proc.poll()
                    if rc is not None:
                        active.remove(proc)
                        if rc:
                            raise RuntimeError("Diagnostic worker failed; see per-job log. Checkpoints retained.")
                runner.status(output)
                if active:
                    sample_gpus(output, manifest["jobs"])
                    time.sleep(15)
            summarize(output)
            runner.write_json(output / "status.json", dict(state="completed", finished_utc=runner.now()))
        except BaseException as error:
            runner.write_json(output / "status.json", dict(state="failed", error=str(error), updated_utc=runner.now()))
            raise
        finally:
            for proc in active:
                proc.terminate()
            for proc in active:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "summarize", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--gpus", nargs=4, default=["0", "1", "2", "3"])
    parser.add_argument("--learning-rates", type=float, nargs=2, default=[.01, 1.0])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--train-seconds", type=float, default=3600.)
    parser.add_argument("--eval-problems", type=int, default=32)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "runs" or not output.is_relative_to(ROOT / "runs"):
        parser.error("Output must be a subdirectory of repository runs")
    if args.command == "prepare":
        if args.source_run is None:
            parser.error("prepare requires --source-run")
        prepare(args, output)
        print("Prepared diagnostic. Run only after the source evaluation completes.")
    elif args.command == "run":
        def stop(signum, frame):
            raise KeyboardInterrupt("Diagnostic interrupted")
        signal.signal(signal.SIGTERM, stop)
        launch(output)
    elif args.command == "worker":
        if args.job not in {j["name"] for j in validate(output)["jobs"]}:
            parser.error("Unknown job")
        import fcntl
        with (output / ("worker-" + args.job + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(output, args.job)
    elif args.command == "status":
        runner.status(output)
    else:
        summarize(output)


if __name__ == "__main__":
    main()
