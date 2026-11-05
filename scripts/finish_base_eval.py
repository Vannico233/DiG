"""Finish an interrupted exploratory base evaluation on several independent GPUs.

Keep the original runner, checkpoint, prompt order, per-prompt seed and batch size.
Only completed per-question files are merged, under the original launcher's locks.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("original_exploratory", ROOT / "scripts/run_exploratory.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def helper_hash():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def seed_for(cfg, index):
    return cfg.seed + 200000000 + 10007 * index


def read_prefix(path, rows, samples):
    raw = path.read_bytes() if path.exists() else b""
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Incomplete final JSONL line; preserve/recover it before parallel evaluation")
    records = [json.loads(line) for line in raw.split(b"\n") if line.strip()]
    if len(records) > len(rows):
        raise ValueError("Too many existing predictions")
    for index, record in enumerate(records):
        validate_record(record, rows[index]["id"], samples)
    return raw, records


def validate_record(record, question_id, samples):
    if record["id"] != question_id or record["responses"] != samples:
        raise ValueError("Prediction question/order/sample count differs")
    rewards = record["rewards"]
    if (len(record["texts"]) != samples or len(rewards) != samples
            or any(r not in (0, 1) for r in rewards)
            or record["score"] != sum(rewards) / samples):
        raise ValueError("Invalid prediction scores")
    if not 0 <= record["parsed"] <= samples or not 0 <= record["truncated"] <= samples:
        raise ValueError("Invalid prediction counts")


def part_path(directory, index):
    return directory / "parts" / f"{index:06d}.json"


def read_part(directory, index, rows, cfg, protocol):
    path = part_path(directory, index)
    if not path.exists():
        return None
    part = runner.read_json(path)
    expected = dict(index=index, seed=seed_for(cfg, index),
                    protocol_sha256=runner.object_hash(protocol), helper_sha256=helper_hash())
    if any(part.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Parallel part protocol changed: {path.name}")
    validate_record(part["record"], rows[index]["id"], protocol["samples"])
    return part["record"]


def evaluate_indices(engine, rows, cfg, indices, directory, protocol, slot):
    for index in indices:
        if read_part(directory, index, rows, cfg, protocol) is not None:
            continue
        row = rows[index]
        runner.write_json(directory / f"slot-{slot}.json", dict(
            phase="evaluating", index=index, question_id=row["id"], pid=os.getpid(),
            updated_utc=runner.now()))
        sequences, rewards, stats = engine.rollout(row, seed_for(cfg, index), protocol["samples"])
        prompt_len = engine.tensors(row)[0].shape[1]
        texts = [engine.tokenizer.decode(s[0, prompt_len:], skip_special_tokens=True) for s in sequences]
        record = dict(id=row["id"], score=float(rewards.mean()), texts=texts, **stats)
        validate_record(record, row["id"], protocol["samples"])
        runner.write_json(part_path(directory, index), dict(
            index=index, seed=seed_for(cfg, index), protocol_sha256=runner.object_hash(protocol),
            helper_sha256=helper_hash(), record=record))
        print(f"slot {slot}: question {index + 1}/{len(rows)} complete", flush=True)
    runner.write_json(directory / f"slot-{slot}.json", dict(phase="completed", updated_utc=runner.now()))


def merge_parts(path, before, rows, cfg, directory, protocol):
    current, prefix = read_prefix(path, rows, protocol["samples"])
    if current != before:
        raise ValueError("Original predictions changed during parallel evaluation; merge refused")
    suffix = []
    for index in range(len(prefix), len(rows)):
        record = read_part(directory, index, rows, cfg, protocol)
        if record is None:
            raise ValueError(f"Missing parallel prediction: {index}")
        suffix.append((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode())
    if not suffix:
        return
    backup = directory / ("prefix-" + hashlib.sha256(before).hexdigest() + ".jsonl")
    if not backup.exists():
        backup.write_bytes(before)
    temp = path.with_name("predictions.parallel.tmp")
    temp.write_bytes(before + b"".join(suffix))
    os.replace(temp, path)


def load_context(output, dataset):
    from dig_rl.config import Config
    from dig_rl.io import read_rows
    manifest = runner.validate_manifest(output)
    cfg = Config.load(output / "config.json")
    job = next(j for j in manifest["jobs"] if j["name"] == dataset + "_rloo")
    rows = read_rows(ROOT / "assets/data" / dataset / "validation.jsonl")[:manifest["controls"]["eval_problems"]]
    protocol = runner.read_json(output / "eval" / (dataset + "_base") / "protocol.json")
    initial = runner.read_json(output / "train" / job["name"] / "protocol.json")
    expected = dict(model=dict(initial_parameters_sha256=initial["initial_parameters_sha256"],
                               core=initial["identity"]["core"]), prompts=[r["id"] for r in rows],
                    samples=manifest["controls"]["eval_samples"], seed=cfg.seed, runner=runner.runner_hash())
    if len(rows) != manifest["controls"]["eval_problems"] or protocol != expected:
        raise ValueError("Original base evaluation protocol differs")
    for item in manifest["jobs"]:
        training = runner.read_json(output / "train" / item["name"] / "summary.json")
        if training["checkpoint_sha256"] != runner.file_hash(output / "train" / item["name"] / "checkpoint.pt"):
            raise ValueError("Training checkpoint differs")
        runner.read_json(output / "eval" / item["name"] / "summary.json")
        if item["dataset"] != dataset:
            runner.read_json(output / "eval" / (item["dataset"] + "_base") / "summary.json")
    return manifest, cfg, job, rows, protocol


def worker(output, dataset, slot):
    from dig_rl.io import experiment_identity, verify_data
    from dig_rl.runtime import Engine
    manifest, cfg, job, rows, protocol = load_context(output, dataset)
    directory = output / "eval" / (dataset + "_base") / "parallel"
    plan = runner.read_json(directory / "plan.json")
    if (plan["helper_sha256"] != helper_hash() or plan["protocol"] != protocol
            or os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpus"][slot]):
        raise ValueError("Parallel worker assignment/identity changed")
    engine = Engine(ROOT / "assets/model", cfg)
    data = ROOT / "assets/data" / dataset
    verify_data(data, cfg, engine.model_id)
    if experiment_identity(cfg, engine.model_id, data) != protocol["model"]["core"]:
        raise ValueError("Runtime no longer matches original experiment")
    if runner.parameter_hash(engine.parameter_state()) != protocol["model"]["initial_parameters_sha256"]:
        raise ValueError("Initial model parameters differ")
    evaluate_indices(engine, rows, cfg, plan["assignments"][slot], directory, protocol, slot)


def finish(output, dataset, gpus):
    import fcntl
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError("Specify distinct numeric GPU IDs")
    directory = output / "eval" / (dataset + "_base") / "parallel"
    with ExitStack() as stack:
        locks = []
        # Original launcher and workers cannot concurrently rewrite evaluation.
        lock_paths = [output.parent / (output.name + ".lock")]
        lock_paths += [output / ("worker-" + name + ".lock") for name in
                       ("gsm8k_rloo", "gsm8k_dig", "math_numeric_rloo", "math_numeric_dig")]
        for path in lock_paths:
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        manifest, cfg, job, rows, protocol = load_context(output, dataset)
        predictions = output / "eval" / (dataset + "_base") / "predictions.jsonl"
        before, prefix = read_prefix(predictions, rows, protocol["samples"])
        pending = [j for j in range(len(prefix), len(rows))
                   if read_part(directory, j, rows, cfg, protocol) is None]
        assignments = [pending[slot::len(gpus)] for slot in range(len(gpus))]
        runner.write_json(directory / "plan.json", dict(protocol=protocol, helper_sha256=helper_hash(),
            prefix_sha256=hashlib.sha256(before).hexdigest(), prefix_count=len(prefix),
            assignments=assignments, gpus=gpus, started_utc=runner.now()))
        print(f"Retained {len(prefix)} original questions; {len(pending)} pending on {len(gpus)} GPUs", flush=True)
        active = []
        start = time.monotonic()
        try:
            for slot, indices in enumerate(assignments):
                if not indices:
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[slot], PYTHONUNBUFFERED="1")
                log = stack.enter_context((directory / f"slot-{slot}.log").open("a", encoding="utf-8"))
                cmd = [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                       "--dataset", dataset, "--worker-slot", str(slot)]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        pass_fds=tuple(lock.fileno() for lock in locks))
                active.append(proc)
            while active:
                for proc in list(active):
                    rc = proc.poll()
                    if rc is not None:
                        active.remove(proc)
                        if rc:
                            raise RuntimeError("Evaluation worker failed; see parallel/slot-*.log; completed parts retained")
                finished = sum(part_path(directory, j).exists() for j in range(len(prefix), len(rows)))
                runner.write_json(directory / "status.json", dict(phase="evaluating", retained=len(prefix),
                    completed=len(prefix) + finished, total=len(rows), updated_utc=runner.now()))
                print(f"Base evaluation: {len(prefix) + finished}/{len(rows)}", flush=True)
                if active:
                    time.sleep(10)
            merge_parts(predictions, before, rows, cfg, directory, protocol)
            # All rows are already present: the original evaluator only summarizes.
            runner.evaluate_job(None, rows, cfg, output, job, manifest["controls"],
                                dataset + "_base", protocol["model"])
            runner.summarize(output)
            training = runner.read_json(output / "train" / job["name"] / "summary.json")
            runner.progress(output, job, "completed", steps=training["steps"], training_seconds=training["training_seconds"])
            runner.write_json(output / "status.json", dict(state="completed", finished_utc=runner.now()))
            runner.write_json(directory / "status.json", dict(phase="completed", completed=len(rows), total=len(rows),
                updated_utc=runner.now(), elapsed_this_attempt_seconds=time.monotonic() - start,
                helper_sha256=helper_hash(), gpus=gpus))
        except BaseException as error:
            runner.write_json(directory / "status.json", dict(phase="interrupted", error=str(error), updated_utc=runner.now()))
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=runner.DATASETS, default="math_numeric")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--worker-slot", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "runs" or not output.is_relative_to(ROOT / "runs"):
        parser.error("Output must be an existing repository run")
    if args.worker_slot is None:
        def stop(signum, frame):
            raise KeyboardInterrupt("Parallel evaluation interrupted")
        signal.signal(signal.SIGTERM, stop)
        finish(output, args.dataset, args.gpus)
    else:
        worker(output, args.dataset, args.worker_slot)


if __name__ == "__main__":
    main()
