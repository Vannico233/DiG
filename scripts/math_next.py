"""Math-only, isolated-GPU pilot: profile -> RLOO/zero/online -> paired validation."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dig_rl.io import read_rows, sha256, write_json, append_json
from dig_rl.round2_protocol import Protocol, object_hash
from dig_rl.round2_train import records, evaluate
from dig_rl.math_profile_recovery import memory_assessment, migrate_profiles
from round2 import runtime_identity, read, now

DATASETS = ("gsm8k", "math")
ARMS = ("rloo", "zero", "online")


class NoSafeProfile(RuntimeError):
    pass


def identity():
    return dict(**runtime_identity(), math_script=sha256(__file__))


def plan(args):
    from dig_rl.round2_data import verify
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    if len(set(args.gpus)) != 4:
        raise ValueError("Four distinct GPUs required")
    configs, manifests = {}, {}
    model_id = sha256(args.model / "dig_model.json")
    for dataset in DATASETS:
        manifest = verify(args.assets / dataset)
        if manifest["model_id"] != model_id:
            raise ValueError("Prepared data/model identity mismatch")
        cfg = Protocol(dataset, max_new_tokens=4096 if dataset == "gsm8k" else 8192,
            max_input_tokens=2048 if dataset == "gsm8k" else 4096,
            candidates=args.candidates, max_steps=args.steps, train_seconds=args.train_seconds,
            validation_size=256, learning_rate=1e-6)
        configs[dataset] = asdict(cfg)
        manifests[dataset] = sha256(args.assets / dataset / "manifest.json")
    args.output.mkdir(parents=True)
    write_json(args.output / "experiment.json", dict(schema=1, runtime=identity(), created_utc=now(),
        output=str(args.output), model=str(args.model.resolve()), assets=str(args.assets.resolve()),
        model_id=model_id, data=manifests, configs=configs, gpus=args.gpus,
        arms=list(ARMS), comparison="Same candidate steps and configs; report realized sampling/time separately",
        selection="Final step, no validation checkpoint selection. Official test is a separate frozen-protocol command.",
        hypothesis="Online block predictor may improve on zero-predictor subsampling; NOT an established result"))


def validate(output):
    m = read(output / "experiment.json")
    if m["output"] != str(output) or m["runtime"] != identity():
        raise ValueError("Code/environment/path changed; use exact version or a fresh run")
    if sha256(Path(m["model"]) / "dig_model.json") != m["model_id"]:
        raise ValueError("Model identity changed")
    for dataset, digest in m["data"].items():
        if sha256(Path(m["assets"]) / dataset / "manifest.json") != digest:
            raise ValueError("Data identity changed")
    return m


def task_specs(phase):
    if phase == "verify":
        return [dict(name=d+"_verify",dataset=d,phase=phase) for d in DATASETS]
    if phase == "profile":
        return [dict(name=f"{d}_{mode}_{size}", dataset=d, mode=mode, size=size, phase=phase)
                for mode,size in (("legacy",16),("buffered",16),("buffered",32),("buffered",64)) for d in DATASETS]
    # Long math arms start first; remaining jobs use whichever physical GPU frees up.
    return [dict(name=f"{d}_{arm}", dataset=d, arm=arm, phase=phase)
            for arm in ("rloo", "online", "zero", "base") for d in ("math", "gsm8k")]


def choose_profiles(output):
    decision = {}
    diagnostics = {}
    for dataset in DATASETS:
        rows = [read(output / "profile" / (t["name"]+".json")) for t in task_specs("profile") if t["dataset"]==dataset]
        diagnostics[dataset] = [dict(mode=r.get("mode"),active_sequences=r.get("active_sequences"),
                                    **memory_assessment(r)) for r in rows]
        safe = [r for r in rows if memory_assessment(r)["accepted"]]
        if not safe:
            write_json(output/"profile_selection.json",diagnostics)
            raise NoSafeProfile("No accepted live-memory profile: " + dataset + "; details: profile_selection.json")
        best = max(safe, key=lambda r:r["tokens_per_second"])
        decision[dataset] = dict(mode=best["mode"], active_sequences=best["active_sequences"],
            tokens_per_second=best["tokens_per_second"], profile=best,
            note="One full-cap calibration-prefix sample. Not sustained-training performance or worst-case memory proof.")
    write_json(output/"profile_selection.json",diagnostics)
    write_json(output / "profile_decision.json", decision)


def verify_capacity(engine, data, output, dataset, progress):
    """One conservative check per dataset; lower batch only on OOM/headroom failure."""
    import torch
    row = read_rows(data/"calibration.jsonl")[0]
    history = output/("historical_profiles" if (output/"historical_profiles").exists() else "profile")
    # Prefer the already measured original sampler. Do not select larger batches
    # from historical reserved-only measurements or claim their timing is fresh.
    mode = "legacy"
    engine.buffered_sampling = False
    for size in (16,8,4):
        engine.protocol = replace(engine.protocol,active_sequences=size)
        try:
            report = engine.capacity_check(row,progress)
        except Exception as error:
            if not isinstance(error,torch.OutOfMemoryError) and "out of memory" not in str(error).lower():
                raise
            report = dict(passed=False,error=str(error))
        report.update(mode=mode,active_sequences=size)
        assessment = memory_assessment(report)
        report["assessment"] = assessment
        write_json(output/"capacity"/f"{dataset}_{size}.json",report)
        if assessment["accepted"]:
            previous = history/f"{dataset}_{mode}_{size}.json"
            old = read(previous) if previous.exists() else {}
            decision = dict(mode=mode,active_sequences=size,capacity=report,
                historical_tokens_per_second=old.get("tokens_per_second"),
                note="Conservative batch with new synthetic maximum-shape live-memory check; historical speed is not remeasured throughput.")
            write_json(output/"capacity"/f"{dataset}_decision.json",decision)
            return
        progress("reducing_batch",rejected_size=size,reason=assessment["reason"])
        engine.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raise RuntimeError("Capacity checks failed at batch 16/8/4: "+dataset+"; see capacity/*.json")


def select_verified_capacity(output):
    decisions = {d:read(output/"capacity"/f"{d}_decision.json") for d in DATASETS}
    if not all(memory_assessment(r["capacity"])["accepted"] for r in decisions.values()):
        raise RuntimeError("Capacity verification missing/invalid")
    write_json(output/"profile_decision.json",decisions)


def worker(output, task):
    import torch
    from dig_rl.math_engine import MathEngine
    from dig_rl.math_train import train_math
    m = validate(output)
    dataset, phase, name = task["dataset"], task["phase"], task["name"]
    state = output / f"progress_{phase}_{name}.json"
    def progress(stage, **values):
        write_json(state, dict(phase=stage, utc=now(), pid=os.getpid(),
                   gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), **values))
    progress("loading")
    cfg = Protocol(**m["configs"][dataset])
    data = Path(m["assets"]) / dataset
    engine = None
    try:
        if phase == "profile":
            cfg = replace(cfg, active_sequences=task["size"])
            mode = task["mode"]
        elif phase == "verify":
            mode = "legacy"
        else:
            decision = read(output / "profile_decision.json")[dataset]
            cfg = replace(cfg, active_sequences=decision["active_sequences"])
            mode = decision["mode"]
        engine = MathEngine(m["model"], cfg, data)
        engine.buffered_sampling = mode=="buffered"
        binding = dict(manifest=object_hash(m), execution=asdict(cfg))
        if phase == "verify":
            verify_capacity(engine,data,output,dataset,progress)
        elif phase == "profile":
            rows = read_rows(data / "calibration.jsonl")[:8]
            progress("warming")
            engine.rollout_many([(rows[0], cfg.seed)], count=2, max_tokens=32)
            engine.sync(); torch.cuda.reset_peak_memory_stats()
            progress("profiling", active_sequences=cfg.active_sequences, max_new_tokens=cfg.max_new_tokens)
            start = time.monotonic()
            groups = engine.rollout_many([(r, cfg.seed+10007*i) for i,r in enumerate(rows)])
            engine.sync(); elapsed = time.monotonic()-start
            generation_peak = torch.cuda.max_memory_reserved()/2**30
            generation_live = torch.cuda.max_memory_allocated()/2**30
            total_memory = torch.cuda.get_device_properties(engine.device).total_memory/2**30
            tokens = sum(r["generated_tokens"] for r in groups)
            slots = sum(r["decode_slots"] for r in groups)
            progress("full_length_backward")
            memory = engine.memory_check(read_rows(data / "calibration.jsonl"))
            write_json(output / "profile" / (name+".json"), dict(passed=True, mode=task["mode"],
                active_sequences=cfg.active_sequences, tokens=tokens, seconds=elapsed,
                tokens_per_second=tokens/elapsed, useful_decode_fraction=tokens/slots,
                device_total_gib=total_memory,generation_peak_allocated_gib=generation_live,
                generation_peak_reserved_gib=generation_peak,
                peak_reserved_gib=max(generation_peak,memory["peak_reserved_gib"]), memory=memory,
                max_new_tokens=cfg.max_new_tokens, responses=sum(r["responses"] for r in groups)))
        else:
            arm = task["arm"]
            model_identity = binding
            if arm != "base":
                directory = output / "train" / name
                if phase == "train":
                    summary = train_math(engine, read_rows(data / "train.jsonl"), cfg,
                                         directory, binding, arm, progress)
                else:
                    summary = read(directory / "summary.json")
                    if not summary["matched_steps_complete"]:
                        raise ValueError("Matched-step pilot incomplete; official test is not enabled")
                    if sha256(directory / "checkpoint.pt") != summary["checkpoint_sha256"]:
                        raise ValueError("Checkpoint changed")
                    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
                    if checkpoint["binding"]["identity"] != binding or checkpoint["binding"]["arm"] != arm:
                        raise ValueError("Checkpoint identity differs")
                    engine.load_parameters(checkpoint["parameters"])
                model_identity = dict(checkpoint=summary["checkpoint_sha256"])
            split = "validation" if phase == "train" else "test"
            rows = read_rows(data / (split+".jsonl"))
            if phase == "train":
                rows = rows[:cfg.validation_size]
            evaluate(engine, rows, cfg, output / ("eval" if phase=="train" else "test") / name,
                     model_identity, progress)
        progress("completed")
    except Exception as error:
        if phase=="profile" and (isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()):
            write_json(output / "profile" / (name+".json"), dict(passed=False, mode=task["mode"],
                active_sequences=cfg.active_sequences, error=str(error)))
            progress("completed", profile_passed=False)
        else:
            progress("failed", error=f"{type(error).__name__}: {error}")
            raise


def launch(output, phase, m):
    occupied = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if occupied:
        raise RuntimeError("GPU compute processes already running; no jobs launched")
    pending, active = task_specs(phase), []
    started = time.monotonic()
    try:
        while pending or active:
            for task in list(pending):
                state = output / f"progress_{phase}_{task['name']}.json"
                if state.exists() and read(state)["phase"]=="completed":
                    pending.remove(task); continue
                free = [g for g in m["gpus"] if g not in {x[1] for x in active}]
                if not free:
                    break
                gpu = free[0]
                log = (output / f"{phase}_{task['name']}.log").open("a", encoding="utf-8")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                try:
                    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                        "--phase", phase, "--task", task["name"]], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                except BaseException:
                    log.close(); raise
                active.append((task,gpu,proc,log)); pending.remove(task)
            for item in list(active):
                task,gpu,proc,log = item
                rc = proc.poll()
                if rc is not None:
                    active.remove(item); log.close()
                    if rc:
                        raise RuntimeError(f"{phase}/{task['name']} failed (exit {rc}); see {phase}_{task['name']}.log")
            if active:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                append_json(output / "gpu_samples.jsonl", dict(utc=now(), phase=phase, samples=gpu.stdout.splitlines()))
                time.sleep(10)
    finally:
        for _,_,proc,_ in active:
            proc.terminate()
        for _,_,proc,log in active:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            log.close()
        append_json(output / "phase_timing.jsonl", dict(phase=phase, seconds=time.monotonic()-started,
            utc=now(), reserved_gpu_count=len(m["gpus"])))


def summarize(output, official=False):
    import numpy as np
    result, comparisons = [], {}
    folder = "test" if official else "eval"
    for dataset in DATASETS:
        predictions = {}
        for arm in ("base", *ARMS):
            name = f"{dataset}_{arm}"
            summary_path = output / folder / name / "summary.json"
            if not summary_path.exists():
                result.append(dict(job=name, state="incomplete")); continue
            summary = read(summary_path)
            predictions[arm] = records(summary_path.with_name("predictions.jsonl"))
            row = dict(job=name, accuracy=summary["accuracy"], prompts=summary["prompts"],
                       truncated_fraction=summary["truncated_fraction"])
            if arm != "base":
                t = read(output / "train" / name / "summary.json")
                metrics = records(output / "train" / name / "metrics.jsonl")
                row.update(t)
                for key in ("generated_tokens", "responses", "generation_seconds", "backward_seconds", "mixed_groups", "selected", "decode_slots"):
                    row[key] = sum(r[key] for r in metrics)
                row["useful_decode_fraction"] = row["generated_tokens"]/row["decode_slots"] if row["decode_slots"] else None
                gg = sum(r["preupdate_rl_squared_norm"] for r in metrics)
                ee = sum(r["preupdate_residual_squared_norm"] for r in metrics)
                row["lagged_predictor_error_reduction"] = 1-ee/gg if gg>0 else None
                row["zero_gradient_steps"] = sum(r["gradient_norm"]==0 for r in metrics)
            result.append(row)
        comparisons[dataset] = {}
        for left,right in (("base","rloo"),("base","online"),("rloo","online"),("zero","online")):
            if left not in predictions or right not in predictions:
                continue
            a,b = predictions[left], predictions[right]
            if [r["id"] for r in a] != [r["id"] for r in b]:
                raise ValueError("Evaluation IDs differ")
            diff = np.array([y["score"]-x["score"] for x,y in zip(a,b)])
            boot = np.random.default_rng(20261106).choice(diff, (5000,len(diff)), replace=True).mean(1)
            comparisons[dataset][right+"_minus_"+left] = dict(difference=float(diff.mean()),
                paired_prompt_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
                right_only_correct=int((diff>0).sum()), left_only_correct=int((diff<0).sum()))
    payload = dict(results=result, comparisons=comparisons, split="official_test" if official else "heldout_validation",
        note="Single seed; zero is rho=.5 without prediction. Includes all final arms, no winner selection. Profiling/evaluation costs are separate.")
    write_json(output / ("test_results.json" if official else "results.json"), payload)
    print(json.dumps(payload, indent=2), flush=True)


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan","migrate-profiles","pipeline","profile","test","status","summarize","worker"))
    p.add_argument("--output", type=Path, default=Path("runs/math_online_20261106"))
    p.add_argument("--assets", type=Path, default=Path("assets/round2"))
    p.add_argument("--model", type=Path, default=Path("assets/model"))
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--gpus", nargs=4, default=["0","1","2","3"])
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--train-seconds", type=float, default=14400.)
    p.add_argument("--phase", choices=("profile","verify","train","test"))
    p.add_argument("--task")
    args = p.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT/"runs") or args.output == ROOT/"runs":
        p.error("Output must be a child of repository runs")
    if args.command == "plan":
        plan(args); return
    if args.command == "migrate-profiles":
        if args.resume_from is None:
            p.error("migrate-profiles requires --resume-from")
        migrate_profiles(args.resume_from,args.output,identity()); return
    if args.command == "status":
        print(json.dumps(read(args.output / "status.json"), indent=2) if (args.output/"status.json").exists() else "Not launched")
        for path in sorted(args.output.glob("progress_*.json")):
            print(path.stem, json.dumps(read(path)))
        return
    if args.command == "summarize":
        summarize(args.output); return
    import fcntl
    m = validate(args.output)
    if args.command == "worker":
        tasks = task_specs(args.phase) if args.phase else []
        task = next((t for t in tasks if t["name"]==args.task), None)
        if task is None:
            p.error("Unknown task")
        with (args.output / f"worker_{args.phase}_{args.task}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(args.output, task)
        return
    def stop(signum, frame):
        raise KeyboardInterrupt("Launcher stopped; completed checkpoints retained")
    signal.signal(signal.SIGTERM, stop)
    with (args.output.parent / (args.output.name+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(args.output/"status.json", dict(state="running", command=args.command, utc=now(), pid=os.getpid()))
        try:
            if args.command in ("pipeline","profile"):
                if m.get("profile_recovery"):
                    launch(args.output,"verify",m)
                    select_verified_capacity(args.output)
                else:
                    launch(args.output, "profile", m)
                    try:
                        choose_profiles(args.output)
                    except NoSafeProfile:
                        launch(args.output,"verify",m)
                        select_verified_capacity(args.output)
            if args.command == "pipeline":
                launch(args.output, "train", m)
                summarize(args.output)
                if any(not read(args.output/"train"/f"{d}_{a}"/"summary.json")["matched_steps_complete"]
                       for d in DATASETS for a in ARMS):
                    raise RuntimeError("Wall-time guard reached before matched steps. Partial results saved; this is not a completed comparison.")
            if args.command == "test":
                # Refuse to spend official-test compute on an incomplete paired protocol.
                for dataset in DATASETS:
                    for arm in ARMS:
                        if not read(args.output/"train"/f"{dataset}_{arm}"/"summary.json")["matched_steps_complete"]:
                            raise ValueError("All arms must complete matched steps before official test")
                launch(args.output, "test", m)
                summarize(args.output, official=True)
            write_json(args.output/"status.json", dict(state="completed", command=args.command, utc=now()))
        except BaseException as error:
            write_json(args.output/"status.json", dict(state="failed", command=args.command, utc=now(), error=str(error)))
            raise


if __name__ == "__main__":
    main()
