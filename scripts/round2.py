"""CPU prepare; GPU profile -> calibrate -> run. Eight bounded, resumable workers."""
import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dig_rl.io import sha256, write_json, code_id, version_info, read_rows
from dig_rl.round2_protocol import Protocol, DATASETS, jobs, object_hash
from dig_rl.round2_data import verify


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def runtime_identity():
    versions = version_info()
    for name in ("math-verify", "latex2sympy2-extended", "pillow"):
        versions[name] = importlib.metadata.version(name)
    return dict(code=code_id(), script=sha256(__file__), versions=versions)


def plan(args):
    if args.output.exists():
        raise ValueError("Use a fresh round-two output directory")
    configs, manifests = {}, {}
    model = sha256(args.model / "dig_model.json")
    for dataset in DATASETS:
        manifest = verify(args.assets / dataset)
        if manifest["model_id"] != model:
            raise ValueError("Data/model mismatch")
        configs[dataset] = Protocol(dataset=dataset, max_new_tokens=8192 if dataset == "math" else 4096,
            max_input_tokens=2048 if dataset == "gsm8k" else (4096 if dataset == "math" else 8192),
            candidates=args.candidates, active_sequences=args.active_sequences,
            max_steps=args.steps, train_seconds=args.train_seconds, seed=args.seed,
            optimizer=args.optimizer, learning_rate=args.learning_rate if args.learning_rate is not None else (
                .01 if args.optimizer == "sgd" else 1e-6))
        manifests[dataset] = sha256(args.assets / dataset / "manifest.json")
    args.output.mkdir(parents=True)
    write_json(args.output / "experiment.json", dict(schema=2, created_utc=now(),
        assets=str(args.assets.resolve()), model=str(args.model.resolve()), output=str(args.output.resolve()),
        model_id=model, data=manifests, runtime=runtime_identity(), jobs=jobs(configs, args.gpus),
        interpretation="Single-seed exploratory fixed-rho pilot. Three-seed and isolated timing stages require separate runs.",
        budgets=dict(training_seconds_per_worker=args.train_seconds, max_steps=args.steps),
        notes="No test split opened by profile/calibrate/run. Base evaluation is shared by both arms."))


def validate(output):
    m = read(output / "experiment.json")
    if m["output"] != str(output.resolve()) or m["runtime"] != runtime_identity():
        raise ValueError("Run path, code or libraries changed; use exact environment or a new run")
    if sha256(Path(m["model"]) / "dig_model.json") != m["model_id"]:
        raise ValueError("Model changed")
    for dataset, digest in m["data"].items():
        if sha256(Path(m["assets"]) / dataset / "manifest.json") != digest:
            raise ValueError("Data manifest changed")
    return m


def progress(output, name, phase, **values):
    write_json(output / ("progress_"+name+".json"), dict(utc=now(), pid=os.getpid(), phase=phase, **values))


def worker(args):
    import torch
    from dig_rl.round2_engine import BatchedEngine
    from dig_rl.round2_train import train, evaluate, records
    from dig_rl.audit import measured, fit_scale
    m = validate(args.output)
    job = next(j for j in m["jobs"] if j["name"] == args.job)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu"]:
        raise ValueError("GPU assignment differs")
    cfg = Protocol(**job["config"])
    label = args.job + ("_"+args.phase if args.phase != "train" else "")
    update = lambda phase, **v: progress(args.output, label, phase, **v)
    data = Path(m["assets"]) / job["dataset"]
    verify(data)
    if args.phase == "train":
        profiles = read(args.output / "profile_decision.json")
        cfg = replace(cfg, active_sequences=profiles[job["dataset"]]["active_sequences"])
    update("loading")
    engine = BatchedEngine(m["model"], cfg, data)
    identity = dict(manifest=object_hash(m), config=cfg.fingerprint())
    try:
        if args.phase.startswith("profile"):
            mode = int(args.phase[-1])
            rows = read_rows(data / "train.jsonl")[:4]
            engine.rollout_many([(rows[0], cfg.seed)], count=2, max_tokens=32)
            # A real image-aware backward is mandatory, not an import-only kernel check.
            engine.zero_grad(); engine.demo_backward(rows[0]); engine.gradient(); engine.zero_grad()
            if mode == 2:
                token = os.environ["DIG_PROFILE_TOKEN"]
                barrier = args.output / "profile" / "barriers" / token
                write_json(barrier / (args.job+".json"), dict(ready=True))
                deadline = time.monotonic()+600
                while not all((barrier / (job["dataset"]+"_"+arm+".json")).exists() for arm in ("rloo", "dig")):
                    if time.monotonic() > deadline:
                        raise TimeoutError("The paired profiling worker did not become ready")
                    time.sleep(.1)
            torch.cuda.reset_peak_memory_stats()
            engine.sync(); started = time.monotonic()
            groups = engine.rollout_many([(r, cfg.seed+10007*j) for j,r in enumerate(rows)], max_tokens=512)
            engine.sync(); duration = time.monotonic()-started
            write_json(args.output / "profile" / args.phase / (args.job+".json"), dict(
                started=time.time()-duration, finished=time.time(), seconds=duration,
                generated_tokens=sum(g["generated_tokens"] for g in groups),
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                active_sequences=cfg.active_sequences, workers=mode))
        elif args.phase == "calibrate":
            directory = args.output / "setup" / job["dataset"]
            directory.mkdir(parents=True, exist_ok=True)
            meta = directory / "identity.json"
            if meta.exists() and read(meta) != identity:
                raise ValueError("Setup identity differs")
            write_json(meta, identity)
            rows = read_rows(data / "calibration.jsonl")[:cfg.calibration_size]
            saved = records(directory / "calibration.jsonl", repair=True)
            if [r["id"] for r in saved] != [r["id"] for r in rows[:len(saved)]] or len(saved)>len(rows):
                raise ValueError("Invalid calibration prefix")
            from dig_rl.io import append_json
            for j in range(len(saved), len(rows)):
                update("calibrating", completed=j, total=len(rows))
                start = time.monotonic()
                record, _ = measured(engine, rows[j], cfg.seed+10007*j, 0.)
                record["wall_seconds"] = time.monotonic()-start
                append_json(directory / "calibration.jsonl", record)
                saved.append(record)
            write_json(directory / "calibration.json", dict(a=fit_scale(saved), identity=identity,
                seconds=sum(r["wall_seconds"] for r in saved), generated_tokens=sum(r["generated_tokens"] for r in saved),
                status="calibrated_only_not_a_passed_audit", questions=len(saved)))
            evaluate(engine, read_rows(data / "validation.jsonl")[:cfg.validation_size], cfg,
                args.output / "eval" / (job["dataset"]+"_base"), identity, update)
        elif args.phase == "train":
            setup = read(args.output / "setup" / job["dataset"] / "calibration.json")
            # Profile currently changes concurrency only, never mathematical protocol.
            if setup["identity"]["manifest"] != object_hash(m):
                raise ValueError("Calibration from a different experiment")
            a = setup["a"] if job["arm"] == "dig" else 0.
            trained = train(engine, read_rows(data / "train.jsonl"), cfg,
                args.output / "train" / args.job, identity, a, job["rho"], update)
            evaluate(engine, read_rows(data / "validation.jsonl")[:cfg.validation_size], cfg,
                args.output / "eval" / args.job, dict(checkpoint=trained["checkpoint_sha256"]), update)
        update("completed")
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


def profile_decision(output, m):
    decisions = {}
    for dataset in DATASETS:
        one = read(output / "profile/profile1" / (dataset+"_rloo.json"))
        two = [read(output / "profile/profile2" / (dataset+"_"+arm+".json")) for arm in ("rloo", "dig")]
        single = one["generated_tokens"] / one["seconds"]
        duration = max(r["finished"] for r in two)-min(r["started"] for r in two)
        double = sum(r["generated_tokens"] for r in two)/duration
        fit = sum(r["peak_reserved_gib"] for r in two) < 64
        decisions[dataset] = dict(slots=2 if fit and double >= 1.15*single else 1,
            single_tokens_per_second=single, dual_tokens_per_second=double,
            active_sequences=one["active_sequences"], peak_reserved_sum_gib=sum(r["peak_reserved_gib"] for r in two),
            note="512-token screening; full-length OOM still fails loudly. This is not an algorithm speedup measurement.")
    write_json(output / "profile_decision.json", decisions)
    print(json.dumps(decisions, indent=2), flush=True)


def launch(output, phase, m, slots):
    from contextlib import ExitStack
    # Pre-existing work is never killed or silently co-located.
    spec = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if spec:
        raise RuntimeError("GPU compute processes already running; no new workers launched")
    pending = list(m["jobs"] if phase in ("train", "profile2") else [j for j in m["jobs"] if j["arm"]=="rloo"])
    active = []
    profile_token = str(time.time_ns())
    stage_start = time.monotonic()
    with ExitStack() as stack:
        try:
            while pending or active:
                for job in list(pending):
                    label = job["name"]+("_"+phase if phase != "train" else "")
                    state = output / ("progress_"+label+".json")
                    if not phase.startswith("profile") and state.exists() and read(state)["phase"] == "completed":
                        pending.remove(job); continue
                    count = sum(j["gpu"]==job["gpu"] for j, _ in active)
                    if count >= slots[job["dataset"]]:
                        continue
                    log = stack.enter_context((output / (label+".log")).open("a", encoding="utf-8"))
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpu"], PYTHONUNBUFFERED="1",
                        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                        DIG_PROFILE_TOKEN=profile_token)
                    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                        "--job", job["name"], "--phase", phase], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                    active.append((job, proc)); pending.remove(job)
                for job, proc in list(active):
                    rc = proc.poll()
                    if rc is not None:
                        active.remove((job, proc))
                        if rc:
                            raise RuntimeError(job["name"]+" failed; inspect its log. Checkpoints retained.")
                if active:
                    status(output)
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                    from dig_rl.io import append_json
                    append_json(output / "gpu_samples.jsonl", dict(utc=now(), phase=phase, samples=gpu.stdout.splitlines()))
                    time.sleep(10)
        finally:
            for _, proc in active:
                proc.terminate()
            for _, proc in active:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
            from dig_rl.io import append_json
            seconds = time.monotonic()-stage_start
            append_json(output / "phase_timing.jsonl", dict(utc=now(), phase=phase, wall_seconds=seconds,
                reserved_gpu_seconds=seconds*len({j["gpu"] for j in m["jobs"]}),
                convention="All four physical cards reserved for the phase; co-resident workers are not counted twice"))


def status(output):
    states = []
    for path in sorted(output.glob("progress_*.json")):
        row = read(path)
        states.append(dict(job=path.stem.removeprefix("progress_"), **row))
    print(json.dumps(dict(utc=now(), workers=states), ensure_ascii=False), flush=True)


def summarize(output):
    m = validate(output)
    result = []
    for job in m["jobs"]:
        name, dataset = job["name"], job["dataset"]
        train_path = output / "train" / name / "summary.json"
        eval_path = output / "eval" / name / "summary.json"
        if not train_path.exists() or not eval_path.exists():
            result.append(dict(job=name, status="incomplete")); continue
        t, e = read(train_path), read(eval_path)
        base = read(output / "eval" / (dataset+"_base") / "summary.json")
        setup = read(output / "setup" / dataset / "calibration.json")
        from dig_rl.round2_train import records
        metrics = records(output / "train" / name / "metrics.jsonl")
        result.append(dict(job=name, **t, accuracy=e["accuracy"], base_accuracy=base["accuracy"],
            gain=e["accuracy"]-base["accuracy"], truncated_fraction=e["truncated_fraction"],
            zero_gradient_steps=sum(r["gradient_norm"]==0 for r in metrics),
            mixed_groups=sum(r["mixed_groups"] for r in metrics), selected_groups=sum(r["selected"] for r in metrics),
            setup_seconds=setup["seconds"] if job["arm"]=="dig" else 0.,
            gate="not_audited", effective_arm=("dig_zero_predictor" if t["a"]==0 and job["arm"]=="dig" else job["arm"])))
    import numpy as np
    from dig_rl.round2_train import records
    comparisons = {}
    for dataset in DATASETS:
        paths = [output / "eval" / (dataset+"_"+arm) / "predictions.jsonl" for arm in ("rloo","dig")]
        if not all(p.exists() for p in paths) or any(r.get("status")=="incomplete" for r in result if r["job"].startswith(dataset+"_")):
            continue
        left,right = [records(p) for p in paths]
        if [r["id"] for r in left] != [r["id"] for r in right]:
            raise ValueError("Paired evaluations cover different questions")
        difference = np.array([b["score"]-a["score"] for a,b in zip(left,right)])
        means = np.random.default_rng(20261105).choice(difference,(2000,len(difference)),replace=True).mean(1)
        comparisons[dataset] = dict(dig_minus_rloo=float(difference.mean()),
                                    paired_prompt_bootstrap_95=np.quantile(means,[.025,.975]).tolist())
    write_json(output / "results.json", dict(results=result, comparisons=comparisons,
        phase_timing=records(output / "phase_timing.jsonl"),
        note="Single-seed validation pilot. Co-resident worker wall time is not exclusive GPU-hours; no isolated speedup claim."))
    print(json.dumps(result, indent=2), flush=True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "plan", "profile", "calibrate", "run", "pipeline", "status", "summarize", "worker"))
    parser.add_argument("--output", type=Path, default=Path("runs/round2_20261105"))
    parser.add_argument("--assets", type=Path, default=Path("assets/round2"))
    parser.add_argument("--model", type=Path, default=Path("assets/model"))
    parser.add_argument("--gpus", nargs=4, default=["0","1","2","3"])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--active-sequences", type=int, default=16)
    parser.add_argument("--train-seconds", type=float, default=7200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=None, help="Default AdamW=1e-6; SGD=0.01")
    parser.add_argument("--phase", choices=("profile1", "profile2", "calibrate", "train"))
    parser.add_argument("--job")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "runs") or args.output == ROOT / "runs":
        parser.error("Output must be a child of repository runs")
    if args.command == "prepare":
        from dig_rl.round2_data import prepare
        prepare(args.assets, args.model)
    elif args.command == "plan":
        plan(args)
    elif args.command == "status":
        status(args.output)
    elif args.command == "summarize":
        summarize(args.output)
    elif args.command == "worker":
        import fcntl
        if args.job not in {j["name"] for j in validate(args.output)["jobs"]} or args.phase is None:
            parser.error("Worker requires a declared job and phase")
        with (args.output / ("worker-"+args.job+".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(args)
    else:
        import fcntl
        def stop(signum, frame):
            raise KeyboardInterrupt("Round-two launcher stopped")
        signal.signal(signal.SIGTERM, stop)
        with (args.output.parent / (args.output.name+".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            m = validate(args.output)
            write_json(args.output / "status.json", dict(state="running", command=args.command, pid=os.getpid(), utc=now()))
            try:
                if args.command == "profile" or (args.command == "pipeline" and not (args.output / "profile_decision.json").exists()):
                    launch(args.output, "profile1", m, {d:1 for d in DATASETS})
                    launch(args.output, "profile2", m, {d:2 for d in DATASETS})
                    profile_decision(args.output, m)
                if args.command in ("calibrate", "pipeline"):
                    launch(args.output, "calibrate", m, {d:1 for d in DATASETS})
                if args.command in ("run", "pipeline"):
                    decision = read(args.output / "profile_decision.json")
                    launch(args.output, "train", m, {d:decision[d]["slots"] for d in DATASETS})
                    summarize(args.output)
                write_json(args.output / "status.json", dict(state="completed", command=args.command, utc=now()))
            except BaseException as error:
                write_json(args.output / "status.json", dict(state="failed", command=args.command, error=str(error), utc=now()))
                raise


if __name__ == "__main__":
    main()
