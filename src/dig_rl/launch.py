"""Two independent GPUs; bounded stages, no implicit full experiment sweep."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import write_json


def plan(stage, assets, output, config, datasets, gpus):
    root = Path(output)
    common = ["--model-dir", str(Path(assets) / "model"), "--config", str(config)]
    jobs = []
    for j, dataset in enumerate(datasets):
        data = ["--data-dir", str(Path(assets) / "data" / dataset)]
        if stage in ("audit", "smoke"):
            dest = root / stage / dataset
            jobs.append((f"{stage}_{dataset}", gpus[j % 2], [sys.executable, "-m", f"dig_rl.{stage}", *common, *data, "--output", str(dest)]))
        elif stage == "preflight":
            for gpu in gpus:
                dest = root / f"preflight_{dataset}_gpu{gpu}.json"
                jobs.append((f"preflight_{dataset}_{gpu}", gpu, [sys.executable, "-m", "dig_rl.preflight", *common, *data, "--output", str(dest)]))
        elif stage == "train":
            for arm, gpu in zip(("rloo", "dig"), gpus):
                dest = root / "train" / f"{dataset}_{arm}"
                cmd = [sys.executable, "-m", "dig_rl.train", *common, *data, "--arm", arm, "--output", str(dest)]
                if arm == "dig":
                    cmd += ["--audit-file", str(root / "audit" / dataset / "decision.json")]
                jobs.append((f"{dataset}_{arm}", gpu, cmd))
        else:
            for arm, gpu in zip(("rloo", "dig"), gpus):
                checkpoint = root / "train" / f"{dataset}_{arm}" / "checkpoint.pt"
                dest = root / "eval" / f"{dataset}_{arm}"
                jobs.append((f"eval_{dataset}_{arm}", gpu, [sys.executable, "-m", "dig_rl.evaluate", *common, *data,
                    "--checkpoint", str(checkpoint), "--output", str(dest)]))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["preflight", "smoke", "audit", "train", "evaluate"], required=True)
    p.add_argument("--assets", default="assets")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", default="configs/qwen35_4b.json")
    p.add_argument("--datasets", nargs="+", choices=["gsm8k", "math_numeric"], default=["gsm8k", "math_numeric"])
    p.add_argument("--gpus", nargs=2, default=["0", "1"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.gpus[0] == args.gpus[1]:
        p.error("Two distinct GPUs required")
    jobs = plan(args.stage, args.assets, args.output, args.config, args.datasets, args.gpus)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    if args.stage in ("train", "evaluate"):
        for dataset in args.datasets:
            decision = args.output / "audit" / dataset / "decision.json"
            if not decision.is_file():
                raise ValueError(f"Run audit first: {decision}")
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / f"{args.stage}_status.json"
    if status_path.exists():
        raise ValueError("Stage already launched; inspect status and resume individual commands explicitly")
    active, pending, status = {}, list(jobs), {}
    try:
        while pending or active:
            for job in list(pending):
                name, gpu, cmd = job
                if gpu in active:
                    continue
                pending.remove(job)
                if args.stage in ("train", "evaluate") and name.endswith("_dig"):
                    dataset = name.removeprefix("eval_").removesuffix("_dig")
                    decision = args.output / "audit" / dataset / "decision.json"
                    if not decision.is_file():
                        raise ValueError(f"Run audit first: {decision}")
                    if not json.loads(decision.read_text())["enabled"]:
                        status[name] = dict(status="skipped", reason="audit gate failed; no duplicate fallback training")
                        write_json(status_path, status)
                        continue
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
                log = (args.output / f"{name}.log").open("w", encoding="utf-8")
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (name, proc, log)
                status[name] = dict(status="running", pid=proc.pid, gpu=gpu, command=cmd)
                write_json(status_path, status)
            for gpu, (name, proc, log) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                del active[gpu]
                status[name].update(status="completed" if rc == 0 else "failed", returncode=rc)
                write_json(status_path, status)
                if rc:
                    raise RuntimeError(f"{name} failed; see its log. Remaining jobs stopped.")
            time.sleep(1)
    finally:
        for name, proc, log in active.values():
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            log.close()
            status[name].update(status="interrupted", returncode=proc.returncode)
        write_json(status_path, status)


if __name__ == "__main__":
    main()
