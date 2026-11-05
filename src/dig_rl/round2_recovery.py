"""Explicit, provenance-preserving recovery of the pre-training OOM run."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

from .io import read_rows, sha256, write_json, write_jsonl
from .round2_data import verify
from .round2_protocol import DATASETS, Protocol, object_hash
from .round2_train import records


LEGACY_CODE = "384e39f9ec55b78bb5609744232fde26a76f25e0e4e44237b92ad5e8c48fed0a"
LEGACY_SCRIPT = "6856cbdf82ba2086caa4f8e17d98b6e3790b2e0edf11fc63d264db0a264a08ff"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def locked(path):
    with Path(path).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def collect_prefixes(source, old, runtime):
    if (old["runtime"]["code"], old["runtime"]["script"]) != (LEGACY_CODE, LEGACY_SCRIPT):
        raise ValueError("Only the reviewed 491a0d6 pre-training run can be migrated")
    if old["output"] != str(source.resolve()) or old["runtime"]["versions"] != runtime["versions"]:
        raise ValueError("Source path/libraries changed; do not mix environments")
    if (source / "train").exists() and any((source / "train").rglob("*")):
        raise ValueError("Training outputs exist; this migration only preserves pre-training calibration")
    if sha256(Path(old["model"]) / "dig_model.json") != old["model_id"]:
        raise ValueError("Model changed")
    prefixes, provenance = {}, {}
    for dataset in DATASETS:
        data = Path(old["assets"]) / dataset
        manifest = verify(data)
        if sha256(data / "manifest.json") != old["data"][dataset] or manifest["model_id"] != old["model_id"]:
            raise ValueError("Dataset/model changed: " + dataset)
        pair = [j for j in old["jobs"] if j["dataset"] == dataset]
        if len(pair) != 2 or {j["arm"] for j in pair} != {"rloo", "dig"} or pair[0]["config"] != pair[1]["config"]:
            raise ValueError("Paired experiment configuration differs")
        cfg = Protocol(**pair[0]["config"])
        directory = source / "setup" / dataset
        path = directory / "calibration.jsonl"
        saved = records(path)
        rows = read_rows(data / "calibration.jsonl")[:cfg.calibration_size]
        identity = dict(manifest=object_hash(old), config=cfg.fingerprint())
        if saved and (read(directory / "identity.json") != identity or len(saved) > len(rows)
                      or [r["id"] for r in saved] != [r["id"] for r in rows[:len(saved)]]):
            raise ValueError("Invalid source calibration identity/prefix: " + dataset)
        for r in saved:
            for k in ("uu", "ug", "cross", "nu", "epsilon", "demo_seconds", "rl_seconds", "wall_seconds",
                      "responses", "generated_tokens", "scored_tokens"):
                if not isinstance(r.get(k), (float, int)) or not math.isfinite(r[k]):
                    raise ValueError("Invalid calibration statistic: " + k)
            if r["responses"] != 2*cfg.group_size:
                raise ValueError("Calibration response count changed")
        prefixes[dataset] = saved
        provenance[dataset] = dict(records=len(saved), source_sha256=sha256(path) if path.exists() else None)
    return prefixes, provenance


def migrate(source, output, runtime):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError("Migration requires a fresh output directory; original files are retained")
    old = read(source / "experiment.json")
    with ExitStack() as locks:
        locks.enter_context(locked(source.parent / (source.name+".lock")))
        for job in old["jobs"]:
            if not job["name"].replace("_", "").isalnum():
                raise ValueError("Invalid job name")
            locks.enter_context(locked(source / ("worker-"+job["name"]+".lock")))
        prefixes, provenance = collect_prefixes(source, old, runtime)
        manifest = deepcopy(old)
        manifest.update(output=str(output), runtime=runtime, created_utc=datetime.now(timezone.utc).isoformat())
        manifest["recovery"] = dict(source=str(source), source_manifest_sha256=sha256(source / "experiment.json"),
            source_runtime=old["runtime"], prefixes=provenance,
            note="Unchanged estimator/config/data; completed calibration prefixes reused across numerical-equivalent "
                 "memory execution policies. Setup timings mix policies and are not a speedup comparison. "
                 "Profiles, base evaluation and training are recomputed; original run remains intact.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".round2-recovery-", dir=output.parent) as tmp:
            staged = Path(tmp) / "run"
            write_json(staged / "experiment.json", manifest)
            for dataset, saved in prefixes.items():
                cfg = Protocol(**next(j["config"] for j in manifest["jobs"] if j["dataset"] == dataset))
                directory = staged / "setup" / dataset
                write_json(directory / "identity.json", dict(manifest=object_hash(manifest), config=cfg.fingerprint()))
                write_jsonl(directory / "calibration.jsonl", saved)
            os.replace(staged, output)
    print("Recovered calibration prefixes: " + json.dumps({k:len(v) for k,v in prefixes.items()}), flush=True)
    print("New run: " + str(output), flush=True)
