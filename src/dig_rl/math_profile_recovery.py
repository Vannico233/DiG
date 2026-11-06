"""Reuse historical measurements, never infer live memory from allocator reserve."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from .io import sha256, write_json
from .round2_data import verify
from .round2_protocol import Protocol
from .round2_recovery import locked

LEGACY = dict(code="aa600372a60516de7b45c9e57ef494c4053d00e76496d3d796d609aedeb4529e",
    script="3befad77e9b43c4ddace1cb866faa808b73966f66adc302384b03a2060899c6e",
    math_script="895b26210cb5d639e868471f867c791f8ed6d4b19bcad1b06945ebaa737673f1")


def memory_assessment(report):
    if not report.get("passed"):
        return dict(accepted=False, reason=report.get("error", "Profile did not pass"))
    try:
        total = report["device_total_gib"]
        generation = report["generation_peak_allocated_gib"]
        backward = report["memory"]["peak_allocated_gib"]
        if not report["memory"]["passed"] or any(not math.isfinite(v) or v <= 0 for v in (total,generation,backward)):
            raise ValueError("Invalid memory statistics")
    except (KeyError,TypeError,ValueError):
        return dict(accepted=False, reason="Missing/invalid live generation/backward peaks; a capacity check is required")
    limit = min(.85*total, total-8.)
    peak = max(generation,backward)
    return dict(accepted=peak<limit, peak_allocated_gib=peak, limit_gib=limit,
        reason="Live peaks leave required headroom" if peak<limit else "Live peak exceeds headroom limit",
        note="Allocator reserve includes cached blocks and is recorded separately, not used as live tensor memory")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def migrate_profiles(source, output, runtime):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source==output:
        raise ValueError("Use a fresh recovery directory; original results are preserved")
    with ExitStack() as stack:
        stack.enter_context(locked(source.parent/(source.name+".lock")))
        old = read(source/"experiment.json")
        for path in source.glob("worker_*.lock"):
            stack.enter_context(locked(path))
        if any(old["runtime"].get(k)!=v for k,v in LEGACY.items()):
            raise ValueError("Only the reviewed 52f0f47 pre-training profile run can be recovered")
        if old["runtime"]["versions"] != runtime["versions"] or old["output"] != str(source):
            raise ValueError("Source path/libraries differ")
        if any((source/folder).exists() and any((source/folder).rglob("*")) for folder in ("train","eval","test")):
            raise ValueError("Training/evaluation already exists; this recovery is pre-training only")
        if sha256(Path(old["model"])/"dig_model.json") != old["model_id"]:
            raise ValueError("Model changed")
        profiles = {}
        for dataset in ("gsm8k","math"):
            cfg = Protocol(**old["configs"][dataset])
            data = Path(old["assets"])/dataset
            verified = verify(data)
            if sha256(data/"manifest.json")!=old["data"][dataset] or verified["model_id"]!=old["model_id"]:
                raise ValueError("Prepared data/model differs")
            for mode,size in (("legacy",16),("buffered",16),("buffered",32),("buffered",64)):
                name = f"{dataset}_{mode}_{size}"
                path = source/"profile"/(name+".json")
                r = read(path)
                if r.get("mode")!=mode or r.get("active_sequences")!=size:
                    raise ValueError("Profile identity differs: "+name)
                if r.get("passed"):
                    if r.get("max_new_tokens")!=cfg.max_new_tokens or r.get("responses")!=8*cfg.group_size:
                        raise ValueError("Profile protocol differs: "+name)
                    if any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) or r[k]<=0
                           for k in ("tokens","seconds","tokens_per_second")):
                        raise ValueError("Invalid historical profile: "+name)
                profiles[name] = dict(record=r, sha256=sha256(path))
        manifest = deepcopy(old)
        manifest.update(output=str(output), runtime=runtime, created_utc=datetime.now(timezone.utc).isoformat())
        manifest["profile_recovery"] = dict(source=str(source), source_manifest_sha256=sha256(source/"experiment.json"),
            source_runtime=old["runtime"], profiles={n:v["sha256"] for n,v in profiles.items()},
            note="Historical throughput preserved, not reused as a fresh memory measurement. Conservative <=16 capacity checks required before training; no old training outputs copied.")
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".math-profile-recovery-",dir=output.parent) as temp:
            staged = Path(temp)/"run"
            write_json(staged/"experiment.json",manifest)
            for name,value in profiles.items():
                write_json(staged/"historical_profiles"/(name+".json"),value["record"])
            for name in ("phase_timing.jsonl","gpu_samples.jsonl"):
                if (source/name).exists():
                    (staged/("historical_"+name)).write_bytes((source/name).read_bytes())
            os.replace(staged,output)
    print("Historical profiles preserved; GPU will only run capacity checks, then training.",flush=True)
    print("New run: "+str(output),flush=True)
