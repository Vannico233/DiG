import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def read_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows or any(not all(r.get(k) for k in ("id", "question", "answer", "demo")) for r in rows):
        raise ValueError(f"Invalid/empty prepared dataset: {path}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    return rows


def verify_data(directory, cfg=None, model_id=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if cfg is not None:
        keys = ("horizon", "max_prompt_tokens", "calibration_size", "audit_size", "validation_size", "split_seed")
        if any(manifest["config"][k] != getattr(cfg, k) for k in keys):
            raise ValueError("Prepared dataset protocol differs; use matching config or prepare new assets")
    if model_id is not None and manifest["model_id"] != model_id:
        raise ValueError("Prepared tokenizer/model identity differs")
    seen = set()
    for split, meta in manifest["files"].items():
        path = directory / f"{split}.jsonl"
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"Data checksum mismatch: {path}")
        rows = read_rows(path)
        ids = {r["id"] for r in rows}
        if len(rows) != meta["rows"] or seen & ids:
            raise ValueError("Data count mismatch or split overlap")
        seen.update(ids)
    return manifest


def atomic_save(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def code_id():
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for p in sorted((root / "src").rglob("*.py")):
        h.update(str(p.relative_to(root)).replace("\\", "/").encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def version_info():
    import importlib.metadata as m
    info = {}
    for name in ("torch", "transformers", "peft", "datasets", "huggingface-hub", "fla-core", "causal-conv1d"):
        try:
            info[name] = m.version(name)
        except m.PackageNotFoundError:
            info[name] = None
    return info


def experiment_identity(cfg, model_id, data_dir, initial_checkpoint=None):
    return dict(config=cfg.fingerprint(), data=sha256(Path(data_dir) / "manifest.json"),
                model=model_id, code=code_id(), libraries=version_info(), cuda=torch.version.cuda,
                initial_checkpoint=sha256(initial_checkpoint) if initial_checkpoint else None)
