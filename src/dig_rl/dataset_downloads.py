"""Pinned VLM Parquet downloads with persistent ranges and SHA256 verification."""
import json
import os
from pathlib import Path
import re
import shutil

from .hf_downloads import hf_retry
from .io import sha256, write_json
from .model_transfer import Route, download_parallel, transfer_lock


# These public repositories embed images in their Parquet files. Fail closed if
# their layout changes, rather than silently losing shards or inferring splits.
REPOS = {"hiyouga/geometry3k", "derek-thomas/ScienceQA"}
SHARD = re.compile(r"data/(train|test)-(\d{5})-of-(\d{5})(?:-[0-9a-f]+)?\.parquet")


def validate_files(items):
    if not isinstance(items, list) or not items:
        raise ValueError("Empty dataset download plan")
    names, groups = set(), {"train": [], "test": []}
    for item in items:
        match = SHARD.fullmatch(item["filename"])
        if (not match or item["filename"] in names or type(item["size"]) is not int
                or item["size"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Invalid pinned dataset file metadata")
        names.add(item["filename"])
        split, index, count = match.groups()
        groups[split].append((int(index), int(count)))
    for shards in groups.values():
        if (not shards or len({n for _, n in shards}) != 1
                or sorted(i for i, _ in shards) != list(range(shards[0][1]))):
            raise ValueError("Missing or inconsistent dataset shards")
    return items


def dataset_plan(api, repo, revision, directory):
    if repo not in REPOS or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Unsupported dataset or non-immutable revision")
    path = Path(directory) / "download_plan.json"
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if (plan["repo"], plan["revision"]) != (repo, revision):
            raise ValueError("Saved dataset plan belongs to another revision")
        return validate_files(plan["files"])
    info = hf_retry("Reading pinned dataset hashes: " + repo, lambda: api.dataset_info(
        repo, revision=revision, files_metadata=True, timeout=30))
    if info.sha != revision or info.private or info.gated:
        raise ValueError("Dataset downloads require the pinned public, ungated repository")
    items = []
    for entry in info.siblings:
        if SHARD.fullmatch(entry.rfilename):
            if not entry.lfs:
                raise ValueError("Dataset file has no SHA256 metadata")
            items.append(dict(filename=entry.rfilename, size=entry.lfs.size, sha256=entry.lfs.sha256))
        elif entry.rfilename.endswith(".parquet") and not entry.rfilename.startswith("data/validation-"):
            raise ValueError("Unexpected dataset Parquet layout: " + entry.rfilename)
    items.sort(key=lambda item: item["filename"])
    validate_files(items)
    write_json(path, dict(repo=repo, revision=revision, files=items))
    return items


def dataset_routes(repo, revision, filename, source="auto"):
    from huggingface_hub import hf_hub_url
    routes = [Route("official", hf_hub_url(repo, filename, repo_type="dataset", revision=revision,
                                         endpoint="https://huggingface.co")),
              Route("hf-mirror", hf_hub_url(repo, filename, repo_type="dataset", revision=revision,
                                          endpoint="https://hf-mirror.com"), True),
              Route("official-direct", hf_hub_url(repo, filename, repo_type="dataset", revision=revision,
                                                endpoint="https://huggingface.co"), True)]
    if source != "auto":
        routes = [r for r in routes if r.name == source]
    if not routes:
        raise ValueError("DIG_DATA_SOURCE must be auto, official, hf-mirror or official-direct")
    return routes


def reuse_hub_cache(repo, revision, item, path):
    """Copy, never move or modify, Hub files left by the previous loader."""
    from huggingface_hub import constants, try_to_load_from_cache
    if path.exists():
        return
    cached = try_to_load_from_cache(repo, item["filename"], revision=revision, repo_type="dataset")
    if (isinstance(cached, str) and Path(cached).is_file()
            and Path(cached).stat().st_size == item["size"] and sha256(cached) == item["sha256"]):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".importing")
        shutil.copyfile(cached, temporary)
        os.replace(temporary, path)
        print("Reused verified Hub dataset file: " + path.name, flush=True)
        return
    partial = path.parent / ".dig-downloads" / (path.name + "." + item["sha256"] + ".partial")
    if partial.exists() or partial.with_suffix(".chunks").exists():
        return
    old = Path(constants.HF_HUB_CACHE) / ("datasets--" + repo.replace("/", "--")) / "blobs" / (
        item["sha256"] + ".incomplete")
    if old.is_file() and 0 < old.stat().st_size <= item["size"]:
        partial.parent.mkdir(parents=True, exist_ok=True)
        temporary = partial.with_suffix(".importing")
        shutil.copyfile(old, temporary)
        os.replace(temporary, partial)
        print(f"Imported Hub partial: {path.name}, {partial.stat().st_size / 1e6:.2f} MB; "
              "full SHA256 will be checked after download", flush=True)


def load_pinned_parquet(api, repo, revision, directory):
    from datasets import load_dataset
    directory = Path(directory)
    workers = int(os.environ.get("DIG_DATA_CONNECTIONS", "4"))
    source = os.environ.get("DIG_DATA_SOURCE", "auto")
    data_files = {"train": [], "test": []}
    with transfer_lock(directory):
        items = dataset_plan(api, repo, revision, directory)
        print(f"Dataset {repo}: {len(items)} files; "
              f"{sum(i['size'] for i in items) / 1e6:.1f} MB before cache reuse", flush=True)
        for item in items:
            path = directory / item["filename"]
            reuse_hub_cache(repo, revision, item, path)
            # Larger ranges reduce CDN redirect/rate-limit overhead. Interrupted
            # ranges still retain their validated prefixes, even below 8 MiB.
            download_parallel(dataset_routes(repo, revision, item["filename"], source), path, item,
                              workers=workers, chunk_size=8 * 1024 * 1024, attempts=12, label="Dataset")
            data_files[SHARD.fullmatch(item["filename"]).group(1)].append(str(path.resolve()))
    # Built-in local Parquet reader: no new Hub download/redirect, preserving
    # embedded Image/Sequence feature metadata and the original split ordering.
    return load_dataset("parquet", data_files=data_files, cache_dir=str(directory / ".arrow-cache"))
