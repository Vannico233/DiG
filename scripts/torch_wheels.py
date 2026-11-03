"""Download pinned CUDA wheels persistently, then install them without another download."""
import argparse
from email.parser import Parser
import http.client
import json
import os
from pathlib import Path
import platform
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from offline_setup import RUNTIME, digest, download, open_verified
from pypi_routes import PYPI_MIRRORS, MIRROR_HOSTS, configure_mirror, wheel_download_url

TORCH = dict(filename="torch-2.10.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
             sha256="1d01ffaebf64715c0f507a39463149cb19e596ff702bd4bcf862601f2881dabc", size=916750307)
VISION = dict(filename="torchvision-0.25.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
              sha256="ebf2b495c76097796b9a2eac9290efbcae96e0fd9e5ae52c40eff188610bb440", size=8070804)
TORCH_METADATA_SHA256 = "7f56e1b49211bfcfe45a8c7df42d96a054f69a96fbf6bc4f1b6b6fe8d3f097a7"
# Links and hashes from https://download.pytorch.org/whl/cu128/{torch,torchvision}/
TORCH_CDN = "https://download-r2.pytorch.org/whl/cu128/"


def accelerator_requirements(text):
    result = []
    message = Parser().parsestr(text)
    for raw in message.get_all("Requires-Dist", []):
        req = Requirement(raw)
        name = canonicalize_name(req.name)
        if not (name.startswith(("nvidia-", "cuda-")) or name == "triton"):
            continue
        if req.marker and not req.marker.evaluate({"platform_system": "Linux", "extra": ""}):
            continue
        specs = list(req.specifier)
        if len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version:
            raise RuntimeError("Expected an exact accelerator dependency pin: " + raw)
        result.append((name, specs[0].version))
    if not any(name == "nvidia-cudnn-cu12" for name, _ in result):
        raise RuntimeError("Unexpected CUDA wheel metadata: cuDNN dependency missing")
    return result


def select_wheel(info, name, version, tags):
    ranks = {tag: j for j, tag in enumerate(tags)}
    choices = []
    for entry in info["urls"]:
        filename = entry["filename"]
        if entry.get("yanked") or not filename.endswith(".whl") or Path(filename).name != filename:
            continue
        candidate_name, candidate_version, _, wheel_tags = parse_wheel_filename(filename)
        if candidate_name != canonicalize_name(name) or candidate_version != Version(version):
            continue
        matching = [ranks[tag] for tag in wheel_tags if tag in ranks]
        if not matching:
            continue
        parsed = urllib.parse.urlparse(entry["url"])
        if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
            raise RuntimeError("Unexpected PyPI download host")
        choices.append((min(matching), filename, entry))
    if not choices:
        raise RuntimeError("No compatible PyPI wheel for {}=={}".format(name, version))
    entry = min(choices, key=lambda x: (x[0], x[1]))[2]
    return dict(filename=entry["filename"], url=entry["url"],
                sha256=entry["digests"]["sha256"], size=entry["size"])


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def pypi_info(name, version, directory, tags, attempts=4):
    """Cache only complete usable responses; retry short HTTP bodies as well as connection failures."""
    path = directory / "metadata" / (name + "-" + version + ".json")
    if path.is_file():
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            select_wheel(info, name, version, tags)
            print("Verified cached package metadata:", name, flush=True)
            return info
        except (ValueError, KeyError, TypeError, RuntimeError):
            print("Rebuilding invalid package metadata cache:", name, flush=True)
    url = "https://pypi.org/pypi/{}/{}/json".format(name, version)
    for attempt in range(attempts):
        try:
            print("Reading package metadata: {}=={} ({}/{})".format(name, version, attempt + 1, attempts), flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "DiG-offline-setup"})
            with open_verified(request) as response:
                info = json.load(response)
            select_wheel(info, name, version, tags)
            write_json_atomic(path, info)
            return info
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            failure = error
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead,
                ssl.SSLEOFError, json.JSONDecodeError, UnicodeDecodeError) as error:
            failure = error
        if attempt + 1 < attempts:
            print("Package metadata transfer interrupted; retrying. Wheel partial files are unchanged.", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Cannot read {}=={} metadata after {} attempts. Saved wheel progress is unchanged: {}".format(
        name, version, attempts, failure)) from failure


def valid_cached_plan(plan, core, requirements, tags):
    """Accept the previously saved list format only if versions, coverage and platform still match."""
    if not isinstance(plan, list) or len(plan) != len(core) + len(requirements) or plan[:len(core)] != core:
        return False
    expected = {name: Version(version) for name, version in requirements}
    seen = set()
    try:
        for item in plan[len(core):]:
            filename = item["filename"]
            if "/" in filename or "\\" in filename:
                return False
            name, version, _, wheel_tags = parse_wheel_filename(filename)
            url = urllib.parse.urlparse(item["url"])
            if (expected.get(name) != version or name in seen or not (wheel_tags & set(tags))
                    or type(item["size"]) is not int or item["size"] <= 0
                    or not re.fullmatch("[0-9a-f]{64}", item["sha256"])
                    or url.scheme != "https" or url.hostname != "files.pythonhosted.org"
                    or url.username is not None or url.password is not None):
                return False
            seen.add(name)
    except (ValueError, KeyError, TypeError, AttributeError):
        return False
    return seen == set(expected)


def build_plan(directory, tags=None):
    tags = list(sys_tags()) if tags is None else list(tags)
    plan = [dict(item, url=TORCH_CDN + urllib.parse.quote(item["filename"])) for item in (TORCH, VISION)]
    for item in plan:
        if not (parse_wheel_filename(item["filename"])[3] & set(tags)):
            raise RuntimeError("This host cannot install the pinned Linux Python 3.11 wheels (glibc >= 2.28 required)")
    metadata_file = download(plan[0]["url"] + ".metadata", directory / "torch.metadata", TORCH_METADATA_SHA256)
    requirements = accelerator_requirements(metadata_file.read_text(encoding="utf-8"))
    plan_path = directory / "torch-downloads.json"
    if plan_path.is_file():
        try:
            cached = json.loads(plan_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            cached = None
        if valid_cached_plan(cached, plan, requirements, tags):
            print("Using cached CUDA download plan; skipping PyPI metadata requests.", flush=True)
            return cached
        print("Saved download plan is incomplete or incompatible; rebuilding metadata only.", flush=True)
    for name, version in requirements:
        info = pypi_info(name, version, directory, tags)
        plan.append(select_wheel(info, name, version, tags))
    write_json_atomic(plan_path, plan)
    return plan


def reuse_pip_cache(plan, directory, cache):
    """Opportunistically recover complete old pip HTTP bodies; never trust a cache filename."""
    wanted = {}
    for item in plan:
        if not (directory / item["filename"]).is_file():
            wanted.setdefault(item["size"], []).append(item)
    for source in cache.rglob("*.body"):
        if not source.is_file():
            continue
        byte_count = source.stat().st_size
        candidates = wanted.get(byte_count, [])
        if not candidates:
            continue
        checksum = digest(source)
        for item in candidates:
            if checksum == item["sha256"]:
                target = directory / item["filename"]
                partial = target.with_name(target.name + ".partial")
                shutil.copyfile(source, partial)
                partial.replace(target)
                print("Recovered verified pip cache:", target.name, flush=True)
        wanted[byte_count] = [item for item in candidates if item["sha256"] != checksum]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Resolve metadata and print download sizes; do not download/install wheels")
    args = parser.parse_args()
    if (platform.system(), platform.machine(), sys.version_info[:2]) != ("Linux", "x86_64", (3, 11)):
        raise RuntimeError("Run with the project's Linux x86_64 Python 3.11")
    if Path(sys.prefix).resolve() != (RUNTIME / "py311").resolve():
        raise RuntimeError("Use .runtime/py311/bin/python")
    mirror = configure_mirror()
    directory = RUNTIME / "wheels"
    directory.mkdir(parents=True, exist_ok=True)
    plan = build_plan(directory)
    total = sum(item["size"] for item in plan)
    print("Pinned CUDA wheel set: {} files, {:.2f} GB before cache reuse.".format(len(plan), total / 1e9), flush=True)
    for item in plan:
        print("  {:.1f} MB  {}".format(item["size"] / 1e6, item["filename"]), flush=True)
    if args.plan_only:
        return
    reuse_pip_cache(plan, directory, RUNTIME / "pip-cache")
    wheels = [download(wheel_download_url(item, mirror), directory / item["filename"], item["sha256"], item["size"])
              for item in plan]
    # General Python dependencies are resolved by the next project pip-install step.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *map(str, wheels)], check=True)


if __name__ == "__main__":
    main()
