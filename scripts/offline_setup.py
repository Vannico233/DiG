"""Shared Linux runtime preparation; bootstrap commands also work with Python 3.9."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
WHEEL = "causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
WHEEL_SHA256 = "b872d04d98c80856fd897660eac7129cc2ab51db9694530ce0cd24e1b57200fe"
WHEEL_URL = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/" + WHEEL.replace("+", "%2B")
PACKAGES = {"torch": "2.10.0", "torchvision": "0.25.0", "transformers": "5.22.1",
            "peft": "0.21.2", "accelerate": "1.15.0", "huggingface-hub": "1.33.0",
            "fla-core": "0.6.0", "causal-conv1d": "1.7.0"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected=None):
    path = Path(path)
    if path.is_file() and expected is not None and digest(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "DiG-offline-setup"})
    print("Downloading", path.name, flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if expected is not None and digest(partial) != expected:
        raise RuntimeError("Downloaded wheel checksum mismatch; re-run CPU preparation")
    os.replace(partial, path)
    return path


def install_micromamba():
    binary = RUNTIME / "tools/bin/micromamba"
    if binary.is_file():
        return
    archive = download("https://micro.mamba.pm/api/micromamba/linux-64/latest",
                       RUNTIME / "downloads/micromamba.tar.bz2")
    # Extract just the expected executable, never archive-supplied paths or links.
    with tarfile.open(archive, "r:bz2") as tar:
        member = tar.getmember("bin/micromamba")
        if not member.isfile():
            raise RuntimeError("Unexpected micromamba archive")
        binary.parent.mkdir(parents=True, exist_ok=True)
        tmp = binary.with_suffix(".partial")
        with tar.extractfile(member) as source, tmp.open("wb") as out:
            shutil.copyfileobj(source, out)
        tmp.chmod(0o755)
        os.replace(tmp, binary)


def validate_target(system, machine, python_minor, torch_version, cuda, abi):
    if (system, machine, tuple(python_minor)) != ("Linux", "x86_64", (3, 11)):
        raise RuntimeError("The offline runtime requires Linux x86_64 and its own Python 3.11")
    if torch_version.split("+")[0] != "2.10.0" or cuda != "12.8" or not abi:
        raise RuntimeError("Expected torch 2.10.0 / CUDA 12.8 / CXX11 ABI TRUE; do not install CPU-only torch")


def check_runtime():
    import torch
    validate_target(platform.system(), platform.machine(), sys.version_info[:2],
                    torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)
    expected_prefix = (RUNTIME / "py311").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError("Use source scripts/activate_offline.sh to select the shared Python")
    return torch


def installed_packages():
    for name, expected in PACKAGES.items():
        actual = metadata.version(name)
        if actual.split("+")[0] != expected:
            raise RuntimeError("Unexpected {} version: {}; expected {}".format(name, actual, expected))
    for name in ("datasets", "triton", "numpy", "safetensors", "tokenizers", "ninja", "packaging"):
        metadata.version(name)  # All these packages must already exist before going offline.
    return {dist.metadata["Name"].lower().replace("_", "-"): dist.version
            for dist in metadata.distributions() if dist.metadata["Name"]}


def check_compiler():
    compiler = RUNTIME / "py311/bin/x86_64-conda-linux-gnu-cc"
    if not compiler.is_file():
        raise RuntimeError("C compiler missing: re-run CPU preparation; Triton needs it on GPU")
    # Triton creates a small C launcher on the GPU host even with prebuilt CUDA extensions.
    with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
        src, lib = Path(directory) / "check.c", Path(directory) / "check.so"
        src.write_text("int dig_check(void) { return 7; }\n", encoding="utf-8")
        subprocess.run([str(compiler), "-shared", "-fPIC", str(src), "-o", str(lib)], check=True)
        import ctypes
        if ctypes.CDLL(str(lib)).dig_check() != 7:
            raise RuntimeError("C compiler/loader check failed")


def local_assets():
    from dig_rl.config import Config
    from dig_rl.io import verify_data
    from safetensors import safe_open
    from transformers import AutoProcessor
    model = ROOT / "assets/model"
    identity = digest(model / "dig_model.json")
    AutoProcessor.from_pretrained(model, local_files_only=True)
    index = model / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
    if not shards:
        raise RuntimeError("Empty model shard index")
    for name in shards:
        path = (model / name).resolve()
        if not path.is_relative_to(model.resolve()):
            raise RuntimeError("Model shard path escapes the local model directory")
        # Checks header and expected data length without loading 4B parameters into RAM.
        with safe_open(path, framework="pt", device="cpu") as handle:
            if not list(handle.keys()):
                raise RuntimeError("Empty model shard: " + name)
    cfg = Config.load(ROOT / "configs/qwen35_4b.json")
    files = [p for p in model.rglob("*") if p.is_file() and ".cache" not in p.relative_to(model).parts]
    for name in ("gsm8k", "math_numeric"):
        directory = ROOT / "assets/data" / name
        manifest = verify_data(directory, cfg, identity)
        required = {"train", "calibration", "audit", "validation", "test"}
        if set(manifest["files"]) != required:
            raise RuntimeError("Incomplete prepared splits: " + name)
        files += [directory / "manifest.json"] + [directory / (split + ".jsonl") for split in required]
    files += [ROOT / "assets/download_lock.json", ROOT / "configs/qwen35_4b.json",
              ROOT / "requirements/h200-constraints.txt", ROOT / "pyproject.toml",
              RUNTIME / "pip-freeze.txt", RUNTIME / "conda-explicit.txt"]
    files += sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.*"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(files))}


def validate_ready(saved, root, prefix, packages, files):
    if saved.get("root") != str(root) or saved.get("prefix") != str(prefix):
        raise RuntimeError("Shared directory mount path changed. Use the same absolute path in CPU and GPU spaces")
    if saved.get("packages") != packages:
        raise RuntimeError("Runtime packages changed since CPU preparation")
    if saved.get("files") != files:
        raise RuntimeError("Prepared model/data/code/config changed; return to CPU and re-run preparation")


def readiness(seal=False, gpu=False):
    # Set offline mode before any Hugging Face/Transformers import, even when sealing on CPU.
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", USE_HUB_KERNELS="NO")
    torch = check_runtime()
    packages = installed_packages()
    check_compiler()
    print("Verifying local assets (including weight-file hashes)...", flush=True)
    files = local_assets()
    ready = dict(root=str(ROOT), prefix=str(Path(sys.prefix).resolve()), packages=packages,
                 files=files, python=sys.version, platform=platform.platform(), libc=platform.libc_ver())
    marker = RUNTIME / "CPU_READY.json"
    if seal:
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, marker)
        print("CPU_READY written. No GPU execution has been tested.", flush=True)
    else:
        if not marker.is_file():
            raise RuntimeError("CPU_READY.json missing: finish CPU preparation first")
        validate_ready(json.loads(marker.read_text()), ROOT, Path(sys.prefix).resolve(), packages, files)
        print("Shared runtime and assets verified.", flush=True)
    if gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("Two visible CUDA GPUs required. Check nvidia-smi, driver and CUDA_VISIBLE_DEVICES")
        subprocess.run(["nvidia-smi"], check=True)
        for device in range(2):
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("BF16 unavailable on GPU {}".format(device))
                x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
                assert float((x @ x).sum().item()) == 32768.0
                torch.cuda.synchronize()
                print("GPU {}: {}".format(device, torch.cuda.get_device_name(device)), flush=True)
        from dig_rl.runtime import fast_kernel_report
        report = fast_kernel_report()
        print(json.dumps(report, indent=2), flush=True)
        if not report["fast_path"]:
            raise RuntimeError("Fast kernels failed to bind. Save this output; do not start training")
        print("GPU environment check passed. Run launch --stage preflight, then --stage smoke.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("micromamba", "causal-wheel", "seal", "check", "gpu"))
    args = parser.parse_args()
    if args.stage == "micromamba":
        install_micromamba()
    elif args.stage == "causal-wheel":
        check_runtime()
        download(WHEEL_URL, RUNTIME / "downloads" / WHEEL, WHEEL_SHA256)
    else:
        readiness(seal=args.stage == "seal", gpu=args.stage == "gpu")


if __name__ == "__main__":
    main()
