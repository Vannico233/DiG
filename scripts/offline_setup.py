"""Shared Linux runtime preparation; bootstrap commands also work with Python 3.9."""
import argparse
import hashlib
import http.client
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
SYSTEM_CA_FILES = ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                   "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", "/etc/ssl/ca-bundle.pem")
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


def prepare_certificates():
    """Read existing trusted roots, then write a project-local bundle; never disable TLS checks."""
    context = ssl.create_default_context()
    candidates = []
    for name in ("DIG_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT"):
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise RuntimeError("{} points to a missing CA file: {}".format(name, path))
            candidates.append(path)
    candidates.extend(Path(path) for path in SYSTEM_CA_FILES if Path(path).is_file())
    # This uses an existing installation only; no pip/network needed to bootstrap Python 3.9.
    try:
        import certifi
    except ImportError:
        pass
    else:
        candidates.append(Path(certifi.where()))
    loaded = []
    for path in dict.fromkeys(candidates):
        try:
            context.load_verify_locations(cafile=str(path))
        except (OSError, ssl.SSLError) as error:
            raise RuntimeError("Cannot read CA certificates from {}: {}".format(path, error)) from None
        loaded.append(str(path))
    certificates = sorted(set(context.get_ca_certs(binary_form=True)))
    if not certificates:
        raise RuntimeError("No trusted CA certificates found. Ask the server administrator for a CA bundle and set DIG_CA_BUNDLE")
    directory = RUNTIME / "certs"
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / "ca-bundle.pem"
    tmp = bundle.with_suffix(".tmp")
    tmp.write_text("".join(ssl.DER_cert_to_PEM_cert(cert) for cert in certificates), encoding="ascii")
    # Re-open the generated bundle with normal hostname/certificate verification enabled.
    ssl.create_default_context(cafile=str(tmp))
    os.replace(tmp, bundle)
    (directory / "sources.json").write_text(json.dumps(dict(
        python=sys.version.split()[0], openssl=ssl.OPENSSL_VERSION,
        default_paths=ssl.get_default_verify_paths()._asdict(),
        additional_files=loaded, ca_count=len(certificates)), indent=2) + "\n", encoding="utf-8")
    print("Trusted CA bundle: {} ({} certificates)".format(bundle, len(certificates)), flush=True)
    return bundle


def open_verified(request, **kwargs):
    if "context" not in kwargs:
        kwargs["context"] = ssl.create_default_context()
    try:
        return urllib.request.urlopen(request, timeout=120, **kwargs)
    except urllib.error.URLError as error:
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                "TLS certificate verification failed. Run: python3 scripts/offline_setup.py tls-check. "
                "If the server uses a private CA, obtain its CA bundle from the administrator and "
                "set DIG_CA_BUNDLE to that file before rerunning preparation. TLS verification stays enabled."
            ) from None
        raise


def check_tls():
    bundle = prepare_certificates()
    context = ssl.create_default_context(cafile=str(bundle))
    request = urllib.request.Request(MICROMAMBA_URL, headers={"User-Agent": "DiG-offline-setup"})
    # Includes the download service's redirects, but does not download the full archive.
    with open_verified(request, context=context) as response:
        response.read(1)
    print("TLS check passed for the micromamba download and its redirects.", flush=True)


def run_curl(command):
    """Keep curl in the foreground and reap it before returning from Ctrl+C."""
    process = subprocess.Popen(command)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def download_curl(url, path, expected, size, attempts):
    """Resume the same pinned partial as urllib; never publish unchecked curl output."""
    executable = shutil.which("curl")
    if not executable:
        raise RuntimeError("DIG_DOWNLOAD_TRANSPORT=curl requires curl on PATH")
    partial = path.with_name(path.name + ".partial")
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.is_file() else 0
        if partial.is_file() and (size is None or offset == size):
            if digest(partial) == expected:
                os.replace(partial, path)
                return path
            if size is not None and offset == size:
                raise RuntimeError("Downloaded wheel checksum mismatch; partial retained: " + str(partial))
        if size is not None and offset > size:
            raise RuntimeError("Partial exceeds the pinned file size; partial retained: " + str(partial))
        print("{} {} with curl (saved {:.1f} MB; attempt {}/{})".format(
            "Resuming" if offset else "Downloading", path.name, offset / 1e6, attempt + 1, attempts), flush=True)
        # -q disables ~/.curlrc. Proxy variables remain inherited, as with urllib.
        # Let our outer loop recompute the saved offset after every failed transfer.
        command = [executable, "-q", "--fail", "--location", "--http1.1",
                   "--proto", "=http,https", "--proto-redir", "=https",
                   "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "60",
                   "--header", "Accept-Encoding: identity", "--continue-at", "-",
                   "--output", str(partial), "--write-out",
                   "\nCURL HTTP=%{http_code} received=%{size_download} average=%{speed_download} B/s\n"]
        ca_file = os.environ.get("SSL_CERT_FILE") or os.environ.get("CURL_CA_BUNDLE")
        if ca_file:
            command.extend(["--cacert", ca_file])
        command.extend(["--url", url])
        code = run_curl(command)
        if code == 0:
            if size is not None and partial.stat().st_size != size:
                raise RuntimeError("curl download size differs from pinned metadata; partial retained: " + str(partial))
            if digest(partial) != expected:
                raise RuntimeError("Downloaded wheel checksum mismatch; partial retained: " + str(partial))
            os.replace(partial, path)
            print("Verified downloaded", path.name, flush=True)
            return path
        if code in (33, 36):
            raise RuntimeError("Server refused curl resume; saved bytes are unchanged. Partial retained: " + str(partial))
        if code not in (5, 6, 7, 18, 28, 35, 52, 55, 56, 92):
            raise RuntimeError("curl failed with exit code {}; partial retained: {}".format(code, partial))
        if attempt + 1 < attempts:
            print("curl interrupted (exit {}); retrying from saved bytes.".format(code), flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("curl download interrupted; rerun preparation. Partial retained: " + str(partial))


def download(url, path, expected=None, size=None, attempts=4):
    """Keep pinned downloads across interruptions; only publish fully verified files."""
    path = Path(path)
    if path.is_file() and expected is not None and digest(path) == expected:
        print("Verified cached", path.name, flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    transport = os.environ.get("DIG_DOWNLOAD_TRANSPORT", "urllib")
    if transport not in ("urllib", "curl"):
        raise RuntimeError("DIG_DOWNLOAD_TRANSPORT must be urllib or curl")
    if transport == "curl" and expected is not None:
        return download_curl(url, path, expected, size, attempts)
    partial = path.with_name(path.name + ".partial")
    for attempt in range(attempts):
        # Covers Ctrl+C after the last byte but before verification/rename.
        if expected and partial.is_file() and (size is None or partial.stat().st_size == size):
            if digest(partial) == expected:
                os.replace(partial, path)
                return path
        offset = partial.stat().st_size if expected and partial.is_file() else 0
        headers = {"User-Agent": "DiG-offline-setup", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes={}-".format(offset)
        print("{} {} (saved {:.1f} MB)".format("Resuming" if offset else "Downloading",
                                             path.name, offset / 1e6), flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with open_verified(request) as response:
                status = getattr(response, "status", 200)
                response_headers = getattr(response, "headers", {})
                total = size
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response_headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise RuntimeError("Invalid Content-Range; refusing to append download bytes")
                    total = int(match[3])
                    if size is not None and total != size:
                        raise RuntimeError("Download size differs from the pinned metadata")
                elif status == 200:
                    if offset:
                        print("Server ignored resume request; restarting this file.", flush=True)
                    offset = 0
                    length = response_headers.get("Content-Length")
                    if length is not None:
                        if size is not None and int(length) != size:
                            raise RuntimeError("Download size differs from the pinned metadata")
                        total = int(length)
                else:
                    raise RuntimeError("Unexpected download HTTP status: {}".format(status))
                received, started, last_report = offset, time.monotonic(), time.monotonic()
                last_received = offset
                with partial.open("ab" if offset else "wb") as stream:
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        received += len(block)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            rate = (received - offset) / max(now - started, 0.001) / 1e6
                            recent_rate = (received - last_received) / max(now - last_report, 0.001) / 1e6
                            target = " / {:.1f} MB".format(total / 1e6) if total else ""
                            print("  {:.1f} MB{} recent={:.2f} MB/s average={:.2f} MB/s".format(
                                received / 1e6, target, recent_rate, rate), flush=True)
                            last_report = now
                            last_received = received
                if total is not None and received != total:
                    raise ConnectionError("Incomplete response: {} / {} bytes".format(received, total))
            if expected is not None and digest(partial) != expected:
                partial.unlink()  # This file was downloaded by us and has failed integrity checking.
                raise RuntimeError("Downloaded wheel checksum mismatch; re-run CPU preparation")
            os.replace(partial, path)
            return path
        except urllib.error.HTTPError as error:
            if error.code == 416 and offset:
                partial.unlink()  # Stale/overlong partial; a valid complete file was handled above.
            elif error.code not in (408, 429, 500, 502, 503, 504):
                raise
            failure = error
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead, ssl.SSLEOFError) as error:
            failure = error
        if attempt + 1 < attempts:
            print("Connection interrupted ({}); retrying from saved bytes.".format(failure), flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Download interrupted; rerun preparation. Pinned partial files are retained: {} ({})".format(partial, failure))


def install_micromamba():
    binary = RUNTIME / "tools/bin/micromamba"
    if binary.is_file():
        return
    archive = download(MICROMAMBA_URL,
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
    for name in ("datasets", "triton", "numpy", "safetensors", "tokenizers", "ninja", "packaging", "httpx", "socksio"):
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
    parser.add_argument("stage", choices=("certificates", "tls-check", "micromamba", "causal-wheel", "seal", "check", "gpu"))
    args = parser.parse_args()
    if args.stage == "certificates":
        prepare_certificates()
    elif args.stage == "tls-check":
        check_tls()
    elif args.stage == "micromamba":
        install_micromamba()
    elif args.stage == "causal-wheel":
        check_runtime()
        download(WHEEL_URL, RUNTIME / "downloads" / WHEEL, WHEEL_SHA256)
    else:
        readiness(seal=args.stage == "seal", gpu=args.stage == "gpu")


if __name__ == "__main__":
    main()
