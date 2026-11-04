"""Download public model weights with persistent, hash-checked curl partials."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .hf_downloads import hf_retry
from .io import sha256, write_json


def validate_weights(items):
    if not isinstance(items, list) or not items:
        raise ValueError("Model manifest has no weights")
    names = set()
    for item in items:
        name = item["filename"]
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors", name)
                or name in names or type(item["size"]) is not int or item["size"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Invalid pinned model weight metadata")
        names.add(name)
    return items


def weight_plan(api, repo, revision, directory, attempts):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model downloads require an immutable revision")
    key = hashlib.sha256((repo + "@" + revision).encode()).hexdigest()
    path = directory / ".dig-downloads" / (key + ".json")
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if (plan["repo"], plan["revision"]) != (repo, revision):
            raise ValueError("Saved model manifest belongs to a different revision")
        return validate_weights(plan["weights"])
    info = hf_retry("Reading pinned model weight hashes", lambda: api.model_info(
        repo, revision=revision, files_metadata=True, timeout=30), attempts)
    if info.sha != revision or info.private or info.gated:
        raise ValueError("Persistent curl weights require the pinned public, ungated model")
    items = []
    for entry in info.siblings:
        if entry.rfilename.endswith(".safetensors"):
            if not entry.lfs:
                raise ValueError("Model weight has no SHA256 metadata")
            items.append(dict(filename=entry.rfilename, size=entry.lfs.size, sha256=entry.lfs.sha256))
    validate_weights(items)
    write_json(path, dict(repo=repo, revision=revision, weights=items))
    return items


def run_curl(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        output, _ = process.communicate()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    return process.returncode, output


def download_weight(url, path, item, attempts=4):
    """Never overwrite a saved prefix on refusal, nor publish an unchecked file."""
    if attempts < 1:
        raise ValueError("Download attempts must be positive")
    path = Path(path)
    size, expected = item["size"], item["sha256"]
    if path.is_file() and path.stat().st_size == size and sha256(path) == expected:
        print("Verified cached model weight:", path.name, flush=True)
        return
    executable = shutil.which("curl")
    if not executable:
        raise RuntimeError("Persistent model downloads require curl")
    partial = path.parent / ".dig-downloads" / (path.name + "." + expected + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError("Oversized model partial retained: " + str(partial))
        if offset == size:
            if sha256(partial) != expected:
                raise RuntimeError("Model SHA256 mismatch; partial retained: " + str(partial))
            os.replace(partial, path)
            print("Verified downloaded model weight:", path.name, flush=True)
            return
        print("Model curl: {} (saved {:.1f}/{:.1f} MB; attempt {}/{})".format(
            path.name, offset / 1e6, size / 1e6, attempt, attempts), flush=True)
        command = [executable, "-q", "--fail", "--location", "--http1.1",
                   "--proto", "=http,https", "--proto-redir", "=https", "--max-redirs", "5",
                   "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "60",
                   "--header", "Accept-Encoding: identity", "--continue-at", "-",
                   "--max-filesize", str(size), "--output", str(partial),
                   "--write-out", "%{http_code} %{size_download} %{speed_download}"]
        ca_file = os.environ.get("SSL_CERT_FILE") or os.environ.get("CURL_CA_BUNDLE")
        if ca_file:
            command += ["--cacert", ca_file]
        # Resolve the immutable Hub URL afresh on each retry, not a saved expiring CDN URL.
        command += ["--url", url]
        code, output = run_curl(command)
        fields = output.strip().split()
        status = fields[0] if len(fields) == 3 else "unknown"
        if len(fields) == 3:
            print("Model HTTP={} received={} B speed={} B/s".format(*fields), flush=True)
        if code == 0:
            if partial.stat().st_size != size or sha256(partial) != expected:
                raise RuntimeError("Model size/SHA256 mismatch; partial retained: " + str(partial))
            os.replace(partial, path)
            print("Verified downloaded model weight:", path.name, flush=True)
            return
        retryable = code in (5, 6, 7, 18, 28, 35, 52, 55, 56, 92) or (
            code == 22 and status in ("408", "429", "500", "502", "503", "504"))
        if not retryable or attempt == attempts:
            raise RuntimeError("Model curl failed (exit {}, HTTP {}); partial retained: {}".format(
                code, status, partial))
        print("Model transfer interrupted; retrying from saved bytes.", flush=True)
        time.sleep(min(2 ** attempt, 16))


def download_model_weights(api, repo, revision, directory, attempts=4):
    from huggingface_hub import hf_hub_url
    directory = Path(directory)
    items = weight_plan(api, repo, revision, directory, attempts)
    print("Persistent model weights: {} files; {:.2f} GB before cache reuse.".format(
        len(items), sum(item["size"] for item in items) / 1e9), flush=True)
    for item in items:
        url = hf_hub_url(repo, filename=item["filename"], revision=revision,
                         endpoint="https://huggingface.co")
        download_weight(url, directory / item["filename"], item, attempts)


def download_model_snapshot(api, repo, revision, directory, attempts=4, transport="curl"):
    from huggingface_hub import snapshot_download
    if transport not in ("curl", "hub"):
        raise ValueError("Model transport must be curl or hub")
    if transport == "curl":
        download_model_weights(api, repo, revision, directory, attempts)
    return hf_retry("Downloading pinned model snapshot", lambda: snapshot_download(
        repo, revision=revision, local_dir=directory,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"],
        ignore_patterns=["*.safetensors"] if transport == "curl" else None), attempts)
