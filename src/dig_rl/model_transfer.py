"""Bounded parallel Range transfers, preserving legacy contiguous weight prefixes."""
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlencode

from .io import sha256, write_json


@dataclass(frozen=True)
class Route:
    name: str
    url: str
    direct: bool = False


# ModelScope's immutable upload contains byte-identical HF weight shards. Do not
# reuse this mapping for other revisions or infer equivalence from filenames.
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MS_REVISION = "fcb1a040bb418b0b8add6f6f6c475386abc2cb97"
MS_WEIGHTS = {
    "model.safetensors-00001-of-00002.safetensors": (
        5329398688, "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"),
    "model.safetensors-00002-of-00002.safetensors": (
        3990429408, "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"),
}


def weight_routes(repo, revision, item, source="auto"):
    from huggingface_hub import hf_hub_url
    if source not in ("auto", "official", "hf-mirror", "modelscope", "modelscope-proxy"):
        raise ValueError("DIG_MODEL_SOURCE must be auto, official, hf-mirror, modelscope or modelscope-proxy")
    routes = [Route("official", hf_hub_url(repo, item["filename"], revision=revision,
                                          endpoint="https://huggingface.co")),
              Route("hf-mirror", hf_hub_url(repo, item["filename"], revision=revision,
                                           endpoint="https://hf-mirror.com"), True)]
    if (repo == "Qwen/Qwen3.5-4B" and revision == QWEN_REVISION
            and MS_WEIGHTS.get(item["filename"]) == (item["size"], item["sha256"])):
        url = "https://modelscope.cn/api/v1/models/" + repo + "/repo?" + urlencode(
            {"Revision": MS_REVISION, "FilePath": item["filename"]})
        routes.insert(0, Route("modelscope", url, True))
        if source == "modelscope-proxy" or any(os.environ.get(key) for key in (
                "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")):
            routes.insert(1, Route("modelscope-proxy", url))
    if source != "auto":
        routes = [route for route in routes if route.name == source]
    if not routes:
        raise ValueError("No verified ModelScope mapping for this pinned weight")
    return routes


@contextmanager
def transfer_lock(directory):
    """OS-released lock: a crash must not leave a permanent stale lock."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".transfer.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another model downloader is running in this directory") from error
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another model downloader is running in this directory") from error
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def response_range(headers, start, end, total):
    """Inspect the final response, never a proxy CONNECT or redirect header."""
    blocks = re.split(r"\r?\n\r?\n", headers.strip())
    responses = [block for block in blocks if block.startswith("HTTP/")]
    if not responses:
        return False
    lines = responses[-1].splitlines()
    status = lines[0].split()
    fields = dict(line.split(":", 1) for line in lines[1:] if ":" in line)
    fields = {key.lower().strip(): value.strip() for key, value in fields.items()}
    return (len(status) >= 2 and status[1] == "206"
            and fields.get("content-range") == f"bytes {start}-{end}/{total}"
            and fields.get("content-encoding", "identity").lower() == "identity")


def stored_size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def fetch_range(route, start, end, total, output, stop, seconds=90):
    """Return only validated prefix bytes. Untrusted/error bodies never get merged."""
    executable = shutil.which("curl")
    if not executable:
        raise RuntimeError("Model downloads require curl")
    output = Path(output)
    headers = output.with_suffix(output.suffix + ".headers")
    output.unlink(missing_ok=True)
    headers.unlink(missing_ok=True)
    command = [executable, "-q", "--silent", "--show-error", "--fail", "--location",
               "--http1.1", "--proto", "=http,https", "--proto-redir", "=https",
               "--max-redirs", "5", "--connect-timeout", "10", "--max-time", str(seconds),
               "--speed-limit", "1024", "--speed-time", "30",
               "--header", "Accept-Encoding: identity", "--range", f"{start}-{end}",
               "--max-filesize", str(end - start + 1), "--dump-header", str(headers),
               "--output", str(output), "--write-out", "%{http_code}"]
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("CURL_CA_BUNDLE")
    if ca:
        command += ["--cacert", ca]
    if route.direct:
        command += ["--noproxy", "*"]
    command += ["--url", route.url]
    began = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        while True:
            try:
                status, _ = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if stop.is_set():
                    process.terminate()
                    try:
                        status, _ = process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        status, _ = process.communicate()
                    break
    except BaseException:
        process.kill()
        process.communicate()
        raise
    count = output.stat().st_size if output.exists() else 0
    valid = headers.exists() and response_range(
        headers.read_text(encoding="iso-8859-1"), start, end, total)
    # SSL errors and timeouts may still deliver a useful, validated prefix.
    accepted = count if valid and 0 < count <= end - start + 1 else 0
    elapsed = max(time.monotonic() - began, 0.001)
    headers.unlink(missing_ok=True)  # Headers can contain expiring CDN URLs.
    return accepted, elapsed, f"curl={process.returncode} HTTP={status.strip() or 'unknown'}"


def rank_routes(routes, item, offset, directory, stop, label="Model"):
    if len(routes) == 1:
        return routes
    end = min(item["size"] - 1, offset + 2 * 1024 * 1024 - 1)
    print(f"Testing actual {label.lower()} routes (up to 15 seconds each, in parallel)...", flush=True)
    scores = []
    with tempfile.TemporaryDirectory(prefix="probe-", dir=directory) as tmp:
        pool = ThreadPoolExecutor(max_workers=len(routes))
        try:
            futures = [(route, pool.submit(fetch_range, route, offset, end, item["size"],
                        Path(tmp) / str(i), stop, 15)) for i, route in enumerate(routes)]
            for route, future in futures:
                count, elapsed, result = future.result()
                speed = count / elapsed
                print(f"  {route.name}: {speed / 1e6:.2f} MB/s; {result}; "
                      + ("Range OK" if count else "unavailable"), flush=True)
                if count:
                    scores.append((speed, route))
        except BaseException:
            stop.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    if not scores:
        raise RuntimeError(f"All {label.lower()} routes failed their Range checks; existing progress retained")
    ranked = [route for _, route in sorted(scores, key=lambda pair: pair[0], reverse=True)]
    print(f"Selected {label.lower()} route: " + ranked[0].name, flush=True)
    return ranked


def fill_chunk(routes, start, end, total, piece, stop, attempts):
    """Retry budget counts consecutive no-progress requests, not whole-file drops."""
    length = end - start + 1
    failures, route_index = 0, 0
    temporary = piece.with_suffix(".transfer")
    while not stop.is_set():
        saved = piece.stat().st_size if piece.exists() else 0
        if saved == length:
            return
        if saved > length:
            raise RuntimeError("Oversized saved chunk retained: " + str(piece))
        route = routes[route_index % len(routes)]
        count, elapsed, result = fetch_range(route, start + saved, end, total, temporary, stop)
        if count:
            with piece.open("ab") as out, temporary.open("rb") as source:
                shutil.copyfileobj(source, out)
                out.flush()
                os.fsync(out.fileno())
            failures = 0
        else:
            failures += 1
        temporary.unlink(missing_ok=True)
        if saved + count == length:
            return
        if stop.is_set():
            raise InterruptedError("Download paused; validated chunk prefixes retained")
        route_index += 1
        print(f"Range {start + saved}-{end}: {result}, saved +{count / 1e6:.2f} MB; "
              f"next={routes[route_index % len(routes)].name}", flush=True)
        if failures >= attempts:
            raise RuntimeError(f"No progress for {attempts} consecutive requests; chunks retained: {piece.parent}")
        if not count:
            stop.wait(min(2 ** min(failures, 4), 16))
    raise InterruptedError("Download paused; chunk progress retained")


def download_parallel(routes, path, item, *, workers=4, chunk_size=8 * 1024 * 1024, attempts=12, label="Model"):
    if not 1 <= workers <= 8 or chunk_size <= 0 or attempts < 1:
        raise ValueError("Model connections must be 1..8; chunk size and attempts must be positive")
    path = Path(path)
    size, expected = item["size"], item["sha256"]
    if path.is_file() and path.stat().st_size == size and sha256(path) == expected:
        print(f"Verified cached {label.lower()} file: " + path.name, flush=True)
        return
    scratch = path.parent / ".dig-downloads"
    scratch.mkdir(parents=True, exist_ok=True)
    partial = scratch / (path.name + "." + expected + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise RuntimeError("Oversized model partial retained: " + str(partial))
    chunks = partial.with_suffix(".chunks")
    stop = threading.Event()
    if offset < size:
        chunks.mkdir(exist_ok=True)
        layout_path = chunks / "layout.json"
        if layout_path.exists():
            layout = json.loads(layout_path.read_text())
            if (layout["size"] != size or layout["sha256"] != expected
                    or type(layout["base"]) is not int or not 0 <= layout["base"] <= offset
                    or type(layout["chunk_size"]) is not int or layout["chunk_size"] <= 0):
                raise ValueError("Invalid saved chunk layout")
            base, chunk_size = layout["base"], layout["chunk_size"]
        else:
            base = offset
            write_json(layout_path, dict(base=base, chunk_size=chunk_size, size=size, sha256=expected))
        ranges = [(start, min(size - 1, start + chunk_size - 1))
                  for start in range(base, size, chunk_size) if min(size, start + chunk_size) > offset]
        # If a previous run completed every outstanding chunk, no network is needed.
        def piece_path(start, end):
            return chunks / f"{start}-{end}.part"
        pending = any(not piece_path(a, b).exists() or piece_path(a, b).stat().st_size != b-a+1
                      for a, b in ranges)
        if pending:
            routes = rank_routes(routes, item, offset, scratch, stop, label)
        print(f"Parallel {label.lower()}: {path.name}; saved prefix {offset / 1e6:.1f}/{size / 1e6:.1f} MB; "
              f"connections={workers}; chunk={chunk_size / 1048576:.1f} MiB", flush=True)
        began, initial = time.monotonic(), offset
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            # A bounded window prevents a slow early range from accumulating GBs of pieces.
            for index in range(0, len(ranges), workers):
                batch = ranges[index:index + workers]
                futures = [pool.submit(fill_chunk, routes, a, b, size, piece_path(a, b), stop, attempts)
                           for a, b in batch]
                while True:
                    done, unfinished = wait(futures, timeout=10)
                    for future in done:
                        future.result()
                    extra = sum(max(0, stored_size(piece_path(a, b)) - max(0, offset-a))
                                for a, b in batch)
                    inflight = sum(stored_size(piece_path(a, b).with_suffix(".transfer")) for a, b in batch)
                    print(f"{label} progress: {min(size, offset + extra) / 1e6:.1f}/{size / 1e6:.1f} MB; "
                          f"in-flight {inflight / 1e6:.1f} MB; "
                          f"session {(offset + extra + inflight - initial) / max(time.monotonic()-began, .001) / 1e6:.2f} MB/s",
                          flush=True)
                    if not unfinished:
                        break
                for a, b in batch:
                    piece = piece_path(a, b)
                    with partial.open("ab") as out, piece.open("rb") as source:
                        source.seek(max(0, offset - a))  # Recover a crash during an earlier append.
                        shutil.copyfileobj(source, out)
                        out.flush()
                        os.fsync(out.fileno())
                    offset = partial.stat().st_size
                    piece.unlink()
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
    if partial.stat().st_size != size or sha256(partial) != expected:
        raise RuntimeError(f"{label} size/SHA256 mismatch; partial retained: " + str(partial))
    os.replace(partial, path)
    # Only the downloader's own empty chunk directory/manifest is removed.
    if chunks.exists():
        (chunks / "layout.json").unlink(missing_ok=True)
        if not any(chunks.iterdir()):
            chunks.rmdir()
    print(f"Verified downloaded {label.lower()} file: " + path.name, flush=True)
