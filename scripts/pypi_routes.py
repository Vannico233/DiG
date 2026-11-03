"""Choose a PyPI file route on the actual download host, without changing saved wheel bytes."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

from offline_setup import RUNTIME

PYPI_MIRRORS = {
    "tuna": "https://pypi.tuna.tsinghua.edu.cn",
    "ustc": "https://mirrors.ustc.edu.cn/pypi",
    "aliyun": "https://mirrors.aliyun.com/pypi",
    "huawei": "https://repo.huaweicloud.com/repository/pypi",
}
MIRROR_HOSTS = ("pypi.tuna.tsinghua.edu.cn", "mirrors.tuna.tsinghua.edu.cn", "mirrors.ustc.edu.cn",
                "mirrors.aliyun.com", "repo.huaweicloud.com", "mirrors.huaweicloud.com")
# Small pinned CUDA package, used only if no previous wheel plan is available.
PROBE_FILE = dict(
    filename="nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    url="https://files.pythonhosted.org/packages/a2/eb/86626c1bbc2edb86323022371c39aa48df6fd8b0a1647bc274577f72e90b/"
        "nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    size=89954)


def wheel_download_url(item, mirror):
    """Keep the official lock and pins intact; change only where bytes are fetched."""
    parsed = urllib.parse.urlsplit(item["url"])
    if mirror == "official" or parsed.hostname != "files.pythonhosted.org":
        return item["url"]
    if (parsed.scheme != "https" or not parsed.path.startswith("/packages/")
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]) != item["filename"]):
        raise RuntimeError("Unexpected pinned PyPI file URL; refusing mirror rewrite")
    return PYPI_MIRRORS[mirror] + parsed.path


def configure_mirror():
    mirror = os.environ.get("DIG_PYPI_MIRROR", "official")
    if mirror == "auto":
        mirror = select_route()
        os.environ["DIG_PYPI_MIRROR"] = mirror
    if mirror not in ("official", *PYPI_MIRRORS):
        raise RuntimeError("DIG_PYPI_MIRROR must be official, auto, tuna, ustc, aliyun or huawei")
    if mirror != "official":
        exclusions = os.environ.get("no_proxy", os.environ.get("NO_PROXY", "")).split(",")
        exclusions = [item for item in dict.fromkeys(exclusions + list(MIRROR_HOSTS)) if item]
        os.environ["no_proxy"] = os.environ["NO_PROXY"] = ",".join(exclusions)
        print("PyPI wheel mirror: {} (direct); original versions and SHA256 pins retained.".format(mirror), flush=True)
    return mirror


def probe_target(directory):
    plan_path = directory / "torch-downloads.json"
    if not plan_path.is_file():
        return dict(PROBE_FILE), 0
    for item in json.loads(plan_path.read_text(encoding="utf-8")):
        if urllib.parse.urlsplit(item["url"]).hostname != "files.pythonhosted.org":
            continue
        name = item["filename"]
        if "/" in name or "\\" in name or type(item["size"]) is not int or item["size"] <= 0:
            raise RuntimeError("Invalid wheel plan entry for route probe")
        wheel_download_url(item, "tuna")  # Validate the original host/path before sending a probe.
        path = directory / name
        if path.is_file() and path.stat().st_size == item["size"]:
            continue
        partial = path.with_name(name + ".partial")
        offset = partial.stat().st_size if partial.is_file() else 0
        return item, min(offset, item["size"] - 1)
    return dict(PROBE_FILE), 0


def probe_route(item, offset, mirror, temp_directory, seconds=10):
    executable = shutil.which("curl")
    if not executable:
        raise RuntimeError("Automatic route selection requires curl on PATH")
    end = min(offset + 2 * 1024 * 1024, item["size"]) - 1
    count = end - offset + 1
    with tempfile.TemporaryDirectory(dir=temp_directory) as scratch:
        headers = Path(scratch) / "headers.txt"
        command = [executable, "-q", "--fail", "--location", "--http1.1", "--silent", "--show-error",
                   "--proto", "=http,https", "--proto-redir", "=https", "--connect-timeout", "6",
                   "--max-time", str(seconds), "--max-filesize", str(count), "--max-redirs", "5",
                   "--range", "{}-{}".format(offset, end), "--header", "Accept-Encoding: identity",
                   "--output", os.devnull, "--dump-header", str(headers),
                   "--write-out", "%{http_code} %{size_download} %{speed_download}"]
        if mirror != "official":
            command.extend(["--noproxy", "*"])
        ca_file = os.environ.get("SSL_CERT_FILE") or os.environ.get("CURL_CA_BUNDLE")
        if ca_file:
            command.extend(["--cacert", ca_file])
        command.extend(["--url", wheel_download_url(item, mirror)])
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=seconds + 3)
        except subprocess.TimeoutExpired:
            return dict(mirror=mirror, ok=False, status="timeout", speed=0.0, received=0, error="Probe process timed out")
        try:
            status, received, speed = result.stdout.strip().split()
            received, speed = int(received), float(speed)
        except ValueError:
            return dict(mirror=mirror, ok=False, status="curl-exit-{}".format(result.returncode), speed=0.0, received=0,
                        error=result.stderr.strip()[-300:])
        raw = headers.read_text(encoding="iso-8859-1") if headers.is_file() else ""
        ranges = re.findall(r"(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$", raw)
        right_range = bool(ranges) and tuple(map(int, ranges[-1])) == (offset, end, item["size"])
        ok = (result.returncode in (0, 28) and status == "206" and right_range
              and 0 < received <= count and math.isfinite(speed) and speed > 0)
        return dict(mirror=mirror, ok=ok, status=status, speed=speed, received=received,
                    error="" if ok else result.stderr.strip()[-300:] or "Range/size verification failed")


def select_route(directory=None):
    directory = RUNTIME / "wheels" if directory is None else Path(directory)
    item, offset = probe_target(directory)
    temp_directory = RUNTIME / "tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    print("Checking routes for {} at saved offset {}. Wheel files will not be modified.".format(
        item["filename"], offset), file=sys.stderr, flush=True)
    results = []
    for mirror in ("aliyun", "huawei", "tuna", "ustc", "official"):
        print("Probing {} (up to 10 seconds)...".format(mirror), file=sys.stderr, flush=True)
        result = probe_route(item, offset, mirror, temp_directory)
        results.append(result)
        print("  {} HTTP={} sample={:.2f} MB/s {}".format(
            mirror, result["status"], result["speed"] / 1e6, "OK" if result["ok"] else "SKIP"),
            file=sys.stderr, flush=True)
        if not result["ok"] and result.get("error"):
            print("  " + result["error"], file=sys.stderr, flush=True)
    usable = [r for r in results if r["ok"]]
    if not usable:
        raise RuntimeError("No route passed the range probe. Saved wheel files are unchanged; retain the probe output for diagnosis.")
    chosen = max(usable, key=lambda r: r["speed"])["mirror"]
    print("Selected {} from short probes; sustained speed may differ.".format(chosen), file=sys.stderr, flush=True)
    return chosen


if __name__ == "__main__":
    print(select_route())
