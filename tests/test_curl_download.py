"""Exercise the optional transport with real curl and a local HTTP server."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("offline_curl_test", Path(__file__).resolve().parents[1] / "scripts/offline_setup.py")
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


@pytest.fixture
def curl_server(monkeypatch):
    if not shutil.which("curl"):
        pytest.skip("curl unavailable")
    monkeypatch.setenv("DIG_DOWNLOAD_TRANSPORT", "curl")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(offline.time, "sleep", lambda seconds: None)
    state = SimpleNamespace(body=b"abcdef", offsets=[], behavior="normal")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            raw = self.headers.get("Range")
            offset = int(raw.split("=")[1].split("-")[0]) if raw else 0
            state.offsets.append(offset)
            if state.behavior == "forbidden":
                self.send_response(403)
                self.end_headers()
                return
            if state.behavior == "ignore":
                offset = 0
                raw = None
            body = state.body[offset:]
            self.send_response(206 if raw else 200)
            if raw:
                start = 0 if state.behavior == "wrong_range" else offset
                self.send_header("Content-Range", "bytes {}-{}/{}".format(start, len(state.body) - 1, len(state.body)))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state.behavior == "short" and len(state.offsets) == 1:
                body = body[:2]
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    state.url = "http://127.0.0.1:{}/wheel".format(server.server_port)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_curl_resumes_existing_urllib_partial_and_verifies_cache(curl_server, tmp_path):
    path = tmp_path / "torch.whl"
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(b"abc")
    expected = hashlib.sha256(curl_server.body).hexdigest()
    offline.download(curl_server.url, path, expected, size=6)
    assert path.read_bytes() == b"abcdef"
    assert not partial.exists()
    assert curl_server.offsets == [3]
    offline.download(curl_server.url, path, expected, size=6)
    assert curl_server.offsets == [3]


def test_curl_retry_uses_bytes_saved_by_the_interrupted_request(curl_server, tmp_path):
    curl_server.behavior = "short"
    path = tmp_path / "torch.whl"
    offline.download(curl_server.url, path, hashlib.sha256(curl_server.body).hexdigest(), size=6)
    assert path.read_bytes() == b"abcdef"
    assert curl_server.offsets == [0, 2]


def test_curl_refused_resume_preserves_previous_progress(curl_server, tmp_path):
    curl_server.behavior = "ignore"
    path = tmp_path / "torch.whl"
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="refused curl resume"):
        offline.download(curl_server.url, path, hashlib.sha256(curl_server.body).hexdigest(), size=6)
    assert partial.read_bytes() == b"abc"
    assert not path.exists()


def test_curl_http_403_does_not_truncate_the_saved_partial(curl_server, tmp_path):
    curl_server.behavior = "forbidden"
    path = tmp_path / "torch.whl"
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="exit code 22"):
        offline.download(curl_server.url, path, hashlib.sha256(curl_server.body).hexdigest(), size=6)
    assert partial.read_bytes() == b"abc"
    assert not path.exists()


@pytest.mark.parametrize("behavior,ok", [("normal", True), ("forbidden", False), ("ignore", False), ("wrong_range", False)])
def test_mirror_probe_uses_saved_offset_and_requires_matching_range(curl_server, tmp_path, monkeypatch, behavior, ok):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import pypi_routes as routes
    monkeypatch.setitem(routes.PYPI_MIRRORS, "local", curl_server.url.rsplit("/", 1)[0])
    curl_server.behavior = behavior
    item = dict(filename="test.whl", url="https://files.pythonhosted.org/packages/test.whl", size=6)
    partial = tmp_path / "test.whl.partial"
    partial.write_bytes(b"abc")
    result = routes.probe_route(item, 3, "local", tmp_path)
    assert result["ok"] is ok
    assert curl_server.offsets == [3]
    assert partial.read_bytes() == b"abc"


@pytest.mark.parametrize("wrong_size", [False, True])
def test_curl_never_publishes_wrong_hash_or_size(curl_server, tmp_path, wrong_size):
    path = tmp_path / "torch.whl"
    expected = hashlib.sha256(curl_server.body).hexdigest() if wrong_size else "0" * 64
    with pytest.raises(RuntimeError, match="checksum|size differs"):
        offline.download(curl_server.url, path, expected, size=7 if wrong_size else 6)
    assert not path.exists()
    assert path.with_name(path.name + ".partial").read_bytes() == curl_server.body


def test_curl_complete_partial_is_verified_without_a_request(curl_server, tmp_path):
    path = tmp_path / "torch.whl"
    path.with_name(path.name + ".partial").write_bytes(curl_server.body)
    offline.download(curl_server.url, path, hashlib.sha256(curl_server.body).hexdigest(), size=6)
    assert path.read_bytes() == curl_server.body
    assert curl_server.offsets == []


def test_curl_uses_explicit_ca_and_stops_on_certificate_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("DIG_DOWNLOAD_TRANSPORT", "curl")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "trusted.pem"))
    monkeypatch.setattr(offline.shutil, "which", lambda name: "curl")
    calls = []
    def reject(command):
        calls.append(command)
        return 60
    monkeypatch.setattr(offline, "run_curl", reject)
    path = tmp_path / "torch.whl"
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="exit code 60"):
        offline.download("https://example.invalid/wheel", path, "0" * 64, size=6)
    assert len(calls) == 1
    assert calls[0][1] == "-q"
    assert calls[0][calls[0].index("--cacert") + 1] == str(tmp_path / "trusted.pem")
    assert "--insecure" not in calls[0] and "-k" not in calls[0]
    assert partial.read_bytes() == b"abc"


def test_ctrl_c_reaps_curl_and_leaves_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("DIG_DOWNLOAD_TRANSPORT", "curl")
    monkeypatch.setattr(offline.shutil, "which", lambda name: "curl")
    path = tmp_path / "torch.whl"
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(b"abc")
    class Process:
        stopped = False
        def wait(self, **kwargs):
            if not self.stopped:
                raise KeyboardInterrupt()
            return -15
        def terminate(self):
            self.stopped = True
    process = Process()
    monkeypatch.setattr(offline.subprocess, "Popen", lambda command: process)
    with pytest.raises(KeyboardInterrupt):
        offline.download("https://example.invalid/wheel", path, "0" * 64, size=6)
    assert process.stopped
    assert partial.read_bytes() == b"abc"
    assert not path.exists()
