import hashlib
import shutil
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from dig_rl import model_downloads as model


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    state = dict(body=b"model checkpoint data\n" * 20000, mode="interrupt", requests=[])
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            range_header = self.headers.get("Range")
            state["requests"].append(range_header)
            if state["mode"] == "forbidden":
                self.send_error(403)
                return
            start = int(range_header[6:-1]) if range_header else 0
            body = state["body"]
            resume = bool(range_header) and state["mode"] != "ignore"
            data = body[start:] if resume else body
            if state["mode"] == "corrupt":
                data = b"x" * len(data)
            self.send_response(206 if resume else 200)
            self.send_header("Content-Length", str(len(data)))
            if resume:
                self.send_header("Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}")
            self.end_headers()
            try:
                if state["mode"] == "interrupt":
                    self.wfile.write(data[:100000])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                else:
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}/weight"
    state["item"] = dict(filename="model.safetensors", size=len(state["body"]),
                         sha256=hashlib.sha256(state["body"]).hexdigest())
    yield state
    srv.shutdown()
    srv.server_close()
    thread.join()


@pytest.mark.skipif(not shutil.which("curl"), reason="requires actual curl")
def test_real_connection_drop_survives_new_invocation_and_resumes(server, tmp_path):
    path = tmp_path / "model.safetensors"
    with pytest.raises(RuntimeError, match="partial retained"):
        model.download_weight(server["url"], path, server["item"], attempts=1)
    partial = next((tmp_path / ".dig-downloads").glob("*.partial"))
    assert partial.read_bytes() == server["body"][:100000]
    assert not path.exists()
    server["mode"] = "ok"
    model.download_weight(server["url"], path, server["item"], attempts=1)
    assert server["requests"] == [None, "bytes=100000-"]
    assert path.read_bytes() == server["body"] and not partial.exists()
    model.download_weight("https://must-not-contact.invalid", path, server["item"])
    assert len(server["requests"]) == 2


@pytest.mark.skipif(not shutil.which("curl"), reason="requires actual curl")
@pytest.mark.parametrize("mode", ["ignore", "forbidden"])
def test_server_refusal_keeps_saved_prefix(server, tmp_path, mode):
    path = tmp_path / "model.safetensors"
    with pytest.raises(RuntimeError):
        model.download_weight(server["url"], path, server["item"], attempts=1)
    partial = next((tmp_path / ".dig-downloads").glob("*.partial"))
    saved = partial.read_bytes()
    server["mode"] = mode
    with pytest.raises(RuntimeError, match="partial retained"):
        model.download_weight(server["url"], path, server["item"], attempts=1)
    assert partial.read_bytes() == saved and not path.exists()


@pytest.mark.skipif(not shutil.which("curl"), reason="requires actual curl")
def test_corrupt_bytes_are_not_published_or_used_on_rerun(server, tmp_path):
    server["mode"] = "corrupt"
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"previous file must not be destroyed")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            model.download_weight(server["url"], path, server["item"], attempts=1)
    assert path.read_bytes() == b"previous file must not be destroyed"
    assert len(server["requests"]) == 1


def test_complete_partial_is_verified_without_network(tmp_path, monkeypatch):
    item = dict(filename="model.safetensors", size=3, sha256=hashlib.sha256(b"abc").hexdigest())
    scratch = tmp_path / ".dig-downloads"
    scratch.mkdir()
    (scratch / (item["filename"] + "." + item["sha256"] + ".partial")).write_bytes(b"abc")
    monkeypatch.setattr(model.shutil, "which", lambda name: "curl")
    monkeypatch.setattr(model, "run_curl", lambda cmd: pytest.fail("must not download"))
    model.download_weight("unused", tmp_path / item["filename"], item)
    assert (tmp_path / item["filename"]).read_bytes() == b"abc"


def test_manifest_reuse_and_revision_isolation(tmp_path):
    calls = []
    def info(repo, revision, **kwargs):
        calls.append(revision)
        return SimpleNamespace(sha=revision, private=False, gated=False, siblings=[SimpleNamespace(
            rfilename="model.safetensors", lfs=SimpleNamespace(size=4, sha256=revision[0] * 64))])
    api = SimpleNamespace(model_info=info)
    first = model.weight_plan(api, "public/repo", "a" * 40, tmp_path, 1)
    assert model.weight_plan(api, "public/repo", "a" * 40, tmp_path, 1) == first
    second = model.weight_plan(api, "public/repo", "b" * 40, tmp_path, 1)
    assert first != second and calls == ["a" * 40, "b" * 40]


@pytest.mark.parametrize("filename", ["../escape.safetensors", "a/b.safetensors", "C:bad.safetensors"])
def test_manifest_rejects_unsafe_names(filename):
    with pytest.raises(ValueError):
        model.validate_weights([dict(filename=filename, size=1, sha256="a" * 64)])


def test_ctrl_c_retains_partial(monkeypatch, tmp_path):
    item = dict(filename="model.safetensors", size=6, sha256=hashlib.sha256(b"abcdef").hexdigest())
    monkeypatch.setattr(model.shutil, "which", lambda name: "curl")
    def interrupt(cmd):
        from pathlib import Path
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"abc")
        raise KeyboardInterrupt()
    monkeypatch.setattr(model, "run_curl", interrupt)
    with pytest.raises(KeyboardInterrupt):
        model.download_weight("unused", tmp_path / item["filename"], item)
    assert next((tmp_path / ".dig-downloads").glob("*.partial")).read_bytes() == b"abc"


@pytest.mark.parametrize("transport", ["curl", "hub"])
def test_snapshot_never_redownloads_curl_weights(monkeypatch, tmp_path, transport):
    hub = pytest.importorskip("huggingface_hub")
    events = []
    monkeypatch.setattr(model, "download_model_weights", lambda *args: events.append("weights"))
    def snapshot(repo, **kwargs):
        events.append("snapshot")
        assert kwargs["revision"] == "a" * 40
        assert kwargs["local_dir"] == tmp_path
        assert kwargs["ignore_patterns"] == (["*.safetensors"] if transport == "curl" else None)
        return str(tmp_path)
    monkeypatch.setattr(hub, "snapshot_download", snapshot)
    assert model.download_model_snapshot(None, "public/repo", "a" * 40, tmp_path,
                                         transport=transport) == str(tmp_path)
    assert events == (["weights", "snapshot"] if transport == "curl" else ["snapshot"])
