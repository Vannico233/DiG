import hashlib
import json
from pathlib import Path
import shutil
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dig_rl import model_transfer as transfer


@pytest.fixture
def origin(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    state = dict(body=bytes(range(256)) * 1600, requests=[], mode="ok", active=0, peak=0)
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            start, end = map(int, self.headers["Range"][6:].split("-"))
            with lock:
                state["requests"].append((self.path, start, end))
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                mode = "ignore" if self.path == "/bad" else state["mode"]
                if mode == "forbidden":
                    self.send_error(403)
                    return
                time.sleep(.02)
                body = state["body"]
                data = body[start:end + 1]
                if mode == "ignore":
                    self.send_response(200)
                else:
                    self.send_response(206)
                    actual_start = start + 1 if mode == "wrong" else start
                    self.send_header("Content-Range", f"bytes {actual_start}-{end}/{len(body)}")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if mode == "corrupt":
                    data = b"x" * len(data)
                if mode == "drop" and len(data) > 16000:
                    self.wfile.write(data[:16000])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                else:
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    state["active"] -= 1
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["route"] = transfer.Route("local", f"http://127.0.0.1:{server.server_port}/ok", True)
    state["item"] = dict(filename="model.safetensors", size=len(state["body"]),
                         sha256=hashlib.sha256(state["body"]).hexdigest())
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def partial_path(path, item):
    scratch = path.parent / ".dig-downloads"
    scratch.mkdir(exist_ok=True)
    return scratch / (path.name + "." + item["sha256"] + ".partial")


@pytest.mark.skipif(not shutil.which("curl"), reason="requires curl")
def test_parallel_preserves_legacy_prefix_and_survives_more_than_four_drops(origin, tmp_path):
    origin["mode"] = "drop"
    path = tmp_path / "model.safetensors"
    prefix = 23345
    partial_path(path, origin["item"]).write_bytes(origin["body"][:prefix])
    transfer.download_parallel([origin["route"]], path, origin["item"],
                               workers=4, chunk_size=100000, attempts=2)
    assert path.read_bytes() == origin["body"]
    assert min(start for _, start, _ in origin["requests"]) == prefix
    assert len(origin["requests"]) > 16
    assert origin["peak"] > 1
    calls = len(origin["requests"])
    transfer.download_parallel([], path, origin["item"])
    assert len(origin["requests"]) == calls


@pytest.mark.parametrize("mode", ["ignore", "wrong", "forbidden"])
def test_bad_responses_do_not_overwrite_legacy_prefix(origin, tmp_path, mode):
    origin["mode"] = mode
    path = tmp_path / "model.safetensors"
    partial = partial_path(path, origin["item"])
    partial.write_bytes(origin["body"][:2345])
    with pytest.raises(RuntimeError, match="No progress"):
        transfer.download_parallel([origin["route"]], path, origin["item"],
                                   workers=2, chunk_size=100000, attempts=1)
    assert partial.read_bytes() == origin["body"][:2345]
    assert not path.exists()
    origin["mode"] = "ok"
    transfer.download_parallel([origin["route"]], path, origin["item"], workers=2)
    assert path.read_bytes() == origin["body"]


def test_probe_excludes_ignored_ranges_and_ranks_usable_source(origin, tmp_path):
    bad = transfer.Route("bad", origin["route"].url.replace("/ok", "/bad"), True)
    routes = transfer.rank_routes([bad, origin["route"]], origin["item"], 123,
                                  tmp_path, threading.Event())
    assert routes == [origin["route"]]
    assert not list(tmp_path.iterdir())


def test_cancel_keeps_validated_chunk_prefix(tmp_path, monkeypatch):
    stop = threading.Event()
    piece = tmp_path / "0-99.part"
    def fetch(route, start, end, total, output, cancel):
        output.write_bytes(b"a" * 13)
        cancel.set()
        return 13, .1, "interrupted"
    monkeypatch.setattr(transfer, "fetch_range", fetch)
    with pytest.raises(InterruptedError):
        transfer.fill_chunk([transfer.Route("test", "unused")], 0, 99, 100, piece, stop, 2)
    assert piece.read_bytes() == b"a" * 13


def test_crash_during_append_reuses_completed_chunk_without_network(tmp_path, monkeypatch):
    body = b"0123456789" * 10
    item = dict(size=len(body), sha256=hashlib.sha256(body).hexdigest())
    path = tmp_path / "model.safetensors"
    partial = partial_path(path, item)
    partial.write_bytes(body[:37])  # Original prefix 13, then crash 24 bytes into append.
    chunks = partial.with_suffix(".chunks")
    chunks.mkdir()
    (chunks / "layout.json").write_text(json.dumps(dict(base=13, chunk_size=100, **item)))
    (chunks / "13-99.part").write_bytes(body[13:])
    monkeypatch.setattr(transfer, "fetch_range", lambda *a, **k: pytest.fail("must reuse cached chunk"))
    transfer.download_parallel([], path, item)
    assert path.read_bytes() == body


def test_hash_mismatch_never_publishes_weight(origin, tmp_path):
    origin["mode"] = "corrupt"
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"old file")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        transfer.download_parallel([origin["route"]], path, origin["item"], chunk_size=100000)
    assert path.read_bytes() == b"old file"
    assert partial_path(path, origin["item"]).stat().st_size == origin["item"]["size"]


def test_verified_mapping_does_not_mix_model_versions():
    pytest.importorskip("huggingface_hub")
    filename, (size, digest) = next(iter(transfer.MS_WEIGHTS.items()))
    item = dict(filename=filename, size=size, sha256=digest)
    route = transfer.weight_routes("Qwen/Qwen3.5-4B", transfer.QWEN_REVISION, item, "modelscope")[0]
    assert transfer.MS_REVISION in route.url and route.direct
    for revision, altered in [("a" * 40, item), (transfer.QWEN_REVISION, dict(item, sha256="f" * 64))]:
        with pytest.raises(ValueError, match="No verified"):
            transfer.weight_routes("Qwen/Qwen3.5-4B", revision, altered, "modelscope")


def test_final_header_only_and_wrong_total_rejected():
    assert transfer.response_range("HTTP/1.1 200 Connection established\r\n\r\n"
                                   "HTTP/2 206\r\ncontent-range: bytes 3-8/10\r\n\r\n", 3, 8, 10)
    assert not transfer.response_range("HTTP/2 206\r\nContent-Range: bytes 3-8/11", 3, 8, 10)
    assert not transfer.response_range("HTTP/2 206\r\nContent-Range: bytes 3-8/10\r\n\r\nHTTP/2 403", 3, 8, 10)


def test_lock_rejects_second_writer_and_releases(tmp_path):
    with transfer.transfer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another model"):
            with transfer.transfer_lock(tmp_path):
                pytest.fail("second writer acquired lock")
    with transfer.transfer_lock(tmp_path):
        pass
