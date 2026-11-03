"""Offline preparation must reject incomplete/corrupt files and incompatible wheels."""
import importlib.util
import io
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("offline_setup", Path(__file__).resolve().parents[1] / "scripts/offline_setup.py")
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


def test_download_reuses_verified_cache_and_repairs_corruption(tmp_path, monkeypatch):
    path = tmp_path / "kernel.whl"
    expected = offline.hashlib.sha256(b"complete wheel").hexdigest()
    calls = []

    def fetch(*args, **kwargs):
        calls.append(True)
        return io.BytesIO(b"complete wheel")

    monkeypatch.setattr(offline.urllib.request, "urlopen", fetch)
    path.write_bytes(b"incomplete")
    offline.download("https://example.invalid/wheel", path, expected)
    offline.download("https://example.invalid/wheel", path, expected)
    assert path.read_bytes() == b"complete wheel" and len(calls) == 1
    assert not path.with_name("kernel.whl.partial").exists()


def test_failed_download_is_not_published_as_a_valid_wheel(tmp_path, monkeypatch):
    path = tmp_path / "kernel.whl"
    monkeypatch.setattr(offline.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    with pytest.raises(RuntimeError, match="checksum"):
        offline.download("https://example.invalid/wheel", path, "0" * 64)
    assert not path.exists()


@pytest.mark.parametrize("changed", [dict(cuda=None), dict(torch_version="2.14.0"), dict(abi=False),
                                    dict(python_minor=(3, 9)), dict(machine="aarch64")])
def test_prebuilt_wheel_must_match_the_installed_runtime(changed):
    target = dict(system="Linux", machine="x86_64", python_minor=(3, 11),
                  torch_version="2.10.0+cu128", cuda="12.8", abi=True)
    offline.validate_target(**target)
    target.update(changed)
    with pytest.raises(RuntimeError):
        offline.validate_target(**target)


def test_cpu_seal_does_not_require_a_gpu_and_later_changes_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(offline, "ROOT", tmp_path)
    monkeypatch.setattr(offline, "RUNTIME", tmp_path)
    # This deliberately has no CUDA API: CPU sealing must never use one.
    monkeypatch.setattr(offline, "check_runtime", lambda: object())
    monkeypatch.setattr(offline, "check_compiler", lambda: None)
    monkeypatch.setattr(offline, "installed_packages", lambda: {"torch": "test"})
    files = {"assets/model/shard": "complete hash"}
    monkeypatch.setattr(offline, "local_assets", lambda: dict(files))
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "USE_HUB_KERNELS"):
        monkeypatch.setenv(name, "")
    offline.readiness(seal=True)
    offline.readiness()
    saved = json.loads((tmp_path / "CPU_READY.json").read_text())
    assert saved["files"] == files
    files["assets/model/shard"] = "changed hash"
    with pytest.raises(RuntimeError, match="changed"):
        offline.readiness()


def test_moved_shared_environment_is_not_silently_accepted(tmp_path):
    saved = dict(root=str(tmp_path), prefix=str(tmp_path / "python"), packages={}, files={})
    with pytest.raises(RuntimeError, match="mount path"):
        offline.validate_ready(saved, tmp_path / "different-mount", tmp_path / "python", {}, {})


@pytest.fixture
def isolated_ca_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(offline, "RUNTIME", tmp_path)
    monkeypatch.setattr(offline, "SYSTEM_CA_FILES", ())
    for name in ("DIG_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_ca_bundle_uses_existing_roots_without_changing_source(isolated_ca_setup, monkeypatch):
    roots = offline.ssl.create_default_context().get_ca_certs(binary_form=True)
    assert roots, "Test host needs an existing trusted CA store"
    original = offline.ssl.DER_cert_to_PEM_cert(roots[0])
    source = isolated_ca_setup / "trusted-server-ca.pem"
    source.write_text(original, encoding="ascii")
    monkeypatch.setenv("DIG_CA_BUNDLE", str(source))
    bundle = offline.prepare_certificates()
    context = offline.ssl.create_default_context(cafile=str(bundle))
    assert context.check_hostname
    assert context.verify_mode == offline.ssl.CERT_REQUIRED
    assert roots[0] in context.get_ca_certs(binary_form=True)
    assert source.read_text(encoding="ascii") == original
    assert bundle.parent == isolated_ca_setup / "certs"


@pytest.mark.parametrize("content", [None, "not a CA certificate"])
def test_invalid_explicit_ca_is_not_silently_ignored(isolated_ca_setup, monkeypatch, content):
    source = isolated_ca_setup / "server-ca.pem"
    if content is not None:
        source.write_text(content, encoding="ascii")
    monkeypatch.setenv("DIG_CA_BUNDLE", str(source))
    with pytest.raises(RuntimeError, match="missing CA|Cannot read CA"):
        offline.prepare_certificates()
    assert not (isolated_ca_setup / "certs/ca-bundle.pem").exists()


def test_failed_tls_is_never_retried_without_verification(tmp_path, monkeypatch):
    contexts = []

    def reject(*args, **kwargs):
        contexts.append(kwargs["context"])
        raise offline.urllib.error.URLError(offline.ssl.SSLCertVerificationError(1, "untrusted issuer"))

    monkeypatch.setattr(offline.urllib.request, "urlopen", reject)
    destination = tmp_path / "micromamba.tar.bz2"
    with pytest.raises(RuntimeError, match="DIG_CA_BUNDLE"):
        offline.download("https://example.invalid/archive", destination)
    assert len(contexts) == 1 and contexts[0].check_hostname
    assert contexts[0].verify_mode == offline.ssl.CERT_REQUIRED
    assert not destination.exists()


class DownloadResponse:
    def __init__(self, chunks, status=200, **headers):
        self.chunks, self.status, self.headers = list(chunks), status, headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        chunk = self.chunks.pop(0) if self.chunks else b""
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk


def test_pinned_download_resumes_after_user_interrupt_in_a_later_run(tmp_path, monkeypatch):
    path = tmp_path / "large.whl"
    expected = offline.hashlib.sha256(b"abcdef").hexdigest()
    monkeypatch.setattr(offline, "open_verified", lambda *a, **k: DownloadResponse(
        [b"abc", KeyboardInterrupt()], **{"Content-Length": "6"}))
    with pytest.raises(KeyboardInterrupt):
        offline.download("https://example.invalid/large", path, expected, size=6)
    assert path.with_name("large.whl.partial").read_bytes() == b"abc"

    def remaining(request, **kwargs):
        assert request.get_header("Range") == "bytes=3-"
        return DownloadResponse([b"def"], 206, **{"Content-Range": "bytes 3-5/6"})

    monkeypatch.setattr(offline, "open_verified", remaining)
    offline.download("https://example.invalid/large", path, expected, size=6)
    assert path.read_bytes() == b"abcdef"


def test_short_response_is_retried_from_the_saved_offset(tmp_path, monkeypatch):
    responses = [DownloadResponse([b"abc"], **{"Content-Length": "6"}),
                 DownloadResponse([b"def"], 206, **{"Content-Range": "bytes 3-5/6"})]
    offsets = []
    def respond(request, **kwargs):
        offsets.append(request.get_header("Range"))
        return responses.pop(0)
    monkeypatch.setattr(offline, "open_verified", respond)
    monkeypatch.setattr(offline.time, "sleep", lambda seconds: None)
    path = tmp_path / "large.whl"
    offline.download("https://example.invalid/large", path, offline.hashlib.sha256(b"abcdef").hexdigest(), size=6)
    assert path.read_bytes() == b"abcdef" and offsets == [None, "bytes=3-"]


def test_server_ignoring_range_restarts_without_duplicating_bytes(tmp_path, monkeypatch):
    path = tmp_path / "large.whl"
    path.with_name("large.whl.partial").write_bytes(b"abc")
    monkeypatch.setattr(offline, "open_verified", lambda *a, **k: DownloadResponse(
        [b"abcdef"], **{"Content-Length": "6"}))
    offline.download("https://example.invalid/large", path, offline.hashlib.sha256(b"abcdef").hexdigest(), size=6)
    assert path.read_bytes() == b"abcdef"


def test_wrong_range_never_corrupts_saved_bytes(tmp_path, monkeypatch):
    path = tmp_path / "large.whl"
    partial = path.with_name("large.whl.partial")
    partial.write_bytes(b"abc")
    monkeypatch.setattr(offline, "open_verified", lambda *a, **k: DownloadResponse(
        [b"abcdef"], 206, **{"Content-Range": "bytes 0-5/6"}))
    with pytest.raises(RuntimeError, match="Content-Range"):
        offline.download("https://example.invalid/large", path, "0" * 64, size=6)
    assert partial.read_bytes() == b"abc" and not path.exists()
