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
