import importlib
from pathlib import Path
import pytest
from packaging.tags import Tag


@pytest.fixture
def wheels(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("torch_wheels")


def test_accelerator_versions_come_from_pinned_torch_metadata(wheels):
    metadata = ("Requires-Dist: filelock\n"
                "Requires-Dist: nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux'\n"
                "Requires-Dist: triton==3.6.0; platform_system == 'Linux'\n")
    assert wheels.accelerator_requirements(metadata) == [("nvidia-cudnn-cu12", "9.10.2.21"), ("triton", "3.6.0")]
    with pytest.raises(RuntimeError, match="exact"):
        wheels.accelerator_requirements(metadata.replace("triton==", "triton>="))


def test_pypi_selection_rejects_wrong_platform_and_version(wheels):
    info = {"urls": [dict(filename="triton-3.6.0-cp311-cp311-manylinux_2_27_x86_64.whl",
                         url="https://files.pythonhosted.org/example.whl", digests={"sha256": "a" * 64}, size=10)]}
    tags = [Tag("cp311", "cp311", "manylinux_2_27_x86_64")]
    assert wheels.select_wheel(info, "triton", "3.6.0", tags)["size"] == 10
    with pytest.raises(RuntimeError, match="compatible"):
        wheels.select_wheel(info, "triton", "3.6.0", [Tag("cp311", "cp311", "win_amd64")])
    with pytest.raises(RuntimeError, match="compatible"):
        wheels.select_wheel(info, "triton", "3.7.0", tags)


def test_old_pip_cache_is_only_reused_after_full_hash_verification(wheels, tmp_path):
    import hashlib
    cache, destination = tmp_path / "cache", tmp_path / "wheels"
    cache.mkdir(); destination.mkdir()
    (cache / "unrelated.body").write_bytes(b"wrong bytes")
    (cache / "complete.body").write_bytes(b"right bytes")
    plan = [dict(filename="package.whl", size=11, sha256=hashlib.sha256(b"right bytes").hexdigest())]
    wheels.reuse_pip_cache(plan, destination, cache)
    assert (destination / "package.whl").read_bytes() == b"right bytes"
    assert (cache / "complete.body").read_bytes() == b"right bytes"
