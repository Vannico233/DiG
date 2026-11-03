import importlib
import io
import json
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


def cached_plan_fixture(wheels, tmp_path, monkeypatch):
    metadata = "Requires-Dist: nvidia-cudnn-cu12==9.10.2.21\nRequires-Dist: triton==3.6.0\n"
    (tmp_path / "torch.metadata").write_text(metadata, encoding="utf-8")
    monkeypatch.setattr(wheels, "TORCH_METADATA_SHA256", wheels.digest(tmp_path / "torch.metadata"))
    tags = [Tag("cp311", "cp311", "manylinux_2_28_x86_64"), Tag("py3", "none", "manylinux_2_27_x86_64")]
    core = [dict(item, url=wheels.TORCH_CDN + wheels.urllib.parse.quote(item["filename"]))
            for item in (wheels.TORCH, wheels.VISION)]
    deps = [dict(filename="nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl",
                 url="https://files.pythonhosted.org/cudnn.whl", sha256="a" * 64, size=20),
            dict(filename="triton-3.6.0-cp311-cp311-manylinux_2_28_x86_64.whl",
                 url="https://files.pythonhosted.org/triton.whl", sha256="b" * 64, size=30)]
    return core + deps, tags, wheels.accelerator_requirements(metadata)


def deny_network(*args, **kwargs):
    pytest.fail("A valid cached plan must not require network access")


def test_previous_list_plan_resumes_with_network_unavailable(wheels, tmp_path, monkeypatch):
    plan, tags, _ = cached_plan_fixture(wheels, tmp_path, monkeypatch)
    (tmp_path / "torch-downloads.json").write_text(json.dumps(plan))
    partial = tmp_path / (wheels.TORCH["filename"] + ".partial")
    partial.write_bytes(b"already downloaded bytes")
    monkeypatch.setattr(wheels, "open_verified", deny_network)
    monkeypatch.setattr(importlib.import_module("offline_setup"), "open_verified", deny_network)
    assert wheels.build_plan(tmp_path, tags) == plan
    assert partial.read_bytes() == b"already downloaded bytes"


@pytest.mark.parametrize("problem", ["missing", "version", "hash", "platform"])
def test_incompatible_saved_plan_is_not_reused(wheels, tmp_path, monkeypatch, problem):
    plan, tags, requirements = cached_plan_fixture(wheels, tmp_path, monkeypatch)
    core = list(plan[:2])
    if problem == "missing":
        plan.pop()
    elif problem == "version":
        plan[-1]["filename"] = plan[-1]["filename"].replace("3.6.0", "3.7.0")
    elif problem == "hash":
        plan[-1]["sha256"] = "unfinished hash"
    else:
        plan[-1]["filename"] = plan[-1]["filename"].replace("manylinux_2_28_x86_64", "win_amd64")
    assert not wheels.valid_cached_plan(plan, core, requirements, tags)


def test_short_metadata_response_retries_then_uses_cache(wheels, tmp_path, monkeypatch):
    entry = dict(filename="triton-3.6.0-cp311-cp311-manylinux_2_28_x86_64.whl",
                 url="https://files.pythonhosted.org/triton.whl", size=30, digests={"sha256": "b" * 64})
    info = {"urls": [entry]}
    tags = [Tag("cp311", "cp311", "manylinux_2_28_x86_64")]
    class Truncated(io.BytesIO):
        def read(self, *args):
            raise wheels.http.client.IncompleteRead(b"", 4330)
    replies = [Truncated(), io.BytesIO(json.dumps(info).encode())]
    monkeypatch.setattr(wheels, "open_verified", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(wheels.time, "sleep", lambda seconds: None)
    assert wheels.pypi_info("triton", "3.6.0", tmp_path, tags) == info
    assert not replies
    cache = tmp_path / "metadata/triton-3.6.0.json"
    assert json.loads(cache.read_text()) == info
    assert not cache.with_name(cache.name + ".tmp").exists()
    monkeypatch.setattr(wheels, "open_verified", deny_network)
    assert wheels.pypi_info("triton", "3.6.0", tmp_path, tags) == info


def test_failed_metadata_queries_preserve_wheel_progress(wheels, tmp_path, monkeypatch):
    partial = tmp_path / "torch.whl.partial"
    partial.write_bytes(b"saved progress")
    def fail(*args, **kwargs):
        raise wheels.http.client.IncompleteRead(b"", 4330)
    monkeypatch.setattr(wheels, "open_verified", fail)
    monkeypatch.setattr(wheels.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Saved wheel progress is unchanged"):
        wheels.pypi_info("triton", "3.6.0", tmp_path, [], attempts=2)
    assert partial.read_bytes() == b"saved progress"
    assert not (tmp_path / "metadata/triton-3.6.0.json").exists()


@pytest.mark.parametrize("mirror,base", [
    ("official", "https://files.pythonhosted.org"),
    ("tuna", "https://pypi.tuna.tsinghua.edu.cn"),
    ("ustc", "https://mirrors.ustc.edu.cn/pypi"),
    ("aliyun", "https://mirrors.aliyun.com/pypi"),
    ("huawei", "https://repo.huaweicloud.com/repository/pypi"),
])
def test_mirror_routes_same_object_without_changing_official_plan(wheels, mirror, base):
    item = dict(filename="triton-3.6.0-cp311-cp311-manylinux_2_28_x86_64.whl",
                sha256="b" * 64, size=30)
    path = "/packages/aa/bb/fixed/" + item["filename"]
    item["url"] = "https://files.pythonhosted.org" + path
    original = dict(item)
    assert wheels.wheel_download_url(item, mirror) == base + path
    assert item == original


def test_pypi_mirror_does_not_change_pytorch_url(wheels):
    item = dict(wheels.TORCH, url=wheels.TORCH_CDN + wheels.urllib.parse.quote(wheels.TORCH["filename"]))
    assert wheels.wheel_download_url(item, "tuna") == item["url"]


@pytest.mark.parametrize("url", [
    "http://files.pythonhosted.org/packages/test.whl",
    "https://files.pythonhosted.org/unexpected/test.whl",
    "https://files.pythonhosted.org/packages/wrong.whl",
    "https://files.pythonhosted.org/packages/test.whl?unexpected=1",
])
def test_mirror_refuses_unexpected_package_url(wheels, url):
    with pytest.raises(RuntimeError, match="refusing mirror rewrite"):
        wheels.wheel_download_url(dict(filename="test.whl", url=url), "tuna")


def test_mirror_only_bypasses_proxy_for_mirror_hosts(wheels, monkeypatch):
    monkeypatch.setenv("DIG_PYPI_MIRROR", "tuna")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("no_proxy", "localhost,.cluster.internal")
    monkeypatch.setenv("NO_PROXY", "localhost,.cluster.internal")
    assert wheels.configure_mirror() == "tuna"
    assert wheels.urllib.request.proxy_bypass_environment("pypi.tuna.tsinghua.edu.cn")
    assert wheels.urllib.request.proxy_bypass_environment("node.cluster.internal")
    for host in ("download-r2.pytorch.org", "files.pythonhosted.org", "huggingface.co"):
        assert not wheels.urllib.request.proxy_bypass_environment(host)
    assert wheels.urllib.request.getproxies_environment()["https"] == "http://127.0.0.1:7890"


def test_official_mode_does_not_change_proxy_exclusions(wheels, monkeypatch):
    monkeypatch.setenv("DIG_PYPI_MIRROR", "official")
    monkeypatch.setenv("no_proxy", "localhost")
    assert wheels.configure_mirror() == "official"
    assert wheels.os.environ["no_proxy"] == "localhost"
