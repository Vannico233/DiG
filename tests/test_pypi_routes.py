import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def routes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("pypi_routes")
    monkeypatch.setattr(module, "RUNTIME", tmp_path)
    return module


def save_plan(directory):
    directory.mkdir()
    entries = [dict(filename=name, url="https://files.pythonhosted.org/packages/" + name, size=6)
               for name in ("complete.whl", "pending.whl")]
    (directory / "torch-downloads.json").write_text(json.dumps(entries))
    (directory / "complete.whl").write_bytes(b"abcdef")
    (directory / "pending.whl.partial").write_bytes(b"abc")
    return entries


def test_probe_targets_first_unfinished_file_at_actual_saved_offset(routes, tmp_path):
    directory = tmp_path / "wheels"
    entries = save_plan(directory)
    item, offset = routes.probe_target(directory)
    assert item == entries[1] and offset == 3
    assert (directory / "pending.whl.partial").read_bytes() == b"abc"


def test_auto_skips_403_and_selects_fastest_usable_route(routes, tmp_path, monkeypatch):
    directory = tmp_path / "wheels"
    save_plan(directory)
    seen = []
    def probe(item, offset, mirror, temp_directory):
        seen.append((mirror, item["filename"], offset))
        speeds = dict(aliyun=10, huawei=20, tuna=100, ustc=5, official=1)
        return dict(mirror=mirror, ok=mirror != "tuna", status="403" if mirror == "tuna" else "206",
                    speed=speeds[mirror], received=0 if mirror == "tuna" else 3)
    monkeypatch.setattr(routes, "probe_route", probe)
    assert routes.select_route(directory) == "huawei"
    assert len(seen) == 5 and all(name == "pending.whl" and offset == 3 for _, name, offset in seen)
    assert (directory / "pending.whl.partial").read_bytes() == b"abc"


def test_all_failed_routes_stop_without_modifying_saved_wheels(routes, tmp_path, monkeypatch):
    directory = tmp_path / "wheels"
    save_plan(directory)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(routes, "probe_route", lambda item, offset, mirror, temp_directory:
                        dict(mirror=mirror, ok=False, status="403", speed=0, received=0))
    with pytest.raises(RuntimeError, match="No route passed"):
        routes.select_route(directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_fresh_install_uses_small_probe_when_no_plan_exists(routes, tmp_path):
    item, offset = routes.probe_target(tmp_path / "wheels")
    assert item["size"] < 100000 and offset == 0
    assert not (tmp_path / "wheels").exists()
