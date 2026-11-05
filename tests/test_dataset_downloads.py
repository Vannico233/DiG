import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from dig_rl import dataset_downloads as download
from dig_rl.io import write_json
from dig_rl.round2_data import normalize
from test_model_transfer import origin  # Real HTTP server, including truncated responses.


REPO = "hiyouga/geometry3k"
REV = "a" * 40


def items_for(body):
    return [dict(filename=f"data/{split}-00000-of-00001.parquet", size=len(body),
                 sha256=hashlib.sha256(body).hexdigest()) for split in ("train", "test")]


def no_network(*args, **kwargs):
    pytest.fail("Unexpected network/download operation")


def test_plan_pins_hashes_and_reuses_without_hub(tmp_path):
    items = items_for(b"abc")
    info = SimpleNamespace(sha=REV, private=False, gated=False, siblings=[
        SimpleNamespace(rfilename=x["filename"], lfs=SimpleNamespace(size=x["size"], sha256=x["sha256"]))
        for x in items])
    def query(repo, **kwargs):
        assert repo == REPO and kwargs["revision"] == REV and kwargs["files_metadata"]
        return info
    plan = download.dataset_plan(SimpleNamespace(dataset_info=query), REPO, REV, tmp_path)
    assert sorted(plan, key=lambda x:x["filename"]) == sorted(items, key=lambda x:x["filename"])
    assert download.dataset_plan(SimpleNamespace(dataset_info=no_network), REPO, REV, tmp_path) == plan
    with pytest.raises(ValueError, match="another revision"):
        download.dataset_plan(None, REPO, "b" * 40, tmp_path)


@pytest.mark.parametrize("change", ["missing", "duplicate", "bad_hash", "traversal", "missing_shard"])
def test_invalid_plan_cannot_drop_or_redirect_shards(change):
    items = items_for(b"abc")
    if change == "missing":
        items.pop()
    elif change == "duplicate":
        items.append(items[0])
    elif change == "bad_hash":
        items[0]["sha256"] = "0" * 63
    elif change == "traversal":
        items[0]["filename"] = "../train-00000-of-00001.parquet"
    else:
        items[0]["filename"] = "data/train-00000-of-00002.parquet"
    with pytest.raises(ValueError):
        download.validate_files(items)


def test_dataset_routes_pin_same_object_and_configure_proxy():
    filename = items_for(b"abc")[0]["filename"]
    routes = download.dataset_routes(REPO, REV, filename)
    assert all(f"/datasets/{REPO}/resolve/{REV}/{filename}" in r.url for r in routes)
    assert [(r.name, r.direct) for r in routes] == [
        ("official", False), ("hf-mirror", True), ("official-direct", True)]
    assert len(download.dataset_routes(REPO, REV, filename, "hf-mirror")) == 1
    with pytest.raises(ValueError):
        download.dataset_routes(REPO, REV, filename, "unsupported")


def test_reuse_complete_and_incomplete_hub_files_is_non_destructive(tmp_path, monkeypatch):
    import huggingface_hub as hub
    body = b"0123456789"
    item = items_for(body)[0]
    blob = tmp_path / "hub" / "datasets--hiyouga--geometry3k" / "blobs" / item["sha256"]
    blob.parent.mkdir(parents=True)
    incomplete = blob.with_suffix(".incomplete")
    incomplete.write_bytes(body[:4])
    monkeypatch.setattr(hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *a, **k: None)
    target = tmp_path / "raw" / item["filename"]
    download.reuse_hub_cache(REPO, REV, item, target)
    partial = target.parent / ".dig-downloads" / (target.name + "." + item["sha256"] + ".partial")
    assert partial.read_bytes() == incomplete.read_bytes() == body[:4]
    partial.write_bytes(body[:7])
    download.reuse_hub_cache(REPO, REV, item, target)
    assert partial.read_bytes() == body[:7]  # Do not replace newer download progress.
    blob.write_bytes(body)
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *a, **k: str(blob))
    download.reuse_hub_cache(REPO, REV, item, target)
    assert target.read_bytes() == blob.read_bytes() == body
    assert incomplete.read_bytes() == body[:4]


def test_vlm_loader_recovers_many_drops_and_reopens_locally(origin, tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    from PIL import Image
    # An embedded image large enough to require > 4 interrupted HTTP requests.
    image_bytes = io.BytesIO()
    import random
    Image.frombytes("RGB", (180, 180), random.Random(13).randbytes(180*180*3)).save(image_bytes, format="PNG")
    image = {"bytes":image_bytes.getvalue(), "path":None}
    dataset = datasets.Dataset.from_dict(dict(problem=["<image> Find x"], answer=["4"], images=[[image]]),
        features=datasets.Features(dict(problem=datasets.Value("string"), answer=datasets.Value("string"),
                                        images=datasets.List(datasets.Image()))))
    parquet = tmp_path / "fixture.parquet"
    dataset.to_parquet(str(parquet))
    body = parquet.read_bytes()
    items = items_for(body)
    origin.update(body=body, mode="drop")
    root = tmp_path / "raw"
    write_json(root / "download_plan.json", dict(repo=REPO, revision=REV, files=items))
    monkeypatch.setattr(download, "reuse_hub_cache", lambda *a: None)
    monkeypatch.setattr(download, "dataset_routes", lambda *a: [origin["route"]])
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "arrow-cache"))
    actual = download.load_pinned_parquet(None, REPO, REV, root)
    assert set(actual) == {"train", "test"}
    assert actual["train"].features == dataset.features
    assert len(origin["requests"]) > 8
    assert actual["train"][0]["images"][0].size == (180,180)
    normalized = normalize(actual["train"][0], "geo3k", tmp_path / "prepared")
    assert normalized["answer"] == "4" and len(normalized["images"]) == 1
    for item in items:
        assert (root / item["filename"]).read_bytes() == body
    count = len(origin["requests"])
    reopened = download.load_pinned_parquet(None, REPO, REV, root)
    assert len(reopened["test"]) == 1 and len(origin["requests"]) == count


def test_local_scienceqa_parquet_keeps_optional_images_and_choice_schema(tmp_path):
    datasets = pytest.importorskip("datasets")
    from PIL import Image
    image = io.BytesIO()
    Image.new("RGB", (10,10), "red").save(image, format="PNG")
    row = dict(question="Which color?", hint="", choices=["Red", "Blue"], answer=0,
               solution="The image is red.", image={"bytes":image.getvalue(), "path":None})
    features = datasets.Features(dict(question=datasets.Value("string"), hint=datasets.Value("string"),
        choices=datasets.List(datasets.Value("string")), answer=datasets.Value("int64"),
        solution=datasets.Value("string"), image=datasets.Image()))
    source = datasets.Dataset.from_list([row, dict(row, image=None)], features=features)
    path = tmp_path / "science.parquet"
    source.to_parquet(str(path))
    loaded = datasets.load_dataset("parquet", data_files={"train":str(path)}, cache_dir=str(tmp_path / "cache"))
    assert loaded["train"].features == features
    result = normalize(loaded["train"][0], "scienceqa_image", tmp_path / "prepared")
    assert result["answer"] == "A" and "The image is red." not in result["question"]
    assert normalize(loaded["train"][1], "scienceqa_image", tmp_path / "prepared") is None
