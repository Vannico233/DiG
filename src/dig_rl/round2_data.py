"""CPU-only preparation of pinned text/image data into a fresh namespace."""
import hashlib
import io
import json
import random
from pathlib import Path
from .io import sha256, write_json, write_jsonl, read_rows
from .round2_protocol import boxed_text, DATASETS

SOURCES = {
    "gsm8k": ("openai/gsm8k", "main"),
    "math": ("DigitalLearningGmbH/MATH-lighteval", "default"),
    "math500": ("HuggingFaceH4/MATH-500", "default"),
    "geo3k": ("hiyouga/geometry3k", "default"),
    "scienceqa_image": ("derek-thomas/ScienceQA", "default"),
}


def normalize(raw, dataset, root):
    images, kind, proxy = [], "math", "worked_solution"
    if dataset == "gsm8k":
        from .data import normalize as old_normalize
        row = old_normalize(raw, "gsm8k")
        if row is None:
            return None
        row.update(images=[], kind="numeric", proxy="worked_solution")
        return row
    if dataset in ("math", "math500"):
        question, demo = raw["problem"], raw["solution"]
        answer = raw.get("answer") or boxed_text(demo)
        if not answer:
            return None
    elif dataset == "geo3k":
        question, answer = raw["problem"].replace("<image>", "").strip(), str(raw["answer"])
        images = raw["images"]
        demo, proxy = "\\boxed{" + answer + "}", "answer_only"
    elif dataset == "scienceqa_image":
        if raw["image"] is None:
            return None
        images, kind = [raw["image"]], "choice"
        choices = raw["choices"]
        if not 2 <= len(choices) <= 26 or not 0 <= raw["answer"] < len(choices):
            raise ValueError("Malformed multiple-choice question")
        question = raw["question"] + "\n" + str(raw.get("hint") or "") + "\n" + "\n".join(
            f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
        answer = chr(65 + raw["answer"])
        # Reference explanation is never inserted in the question/rollout prompt.
        explanation = str(raw.get("solution") or "").strip()
        demo = explanation + "\n\\boxed{" + answer + "}"
        proxy = "worked_solution" if explanation else "answer_only"
    else:
        raise ValueError(dataset)
    paths = []
    for img in images:
        from PIL import Image
        if isinstance(img, dict):
            img = Image.open(io.BytesIO(img["bytes"])) if img.get("bytes") else Image.open(img["path"])
        if not hasattr(img, "convert"):
            raise ValueError("Image dataset must supply decoded images")
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        data = buf.getvalue()
        relative = "images/" + hashlib.sha256(data).hexdigest() + ".png"
        dest = Path(root) / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        paths.append(relative)
    identity = " ".join(question.split()) + "|" + "|".join(paths)
    return dict(id=hashlib.sha256(identity.encode()).hexdigest(), question=question, answer=answer,
                demo=demo, images=paths, kind=kind, proxy=proxy, source=dataset)


def split(train, test, seed=20261105):
    test_ids = {r["id"] for r in test}
    if len(test_ids) != len(test):
        raise ValueError("Duplicate test rows; review dataset before preparing")
    seen, clean = set(test_ids), []
    for row in train:
        if row["id"] not in seen:
            seen.add(row["id"])
            clean.append(row)
    random.Random(seed).shuffle(clean)
    if len(clean) <= 384:
        raise ValueError("Too few unique training examples")
    return dict(calibration=clean[:32], audit=clean[32:128], validation=clean[128:384],
                train=clean[384:], test=test)


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    seen = set()
    for split_name, meta in manifest["files"].items():
        path = root / (split_name + ".jsonl")
        if sha256(path) != meta["sha256"]:
            raise ValueError("Dataset hash changed: " + str(path))
        rows = read_rows(path)
        ids = {r["id"] for r in rows}
        if seen & ids or len(rows) != meta["rows"]:
            raise ValueError("Split overlap/count mismatch")
        seen |= ids
    for relative, digest in manifest["images"].items():
        from .round2_protocol import image_path
        if sha256(image_path(root, relative)) != digest:
            raise ValueError("Image hash changed")
    return manifest


def prepare(assets, model):
    from huggingface_hub import HfApi
    from datasets import load_dataset
    from .hf_downloads import resolve_revisions, hf_retry
    from .dataset_downloads import load_pinned_parquet
    assets, model = Path(assets), Path(model)
    assets.mkdir(parents=True, exist_ok=True)
    requested = {k: (v[0], "main") for k, v in SOURCES.items()}
    lock = resolve_revisions(HfApi(), requested, assets / "download_lock.json")
    for dataset in DATASETS:
        dest = assets / dataset
        if (dest / "manifest.json").exists():
            existing = verify(dest)
            if existing["revisions"] != lock["resolved"] or existing["model_id"] != sha256(model / "dig_model.json"):
                raise ValueError("Prepared revisions/model differ; choose a new assets path")
            print("Verified prepared dataset; skipping: " + dataset, flush=True)
            continue
        repo, subset = SOURCES[dataset]
        if dataset in ("geo3k", "scienceqa_image"):
            revision = lock["resolved"][dataset]
            ds = load_pinned_parquet(HfApi(), repo, revision, assets / ".sources" / dataset / revision)
        else:
            ds = hf_retry("Loading " + dataset, lambda: load_dataset(repo, subset, revision=lock["resolved"][dataset]))
        if dataset == "math":
            repo500, subset500 = SOURCES["math500"]
            test_raw = hf_retry("Loading MATH-500", lambda: load_dataset(repo500, subset500,
                revision=lock["resolved"]["math500"], split="test"))
        else:
            test_raw = ds["test"]
        dest.mkdir(parents=True, exist_ok=True)
        train, test, excluded = [], [], []
        for name, rows, result in (("train", ds["train"], train), ("test", test_raw, test)):
            print(f"Preparing {dataset}/{name}: {len(rows)} source rows", flush=True)
            for i, raw in enumerate(rows):
                row = normalize(raw, "math500" if dataset == "math" and name == "test" else dataset, dest)
                if row is None:
                    if name == "test" and dataset != "scienceqa_image":
                        raise ValueError(f"Unparseable official test reference: {dataset}/{i}")
                    excluded.append(dict(split=name, index=i, reason="missing_image_or_reference"))
                else:
                    result.append(row)
                if (i + 1) % 500 == 0 or i + 1 == len(rows):
                    print(f"Prepared {dataset}/{name}: {i + 1}/{len(rows)}; kept {len(result)}", flush=True)
        splits = split(train, test)
        files = {}
        for name, rows in splits.items():
            path = dest / (name + ".jsonl")
            write_jsonl(path, rows)
            files[name] = dict(rows=len(rows), sha256=sha256(path))
        images = {p: sha256(dest / p) for rows in splits.values() for r in rows for p in r["images"]}
        write_json(dest / "excluded.json", excluded)
        write_json(dest / "manifest.json", dict(schema=2, dataset=dataset, sources=SOURCES,
            revisions=lock["resolved"], model_id=sha256(model / "dig_model.json"),
            split_seed=20261105, files=files, images=images,
            test_protocol="MATH-500" if dataset == "math" else dataset,
            eligibility="No test prompt or answer-length filtering; long inputs fail explicitly at runtime",
            note="Geo3K uses answer-only proxy; ScienceQA uses solution when available, never in prompt"))
        verify(dest)
        print(dataset, {k:v["rows"] for k,v in files.items()}, flush=True)
