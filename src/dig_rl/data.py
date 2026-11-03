"""Numeric-answer subset and deterministic, disjoint question splits."""
import hashlib
import math
import random
import re

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def numeric(text):
    text = re.sub(r"\s+", "", str(text)).replace(",", "").replace("\\dfrac", "\\frac")
    try:
        if re.fullmatch(NUMBER, text):
            value = float(text)
        else:
            frac = re.fullmatch(r"([+-]?)\\frac\{(" + NUMBER + r")\}\{(" + NUMBER + r")\}", text)
            slash = re.fullmatch("(" + NUMBER + ")/(" + NUMBER + ")", text)
            if frac:
                value = (-1 if frac[1] == "-" else 1) * float(frac[2]) / float(frac[3])
            elif slash:
                value = float(slash[1]) / float(slash[2])
            else:
                return None
        return value if math.isfinite(value) else None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def extract_boxed(text):
    if text.rfind("<think>") > text.rfind("</think>"):
        return None
    text = text.rsplit("</think>", 1)[-1]
    boxes = list(re.finditer(r"\\boxed\s*\{", text))
    if not boxes:
        return None
    start = boxes[-1].end()
    end, depth = start, 1
    while end < len(text) and depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return numeric(text[start:end - 1]) if depth == 0 else None


def reward(text, answer):
    value, target = extract_boxed(text), numeric(answer)
    return float(value is not None and target is not None and math.isclose(value, target, rel_tol=1e-6, abs_tol=1e-6))


def normalize(row, dataset):
    if dataset == "gsm8k":
        question = row["question"]
        if "####" not in row["answer"]:
            return None
        body, answer = row["answer"].rsplit("####", 1)
    elif dataset == "math_numeric":
        question, body, answer = row["problem"], row["solution"], str(row["extracted_solution"])
    else:
        raise ValueError(dataset)
    if numeric(answer) is None or not body.strip():
        return None
    answer = answer.strip().replace(",", "")
    # Remove GSM8K calculator annotations, retain complete worked solution.
    body = re.sub(r"<<.*?>>", "", body).strip()
    demo = body + "\n\\boxed{" + answer + "}"
    if not reward(demo, answer):
        raise ValueError("Invalid canonical demonstration")
    key = hashlib.sha256(" ".join(question.split()).encode()).hexdigest()
    return dict(id=key, question=question, answer=answer, demo=demo, source=dataset)


def split_rows(train, test, cfg):
    test_ids = {r["id"] for r in test}
    if len(test_ids) != len(test):
        raise ValueError("Duplicate test questions")
    seen, clean = set(), []
    for r in train:
        if r["id"] not in test_ids and r["id"] not in seen:
            seen.add(r["id"])
            clean.append(r)
    random.Random(cfg.split_seed).shuffle(clean)
    sizes = [("calibration", cfg.calibration_size), ("audit", cfg.audit_size), ("validation", cfg.validation_size)]
    if sum(n for _, n in sizes) >= len(clean):
        raise ValueError("Insufficient eligible training data for disjoint splits")
    result, offset = {"test": test}, 0
    for name, n in sizes:
        result[name] = clean[offset:offset + n]
        offset += n
    result["train"] = clean[offset:]
    return result
