import json
import ssl
from types import SimpleNamespace

import pytest

httpx = pytest.importorskip("httpx")
hub = pytest.importorskip("huggingface_hub")
from dig_rl import hf_downloads as hf


@pytest.fixture
def retry_state(monkeypatch):
    state = dict(closes=0, sleeps=[])
    def close():
        state["closes"] += 1
    monkeypatch.setattr(hub.utils, "close_session", close)
    monkeypatch.setattr(hf.time, "sleep", state["sleeps"].append)
    return state


def test_tls_eof_gets_a_fresh_client_and_then_resolves(retry_state, tmp_path):
    calls = []
    def info(repo, **kwargs):
        calls.append((repo, kwargs))
        if len(calls) < 3:
            raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]")
        return SimpleNamespace(sha="a" * 40)
    lock = hf.resolve_revisions(SimpleNamespace(model_info=info, dataset_info=info),
                                {"model": ("Qwen/Qwen3.5-4B", "main")}, tmp_path / "lock.json")
    assert lock["resolved"]["model"] == "a" * 40
    assert len(calls) == 3 and all(kwargs["timeout"] == 30 for _, kwargs in calls)
    assert retry_state == dict(closes=2, sleeps=[2, 4])
    assert json.loads((tmp_path / "lock.json").read_text()) == lock


def test_later_repo_failure_keeps_earlier_sha_and_next_run_queries_only_missing(retry_state, tmp_path):
    requested = {"model": ("model/repo", "main"), "gsm8k": ("data/repo", "main")}
    path = tmp_path / "lock.json"
    def broken_dataset(*args, **kwargs):
        raise httpx.ConnectError("EOF")
    api = SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="a" * 40), dataset_info=broken_dataset)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        hf.resolve_revisions(api, requested, path, attempts=2)
    assert json.loads(path.read_text())["resolved"] == {"model": "a" * 40}
    def no_model_request(*args, **kwargs):
        pytest.fail("Resolved model must not be queried again")
    api = SimpleNamespace(model_info=no_model_request, dataset_info=lambda *a, **k: SimpleNamespace(sha="b" * 40))
    lock = hf.resolve_revisions(api, requested, path)
    assert lock["resolved"] == dict(model="a" * 40, gsm8k="b" * 40)
    assert not path.with_name(path.name + ".tmp").exists()


@pytest.mark.parametrize("failure", ["certificate", "unauthorized", "forbidden", "not_found", "data_error"])
def test_nontransient_errors_are_not_retried_or_weakened(retry_state, failure):
    if failure == "certificate":
        error = httpx.ConnectError("certificate error")
        error.__cause__ = ssl.SSLCertVerificationError(1, "unknown issuer")
    elif failure == "data_error":
        error = ValueError("invalid data")
    else:
        status = dict(unauthorized=401, forbidden=403, not_found=404)[failure]
        response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/example"))
        error = httpx.HTTPStatusError("request denied", request=response.request, response=response)
    calls = []
    def fail():
        calls.append(True)
        raise error
    with pytest.raises(type(error)):
        hf.hf_retry("test", fail)
    assert calls == [True] and retry_state == dict(closes=0, sleeps=[])


def test_temporary_hub_503_is_retried(retry_state):
    response = httpx.Response(503, request=httpx.Request("GET", "https://huggingface.co/example"))
    error = httpx.HTTPStatusError("unavailable", request=response.request, response=response)
    replies = [error, "cached snapshot"]
    def operation():
        result = replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    assert hf.hf_retry("snapshot", operation) == "cached snapshot"
    assert retry_state == dict(closes=1, sleeps=[2])


def test_existing_complete_lock_works_without_network(retry_state, tmp_path):
    path = tmp_path / "lock.json"
    lock = dict(requested={"model": ["repo/model", "main"]}, resolved={"model": "a" * 40})
    path.write_text(json.dumps(lock))
    def fail(*args, **kwargs):
        pytest.fail("Complete revision lock must not need network")
    assert hf.resolve_revisions(SimpleNamespace(model_info=fail, dataset_info=fail),
                                {"model": ("repo/model", "main")}, path) == lock


def test_invalid_revision_is_not_saved(retry_state, tmp_path):
    path = tmp_path / "lock.json"
    api = SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha=None))
    with pytest.raises(ValueError, match="invalid immutable revision"):
        hf.resolve_revisions(api, {"model": ("repo/model", "main")}, path)
    assert not path.exists()
