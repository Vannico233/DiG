"""Retry transient Hub connections and persist each immutable repository revision."""
import json
from pathlib import Path
import re
import ssl
import time

from .io import write_json


def transient_hf_error(error):
    import httpx
    causes, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        causes.append(error)
        error = error.__cause__ or error.__context__
    # Certificate trust errors require a real CA fix, not retries with weaker checks.
    if any(isinstance(e, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e) for e in causes):
        return False
    statuses = [e.response.status_code for e in causes if isinstance(e, httpx.HTTPStatusError)]
    if statuses:
        return all(status in (408, 429, 500, 502, 503, 504) for status in statuses)
    for error in causes:
        if isinstance(error, (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError,
                              httpx.ProxyError, ssl.SSLEOFError, ConnectionError, TimeoutError)):
            return True
    return False


def hf_retry(label, operation, attempts=4):
    from huggingface_hub.utils import close_session
    if attempts < 1:
        raise ValueError("HF network attempts must be positive")
    for attempt in range(1, attempts + 1):
        print("{} (attempt {}/{})".format(label, attempt, attempts), flush=True)
        try:
            return operation()
        except Exception as error:
            if not transient_hf_error(error):
                raise
            close_session()  # Drop a broken TLS/proxy connection before rebuilding the Hub client.
            if attempt == attempts:
                raise RuntimeError(
                    "{} failed after {} attempts. Cached files and resolved revisions are retained. "
                    "The Hugging Face connection is still unavailable; inspect the proxy/endpoint connection.".format(
                        label, attempts)) from error
            delay = min(2 ** attempt, 16)
            print("Transient HF error ({}); retrying with a fresh client in {}s.".format(
                type(error).__name__, delay), flush=True)
            time.sleep(delay)


def resolve_revisions(api, requested, lock_path, attempts=4, timeout=30):
    lock_path = Path(lock_path)
    normalized = {key: list(value) for key, value in requested.items()}
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["requested"] != normalized:
            raise ValueError("Download revisions differ from lock; use a new assets directory")
    else:
        lock = dict(requested=normalized, resolved={})
    for key, (repo, revision) in requested.items():
        if key not in lock["resolved"]:
            method = api.model_info if key == "model" else api.dataset_info
            info = hf_retry("Resolving {}@{}".format(repo, revision),
                            lambda: method(repo, revision=revision, timeout=timeout), attempts)
            if not isinstance(info.sha, str) or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
                raise ValueError("Hub returned an invalid immutable revision for " + repo)
            lock["resolved"][key] = info.sha
            # A later repository failure must not lose already resolved revisions.
            write_json(lock_path, lock)
        elif not isinstance(lock["resolved"][key], str) or not re.fullmatch(r"[0-9a-f]{40}", lock["resolved"][key]):
            raise ValueError("Invalid saved revision for " + repo)
        print("Pinned {}: {}".format(repo, lock["resolved"][key]), flush=True)
    return lock
