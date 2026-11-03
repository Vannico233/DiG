"""HF must initialize with the proxy variables inherited from shared server images."""
import socket

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("huggingface_hub")
from huggingface_hub.utils import close_session, get_session


def test_hf_client_initializes_with_http_and_socks_environment(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:7890")
    for name in ("all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "socks5://127.0.0.1:7890")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "localhost,127.0.0.1")
    def no_network(*args, **kwargs):
        pytest.fail("Initializing the HF client must not send a network request")
    monkeypatch.setattr(socket, "create_connection", no_network)
    close_session()
    try:
        client = get_session()
        assert isinstance(client, httpx.Client)
        assert not client.is_closed
    finally:
        close_session()
