"""The UI can run the API in-process when no external API is configured."""

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from datacleaner.ui import api_client, embedded


@pytest.fixture
def embedded_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.delenv("DATACLEANER_API_URL", raising=False)
    api_client._clients.clear()
    url = embedded.ensure_embedded_api()
    yield url
    embedded.shutdown_embedded_api()
    api_client._clients.clear()


def test_embedded_api_serves_real_http(embedded_api: str) -> None:
    assert embedded_api.startswith("http://127.0.0.1:")
    assert httpx.get(f"{embedded_api}/healthz").json()["status"] == "ok"


def test_ensure_is_idempotent(embedded_api: str) -> None:
    assert embedded.ensure_embedded_api() == embedded_api


def test_default_client_uses_embedded_api_without_env(
    embedded_api: str,
    sample_path: Path,
) -> None:
    client = api_client.default_client()
    assert client.base_url == embedded_api
    assert api_client.default_client() is client  # reused, not rebuilt per call

    report = client.clean(sample_path.read_bytes(), "s.csv", {})["report"]
    assert report["duplicates_removed"] == 1


def test_env_var_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATACLEANER_API_URL", "http://example.test:9000/")
    api_client._clients.clear()

    try:
        assert api_client.default_client().base_url == "http://example.test:9000"
        assert embedded._thread is None or not embedded._thread.is_alive()
    finally:
        api_client._clients.clear()


def test_restarts_after_shutdown(embedded_api: str) -> None:
    embedded.shutdown_embedded_api()
    new_url = embedded.ensure_embedded_api()
    assert httpx.get(f"{new_url}/healthz").status_code == 200


def test_startup_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def never_healthy(_url: str) -> None:
        raise RuntimeError("Embedded API did not become healthy")

    monkeypatch.setattr(embedded, "_wait_until_healthy", never_healthy)
    embedded.shutdown_embedded_api()

    with pytest.raises(RuntimeError, match="healthy"):
        embedded.ensure_embedded_api()

    embedded.shutdown_embedded_api()
