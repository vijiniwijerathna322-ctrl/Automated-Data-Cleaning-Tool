import io
from collections.abc import Iterator

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from datacleaner.api.main import create_app
from datacleaner.api.settings import Settings
from datacleaner.ui.api_client import ApiClient, ApiError, default_client


@pytest.fixture
def api() -> Iterator[ApiClient]:
    with TestClient(create_app(Settings(json_logs=False))) as http:
        yield ApiClient(client=http)


@pytest.fixture
def content(sample_path) -> bytes:  # type: ignore[no-untyped-def]
    return sample_path.read_bytes()


def test_health(api: ApiClient) -> None:
    assert api.health()["status"] == "ok"


def test_profile(api: ApiClient, content: bytes) -> None:
    body = api.profile(content, "s.csv", preview_rows=3)
    assert body["profile"]["rows"] == 19
    assert len(body["preview"]["rows"]) == 3


def test_clean(api: ApiClient, content: bytes) -> None:
    config = {"target": "Exam_Score", "feature_source": "Exam_Score"}
    body = api.clean(content, "s.csv", config)
    assert body["report"]["duplicates_removed"] == 1
    assert "Student_ID" not in body["ml"]["columns"]


def test_download(api: ApiClient, content: bytes) -> None:
    raw = api.download(content, "s.csv", {}, dataset="cleaned", fmt="csv")
    assert len(pd.read_csv(io.BytesIO(raw))) == 18


def test_problem_response_becomes_api_error(api: ApiClient) -> None:
    with pytest.raises(ApiError) as info:
        api.clean(b"x", "bad.txt", {})

    assert info.value.status == 400
    assert "Unsupported" in info.value.detail


def test_unreachable_api_becomes_api_error() -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = ApiClient(
        base_url="http://nowhere",
        client=httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://nowhere"),
    )

    with pytest.raises(ApiError, match="unreachable"):
        client.health()


def test_non_json_error_body() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(502, text="Bad gateway"))
    client = ApiClient(client=httpx.Client(transport=transport, base_url="http://x"))

    with pytest.raises(ApiError) as info:
        client.health()

    assert info.value.status == 502


def test_non_object_json_rejected() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json=[1, 2]))
    client = ApiClient(client=httpx.Client(transport=transport, base_url="http://x"))

    with pytest.raises(ApiError, match="Bad response"):
        client.health()


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATACLEANER_API_URL", "http://example.test:9000/")
    assert default_client().base_url == "http://example.test:9000"
