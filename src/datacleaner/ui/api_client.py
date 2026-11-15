"""The only place the UI talks HTTP. Returns plain dicts, never API/core models."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class ApiError(Exception):
    """The API returned a problem response or could not be reached."""

    def __init__(self, title: str, detail: str, status: int | None = None) -> None:
        super().__init__(f"{title}: {detail}")
        self.title = title
        self.detail = detail
        self.status = status


class ApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        url = base_url or os.getenv("DATACLEANER_API_URL") or DEFAULT_API_URL
        self.base_url = url.rstrip("/")
        # An injected client (e.g. FastAPI's TestClient) already knows its base URL.
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    # ---------------------------------------------------------------- calls
    def health(self) -> dict[str, Any]:
        return self._json(self._send("GET", "/healthz"))

    def profile(self, content: bytes, filename: str, preview_rows: int = 100) -> dict[str, Any]:
        return self._json(
            self._send(
                "POST",
                "/v1/profile",
                params={"preview_rows": preview_rows},
                files={"file": (filename, content)},
            )
        )

    def clean(
        self,
        content: bytes,
        filename: str,
        config: dict[str, Any],
        preview_rows: int = 1000,
    ) -> dict[str, Any]:
        return self._json(
            self._send(
                "POST",
                "/v1/clean",
                params={"preview_rows": preview_rows},
                files={"file": (filename, content)},
                data={"config": json.dumps(config)},
            )
        )

    def download(
        self,
        content: bytes,
        filename: str,
        config: dict[str, Any],
        dataset: str = "cleaned",
        fmt: str = "csv",
    ) -> bytes:
        response = self._send(
            "POST",
            "/v1/clean/download",
            params={"dataset": dataset, "format": fmt},
            files={"file": (filename, content)},
            data={"config": json.dumps(config)},
        )
        return response.content

    # -------------------------------------------------------------- helpers
    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError("API unreachable", f"Could not reach {self.base_url}: {exc}") from exc

        if response.is_error:
            raise _error_from(response)

        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        body = response.json()

        if not isinstance(body, dict):
            raise ApiError("Bad response", "Expected a JSON object.", response.status_code)

        return body


_clients: dict[str, ApiClient] = {}


def default_client() -> ApiClient:
    """Client for the API named by ``DATACLEANER_API_URL``.

    When that variable is unset (e.g. Streamlit Community Cloud, or ``datacleaner-ui`` run
    on its own) the API is started inside this process instead; see :mod:`.embedded`.
    One client per URL is reused so connections are not leaked on every Streamlit rerun.
    """
    url = os.getenv("DATACLEANER_API_URL")

    if not url:
        from datacleaner.ui.embedded import ensure_embedded_api

        url = ensure_embedded_api()

    url = url.rstrip("/")

    if url not in _clients:
        _clients[url] = ApiClient(base_url=url)

    return _clients[url]


def _error_from(response: httpx.Response) -> ApiError:
    try:
        body = response.json()
        return ApiError(
            str(body.get("title", "Error")), str(body.get("detail", "")), response.status_code
        )
    except (ValueError, AttributeError):
        return ApiError("Error", response.text[:200], response.status_code)
