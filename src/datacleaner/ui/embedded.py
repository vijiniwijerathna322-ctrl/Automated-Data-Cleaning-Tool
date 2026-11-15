"""Run the real API inside the UI process, for single-process hosts.

Platforms such as Streamlit Community Cloud run exactly one process, so there is nowhere to
start a separate API. When ``DATACLEANER_API_URL`` is not set, the UI starts the same FastAPI
app on a loopback port in a background thread and talks to it over real HTTP. The layering is
unchanged: the UI still only knows ``api_client``.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request

import uvicorn

from datacleaner.api.main import create_app
from datacleaner.api.settings import Settings

HOST = "127.0.0.1"
STARTUP_TIMEOUT_S = 30.0

_lock = threading.Lock()
_server: uvicorn.Server | None = None
_thread: threading.Thread | None = None
_url: str | None = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def _wait_until_healthy(url: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)

    raise RuntimeError(f"Embedded API did not become healthy within {STARTUP_TIMEOUT_S:.0f}s.")


def ensure_embedded_api() -> str:
    """Start the embedded API once per process and return its base URL."""
    global _server, _thread, _url

    with _lock:
        if _thread is not None and _thread.is_alive() and _url is not None:
            return _url

        port = _free_port()
        url = f"http://{HOST}:{port}"

        # Embedded mode serves one browser session: human-readable logs, default limits.
        app = create_app(Settings(json_logs=False))
        server = uvicorn.Server(
            uvicorn.Config(app, host=HOST, port=port, log_level="warning", access_log=False)
        )
        thread = threading.Thread(target=server.run, name="embedded-api", daemon=True)
        thread.start()

        try:
            _wait_until_healthy(url)
        except RuntimeError:
            server.should_exit = True
            raise

        _server, _thread, _url = server, thread, url
        return url


def shutdown_embedded_api() -> None:
    """Stop the embedded API (used by tests; the daemon thread dies with the process anyway)."""
    global _server, _thread, _url

    with _lock:
        if _server is not None:
            _server.should_exit = True

        if _thread is not None:
            _thread.join(timeout=10)

        _server = _thread = _url = None
