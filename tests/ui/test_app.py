"""Smoke test: the Streamlit script runs end to end against an in-process API."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from datacleaner.api.main import create_app
from datacleaner.api.settings import Settings
from datacleaner.ui import api_client

APP = Path(api_client.__file__).with_name("app.py")
SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_data.csv"


def script(upload: str | None) -> str:
    """Run app.py with ``st.file_uploader`` replaced by a fake upload."""
    patch = (
        "class Fake:\n"
        f"    name = {Path(upload).name!r}\n"
        f"    def getvalue(self): return open({upload!r}, 'rb').read()\n"
        "st.file_uploader = lambda *a, **k: Fake()\n"
        if upload
        else "st.file_uploader = lambda *a, **k: None\n"
    )
    return (
        "import streamlit as st\n"
        + patch
        + f"exec(compile(open({str(APP)!r}, encoding='utf-8').read(), 'app.py', 'exec'))\n"
    )


@pytest.fixture(autouse=True)
def in_process_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    st.cache_data.clear()

    with TestClient(create_app(Settings(json_logs=False))) as http:
        monkeypatch.setattr(
            api_client, "default_client", lambda: api_client.ApiClient(client=http)
        )
        yield

    st.cache_data.clear()


def run(upload: Path | None = SAMPLE) -> AppTest:
    return AppTest.from_string(script(str(upload) if upload else None), default_timeout=60).run()


def test_landing_page_without_file() -> None:
    at = run(None)
    assert not at.exception
    assert "Upload a CSV" in at.info[0].value


def test_sample_dataset_renders_report() -> None:
    at = run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Rows"] == "18"
    assert metrics["Duplicates Removed"] == "1"
    assert metrics["Missing Fixed"] == "4"
    assert metrics["Potential Outlier Columns"] == "1"


def test_toggle_is_respected() -> None:
    at = run()
    box = next(c for c in at.sidebar.checkbox if "Standardize" in c.label)
    box.uncheck().run()
    assert not at.exception, [e.value for e in at.exception]

    cleaned = at.dataframe[1].value
    assert {"MALE", " female "} & set(cleaned["Gender"])


def test_feature_toggle() -> None:
    at = run()
    next(c for c in at.sidebar.checkbox if "Feature" in c.label).uncheck().run()
    created = next(m for m in at.metric if m.label == "Features Created")
    assert created.value == "0"


def test_api_error_is_shown_not_raised(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"")
    at = run(bad)
    assert not at.exception
    assert any("empty" in e.value.lower() for e in at.error)
