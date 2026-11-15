import json
import logging

import pandas as pd
import pytest

from datacleaner.core.errors import InvalidInputError
from datacleaner.core.profile import profile_dataframe
from datacleaner.logging import JsonFormatter, configure_logging


def test_profile_sample(sample_df: pd.DataFrame) -> None:
    profile = profile_dataframe(sample_df)
    assert (profile.rows, profile.columns) == (19, 7)
    assert profile.duplicate_rows == 1
    assert profile.id_columns == ["Student_ID"]

    kinds = {c.name: c.kind for c in profile.column_profiles}
    assert kinds["Age"] == "numeric"
    assert kinds["Gender"] == "text"

    study = next(c for c in profile.column_profiles if c.name == "Study_Hours")
    assert study.missing == 2
    assert study.histogram is not None
    assert sum(study.histogram.counts) == 17


def test_profile_handles_bool_all_nan_and_other() -> None:
    df = pd.DataFrame(
        {
            "flag": [True, False],
            "empty": [float("nan")] * 2,
            "when": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    kinds = {c.name: c for c in profile_dataframe(df).column_profiles}
    assert kinds["flag"].kind == "other"
    assert kinds["when"].kind == "other"
    assert kinds["empty"].kind == "numeric"
    assert kinds["empty"].histogram is None


def test_profile_rejects_empty() -> None:
    with pytest.raises(InvalidInputError):
        profile_dataframe(pd.DataFrame())


def test_json_formatter_merges_extras() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("w",), None)
    record.step = "impute"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello w"
    assert payload["step"] == "impute"
    assert payload["level"] == "INFO"


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "oops", (), sys.exc_info())

    assert "ValueError" in json.loads(JsonFormatter().format(record))["exc_info"]


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level

    try:
        configure_logging("debug", json_logs=False)
        configure_logging("warning", json_logs=True)
        assert len(root.handlers) == 1
        assert root.level == logging.WARNING
    finally:
        root.handlers[:], root.level = saved_handlers, saved_level
