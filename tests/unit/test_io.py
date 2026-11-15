from pathlib import Path

import pandas as pd
import pytest

from datacleaner import Pipeline
from datacleaner.core.errors import InvalidInputError
from datacleaner.io import (
    FileFormat,
    detect_format,
    read_bytes,
    read_path,
    to_bytes,
    write_path,
    write_report,
)


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("a.csv", FileFormat.CSV), ("A.XLSX", FileFormat.XLSX), ("x.parquet", FileFormat.PARQUET)],
)
def test_detect_format(name: str, fmt: FileFormat) -> None:
    assert detect_format(name) is fmt


@pytest.mark.parametrize("name", ["a.txt", "noext", "a.xls"])
def test_unsupported_extension(name: str) -> None:
    with pytest.raises(InvalidInputError, match="Unsupported"):
        detect_format(name)


@pytest.mark.parametrize("fmt", list(FileFormat))
def test_roundtrip_every_format(fmt: FileFormat, sample_df: pd.DataFrame) -> None:
    content = to_bytes(sample_df, fmt)
    back = read_bytes(content, f"data.{fmt.value}")
    assert back.shape == sample_df.shape
    assert list(back.columns) == list(sample_df.columns)


def test_media_types() -> None:
    assert FileFormat.CSV.media_type == "text/csv"
    assert all(f.media_type for f in FileFormat)


def test_all_formats_give_same_pipeline_result(sample_df: pd.DataFrame) -> None:
    reports = []

    for fmt in FileFormat:
        df = read_bytes(to_bytes(sample_df, fmt), f"s.{fmt.value}")
        reports.append(Pipeline().run(df).report)

    assert all(r.duplicates_removed == reports[0].duplicates_removed for r in reports)
    assert all(r.missing_fixed == reports[0].missing_fixed for r in reports)


def test_empty_content() -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        read_bytes(b"", "a.csv")


def test_header_only_csv_has_no_rows() -> None:
    with pytest.raises(InvalidInputError, match="no rows"):
        read_bytes(b"a,b\n", "a.csv")


def test_garbage_is_a_typed_error() -> None:
    with pytest.raises(InvalidInputError, match="Could not read"):
        read_bytes(b"not a parquet file", "a.parquet")


def test_max_rows_enforced() -> None:
    csv = b"a\n" + b"1\n" * 10

    assert len(read_bytes(csv, "a.csv", max_rows=10)) == 10

    with pytest.raises(InvalidInputError, match="more than"):
        read_bytes(csv, "a.csv", max_rows=9)


def test_max_rows_enforced_for_parquet(sample_df: pd.DataFrame) -> None:
    with pytest.raises(InvalidInputError, match="more than"):
        read_bytes(to_bytes(sample_df, FileFormat.PARQUET), "a.parquet", max_rows=3)


def test_read_path_and_write_path(tmp_path: Path, sample_path: Path) -> None:
    df = read_path(sample_path)
    out = write_path(df, tmp_path / "nested" / "out.csv")
    assert out.is_file()
    assert read_path(out).shape == df.shape


def test_read_path_missing(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="not found"):
        read_path(tmp_path / "nope.csv")


def test_write_report(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    report = Pipeline().run(sample_df).report
    out = write_report(report, tmp_path / "r" / "report.json")
    assert '"duplicates_removed": 1' in out.read_text(encoding="utf-8")
