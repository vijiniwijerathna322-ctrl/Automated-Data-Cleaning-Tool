"""Serialise DataFrames and reports."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from datacleaner.core.report import CleaningReport
from datacleaner.io.formats import FileFormat, detect_format


def to_bytes(df: pd.DataFrame, fmt: FileFormat = FileFormat.CSV) -> bytes:
    """Serialise ``df`` in the requested format (index is never written)."""
    if fmt is FileFormat.CSV:
        return df.to_csv(index=False).encode("utf-8")

    buffer = io.BytesIO()

    if fmt is FileFormat.XLSX:
        df.to_excel(buffer, index=False, engine="openpyxl")
    else:
        df.to_parquet(buffer, index=False)

    return buffer.getvalue()


def write_path(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to ``path``; the format comes from its extension."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(to_bytes(df, detect_format(file.name)))
    return file


def write_report(report: CleaningReport, path: str | Path) -> Path:
    """Write the cleaning report as pretty-printed JSON."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return file
