"""Supported tabular file formats."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePath

from datacleaner.core.errors import InvalidInputError


class FileFormat(StrEnum):
    CSV = "csv"
    XLSX = "xlsx"
    PARQUET = "parquet"

    @property
    def media_type(self) -> str:
        return {
            FileFormat.CSV: "text/csv",
            FileFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            FileFormat.PARQUET: "application/vnd.apache.parquet",
        }[self]


def detect_format(filename: str) -> FileFormat:
    """Infer the format from a filename's extension."""
    suffix = PurePath(filename).suffix.lower().lstrip(".")

    try:
        return FileFormat(suffix)
    except ValueError:
        supported = ", ".join(f".{f.value}" for f in FileFormat)
        raise InvalidInputError(
            f"Unsupported file type {('.' + suffix) if suffix else '(none)'!r}. "
            f"Supported: {supported}."
        ) from None
