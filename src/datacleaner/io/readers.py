"""Read CSV / Excel / Parquet into a DataFrame, turning failures into typed errors."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from datacleaner.core.errors import InvalidInputError
from datacleaner.io.formats import FileFormat, detect_format


def read_bytes(content: bytes, filename: str, max_rows: int | None = None) -> pd.DataFrame:
    """Parse ``content`` according to the extension of ``filename``.

    ``max_rows`` bounds memory: a file with more rows is rejected rather than loaded.
    """
    if not content:
        raise InvalidInputError("The uploaded file is empty.")

    fmt = detect_format(filename)
    buffer = io.BytesIO(content)

    try:
        if fmt is FileFormat.CSV:
            # +1 so we can tell "exactly max_rows" from "too many".
            df = pd.read_csv(buffer, nrows=None if max_rows is None else max_rows + 1)
        elif fmt is FileFormat.XLSX:
            df = pd.read_excel(buffer, engine="openpyxl")
        else:
            df = pd.read_parquet(buffer)
    except InvalidInputError:
        raise
    except Exception as exc:
        raise InvalidInputError(f"Could not read {filename!r} as {fmt.value}: {exc}") from exc

    if max_rows is not None and len(df) > max_rows:
        raise InvalidInputError(f"Dataset has more than the allowed {max_rows:,} rows.")

    if df.empty:
        raise InvalidInputError("The dataset contains no rows.")

    return df


def read_path(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Read a dataset from disk."""
    file = Path(path)

    if not file.is_file():
        raise InvalidInputError(f"Input file not found: {file}")

    return read_bytes(file.read_bytes(), file.name, max_rows=max_rows)
