"""Input validation shared by the pipeline and the profiler."""

from __future__ import annotations

import pandas as pd

from datacleaner.core.errors import InvalidInputError


def prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with string column names, or raise :class:`InvalidInputError`."""
    if data.empty or len(data.columns) == 0:
        raise InvalidInputError("The dataset is empty.")

    df = data.copy()
    df.columns = df.columns.astype(str)

    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise InvalidInputError(f"Duplicate column names are not supported: {dupes}")

    return df
