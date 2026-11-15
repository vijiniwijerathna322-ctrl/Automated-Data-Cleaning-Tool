"""Quick, read-only profile of a raw dataset (used to drive UI choices and charts)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from datacleaner.core.columns import detect_id_columns
from datacleaner.core.validation import prepare_dataframe


class Histogram(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: Literal["numeric", "text", "other"]
    missing: int
    unique: int
    histogram: Histogram | None = None


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    id_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    column_profiles: list[ColumnProfile] = Field(default_factory=list)


def _histogram(values: pd.Series, bins: int) -> Histogram | None:
    clean = values.dropna().to_numpy(dtype=float)

    if clean.size == 0:
        return None

    counts, edges = np.histogram(clean, bins=bins)
    return Histogram(bin_edges=[float(e) for e in edges], counts=[int(c) for c in counts])


def profile_dataframe(data: pd.DataFrame, bins: int = 20) -> DatasetProfile:
    """Describe ``data`` without modifying it."""
    df = prepare_dataframe(data)
    profiles: list[ColumnProfile] = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_bool_dtype(series):
            kind: Literal["numeric", "text", "other"] = "other"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "numeric"
        elif pd.api.types.is_string_dtype(series) or series.dtype == object:
            kind = "text"
        else:
            kind = "other"

        profiles.append(
            ColumnProfile(
                name=column,
                dtype=str(series.dtype),
                kind=kind,
                missing=int(series.isna().sum()),
                unique=int(series.nunique(dropna=True)),
                histogram=_histogram(series, bins) if kind == "numeric" else None,
            )
        )

    return DatasetProfile(
        rows=len(df),
        columns=len(df.columns),
        missing_values=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        id_columns=detect_id_columns(df),
        numeric_columns=[p.name for p in profiles if p.kind == "numeric"],
        column_profiles=profiles,
    )
