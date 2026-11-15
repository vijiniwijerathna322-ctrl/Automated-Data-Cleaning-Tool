"""Small helpers for classifying DataFrame columns."""

from __future__ import annotations

import pandas as pd


def text_columns(df: pd.DataFrame) -> list[str]:
    """Names of text (object / string) columns."""
    return [str(c) for c in df.select_dtypes(include=["object", "string"]).columns]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Names of numeric columns."""
    return [str(c) for c in df.select_dtypes(include="number").columns]


def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Columns whose name looks like an identifier (``id``, ``*_id``, ``*identifier*``)."""
    ids: list[str] = []

    for column in df.columns:
        name = str(column).lower()

        if name == "id" or name.endswith("_id") or "identifier" in name:
            ids.append(str(column))

    return ids
