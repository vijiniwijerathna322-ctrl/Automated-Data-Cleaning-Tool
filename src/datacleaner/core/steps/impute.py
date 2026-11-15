"""Missing-value imputation: median for numbers, "Unknown" for text."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from datacleaner.core.columns import numeric_columns, text_columns
from datacleaner.core.steps.base import PipelineContext, Step


def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill gaps; returns ``(dataframe, number_of_values_filled)``."""
    out = df.copy()
    filled = 0

    for column in numeric_columns(out):
        count = int(out[column].isna().sum())

        # An all-empty column has no median; leave it alone.
        if 0 < count < len(out):
            out[column] = out[column].fillna(out[column].median())
            filled += count

    for column in text_columns(out):
        count = int(out[column].isna().sum())

        if count > 0:
            out[column] = out[column].fillna("Unknown")
            filled += count

    return out, filled


class ImputeStep(Step):
    name: ClassVar[str] = "impute"
    toggle: ClassVar[str] = "handle_missing"

    def apply(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        out, ctx.missing_fixed = impute_missing(df)
        return out
