"""IQR outlier detection. Outliers are reported, never removed."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from datacleaner.core.columns import numeric_columns
from datacleaner.core.report import OutlierSummary
from datacleaner.core.steps.base import PipelineContext, Step


def find_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    multiplier: float = 1.5,
) -> tuple[dict[str, OutlierSummary], dict[str, pd.DataFrame]]:
    """Return ``(summaries, offending_rows)`` keyed by column name."""
    summaries: dict[str, OutlierSummary] = {}
    rows: dict[str, pd.DataFrame] = {}

    for column in columns if columns is not None else numeric_columns(df):
        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        mask = (df[column] < lower) | (df[column] > upper)
        count = int(mask.sum())

        if count > 0:
            summaries[column] = OutlierSummary(count=count, lower=lower, upper=upper)
            rows[column] = df[mask]

    return summaries, rows


class OutlierStep(Step):
    name: ClassVar[str] = "outliers"
    toggle: ClassVar[str] = "detect_outliers"

    def apply(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        ctx.outliers, ctx.outlier_rows = find_outliers(
            df, multiplier=ctx.config.outlier_iqr_multiplier
        )
        return df
