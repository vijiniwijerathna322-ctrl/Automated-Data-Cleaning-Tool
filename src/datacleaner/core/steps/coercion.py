"""Convert text columns that are really numbers."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from datacleaner.core.columns import text_columns
from datacleaner.core.steps.base import PipelineContext, Step


def coerce_numeric_columns(
    df: pd.DataFrame, skip: tuple[str, ...] | list[str] = ()
) -> tuple[pd.DataFrame, list[str]]:
    """Convert text columns to numbers when *every* non-null value parses.

    Nothing is converted if even one value would be lost, so this never silently
    turns bad values into NaN.
    """
    out = df.copy()
    converted: list[str] = []

    for column in text_columns(out):
        if column in skip:
            continue

        values = out[column]
        non_null = values.dropna()

        if non_null.empty:
            continue

        parsed = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")

        if parsed.notna().all():
            stripped: Any = values.astype("object").map(
                lambda v: v.strip() if isinstance(v, str) else v
            )
            out[column] = pd.to_numeric(stripped, errors="coerce")
            converted.append(column)

    return out, converted


class NumericCoercionStep(Step):
    name: ClassVar[str] = "coerce_numeric"
    toggle: ClassVar[str] = "coerce_numeric"

    def apply(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        out, ctx.converted_columns = coerce_numeric_columns(df, skip=ctx.id_columns)
        return out
