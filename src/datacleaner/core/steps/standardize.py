"""Normalise whitespace and case in categorical text columns."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from datacleaner.core.columns import text_columns
from datacleaner.core.steps.base import PipelineContext, Step


def _tidy_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower().title()
    return value


def standardize_categories(
    df: pd.DataFrame,
    skip: tuple[str, ...] | list[str] = (),
    max_unique: int = 50,
    max_unique_ratio: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """Trim and title-case categorical text columns.

    Skips ``skip`` (ID columns) and high-cardinality columns, which are almost
    certainly free text or identifiers. Missing values stay missing.
    """
    out = df.copy()
    standardized: list[str] = []

    for column in text_columns(out):
        if column in skip:
            continue

        unique = int(out[column].nunique(dropna=True))
        non_null = int(out[column].notna().sum())

        if unique > max_unique and non_null and unique / non_null > max_unique_ratio:
            continue

        out[column] = out[column].map(_tidy_text)
        standardized.append(column)

    return out, standardized


class StandardizeStep(Step):
    name: ClassVar[str] = "standardize"
    toggle: ClassVar[str] = "standardize_categories"

    def apply(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        out, ctx.standardized_columns = standardize_categories(
            df,
            skip=ctx.id_columns,
            max_unique=ctx.config.max_category_unique,
            max_unique_ratio=ctx.config.max_category_unique_ratio,
        )
        return out
