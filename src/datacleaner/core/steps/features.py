"""Feature engineering: bin one numeric column into a categorical feature."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from datacleaner.core.errors import InvalidConfigError
from datacleaner.core.steps.base import PipelineContext, Step

QUARTILE_LABELS = ["Low", "Medium-Low", "Medium-High", "High"]


def _score_category(score: Any) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 80:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 60:
        return "Average"
    return "Needs Improvement"


def create_category_feature(
    df: pd.DataFrame, source_column: str
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Bin ``source_column`` into ``<source>_Category``.

    ``Exam_Score`` uses fixed grade bands; any other column is split into quartile
    bands. Returns ``(dataframe, new_feature_name_or_None, warning_or_None)``.
    """
    if source_column not in df.columns:
        raise InvalidConfigError(f"feature_source {source_column!r} is not a column.")

    if not pd.api.types.is_numeric_dtype(df[source_column]):
        return df, None, f"feature_source {source_column!r} is not numeric; no feature created."

    name = f"{source_column}_Category"

    if name in df.columns:
        return df, None, f"Column {name!r} already exists; no feature created."

    out = df.copy()

    if source_column == "Exam_Score":
        out[name] = out[source_column].apply(_score_category)
    else:
        try:
            binned = pd.qcut(out[source_column], q=4, labels=QUARTILE_LABELS)
        except ValueError:
            return df, None, f"{source_column!r} has too few distinct values to form quartiles."

        out[name] = binned.astype("object").where(binned.notna(), "Unknown")

    return out, name, None


class FeatureStep(Step):
    name: ClassVar[str] = "features"
    toggle: ClassVar[str] = "create_features"

    def apply(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        source = ctx.config.feature_source

        if source is None:
            return df

        out, feature, warning = create_category_feature(df, source)

        if warning:
            ctx.warnings.append(warning)

        if feature:
            ctx.created_features.append(feature)
            ctx.derived_from[feature] = source

        return out
