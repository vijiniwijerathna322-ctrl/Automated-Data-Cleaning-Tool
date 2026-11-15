"""Build the ML-ready dataset from the cleaned one.

Not a :class:`Step`: it *derives* a second dataset instead of transforming the
cleaned one in place.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from datacleaner.core.columns import numeric_columns, text_columns
from datacleaner.core.steps.base import PipelineContext


def build_ml_dataset(
    df: pd.DataFrame,
    id_columns: list[str] | tuple[str, ...] = (),
    target: str | None = None,
    derived_from: dict[str, str] | None = None,
    encode: bool = True,
    scale: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """One-hot encode and scale a cleaned dataframe for modelling.

    * ID columns are dropped (no signal, and they leak row identity).
    * The target column is never encoded or scaled.
    * Engineered features derived from the target are dropped (leakage).
    * Only original numeric columns are scaled, never the 0/1 dummies.

    Returns ``(ml_df, encoded_columns, scaled_columns, dropped_columns)``.
    """
    derived_from = derived_from or {}
    ml = df.copy()

    dropped = [c for c in id_columns if c in ml.columns]

    if target is not None:
        dropped += [f for f, src in derived_from.items() if src == target and f in ml.columns]

    ml = ml.drop(columns=dropped)

    protected = {target} if target is not None else set()

    scaled: list[str] = []
    encoded: list[str] = []

    to_scale = [c for c in numeric_columns(ml) if c not in protected]

    if scale and to_scale:
        ml[to_scale] = StandardScaler().fit_transform(ml[to_scale])
        scaled = to_scale

    if encode:
        encoded = [c for c in text_columns(ml) if c not in protected]

        if encoded:
            ml = pd.get_dummies(ml, columns=encoded, dtype=int)

    return ml, encoded, scaled, dropped


def build_from_context(df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
    """Run :func:`build_ml_dataset` using, and updating, the pipeline context."""
    ml, ctx.encoded_columns, ctx.scaled_columns, ctx.dropped_from_ml = build_ml_dataset(
        df,
        id_columns=ctx.id_columns,
        target=ctx.config.target,
        derived_from=ctx.derived_from,
        encode=ctx.config.apply_encoding,
        scale=ctx.config.apply_scaling,
    )
    return ml
