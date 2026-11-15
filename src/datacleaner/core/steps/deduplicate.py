"""Drop exact duplicate rows."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from datacleaner.core.steps.base import PipelineContext, Step


class DeduplicateStep(Step):
    name: ClassVar[str] = "deduplicate"
    toggle: ClassVar[str] = "remove_duplicates"

    def apply(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        out = df.drop_duplicates()
        ctx.duplicates_removed = len(df) - len(out)
        return out
