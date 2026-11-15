"""Pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from datacleaner.core.columns import detect_id_columns
from datacleaner.core.config import PipelineConfig
from datacleaner.core.errors import InvalidConfigError
from datacleaner.core.report import CleaningReport, DatasetStats, StepReport
from datacleaner.core.steps import Step, build_steps
from datacleaner.core.steps.base import PipelineContext
from datacleaner.core.steps.ml_dataset import build_from_context
from datacleaner.core.validation import prepare_dataframe

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Output of a run: both datasets, the report and the offending outlier rows."""

    cleaned: pd.DataFrame
    ml: pd.DataFrame
    report: CleaningReport
    outlier_rows: dict[str, pd.DataFrame] = field(default_factory=dict)


class Pipeline:
    """Runs enabled steps in order, then derives the ML-ready dataset."""

    def __init__(self, config: PipelineConfig | None = None, steps: Sequence[Step] | None = None):
        self.config = config or PipelineConfig()
        self.steps: list[Step] = list(steps) if steps is not None else build_steps()

    # ------------------------------------------------------------------ run
    def run(self, data: pd.DataFrame) -> PipelineResult:
        df = prepare_dataframe(data)
        self._validate_config(df)

        started = time.perf_counter()
        ctx = PipelineContext(config=self.config, id_columns=detect_id_columns(df))
        before = DatasetStats.of(df)
        step_reports: list[StepReport] = []

        for step in self.steps:
            if not step.is_enabled(self.config):
                logger.debug("step skipped", extra={"step": step.name})
                continue

            t0 = time.perf_counter()
            df = step.apply(df, ctx)
            elapsed = (time.perf_counter() - t0) * 1000

            logger.info("step done", extra={"step": step.name, "duration_ms": round(elapsed, 2)})
            step_reports.append(
                StepReport(
                    name=step.name,
                    duration_ms=round(elapsed, 3),
                    rows_after=len(df),
                    columns_after=len(df.columns),
                )
            )

        ml = build_from_context(df, ctx)

        report = CleaningReport(
            before=before,
            after=DatasetStats.of(df),
            ml_rows=len(ml),
            ml_columns=len(ml.columns),
            missing_fixed=ctx.missing_fixed,
            duplicates_removed=ctx.duplicates_removed,
            converted_columns=ctx.converted_columns,
            standardized_columns=ctx.standardized_columns,
            outliers=ctx.outliers,
            created_features=ctx.created_features,
            id_columns=ctx.id_columns,
            encoded_columns=ctx.encoded_columns,
            scaled_columns=ctx.scaled_columns,
            dropped_from_ml=ctx.dropped_from_ml,
            steps=step_reports,
            warnings=ctx.warnings,
            total_duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )

        return PipelineResult(cleaned=df, ml=ml, report=report, outlier_rows=ctx.outlier_rows)

    # -------------------------------------------------------------- helpers
    def _validate_config(self, df: pd.DataFrame) -> None:
        feature_source = self.config.feature_source if self.config.create_features else None
        requested = (("target", self.config.target), ("feature_source", feature_source))
        missing = [
            f"{field_name}={value!r}"
            for field_name, value in requested
            if value is not None and value not in df.columns
        ]

        if missing:
            raise InvalidConfigError(
                "Configured column(s) not found in the dataset: " + ", ".join(missing)
            )
