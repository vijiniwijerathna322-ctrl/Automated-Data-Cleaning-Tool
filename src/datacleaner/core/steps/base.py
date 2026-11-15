"""Step contract and the shared state steps read from and write to."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from datacleaner.core.config import PipelineConfig
from datacleaner.core.report import OutlierSummary


@dataclass
class PipelineContext:
    """Mutable run state shared by steps; becomes the :class:`CleaningReport`."""

    config: PipelineConfig
    id_columns: list[str] = field(default_factory=list)
    converted_columns: list[str] = field(default_factory=list)
    standardized_columns: list[str] = field(default_factory=list)
    duplicates_removed: int = 0
    outliers: dict[str, OutlierSummary] = field(default_factory=dict)
    outlier_rows: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing_fixed: int = 0
    created_features: list[str] = field(default_factory=list)
    derived_from: dict[str, str] = field(default_factory=dict)
    encoded_columns: list[str] = field(default_factory=list)
    scaled_columns: list[str] = field(default_factory=list)
    dropped_from_ml: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Step(ABC):
    """One cleaning stage. Must return a new DataFrame, never mutate its input."""

    #: Stable identifier used in logs and the report.
    name: ClassVar[str]
    #: Name of the boolean :class:`PipelineConfig` field that enables this step.
    toggle: ClassVar[str]

    def is_enabled(self, config: PipelineConfig) -> bool:
        return bool(getattr(config, self.toggle))

    @abstractmethod
    def apply(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        """Transform ``df``; record what happened on ``ctx``."""
