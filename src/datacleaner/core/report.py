"""Serialisable description of what a pipeline run did."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class DatasetStats(BaseModel):
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int

    @classmethod
    def of(cls, df: pd.DataFrame) -> DatasetStats:
        return cls(
            rows=len(df),
            columns=len(df.columns),
            missing_values=int(df.isna().sum().sum()),
            duplicate_rows=int(df.duplicated().sum()),
        )


class OutlierSummary(BaseModel):
    count: int
    lower: float
    upper: float


class StepReport(BaseModel):
    name: str
    duration_ms: float
    rows_after: int
    columns_after: int


class CleaningReport(BaseModel):
    before: DatasetStats
    after: DatasetStats
    ml_rows: int
    ml_columns: int

    missing_fixed: int = 0
    duplicates_removed: int = 0
    converted_columns: list[str] = Field(default_factory=list)
    standardized_columns: list[str] = Field(default_factory=list)
    outliers: dict[str, OutlierSummary] = Field(default_factory=dict)
    created_features: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    encoded_columns: list[str] = Field(default_factory=list)
    scaled_columns: list[str] = Field(default_factory=list)
    dropped_from_ml: list[str] = Field(default_factory=list)

    steps: list[StepReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_duration_ms: float = 0.0
