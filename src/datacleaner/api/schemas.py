"""HTTP request/response models."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from datacleaner.core.profile import DatasetProfile
from datacleaner.core.report import CleaningReport


class TablePreview(BaseModel):
    """A JSON-safe head of a DataFrame (NaN becomes ``null``)."""

    columns: list[str]
    rows: list[list[Any]]
    total_rows: int
    truncated: bool

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, limit: int) -> TablePreview:
        head = df.head(limit)
        split = json.loads(head.to_json(orient="split", date_format="iso"))
        return cls(
            columns=[str(c) for c in split["columns"]],
            rows=split["data"],
            total_rows=len(df),
            truncated=len(df) > limit,
        )


class CleanResponse(BaseModel):
    report: CleaningReport
    cleaned: TablePreview
    ml: TablePreview
    outlier_rows: dict[str, TablePreview] = Field(default_factory=dict)


class ProfileResponse(BaseModel):
    profile: DatasetProfile
    preview: TablePreview


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class Problem(BaseModel):
    """RFC 7807-style error body."""

    title: str
    status: int
    detail: str
    request_id: str | None = None
