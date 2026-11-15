"""Validated pipeline configuration (loadable from YAML, JSON or an API body)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from datacleaner.core.errors import InvalidConfigError


class PipelineConfig(BaseModel):
    """Everything that controls a pipeline run. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    # --- step toggles ------------------------------------------------------
    coerce_numeric: bool = Field(True, description="Convert text columns that are really numbers.")
    standardize_categories: bool = Field(True, description="Trim and normalise category text.")
    remove_duplicates: bool = Field(True, description="Drop exact duplicate rows.")
    detect_outliers: bool = Field(True, description="Report IQR outliers (never removed).")
    handle_missing: bool = Field(True, description="Median / 'Unknown' imputation.")
    create_features: bool = Field(True, description="Bin `feature_source` into categories.")
    apply_encoding: bool = Field(True, description="One-hot encode text in the ML dataset.")
    apply_scaling: bool = Field(True, description="Standard-scale numerics in the ML dataset.")

    # --- column selection --------------------------------------------------
    target: str | None = Field(
        None,
        description=(
            "Target column: kept raw in the ML dataset, and features derived from it "
            "are dropped to prevent leakage."
        ),
    )
    feature_source: str | None = Field(
        None, description="Numeric column to bin into a categorical feature."
    )

    # --- tunables ----------------------------------------------------------
    max_category_unique: int = Field(
        50, ge=1, description="Text columns above this many distinct values may be free text."
    )
    max_category_unique_ratio: float = Field(
        0.5, gt=0, le=1, description="...and are skipped when distinct/non-null exceeds this."
    )
    outlier_iqr_multiplier: float = Field(1.5, gt=0, description="IQR fence width.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise InvalidConfigError(_format_validation_error(exc)) from exc

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        """Load a ``.yaml`` / ``.yml`` / ``.json`` config file."""
        file = Path(path)

        if not file.is_file():
            raise InvalidConfigError(f"Config file not found: {file}")

        text = file.read_text(encoding="utf-8")
        suffix = file.suffix.lower()

        try:
            if suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(text) or {}
            elif suffix == ".json":
                data = json.loads(text)
            else:
                raise InvalidConfigError(f"Unsupported config format: {suffix!r}")
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise InvalidConfigError(f"Could not parse {file.name}: {exc}") from exc

        if not isinstance(data, dict):
            raise InvalidConfigError("Config file must contain a mapping at the top level.")

        return cls.from_dict(data)


def _format_validation_error(exc: ValidationError) -> str:
    parts = []

    for error in exc.errors():
        location = ".".join(str(p) for p in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")

    return "; ".join(parts)
