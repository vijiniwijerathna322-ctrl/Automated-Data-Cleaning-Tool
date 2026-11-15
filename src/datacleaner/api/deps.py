"""Shared FastAPI dependencies."""

from __future__ import annotations

import json
from typing import Annotated

import pandas as pd
from fastapi import Depends, File, Form, UploadFile

from datacleaner.api.errors import PayloadTooLargeError
from datacleaner.api.settings import Settings, get_settings
from datacleaner.core.config import PipelineConfig
from datacleaner.core.errors import InvalidConfigError, InvalidInputError
from datacleaner.io import read_bytes

SettingsDep = Annotated[Settings, Depends(get_settings)]


def parse_config(
    config: Annotated[
        str | None,
        Form(description="PipelineConfig as a JSON object. Omit for defaults."),
    ] = None,
) -> PipelineConfig:
    if not config or not config.strip():
        return PipelineConfig()

    try:
        data = json.loads(config)
    except json.JSONDecodeError as exc:
        raise InvalidConfigError(f"`config` is not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise InvalidConfigError("`config` must be a JSON object.")

    return PipelineConfig.from_dict(data)


ConfigDep = Annotated[PipelineConfig, Depends(parse_config)]


def load_upload(
    file: Annotated[UploadFile, File(description="CSV, XLSX or Parquet file.")],
    settings: SettingsDep,
) -> pd.DataFrame:
    """Read the upload with size and row limits enforced."""
    limit = settings.max_upload_bytes
    content = file.file.read(limit + 1)

    if len(content) > limit:
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_mb} MB upload limit.")

    if not file.filename:
        raise InvalidInputError("Uploaded file has no filename.")

    return read_bytes(content, file.filename, max_rows=settings.max_rows)


DataFrameDep = Annotated[pd.DataFrame, Depends(load_upload)]
