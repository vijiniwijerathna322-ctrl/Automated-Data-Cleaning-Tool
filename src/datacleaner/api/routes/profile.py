from typing import Annotated

from fastapi import APIRouter, Query

from datacleaner.api.deps import DataFrameDep
from datacleaner.api.schemas import ProfileResponse, TablePreview
from datacleaner.core.profile import profile_dataframe

router = APIRouter(prefix="/v1", tags=["profile"])


@router.post("/profile", response_model=ProfileResponse, summary="Profile a raw dataset")
def profile(
    df: DataFrameDep,
    preview_rows: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> ProfileResponse:
    return ProfileResponse(
        profile=profile_dataframe(df),
        preview=TablePreview.from_dataframe(df, preview_rows),
    )
