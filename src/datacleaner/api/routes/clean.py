import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Response

from datacleaner.api.deps import ConfigDep, DataFrameDep
from datacleaner.api.schemas import CleanResponse, TablePreview
from datacleaner.core.pipeline import Pipeline
from datacleaner.io import FileFormat, to_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["clean"])


@router.post("/clean", response_model=CleanResponse, summary="Clean a dataset")
def clean(
    df: DataFrameDep,
    config: ConfigDep,
    preview_rows: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> CleanResponse:
    result = Pipeline(config).run(df)

    logger.info(
        "dataset cleaned",
        extra={
            "rows_in": result.report.before.rows,
            "rows_out": result.report.after.rows,
            "duration_ms": result.report.total_duration_ms,
        },
    )

    return CleanResponse(
        report=result.report,
        cleaned=TablePreview.from_dataframe(result.cleaned, preview_rows),
        ml=TablePreview.from_dataframe(result.ml, preview_rows),
        outlier_rows={
            column: TablePreview.from_dataframe(rows, preview_rows)
            for column, rows in result.outlier_rows.items()
        },
    )


@router.post(
    "/clean/download",
    summary="Clean a dataset and download the result as a file",
    response_class=Response,
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def clean_download(
    df: DataFrameDep,
    config: ConfigDep,
    dataset: Annotated[Literal["cleaned", "ml"], Query()] = "cleaned",
    format: Annotated[FileFormat, Query()] = FileFormat.CSV,
) -> Response:
    result = Pipeline(config).run(df)
    frame = result.cleaned if dataset == "cleaned" else result.ml
    filename = f"{dataset}_dataset.{format.value}"

    return Response(
        content=to_bytes(frame, format),
        media_type=format.media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
