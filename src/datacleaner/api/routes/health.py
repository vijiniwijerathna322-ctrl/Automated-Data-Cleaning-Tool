from fastapi import APIRouter

from datacleaner import __version__
from datacleaner.api.schemas import HealthResponse
from datacleaner.core.config import PipelineConfig

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["meta"], summary="Liveness probe")
def healthz() -> HealthResponse:
    return HealthResponse(version=__version__)


@router.get(
    "/v1/config/default",
    response_model=PipelineConfig,
    tags=["meta"],
    summary="Default pipeline configuration",
)
def default_config() -> PipelineConfig:
    return PipelineConfig()
