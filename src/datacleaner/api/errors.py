"""Map domain errors to RFC 7807-style JSON problem responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from datacleaner.api.schemas import Problem
from datacleaner.core.errors import DataCleanerError, InvalidConfigError, InvalidInputError

logger = logging.getLogger(__name__)

PROBLEM_JSON = "application/problem+json"


class PayloadTooLargeError(InvalidInputError):
    """Upload exceeds the configured size limit (HTTP 413)."""


def _problem(request: Request, status: int, title: str, detail: str) -> JSONResponse:
    body = Problem(
        title=title,
        status=status,
        detail=detail,
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(body.model_dump(), status_code=status, media_type=PROBLEM_JSON)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(PayloadTooLargeError)
    async def _too_large(request: Request, exc: PayloadTooLargeError) -> JSONResponse:
        return _problem(request, 413, "Payload Too Large", str(exc))

    @app.exception_handler(InvalidInputError)
    async def _bad_input(request: Request, exc: InvalidInputError) -> JSONResponse:
        return _problem(request, 400, "Invalid Input", str(exc))

    @app.exception_handler(InvalidConfigError)
    async def _bad_config(request: Request, exc: InvalidConfigError) -> JSONResponse:
        return _problem(request, 422, "Invalid Configuration", str(exc))

    @app.exception_handler(DataCleanerError)
    async def _domain(request: Request, exc: DataCleanerError) -> JSONResponse:
        return _problem(request, 400, "Processing Error", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return _problem(request, 422, "Validation Error", detail)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _problem(request, exc.status_code, "HTTP Error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", extra={"path": request.url.path})
        return _problem(request, 500, "Internal Server Error", "An unexpected error occurred.")
