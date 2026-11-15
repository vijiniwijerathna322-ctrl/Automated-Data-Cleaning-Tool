"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from datacleaner import __version__
from datacleaner.api.errors import register_error_handlers
from datacleaner.api.routes import clean, health, profile
from datacleaner.api.settings import Settings, get_settings
from datacleaner.logging import configure_logging

logger = logging.getLogger("datacleaner.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.json_logs)

    app = FastAPI(
        title="Automated Data Cleaning API",
        version=__version__,
        description="Upload a CSV, Excel or Parquet file and get back a cleaned dataset, "
        "an ML-ready dataset and a detailed cleaning report.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response

    # Routes read settings via the `get_settings` dependency; make it return *this* app's.
    app.dependency_overrides[get_settings] = lambda: settings

    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(profile.router)
    app.include_router(clean.router)

    return app


app = create_app()
