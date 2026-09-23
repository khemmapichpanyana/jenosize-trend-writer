"""FastAPI application factory.

Vercel's Python runtime looks for a module-level ASGI app called `app` in this
file (see `vercel.json`), so the factory is invoked at import time.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger, request_id_ctx
from app.db import build_repository
from app.services.llm import build_provider
from app.storage import build_storage

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

DESCRIPTION = """
Generates business trend and future-ideas articles in the tone of Jenosize Ideas.

**Design principle:** fine-tuning teaches style, retrieval supplies facts.

Run it with no cloud accounts at all: `MODEL_PROVIDER=mock`, `PERSISTENCE=none`,
`STORAGE=local`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the pluggable backends once per process."""
    settings: Settings = app.state.settings
    app.state.provider = build_provider(settings)
    app.state.repository = build_repository(settings)
    app.state.storage = build_storage(settings)
    logger.info(
        "startup",
        extra={
            "env": settings.app_env,
            "port": settings.port,
            "model_provider": settings.model_provider,
            "persistence": settings.persistence,
            "storage": settings.storage,
        },
    )
    yield
    logger.info("shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Jenosize Trend Writer",
        description=DESCRIPTION,
        version=settings.app_version,
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,  # no cookies/auth: an allowlist of origins is enough
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a request id to every log line and response.

        Clients may supply their own id so a trace can be followed across the
        browser, this service and the Modal logs.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "request_id": request_id,
            },
        )
        return response

    register_exception_handlers(app)
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Service card. The browser demo page is a separate, later deliverable."""
        return {
            "service": "jenosize-trend-writer",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()
