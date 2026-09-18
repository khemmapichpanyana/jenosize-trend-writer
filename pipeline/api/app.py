"""Jobs API — scrape and fine-tune over HTTP, separate from the article API.

Deployed on Modal (`modal/jobs.py`), not Vercel: a full crawl takes minutes and
training tens of minutes, while Vercel functions stop at 300 s. Write endpoints
*start* a job and return `202` with a run id; the job runs in its own Modal
container and records progress in Postgres, which `GET /v1/runs/{id}` reads.

After the one-time bootstrap (`modal setup`, `make modal-secrets`,
`make deploy-modal`) the whole workflow is HTTP:

| Area | Endpoints |
|---|---|
| setup | `GET /v1/config` · `GET /v1/doctor` · `GET /v1/migrations` · `POST /v1/migrations/apply` |
| corpus | `POST /v1/scrape` · `GET /v1/corpus` · `/corpus/stats` · `/corpus/articles` · `/corpus/article?url=` · `/corpus/sample` |
| labels | `POST /v1/label/preview` · `POST /v1/label` |
| datasets | `POST /v1/datasets` · `GET /v1/datasets` · `/datasets/{v}/validate` · `/card` · `/examples` |
| model | `POST /v1/train` · `POST /v1/eval` · `GET /v1/adapters` · `/adapters/{v}/activate` · `/adapters/{v}/publish` |
| runs | `GET /v1/runs` · `GET /v1/runs/{id}` · `POST /v1/runs/{id}/cancel` |

Every `/v1` route needs the `X-API-Key` header.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.storage.base import Storage
from pipeline.api.dispatch import Dispatcher, InlineDispatcher
from pipeline.api.routes import data, models, runs, setup

logger = get_logger(__name__)


def create_jobs_app(
    dispatcher: Dispatcher,
    settings: Settings | None = None,
    storage: Storage | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not settings.jobs_api_key:
            logger.warning("JOBS_API_KEY is not set; every /v1 request will be refused")
        yield

    app = FastAPI(
        title="Jenosize Trend Writer — Jobs API",
        description=__doc__,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.dispatcher = dispatcher
    app.state.storage_factory = _storage_factory(settings, storage)
    register_exception_handlers(app)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        """Unauthenticated liveness, with which settings are present (never values)."""
        return {
            "status": "ok",
            "configured": {
                "jobs_api_key": bool(settings.jobs_api_key),
                "database": bool(settings.database_url),
                "r2": bool(settings.r2_access_key_id),
                "labeler": bool(settings.labeler_base_url and settings.labeler_model),
            },
        }

    for module in (setup, data, models, runs):
        app.include_router(module.router)
    return app


def _storage_factory(settings: Settings, storage: Storage | None) -> Callable[[], Storage]:
    """R2 client built on first use, so a server with a bad R2 config still boots
    and `GET /v1/doctor` can say what is wrong."""
    cache: list[Storage] = [storage] if storage is not None else []

    def get() -> Storage:
        if not cache:
            from app.storage.r2 import R2Storage

            cache.append(R2Storage(settings))
        return cache[0]

    return get


def create_local_app() -> FastAPI:
    """`make jobs-dev`: the jobs API on your machine, jobs running in-process.

    Scrape, label and datasets work against your real Postgres + R2; train,
    eval and publish need Modal and fail with a clear message. Adapters are
    read from MODELS_DIR (default /models, usually absent locally).
    """
    from app.storage.r2 import R2Storage

    settings = get_settings()
    storage = R2Storage(settings)
    return create_jobs_app(InlineDispatcher(settings, storage), settings, storage)
