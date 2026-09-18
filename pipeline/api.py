"""Jobs API — scrape and fine-tune over HTTP, separate from the article API.

Deployed on Modal (`modal/jobs.py`), not Vercel, because the work it starts is
long: a full crawl is minutes, training is tens of minutes, and Vercel functions
stop at 300 s. Every write endpoint *starts* a job and returns `202` with a run
id straight away; the job runs in a Modal worker and records its progress in
Postgres, which is what `GET /v1/runs/{id}` reads.

    POST /v1/scrape         pull new articles (discover -> crawl -> clean)
    POST /v1/label          reverse-label new/changed articles
    POST /v1/datasets       publish an immutable dataset version (seconds; synchronous)
    GET  /v1/datasets       published versions
    POST /v1/train          QLoRA fine-tune on a dataset version (Modal L4)
    POST /v1/eval           base vs fine-tuned on the version's held-out briefs
    GET  /v1/runs[/{id}]    job status, results and per-stage stats
    POST /v1/runs/{id}/cancel
    GET  /v1/corpus         corpus progress per stage

The dispatcher that actually launches work is injected, so the whole API is
testable in-process without Modal.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Literal, Protocol
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.errors import AppError, NotFoundError, UnauthorizedError, register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.storage.base import Storage
from pipeline import build_dataset
from pipeline._cli import recorded_run
from pipeline.clean import CLEAN_VERSION
from pipeline.jobs import (
    PARAMS,
    VERSION_PATTERN,
    EvalParams,
    GpuRunner,
    JobKind,
    LabelParams,
    ScrapeParams,
    TrainParams,
    run_job,
)
from pipeline.label import LABEL_VERSION
from pipeline.store import ActiveJobExists, CorpusStore, JobStore, connect_jobs

logger = get_logger(__name__)

PollState = Literal["running", "finished", "error"]


class Dispatcher(Protocol):
    """Launches job workers and reports on them. Modal in production."""

    async def spawn(self, kind: str, job_id: UUID, params: dict[str, Any]) -> str: ...

    async def poll(self, call_id: str) -> tuple[PollState, str | None]: ...

    async def cancel(self, call_id: str) -> None: ...


class InlineDispatcher:
    """Runs jobs as asyncio tasks in this process — for tests and local dev.

    Not for production: a Vercel or uvicorn restart would kill the task
    mid-crawl. On Modal every job gets its own container instead.
    """

    def __init__(self, settings: Settings, storage: Storage, gpu_runner: GpuRunner | None = None):
        self._settings = settings
        self._storage = storage
        self._gpu_runner = gpu_runner
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def spawn(self, kind: str, job_id: UUID, params: dict[str, Any]) -> str:
        call_id = f"inline-{uuid4().hex[:12]}"
        self._tasks[call_id] = asyncio.create_task(
            run_job(
                job_id,
                kind,
                params,
                settings=self._settings,
                storage=self._storage,
                gpu_runner=self._gpu_runner,
            )
        )
        return call_id

    async def poll(self, call_id: str) -> tuple[PollState, str | None]:
        task = self._tasks.get(call_id)
        if task is None or task.done():
            if task is not None and not task.cancelled() and task.exception():
                return "error", str(task.exception())
            return "finished", None
        return "running", None

    async def cancel(self, call_id: str) -> None:
        if task := self._tasks.get(call_id):
            task.cancel()

    async def drain(self) -> None:
        """Wait for every spawned job (tests)."""
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)


# ------------------------------------------------------------------ schemas


class StageRun(BaseModel):
    stage: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JobRun(BaseModel):
    id: UUID
    kind: JobKind
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    params: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    stages: list[StageRun] | None = None


class DatasetRequest(BaseModel):
    version: str = Field("v1", pattern=VERSION_PATTERN)
    eval_frac: float = Field(0.1, ge=0.0, lt=0.5)
    seed: int = 13
    overwrite: bool = Field(False, description="Replace an existing version (only if untrained).")


class DatasetVersion(BaseModel):
    version: str
    created_at: datetime
    fingerprint: str
    train_count: int
    eval_count: int
    train_key: str
    eval_key: str
    card_key: str
    params: dict[str, Any]


class CorpusStatus(BaseModel):
    counts: dict[str, int]
    recent_stages: list[StageRun]


# ------------------------------------------------------------------ deps
# Module level on purpose: with `from __future__ import annotations`, FastAPI
# resolves parameter annotations from module globals, so an alias defined inside
# `create_jobs_app` would be unresolvable and silently become a body field.


async def _stores(request: Request) -> AsyncIterator[tuple[CorpusStore, JobStore]]:
    settings: Settings = request.app.state.settings
    if not settings.database_url:
        raise AppError("DATABASE_URL is not configured", code="not_configured", status_code=503)
    async with connect_jobs(settings.database_url) as pair:
        yield pair


Stores = Annotated[tuple[CorpusStore, JobStore], Depends(_stores)]


# ------------------------------------------------------------------ app


def create_jobs_app(dispatcher: Dispatcher, settings: Settings | None = None) -> FastAPI:
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
    register_exception_handlers(app)

    # ------------------------------------------------------------ deps

    async def require_key(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        """Fail closed: these endpoints spend GPU time and labelling budget."""
        expected = settings.jobs_api_key
        if not expected:
            raise AppError(
                "JOBS_API_KEY is not configured on the server",
                code="not_configured",
                status_code=503,
            )
        if not x_api_key or not secrets.compare_digest(x_api_key, expected):
            raise UnauthorizedError("Missing or invalid X-API-Key header")

    guarded = [Depends(require_key)]

    # ------------------------------------------------------------ helpers

    async def reconcile(jobs: JobStore, row: dict[str, Any]) -> dict[str, Any]:
        """Catch workers that died without recording an outcome.

        Workers write their own terminal status. If the underlying call has
        ended but the row is still queued/running, the worker crashed (OOM,
        timeout, preemption) — mark it failed so it stops blocking new jobs.
        """
        if row["status"] not in ("queued", "running") or not row.get("modal_call_id"):
            return row
        state, detail = await dispatcher.poll(row["modal_call_id"])
        if state == "running":
            return row
        fresh = await jobs.get(row["id"])
        if fresh and fresh["status"] in ("queued", "running"):
            await jobs.finish(
                row["id"],
                status="failed",
                error=f"worker exited without recording a result: {detail or 'unknown'}",
            )
            fresh = await jobs.get(row["id"])
        return fresh or row

    async def start(kind: str, params: BaseModel, jobs: JobStore) -> JSONResponse:
        payload = params.model_dump(mode="json")
        try:
            row = await jobs.create(kind, payload)
        except ActiveJobExists as exc:
            # The active one may be a crashed worker; reconcile and retry once.
            if exc.existing:
                await reconcile(jobs, exc.existing)
            try:
                row = await jobs.create(kind, payload)
            except ActiveJobExists as again:
                existing = again.existing or {}
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": {
                            "code": "job_already_active",
                            "message": f"a {kind} job is already {existing.get('status', 'active')}",
                            "run_id": str(existing.get("id")),
                        }
                    },
                )
        call_id = await dispatcher.spawn(kind, row["id"], payload)
        await jobs.set_call_id(row["id"], call_id)
        logger.info("job_started", extra={"job_id": str(row["id"]), "kind": kind})
        return JSONResponse(status_code=202, content=_job(row).model_dump(mode="json"))

    # ------------------------------------------------------------ routes

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "configured": {
                "jobs_api_key": bool(settings.jobs_api_key),
                "database_url": bool(settings.database_url),
                "r2": bool(settings.r2_access_key_id),
                "labeler": bool(settings.labeler_base_url and settings.labeler_model),
            },
        }

    @app.get("/v1/corpus", response_model=CorpusStatus, dependencies=guarded, tags=["data"])
    async def corpus_status(pair: Stores) -> CorpusStatus:
        corpus, _ = pair
        counts = await corpus.counts(clean_version=CLEAN_VERSION, label_version=LABEL_VERSION)
        recent = [StageRun.model_validate(r) for r in await corpus.recent_runs(10)]
        return CorpusStatus(counts=counts, recent_stages=recent)

    @app.post(
        "/v1/scrape", status_code=202, response_model=JobRun, dependencies=guarded, tags=["data"]
    )
    async def start_scrape(pair: Stores, params: ScrapeParams | None = None) -> JSONResponse:
        """Pull new articles into Postgres + R2. Only new/changed content is written."""
        return await start("scrape", params or ScrapeParams(), pair[1])

    @app.post(
        "/v1/label", status_code=202, response_model=JobRun, dependencies=guarded, tags=["data"]
    )
    async def start_label(pair: Stores, params: LabelParams | None = None) -> JSONResponse:
        """Reverse-label new/changed articles with LABELER_MODEL."""
        if not settings.labeler_base_url or not (
            settings.labeler_model or (params and params.model)
        ):
            raise AppError(
                "LABELER_BASE_URL / LABELER_MODEL are not configured",
                code="not_configured",
                status_code=503,
            )
        return await start("label", params or LabelParams(), pair[1])

    @app.get(
        "/v1/datasets", response_model=list[DatasetVersion], dependencies=guarded, tags=["data"]
    )
    async def list_datasets(pair: Stores) -> list[DatasetVersion]:
        corpus, _ = pair
        return [DatasetVersion.model_validate(r) for r in await corpus.dataset_versions()]

    @app.post("/v1/datasets", dependencies=guarded, tags=["data"])
    async def publish_dataset(pair: Stores, request: DatasetRequest) -> JSONResponse:
        """Validate and publish a dataset version to R2.

        Synchronous (seconds). `201` published · `200` unchanged ·
        `409` version exists with different contents · `422` invalid dataset.
        """
        from app.storage.r2 import R2Storage

        corpus, _ = pair
        try:
            async with recorded_run(corpus, "build") as stats:
                stats.update(
                    await build_dataset.run_build(
                        corpus,
                        R2Storage(settings),
                        version=request.version,
                        eval_frac=request.eval_frac,
                        seed=request.seed,
                        overwrite=request.overwrite,
                    )
                )
        except build_dataset.DatasetError as exc:
            message = str(exc)
            immutable = "immutable" in message
            return JSONResponse(
                status_code=409 if immutable else 422,
                content={
                    "error": {
                        "code": "dataset_exists" if immutable else "dataset_invalid",
                        "message": message,
                    }
                },
            )
        return JSONResponse(
            status_code=200 if stats["status"] == "unchanged" else 201, content=stats
        )

    @app.post(
        "/v1/train",
        status_code=202,
        response_model=JobRun,
        dependencies=guarded,
        tags=["fine-tuning"],
    )
    async def start_train(pair: Stores, params: TrainParams) -> JSONResponse:
        """QLoRA fine-tune on a published dataset version (Modal L4, ~15-30 min)."""
        corpus, jobs = pair
        if await corpus.dataset_version(params.version) is None:
            raise NotFoundError(
                f"dataset {params.version} is not published; POST /v1/datasets first"
            )
        return await start("train", params, jobs)

    @app.post(
        "/v1/eval",
        status_code=202,
        response_model=JobRun,
        dependencies=guarded,
        tags=["fine-tuning"],
    )
    async def start_eval(pair: Stores, params: EvalParams) -> JSONResponse:
        """Compare base vs fine-tuned on the version's held-out briefs."""
        corpus, jobs = pair
        if await corpus.dataset_version(params.version) is None:
            raise NotFoundError(f"dataset {params.version} is not published")
        return await start("eval", params, jobs)

    @app.get("/v1/runs", response_model=list[JobRun], dependencies=guarded, tags=["runs"])
    async def list_runs(
        pair: Stores,
        kind: JobKind | None = None,
        limit: int = Query(20, ge=1, le=100),
    ) -> list[JobRun]:
        _, jobs = pair
        rows = [await reconcile(jobs, r) for r in await jobs.list(limit=limit, kind=kind)]
        return [_job(r) for r in rows]

    @app.get("/v1/runs/{run_id}", response_model=JobRun, dependencies=guarded, tags=["runs"])
    async def get_run(run_id: UUID, pair: Stores) -> JobRun:
        corpus, jobs = pair
        row = await jobs.get(run_id)
        if row is None:
            raise NotFoundError(f"no run {run_id}")
        row = await reconcile(jobs, row)
        job = _job(row)
        job.stages = [StageRun.model_validate(s) for s in await corpus.stage_runs_for_job(run_id)]
        return job

    @app.post(
        "/v1/runs/{run_id}/cancel", response_model=JobRun, dependencies=guarded, tags=["runs"]
    )
    async def cancel_run(run_id: UUID, pair: Stores) -> JobRun:
        _, jobs = pair
        row = await jobs.get(run_id)
        if row is None:
            raise NotFoundError(f"no run {run_id}")
        if row["status"] in ("queued", "running"):
            if row.get("modal_call_id"):
                await dispatcher.cancel(row["modal_call_id"])
            await jobs.finish(run_id, status="cancelled", error="cancelled via API")
            row = await jobs.get(run_id) or row
        return _job(row)

    return app


def _job(row: dict[str, Any]) -> JobRun:
    return JobRun.model_validate({k: row.get(k) for k in JobRun.model_fields if k != "stages"})


__all__ = ["PARAMS", "Dispatcher", "InlineDispatcher", "create_jobs_app"]


def create_local_app() -> FastAPI:
    """`make jobs-dev`: the jobs API on your machine, jobs running in-process.

    Scrape and label work fully (they write to your real Postgres + R2); train
    and eval need Modal GPUs and fail with a clear message. For development
    only — use the Modal deployment for real runs.
    """
    from app.storage.r2 import R2Storage

    settings = get_settings()
    return create_jobs_app(InlineDispatcher(settings, R2Storage(settings)), settings)
