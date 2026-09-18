"""Shared plumbing for the pipeline CLIs.

The pipeline writes only to Postgres (`DATABASE_URL`) and Cloudflare R2. There
is deliberately no local fallback: a corpus split across laptops cannot be
reproduced, and the training job on Modal reads the dataset from R2 anyway.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import anyio
import typer

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.storage.r2 import R2Storage
from pipeline.store import CorpusStore, connect


def run_async[T](fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    return anyio.run(lambda: fn(*args, **kwargs))


def settings_or_exit() -> Settings:
    """Fail fast, and say exactly which variables are missing."""
    settings = get_settings()
    configure_logging(settings.log_level)
    missing = [
        name
        for name, value in {
            "DATABASE_URL": settings.database_url,
            "R2_ACCOUNT_ID (or R2_ENDPOINT)": settings.r2_account_id or settings.r2_endpoint,
            "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
            "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
        }.items()
        if not value
    ]
    if missing:
        typer.echo("The pipeline writes to Postgres + R2. Missing in .env: " + ", ".join(missing))
        raise typer.Exit(code=1)
    return settings


@asynccontextmanager
async def pipeline_context() -> AsyncIterator[tuple[Settings, CorpusStore, R2Storage]]:
    settings = settings_or_exit()
    assert settings.database_url is not None
    async with connect(settings.database_url) as store:
        yield settings, store, R2Storage(settings)


@asynccontextmanager
async def recorded_run(
    store: CorpusStore, stage: str, *, job_id: UUID | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Log a stage invocation to `pipeline_runs`, succeeded or failed.

    The stage fills the yielded dict with its counts; it is stored as jsonb.
    `job_id` links the stage to the jobs-API job that started it, if any.
    """
    run_id = await store.start_run(stage, job_id=job_id)
    stats: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        yield stats
    except BaseException as exc:
        stats["seconds"] = round(time.perf_counter() - started, 1)
        with anyio.CancelScope(shield=True):
            await store.finish_run(run_id, stats=stats, error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    stats["seconds"] = round(time.perf_counter() - started, 1)
    await store.finish_run(run_id, stats=stats)


def echo_stats(stage: str, stats: dict[str, Any]) -> None:
    typer.echo(f"{stage}: " + "  ".join(f"{k}={v}" for k, v in stats.items()))
