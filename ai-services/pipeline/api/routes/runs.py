"""Job runs: status, history, cancellation."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.core.errors import NotFoundError
from pipeline.api.deps import GUARDED, DispatcherDep, SettingsDep, Stores
from pipeline.api.runs import job_view, reconcile
from pipeline.api.schemas import JobRun, ProgressPoint, StageRun
from pipeline.jobs import JobKind
from pipeline.store import connect_jobs

TERMINAL = {"succeeded", "failed", "cancelled"}
POLL_S = 2.0
HEARTBEAT_S = 15.0
# Same reason as the article API: stop proxies buffering the event stream.
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}

router = APIRouter(prefix="/v1/runs", tags=["runs"], dependencies=GUARDED)


@router.get("", response_model=list[JobRun])
async def list_runs(
    pair: Stores,
    dispatcher: DispatcherDep,
    kind: JobKind | None = None,
    limit: int = Query(20, ge=1, le=100),
) -> list[JobRun]:
    _, jobs = pair
    rows = [await reconcile(dispatcher, jobs, r) for r in await jobs.list(limit=limit, kind=kind)]
    return [job_view(r) for r in rows]


@router.get("/{run_id}", response_model=JobRun)
async def get_run(run_id: UUID, pair: Stores, dispatcher: DispatcherDep) -> JobRun:
    corpus, jobs = pair
    row = await jobs.get(run_id)
    if row is None:
        raise NotFoundError(f"no run {run_id}")
    job = job_view(await reconcile(dispatcher, jobs, row))
    job.stages = [StageRun.model_validate(s) for s in await corpus.stage_runs_for_job(run_id)]
    return job


@router.post("/{run_id}/cancel", response_model=JobRun)
async def cancel_run(run_id: UUID, pair: Stores, dispatcher: DispatcherDep) -> JobRun:
    _, jobs = pair
    row = await jobs.get(run_id)
    if row is None:
        raise NotFoundError(f"no run {run_id}")
    if row["status"] in ("queued", "running"):
        if row.get("modal_call_id"):
            await dispatcher.cancel(row["modal_call_id"])
        await jobs.finish(run_id, status="cancelled", error="cancelled via API")
        row = await jobs.get(run_id) or row
    return job_view(row)


@router.get("/{run_id}/progress", response_model=list[ProgressPoint])
async def run_progress(
    run_id: UUID, pair: Stores, after_id: int = Query(0, ge=0)
) -> list[ProgressPoint]:
    """Training telemetry rows after `after_id` (for polling clients)."""
    return [
        ProgressPoint.model_validate(r) for r in await pair[1].progress(run_id, after_id=after_id)
    ]


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/{run_id}/events")
async def run_events(
    run_id: UUID, settings: SettingsDep, dispatcher: DispatcherDep, after_id: int = Query(0, ge=0)
) -> StreamingResponse:
    """Live stream for one run: `status`, `progress` (training steps), `heartbeat`, `done`.

    Polls Postgres every 2 s rather than holding a LISTEN connection, which keeps
    it working through Supabase's pooler. The stream ends once the run is
    terminal and every progress row has been sent; clients reconnect with the
    last seen `after_id` to resume without gaps.
    """
    async with connect_jobs(settings.database_url or "") as (_, jobs):
        if await jobs.get(run_id) is None:
            raise NotFoundError(f"no run {run_id}")

    async def stream() -> AsyncIterator[str]:
        last_id, last_status, last_beat = after_id, None, time.monotonic()
        # Own connection: a FastAPI dependency's cleanup may run before a
        # streaming body finishes, which would close the connection mid-stream.
        async with connect_jobs(settings.database_url or "") as (_, jobs):
            while True:
                row = await jobs.get(run_id)
                if row is None:
                    yield _sse("error", {"message": "run disappeared"})
                    return
                row = await reconcile(dispatcher, jobs, row)
                if row["status"] != last_status:
                    last_status = row["status"]
                    yield _sse("status", job_view(row).model_dump(mode="json"))
                points = await jobs.progress(run_id, after_id=last_id)
                for point in points:
                    last_id = point["id"]
                    yield _sse(
                        "progress", ProgressPoint.model_validate(point).model_dump(mode="json")
                    )
                if last_status in TERMINAL and not points:
                    yield _sse("done", {"status": last_status, "after_id": last_id})
                    return
                if time.monotonic() - last_beat > HEARTBEAT_S:
                    last_beat = time.monotonic()
                    yield _sse("heartbeat", {"after_id": last_id})
                await asyncio.sleep(POLL_S)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)
