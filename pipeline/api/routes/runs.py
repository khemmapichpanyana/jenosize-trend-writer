"""Job runs: status, history, cancellation."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from app.core.errors import NotFoundError
from pipeline.api.deps import GUARDED, DispatcherDep, Stores
from pipeline.api.runs import job_view, reconcile
from pipeline.api.schemas import JobRun, StageRun
from pipeline.jobs import JobKind

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
