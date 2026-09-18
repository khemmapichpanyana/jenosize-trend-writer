"""Starting jobs and keeping their status honest."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.logging import get_logger
from pipeline.api.dispatch import Dispatcher
from pipeline.api.schemas import JobRun
from pipeline.store import ActiveJobExists, JobStore

logger = get_logger(__name__)


def job_view(row: dict[str, Any]) -> JobRun:
    return JobRun.model_validate({k: row.get(k) for k in JobRun.model_fields if k != "stages"})


async def reconcile(dispatcher: Dispatcher, jobs: JobStore, row: dict[str, Any]) -> dict[str, Any]:
    """Catch workers that died without recording an outcome.

    Workers write their own terminal status. If the underlying call has ended
    but the row is still queued/running, the worker crashed (OOM, timeout,
    preemption) — mark it failed so it stops blocking new jobs.
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


async def start_job(
    dispatcher: Dispatcher, jobs: JobStore, kind: str, params: BaseModel
) -> JSONResponse:
    """Create the job row, launch the worker, answer 202 (or 409 if one is active)."""
    payload = params.model_dump(mode="json")
    try:
        row = await jobs.create(kind, payload)
    except ActiveJobExists as exc:
        # The active one may be a crashed worker; reconcile and retry once.
        if exc.existing:
            await reconcile(dispatcher, jobs, exc.existing)
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
    return JSONResponse(status_code=202, content=job_view(row).model_dump(mode="json"))
