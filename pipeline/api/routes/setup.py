"""Setup and diagnostics: everything needed after the first deploy, over HTTP."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import anyio
from fastapi import APIRouter

from app.core.errors import AppError
from pipeline import migrate
from pipeline.api.deps import GUARDED, DispatcherDep, SettingsDep, StorageDep, Stores
from pipeline.api.schemas import DoctorCheck, Migration

router = APIRouter(prefix="/v1", tags=["setup"], dependencies=GUARDED)


def _applied(dsn: str) -> set[str]:
    import psycopg

    with psycopg.connect(dsn, prepare_threshold=None, connect_timeout=10) as conn:
        exists = conn.execute("select to_regclass('schema_migrations')").fetchone()
        if not exists or not exists[0]:
            return set()
        return {row[0] for row in conn.execute("select version from schema_migrations")}


@router.get("/config")
async def config(settings: SettingsDep, dispatcher: DispatcherDep) -> dict[str, Any]:
    """What the server resolved from its environment — hosts and names, never secrets."""
    _, active = await dispatcher.list_adapters()
    return {
        "database_host": urlparse(settings.database_url or "").hostname,
        "r2_bucket": settings.r2_bucket,
        "r2_endpoint_host": urlparse(settings.r2_endpoint_url or "").hostname,
        "labeler_model": settings.labeler_model,
        "models_dir": settings.models_dir,
        "active_adapter": active,
        "configured": {
            "database": bool(settings.database_url),
            "r2": bool(settings.r2_access_key_id and settings.r2_secret_access_key),
            "labeler": bool(settings.labeler_base_url and settings.labeler_model),
        },
    }


@router.get("/migrations", response_model=list[Migration])
async def list_migrations(settings: SettingsDep) -> list[Migration]:
    if not settings.database_url:
        raise AppError(
            "Database is not configured (DB_URL)", code="not_configured", status_code=503
        )
    applied = await anyio.to_thread.run_sync(_applied, settings.database_url)
    return [Migration(name=p.name, applied=p.name in applied) for p in migrate.migration_files()]


@router.post("/migrations/apply")
async def apply_migrations(settings: SettingsDep) -> dict[str, Any]:
    """Apply pending migrations. Idempotent: every statement is `if not exists`."""
    if not settings.database_url:
        raise AppError(
            "Database is not configured (DB_URL)", code="not_configured", status_code=503
        )
    applied = await anyio.to_thread.run_sync(migrate.apply_migrations, settings.database_url)
    return {"applied": applied, "status": "applied" if applied else "up to date"}


@router.get("/doctor", response_model=list[DoctorCheck])
async def doctor(
    settings: SettingsDep, storage: StorageDep, dispatcher: DispatcherDep
) -> list[DoctorCheck]:
    """Live checks of every dependency. Always 200; read `ok` per check."""
    checks: list[DoctorCheck] = []

    try:
        applied = await anyio.to_thread.run_sync(_applied, settings.database_url or "")
        pending = [p.name for p in migrate.migration_files() if p.name not in applied]
        checks.append(
            DoctorCheck(
                check="database",
                ok=not pending,
                detail=f"pending migrations: {', '.join(pending)} — POST /v1/migrations/apply"
                if pending
                else f"{len(applied)} migrations applied",
            )
        )
    except Exception as exc:
        checks.append(
            DoctorCheck(check="database", ok=False, detail=f"{type(exc).__name__}: {exc}"[:200])
        )

    try:
        await storage.health()
        checks.append(
            DoctorCheck(check="r2", ok=True, detail=f"bucket {settings.r2_bucket} reachable")
        )
    except Exception as exc:
        checks.append(
            DoctorCheck(check="r2", ok=False, detail=f"{type(exc).__name__}: {exc}"[:200])
        )

    labeler = bool(settings.labeler_base_url and settings.labeler_model)
    checks.append(
        DoctorCheck(
            check="labeler",
            ok=labeler,
            detail=f"model {settings.labeler_model}"
            if labeler
            else "LABELER_BASE_URL / LABELER_MODEL not set",
        )
    )

    try:
        found, active = await dispatcher.list_adapters()
        checks.append(
            DoctorCheck(
                check="adapters",
                ok=True,
                detail=f"{len(found)} trained, active: {active}" if found else "none trained yet",
            )
        )
    except Exception as exc:
        checks.append(
            DoctorCheck(check="adapters", ok=False, detail=f"{type(exc).__name__}: {exc}"[:200])
        )
    return checks


@router.get("/resources")
async def resources(pair: Stores, dispatcher: DispatcherDep) -> dict[str, Any]:
    """Live compute: Modal containers per function, and every active job with
    its latest GPU telemetry (utilisation, memory, loss)."""
    try:
        functions = await dispatcher.function_stats()
    except Exception as exc:  # stats are advisory; never fail the page
        functions = {"error": {"message": f"{type(exc).__name__}: {exc}"[:200]}}  # type: ignore[dict-item]
    return {"functions": functions, "active_jobs": await pair[1].active_with_latest_progress()}
