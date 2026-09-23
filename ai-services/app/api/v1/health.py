"""Liveness + dependency probes."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.deps import ConfigDep, ProviderDep, RepositoryDep, StorageDep
from app.core.logging import get_logger
from app.schemas.common import CheckStatus, HealthResponse

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: ConfigDep,
    repository: RepositoryDep,
    storage: StorageDep,
    provider: ProviderDep,
) -> HealthResponse:
    """Report per-subsystem status.

    Always returns 200: an uptime monitor should read `status`, and a 5xx here
    would make Vercel's own health checks flap while, say, R2 has a bad minute.
    The model is reported as `skipped` rather than probed — waking a
    scaled-to-zero GPU on every health poll would cost more than it proves.
    """
    checks: dict[str, CheckStatus] = {}

    for name, probe in (("database", repository.health), ("storage", storage.health)):
        try:
            await probe()
            checks[name] = "ok"
        except Exception:
            logger.exception("health_check_failed", extra={"check": name})
            checks[name] = "error"

    checks["model"] = "skipped" if settings.model_provider == "mock" else "ok"

    degraded = any(v == "error" for v in checks.values())
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=settings.app_version,
        checks=checks,
    )
