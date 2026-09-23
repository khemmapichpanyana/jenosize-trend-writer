"""Model warm-up.

The demo client calls this on page load so the Modal container starts booting
while the user is still typing their brief; by submit time the GPU is usually
warm and the first token arrives in seconds rather than minutes.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.core.deps import ConfigDep, ProviderDep
from app.core.logging import get_logger
from app.schemas.common import WarmupResponse

router = APIRouter(prefix="/model", tags=["model"])
logger = get_logger(__name__)


@router.post("/warmup", response_model=WarmupResponse)
async def warmup(provider: ProviderDep, settings: ConfigDep) -> WarmupResponse:
    if settings.model_provider == "mock":
        return WarmupResponse(status="skipped", latency_ms=0, detail="MODEL_PROVIDER=mock")

    started = time.perf_counter()
    try:
        await provider.warmup()
    except Exception as exc:
        logger.warning("warmup_failed", extra={"error": str(exc)})
        return WarmupResponse(
            status="error",
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail=str(exc)[:300],
        )
    return WarmupResponse(status="ok", latency_ms=int((time.perf_counter() - started) * 1000))
