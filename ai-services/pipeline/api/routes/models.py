"""The model side: training, evaluation and adapters."""

from __future__ import annotations

from fastapi import APIRouter, Path
from fastapi.responses import JSONResponse

from app.core.errors import NotFoundError
from pipeline.api.deps import GUARDED, DispatcherDep, Stores
from pipeline.api.runs import start_job
from pipeline.api.schemas import Adapter, JobRun
from pipeline.jobs import VERSION_PATTERN, EvalParams, TrainParams

router = APIRouter(prefix="/v1", dependencies=GUARDED)
Version = Path(pattern=VERSION_PATTERN)


@router.post("/train", status_code=202, response_model=JobRun, tags=["jobs"])
async def start_train(pair: Stores, dispatcher: DispatcherDep, params: TrainParams) -> JSONResponse:
    """QLoRA fine-tune on a published dataset version (Modal L4, ~15-30 min)."""
    if await pair[0].dataset_version(params.version) is None:
        raise NotFoundError(f"dataset {params.version} is not published; POST /v1/datasets first")
    return await start_job(dispatcher, pair[1], "train", params)


@router.post("/eval", status_code=202, response_model=JobRun, tags=["jobs"])
async def start_eval(pair: Stores, dispatcher: DispatcherDep, params: EvalParams) -> JSONResponse:
    """Base vs this version's adapter, on the version's held-out briefs."""
    if await pair[0].dataset_version(params.version) is None:
        raise NotFoundError(f"dataset {params.version} is not published")
    found, _ = await dispatcher.list_adapters()
    if not any(a["version"] == params.version for a in found):
        raise NotFoundError(f"no trained adapter for {params.version}; POST /v1/train first")
    return await start_job(dispatcher, pair[1], "eval", params)


@router.get("/adapters", response_model=list[Adapter], tags=["adapters"])
async def list_adapters(dispatcher: DispatcherDep) -> list[Adapter]:
    """Trained adapters on the model volume, with their training metrics."""
    found, active = await dispatcher.list_adapters()
    return [
        Adapter(
            version=a["version"],
            served_as=a["served_as"],
            active=a["version"] == active,
            metrics=a["metrics"],
        )
        for a in found
    ]


@router.post("/adapters/{version}/activate", response_model=list[Adapter], tags=["adapters"])
async def activate_adapter(dispatcher: DispatcherDep, version: str = Version) -> list[Adapter]:
    """Point the `jeno-lora` alias at this version.

    Takes effect when the vLLM server next starts: it scales to zero after
    5 idle minutes, so at the latest on the first request after that. Every
    version stays callable by name (`jeno-lora-v1`, …) in the meantime.
    """
    found, _ = await dispatcher.list_adapters()
    if not any(a["version"] == version for a in found):
        raise NotFoundError(f"no trained adapter for {version}")
    await dispatcher.activate_adapter(version)
    return await list_adapters(dispatcher)
