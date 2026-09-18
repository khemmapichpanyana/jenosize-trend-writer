"""Long-running pipeline work, started through the jobs API.

A *job* is what a caller asks for ("pull new articles", "train on v1"). It can
span several pipeline stages; each stage still logs its own `pipeline_runs` row,
linked to the job. The job's lifecycle lives in `job_runs`:

    queued -> running -> succeeded | failed        (or cancelled)

The executors here are plain async functions with their dependencies passed in,
so the same code runs inside a Modal worker in production and in-process in
tests and local development. Work that needs Modal resources (train and eval
on GPUs, publish from the model volume) goes through `RemoteRunner`, which in
production calls the Modal functions and in tests is a fake.
"""

from __future__ import annotations

import traceback
from collections.abc import Awaitable, Callable
from typing import Any, Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, HttpUrl

from app.core.config import Settings
from app.core.logging import get_logger
from app.storage.base import Storage
from app.storage.keys import dataset_key
from pipeline import clean, label, scrape
from pipeline._cli import recorded_run
from pipeline.store import CorpusStore, connect_jobs

logger = get_logger(__name__)

JobKind = Literal["scrape", "label", "train", "eval", "publish"]
RemoteRunner = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

# Dataset/adapter versions look like v1, v2, … — keeps R2 keys and Modal volume
# paths predictable and stops a typo from creating a stray version.
VERSION_PATTERN = r"^v[0-9]+$"


class ScrapeParams(BaseModel):
    """Pull new articles: discover -> crawl -> clean."""

    limit: int = Field(1000, ge=1, le=5000, description="Maximum pages to fetch.")
    recheck_days: float | None = Field(
        None, ge=0, description="Also re-fetch pages last checked more than N days ago."
    )
    # Floor, not just a default: the API must not be usable to hammer the site.
    delay_s: float = Field(1.0, ge=0.5, le=10, description="Seconds between requests.")


class LabelParams(BaseModel):
    """Reverse-label new or changed articles."""

    limit: int = Field(1000, ge=1, le=5000)
    model: str | None = Field(None, description="Overrides LABELER_MODEL.")


class TrainParams(BaseModel):
    """QLoRA fine-tune on a published dataset version."""

    version: str = Field(pattern=VERSION_PATTERN, description="Dataset version, e.g. v1.")
    epochs: int = Field(3, ge=1, le=10)
    lora_r: Literal[8, 16, 32, 64] = 16
    learning_rate: float = Field(2e-4, gt=0, le=1e-3)
    push_to_hub: bool = False


class EvalParams(BaseModel):
    """Base vs fine-tuned on a version's held-out briefs."""

    version: str = Field(pattern=VERSION_PATTERN)
    endpoint: HttpUrl = Field(description="The vLLM server's OpenAI base URL, ending in /v1.")
    judge: bool = Field(
        default=False,
        description="Blind pairwise judge; uses the labelling LLM unless judge_model is set.",
    )
    judge_model: str | None = None
    judge_endpoint: HttpUrl | None = None
    limit: int = Field(20, ge=1, le=200)


class PublishParams(BaseModel):
    """Upload a trained adapter to the Hugging Face Hub."""

    version: str = Field(pattern=VERSION_PATTERN)
    repo_id: str = Field(
        pattern=r"^[A-Za-z0-9][\w.-]*/[\w.-]+$", description="e.g. your-user/jeno-trend-writer-lora"
    )
    private: bool = False


PARAMS: dict[str, type[BaseModel]] = {
    "scrape": ScrapeParams,
    "label": LabelParams,
    "train": TrainParams,
    "eval": EvalParams,
    "publish": PublishParams,
}


def adapter_dir(version: str, models_dir: str = "/models") -> str:
    """One adapter directory per dataset version, so v2 never overwrites v1."""
    return f"{models_dir}/jeno-lora-{version}"


async def execute(
    kind: str,
    params: dict[str, Any],
    *,
    job_id: UUID | None,
    corpus: CorpusStore,
    storage: Storage,
    settings: Settings,
    remote_runner: RemoteRunner | None,
) -> dict[str, Any]:
    """Run one job to completion and return its result summary."""
    if kind == "scrape":
        p = ScrapeParams.model_validate(params)
        result: dict[str, Any] = {}
        async with scrape.http_client() as client:
            async with recorded_run(corpus, "discover", job_id=job_id) as stats:
                stats.update(await scrape.run_discover(corpus, client))
            result["discover"] = dict(stats)
            async with recorded_run(corpus, "crawl", job_id=job_id) as stats:
                stats.update(
                    await scrape.run_crawl(
                        corpus,
                        storage,
                        client,
                        limit=p.limit,
                        delay_s=p.delay_s,
                        recheck_days=p.recheck_days,
                    )
                )
            result["crawl"] = dict(stats)
        # Cleaning is cheap and deterministic, so "pull data" includes it: the
        # caller gets training-ready markdown, not just archived HTML.
        async with recorded_run(corpus, "clean", job_id=job_id) as stats:
            stats.update(await clean.run_clean(corpus, storage))
        result["clean"] = dict(stats)
        return result

    if kind == "label":
        p_label = LabelParams.model_validate(params)
        model = p_label.model or settings.labeler_model
        if not settings.labeler_base_url or not model:
            raise RuntimeError("LABELER_BASE_URL and LABELER_MODEL must be configured")
        labeler = label.labeler_client(settings.labeler_base_url, settings.labeler_api_key)
        async with recorded_run(corpus, "label", job_id=job_id) as stats:
            result_stats, _ = await label.run_label(corpus, labeler, model, limit=p_label.limit)
            stats.update(result_stats, model=model)
        return dict(stats)

    if kind in ("train", "eval", "publish"):
        if remote_runner is None:
            raise RuntimeError(f"{kind} jobs run on Modal; no remote runner is configured here")
        if kind == "train":
            p_train = TrainParams.model_validate(params)
            return await remote_runner(
                "train",
                {
                    "dataset_uri": f"r2://{dataset_key(p_train.version, 'train.jsonl')}",
                    "epochs": p_train.epochs,
                    "lora_r": p_train.lora_r,
                    "learning_rate": p_train.learning_rate,
                    "push_to_hub": p_train.push_to_hub,
                    "adapter_dir": adapter_dir(p_train.version),
                },
            )
        if kind == "publish":
            p_pub = PublishParams.model_validate(params)
            return await remote_runner(
                "publish",
                {"version": p_pub.version, "repo_id": p_pub.repo_id, "private": p_pub.private},
            )
        p_eval = EvalParams.model_validate(params)
        output = await remote_runner(
            "eval",
            {
                "endpoint": str(p_eval.endpoint),
                # The server registers every trained version under its own name,
                # so eval compares base against *this* version, not the alias.
                "adapter_name": f"jeno-lora-{p_eval.version}",
                "run_name": f"{p_eval.version}-{str(job_id)[:8]}",
                "briefs_uri": f"r2://{dataset_key(p_eval.version, 'eval.jsonl')}",
                "judge": p_eval.judge,
                "judge_model": p_eval.judge_model,
                "judge_endpoint": str(p_eval.judge_endpoint) if p_eval.judge_endpoint else None,
                "limit": p_eval.limit,
            },
        )
        # Per-example outputs go to R2 (eval.py uploads them); the job row keeps
        # only the summary so `GET /runs` stays small.
        return {"summary": output.get("summary", output)}

    raise ValueError(f"unknown job kind: {kind}")


async def run_job(
    job_id: UUID,
    kind: str,
    params: dict[str, Any],
    *,
    settings: Settings,
    storage: Storage,
    remote_runner: RemoteRunner | None = None,
) -> dict[str, Any] | None:
    """Full job lifecycle: running -> execute -> succeeded/failed, recorded in Postgres.

    Never raises: the outcome lives in `job_runs`, which is what callers poll.
    """
    assert settings.database_url, "DATABASE_URL is required to run jobs"
    async with connect_jobs(settings.database_url) as (corpus, jobs):
        await jobs.mark_running(job_id)
        try:
            result = await execute(
                kind,
                params,
                job_id=job_id,
                corpus=corpus,
                storage=storage,
                settings=settings,
                remote_runner=remote_runner,
            )
        except Exception as exc:
            logger.error(
                "job_failed",
                extra={"job_id": str(job_id), "kind": kind, "trace": traceback.format_exc()},
            )
            message = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, httpx.HTTPError):
                message = f"network error: {exc}"
            await jobs.finish(job_id, status="failed", error=message[:1000])
            return None
        await jobs.finish(job_id, status="succeeded", result=result)
        return result
