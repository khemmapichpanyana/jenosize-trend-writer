"""Jobs API + workers on Modal: scrape and fine-tune over HTTP.

    make deploy-modal            # deploys this with serve/train/eval (modal/deploy.py)
    -> https://<workspace>--jenosize-trend-writer-jobs-api.modal.run/docs

Why Modal rather than the Vercel API: a crawl is minutes and training is tens
of minutes, while Vercel functions stop at 300 s. Here each job gets its own
container with no such limit, and the HTTP layer only starts jobs and reports on
them. The FastAPI app itself lives in `pipeline/api.py` and is tested
in-process; this file only binds it to Modal.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

import modal

from common import APP_NAME, MINUTES, app, app_image, models_volume, pipeline_secret, r2_secret

MODELS_DIR = "/models"

# The worker blocks while a training call runs (~15-30 min, up to train's own
# 180-min timeout), so it must outlive that. Its CPU container costs a small
# fraction of the GPU it waits on.
WORKER_TIMEOUT = 200 * MINUTES


async def _run_remote(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run a Modal function and wait for it; those functions don't touch Postgres."""
    fn = modal.Function.from_name(APP_NAME, {"train": "train", "eval": "evaluate"}[kind])
    return await fn.remote.aio(**params)


@app.function(image=app_image, secrets=[r2_secret, pipeline_secret], timeout=WORKER_TIMEOUT)
async def job_worker(job_id: str, kind: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Executes one job and records its outcome in `job_runs` itself."""
    from app.core.config import Settings
    from app.storage.r2 import R2Storage
    from pipeline.jobs import run_job

    settings = Settings()
    return await run_job(
        UUID(job_id),
        kind,
        params,
        settings=settings,
        storage=R2Storage(settings),
        remote_runner=_run_remote,
    )


# One agent turn: minutes at most, but a cold GPU for the writer can add 1-3
# min, and a long article plus a page design adds more. 30 min is a hard ceiling
# on a runaway turn, not an expected duration.
AGENT_TIMEOUT = 30 * MINUTES


@app.function(image=app_image, secrets=[r2_secret, pipeline_secret], timeout=AGENT_TIMEOUT)
async def agent_worker(run_id: str) -> str:
    """Executes one queued chat turn; events go to Postgres as they happen.

    CPU only: the agent's models are remote (Modal vLLM, or the fallback LLM),
    so this container just orchestrates and streams.
    """
    from app.core.config import Settings
    from app.storage.r2 import R2Storage
    from studio.api import default_writer, studio_store
    from studio.llm import build_agent_models
    from studio.runner import execute_agent_run

    settings = Settings()
    async with studio_store(settings) as store:
        return await execute_agent_run(
            UUID(run_id),
            store=store,
            settings=settings,
            storage=R2Storage(settings),
            models_factory=build_agent_models,
            writer_factory=default_writer,
        )


class ModalDispatcher:
    """`pipeline.api.Dispatcher` backed by Modal function calls."""

    async def spawn(self, kind: str, job_id: UUID, params: dict[str, Any]) -> str:
        call = await job_worker.spawn.aio(str(job_id), kind, params)
        return call.object_id

    async def poll(
        self, call_id: str
    ) -> tuple[Literal["running", "finished", "error"], str | None]:
        from builtins import TimeoutError as BuiltinTimeoutError

        from modal.exception import OutputExpiredError
        from modal.exception import TimeoutError as ModalTimeoutError

        call = modal.FunctionCall.from_id(call_id)
        try:
            await call.get.aio(timeout=0)
        # Order matters: OutputExpiredError *subclasses* Modal's TimeoutError
        # (checked in the SDK), and it means "finished long ago", not "running".
        except OutputExpiredError:
            return "finished", "result expired"
        # FunctionCall.get.aio(timeout=0) raises Python's built-in
        # TimeoutError in some Modal SDK versions, while others raise Modal's
        # wrapper. Both mean that the worker is still running here.
        except (ModalTimeoutError, BuiltinTimeoutError):
            return "running", None
        except Exception as exc:  # the worker raised or was killed
            return "error", f"{type(exc).__name__}: {exc}"
        return "finished", None

    async def cancel(self, call_id: str) -> None:
        # Cancels the worker; a GPU call it started is cancelled with it
        # because the awaiting `.remote.aio()` is interrupted.
        await modal.FunctionCall.from_id(call_id).cancel.aio()

    async def list_adapters(self) -> tuple[list[dict[str, Any]], str | None]:
        from pipeline.adapters import read_active, scan_adapters

        # The volume is mounted at container start; reload to see adapters a
        # training job committed since then.
        await models_volume.reload.aio()
        return scan_adapters(MODELS_DIR), read_active(MODELS_DIR)

    async def spawn_agent(self, run_id: UUID) -> str:
        call = await agent_worker.spawn.aio(str(run_id))
        return call.object_id

    async def function_stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for name, fn in (
            ("train", modal.Function.from_name(APP_NAME, "train")),
            ("evaluate", modal.Function.from_name(APP_NAME, "evaluate")),
            ("job_worker", job_worker),
            ("agent_worker", agent_worker),
        ):
            stats = await fn.get_current_stats.aio()
            out[name] = {
                "backlog": stats.backlog,
                "num_total_runners": stats.num_total_runners,
                "num_running_inputs": stats.num_running_inputs,
            }
        return out

    async def activate_adapter(self, version: str) -> None:
        from pipeline.adapters import write_active

        await models_volume.reload.aio()
        write_active(MODELS_DIR, version)
        await models_volume.commit.aio()


@app.function(
    image=app_image,
    secrets=[r2_secret, pipeline_secret],
    volumes={MODELS_DIR: models_volume},  # list/activate adapters
    # One warm CPU container (cheap, unlike the GPU vLLM server, which still
    # scales to zero) so the console never pays a jobs-API cold start after idle.
    min_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="jenosize-trend-writer-jobs-api")
def jobs_api():  # type: ignore[no-untyped-def]
    from pipeline.api import create_jobs_app

    return create_jobs_app(ModalDispatcher())
