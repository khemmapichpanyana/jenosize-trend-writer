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

from typing import Any
from uuid import UUID

import modal

from common import MINUTES, app, app_image, models_volume, pipeline_secret, r2_secret
from eval import evaluate
from train import publish, train

MODELS_DIR = "/models"

# The worker blocks while a training call runs (~15-30 min, up to train's own
# 180-min timeout), so it must outlive that. Its CPU container costs a small
# fraction of the GPU it waits on.
WORKER_TIMEOUT = 200 * MINUTES


async def _run_remote(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run a Modal function and wait for it; those functions don't touch Postgres."""
    fn = {"train": train, "eval": evaluate, "publish": publish}[kind]
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


class ModalDispatcher:
    """`pipeline.api.Dispatcher` backed by Modal function calls."""

    async def spawn(self, kind: str, job_id: UUID, params: dict[str, Any]) -> str:
        call = await job_worker.spawn.aio(str(job_id), kind, params)
        return call.object_id

    async def poll(self, call_id: str) -> tuple[str, str | None]:
        from modal.exception import OutputExpiredError
        from modal.exception import TimeoutError as ModalTimeoutError

        call = modal.FunctionCall.from_id(call_id)
        try:
            await call.get.aio(timeout=0)
        # Order matters: OutputExpiredError *subclasses* Modal's TimeoutError
        # (checked in the SDK), and it means "finished long ago", not "running".
        except OutputExpiredError:
            return "finished", "result expired"
        except ModalTimeoutError:
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

    async def activate_adapter(self, version: str) -> None:
        from pipeline.adapters import write_active

        await models_volume.reload.aio()
        write_active(MODELS_DIR, version)
        await models_volume.commit.aio()


@app.function(
    image=app_image,
    secrets=[r2_secret, pipeline_secret],
    volumes={MODELS_DIR: models_volume},  # list/activate adapters
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="jenosize-trend-writer-jobs-api")
def jobs_api():  # type: ignore[no-untyped-def]
    from pipeline.api import create_jobs_app

    return create_jobs_app(ModalDispatcher())
