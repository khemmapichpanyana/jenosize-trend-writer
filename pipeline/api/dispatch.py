"""How the jobs API launches work, and reads/writes the model volume.

Injected into the app so the whole API runs in-process in tests. Production
uses `modal/jobs.py::ModalDispatcher`; tests and `make jobs-dev` use
`InlineDispatcher`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from app.core.config import Settings
from app.storage.base import Storage
from pipeline import adapters
from pipeline.jobs import RemoteRunner, run_job

PollState = Literal["running", "finished", "error"]


class Dispatcher(Protocol):
    async def spawn(self, kind: str, job_id: UUID, params: dict[str, Any]) -> str: ...

    async def poll(self, call_id: str) -> tuple[PollState, str | None]: ...

    async def cancel(self, call_id: str) -> None: ...

    async def list_adapters(self) -> tuple[list[dict[str, Any]], str | None]:
        """(complete adapters, active version) from the model volume."""
        ...

    async def activate_adapter(self, version: str) -> None: ...

    async def function_stats(self) -> dict[str, dict[str, int]]:
        """Live container stats per Modal function (backlog, runners, running inputs)."""
        ...


class InlineDispatcher:
    """Runs jobs as asyncio tasks in this process — for tests and local dev.

    Not for production: a restart kills a task mid-crawl. On Modal every job
    gets its own container instead.
    """

    def __init__(
        self,
        settings: Settings,
        storage: Storage | None,
        remote_runner: RemoteRunner | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._remote_runner = remote_runner
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def spawn(self, kind: str, job_id: UUID, params: dict[str, Any]) -> str:
        assert self._storage is not None, "InlineDispatcher needs storage to run jobs"
        call_id = f"inline-{uuid4().hex[:12]}"
        self._tasks[call_id] = asyncio.create_task(
            run_job(
                job_id,
                kind,
                params,
                settings=self._settings,
                storage=self._storage,
                remote_runner=self._remote_runner,
            )
        )
        return call_id

    async def poll(self, call_id: str) -> tuple[PollState, str | None]:
        task = self._tasks.get(call_id)
        if task is None or task.done():
            if task is not None and not task.cancelled() and task.exception():
                return "error", str(task.exception())
            return "finished", None
        return "running", None

    async def cancel(self, call_id: str) -> None:
        if task := self._tasks.get(call_id):
            task.cancel()

    async def list_adapters(self) -> tuple[list[dict[str, Any]], str | None]:
        root = self._settings.models_dir
        return adapters.scan_adapters(root), adapters.read_active(root)

    async def activate_adapter(self, version: str) -> None:
        adapters.write_active(self._settings.models_dir, version)

    async def function_stats(self) -> dict[str, dict[str, int]]:
        running = sum(1 for t in self._tasks.values() if not t.done())
        return {
            "inline_worker": {"backlog": 0, "num_total_runners": 1, "num_running_inputs": running}
        }

    async def drain(self) -> None:
        """Wait for every spawned job (tests)."""
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
