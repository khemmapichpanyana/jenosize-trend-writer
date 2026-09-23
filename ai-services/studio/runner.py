"""Executes one agent turn in a background worker and records its event log.

This is what runs inside the Modal `agent_worker` container (and in-process in
tests / `make jobs-dev`). The HTTP layer only queues the turn; this module does
the work and writes every event to `agent_events`, which the console streams.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

import anyio

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.generation import GenerationService
from app.storage.base import Storage
from studio.agent import build_agent, run_turn
from studio.llm import AgentModels
from studio.store import StudioStore
from studio.tools import ToolContext, build_tools

logger = get_logger(__name__)

# Events that arrive many times a second and can be merged without losing
# meaning: their text concatenates. Everything else is written as it happens.
COALESCE = {"token", "artifact_delta"}
FLUSH_INTERVAL_S = 0.25
MAX_BUFFER = 200


class EventSink:
    """Buffers run events and writes them in small batches.

    Consecutive `token` / `artifact_delta` events for the same target are merged
    into one row, and the buffer is flushed every ~250 ms or on any structural
    event (tool start/end, artifact saved, error). That keeps a streamed
    1,000-word article to a few dozen inserts while the console still sees it
    arrive in near real time.
    """

    def __init__(self, store: StudioStore, run_id: UUID) -> None:
        self._store = store
        self._run_id = run_id
        self._buffer: list[tuple[str, dict[str, Any]]] = []
        self._last_flush = time.monotonic()

    async def add(self, kind: str, data: dict[str, Any]) -> None:
        if kind in COALESCE and self._buffer:
            prev_kind, prev = self._buffer[-1]
            if prev_kind == kind and prev.get("artifact_id") == data.get("artifact_id"):
                prev["text"] = prev.get("text", "") + data.get("text", "")
            else:
                self._buffer.append((kind, dict(data)))
        else:
            self._buffer.append((kind, dict(data)))
        due = time.monotonic() - self._last_flush >= FLUSH_INTERVAL_S
        if kind not in COALESCE or due or len(self._buffer) >= MAX_BUFFER:
            await self.flush()

    async def flush(self) -> None:
        if self._buffer:
            batch, self._buffer = self._buffer, []
            await self._store.append_events(self._run_id, batch)
        self._last_flush = time.monotonic()


async def execute_agent_run(
    run_id: UUID,
    *,
    store: StudioStore,
    settings: Settings,
    storage: Storage,
    models_factory: Callable[[Settings], AgentModels],
    writer_factory: Callable[[Settings, Storage], GenerationService],
) -> str:
    """Run a queued turn to completion. Returns the final run status."""
    run = await store.get_run(run_id)
    if run is None:
        raise LookupError(f"no agent run {run_id}")
    if not await store.mark_run_running(run_id):
        return "cancelled"  # cancelled while queued: nothing to do

    sink = EventSink(store, run_id)
    status, error = "succeeded", None
    try:
        messages = await store.list_messages(run["thread_id"])
        user = next(m for m in messages if m["id"] == run["message_id"])
        models = models_factory(settings)
        designer = (
            models.primary.with_fallbacks(models.fallbacks) if models.fallbacks else models.primary
        )
        tools = build_tools(
            ToolContext(
                thread_id=run["thread_id"],
                store=store,
                generation=writer_factory(settings, storage),
                designer=designer,  # type: ignore[arg-type]
                settings=settings,
                storage=storage,
            )
        )
        agent = build_agent(models.primary, models.fallbacks, tools)
        async for event in run_turn(
            agent=agent,
            store=store,
            thread_id=run["thread_id"],
            user_text=user["content"],
            asset_ids=list(user.get("asset_ids") or []),
            model_names=models.names,
            save_user_message=False,
        ):
            kind = event.pop("type")
            if kind == "done":
                continue  # written once below, after the run's final status
            if kind == "error":
                status, error = "failed", str(event.get("message"))
            await sink.add(kind, event)
    except asyncio.CancelledError:
        status, error = "cancelled", "cancelled"
        raise
    except Exception as exc:
        logger.exception("agent_run_failed", extra={"run_id": str(run_id)})
        status, error = "failed", f"{type(exc).__name__}: {exc}"[:500]
        await sink.add("error", {"message": error})
    finally:
        # Shielded: the event log and final status must be written even when
        # the worker is being cancelled.
        with anyio.CancelScope(shield=True):
            await sink.add("done", {"status": status})
            await sink.flush()
            await store.finish_run(run_id, status, error)
    return status
