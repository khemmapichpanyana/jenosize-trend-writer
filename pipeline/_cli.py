"""Shared plumbing for the pipeline CLIs.

Typer commands are sync; every stage is async because it talks to HTTP, SQLite
and object storage. `run_async` is the single bridge, so no stage has to think
about event loops.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import anyio

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.storage import Storage, build_storage
from pipeline.store import CorpusStore, build_store


def run_async[T](fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    return anyio.run(lambda: fn(*args, **kwargs))


def context() -> tuple[Settings, CorpusStore, Storage]:
    """Settings plus the two backends every stage needs."""
    settings = get_settings()
    configure_logging(settings.log_level)
    return settings, build_store(settings), build_storage(settings)


def echo_counts(counts: dict[str, int]) -> None:
    import typer

    typer.echo("corpus: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
