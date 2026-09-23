"""Custom events from tools to the chat stream (LangGraph `custom` stream mode)."""

from __future__ import annotations

from typing import Any


def emit(kind: str, **data: Any) -> None:
    """Send an event to the live chat stream; a no-op outside an agent run."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # not inside a graph run (e.g. a unit test calling a tool)
        return
    writer({"type": kind, **data})
