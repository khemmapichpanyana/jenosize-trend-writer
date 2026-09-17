"""Persistence seam.

The repository speaks in plain dicts rather than ORM objects: the rows are
mostly `jsonb` and the only consumer is this service, so an ORM would add a
dependency and a migration story without buying anything.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class Repository(Protocol):
    name: str

    async def create_generation(
        self, generation_id: UUID, request: dict[str, Any], normalized: dict[str, Any]
    ) -> None:
        """Insert a `queued` row so a crash mid-generation is still observable."""
        ...

    async def update_generation(self, generation_id: UUID, **fields: Any) -> None: ...

    async def get_generation(self, generation_id: UUID) -> dict[str, Any] | None: ...

    async def list_generations(self, limit: int = 20) -> list[dict[str, Any]]: ...

    async def upsert_source_document(self, document: dict[str, Any]) -> dict[str, Any]: ...

    async def get_source_documents(self, source_ids: list[UUID]) -> list[dict[str, Any]]: ...

    async def record_generation_sources(
        self, generation_id: UUID, rows: list[dict[str, Any]]
    ) -> None: ...

    async def health(self) -> None:
        """Raise if the database is unreachable."""
        ...
