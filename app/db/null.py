"""In-memory repository used when PERSISTENCE=none.

Not a no-op: it keeps rows in a process-local dict so `GET /articles/{id}` works
during a local demo. State is lost on restart, which is the documented trade-off
of running without Supabase.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID


class NullRepository:
    name = "none"

    def __init__(self) -> None:
        self._generations: dict[UUID, dict[str, Any]] = {}
        self._sources: dict[UUID, dict[str, Any]] = {}

    async def create_generation(
        self, generation_id: UUID, request: dict[str, Any], normalized: dict[str, Any]
    ) -> None:
        self._generations[generation_id] = {
            "id": str(generation_id),
            "status": "queued",
            "request": request,
            "normalized_params": normalized,
            "created_at": datetime.now(UTC).isoformat(),
        }

    async def update_generation(self, generation_id: UUID, **fields: Any) -> None:
        self._generations.setdefault(generation_id, {"id": str(generation_id)}).update(fields)

    async def get_generation(self, generation_id: UUID) -> dict[str, Any] | None:
        return self._generations.get(generation_id)

    async def list_generations(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = sorted(
            self._generations.values(),
            key=lambda r: str(r.get("created_at") or ""),
            reverse=True,
        )
        return rows[:limit]

    async def upsert_source_document(self, document: dict[str, Any]) -> dict[str, Any]:
        self._sources[UUID(str(document["id"]))] = document
        return document

    async def get_source_documents(self, source_ids: list[UUID]) -> list[dict[str, Any]]:
        return [self._sources[sid] for sid in source_ids if sid in self._sources]

    async def record_generation_sources(
        self, generation_id: UUID, rows: list[dict[str, Any]]
    ) -> None:
        return None

    async def health(self) -> None:
        return None
