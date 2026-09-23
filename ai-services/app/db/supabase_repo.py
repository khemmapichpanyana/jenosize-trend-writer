"""Supabase (Postgres) repository.

Security posture: RLS is enabled on every table with **no** policies, so the
anon key can read nothing. This service authenticates with the service/secret
key, which bypasses RLS — that key must never reach a browser. There is no user
auth in the product, so there is no per-user row filtering to implement.

supabase-py is synchronous (it wraps httpx sync + postgrest), so all calls go
through `anyio.to_thread`.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any
from uuid import UUID

import anyio

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Columns returned by the list endpoint; selecting `*` would drag the full
# markdown body of every article over the wire.
_SUMMARY_COLUMNS = "id,status,title,created_at,latency_ms"


class SupabaseRepository:
    name = "supabase"

    def __init__(self, settings: Settings) -> None:
        if not settings.supabase_url or not settings.supabase_secret_key:
            raise ValueError("PERSISTENCE=supabase requires SUPABASE_URL and SUPABASE_SECRET_KEY")
        self._url = settings.supabase_url
        self._key = settings.supabase_secret_key

    @cached_property
    def _client(self) -> Any:
        from supabase import create_client

        return create_client(self._url, self._key)

    async def _run(self, fn: Any) -> Any:
        try:
            return await anyio.to_thread.run_sync(fn)
        except Exception as exc:
            raise UpstreamError(f"Supabase call failed: {exc}") from exc

    async def create_generation(
        self, generation_id: UUID, request: dict[str, Any], normalized: dict[str, Any]
    ) -> None:
        row = {
            "id": str(generation_id),
            "status": "queued",
            "request": request,
            "normalized_params": normalized,
        }
        await self._run(lambda: self._client.table("generations").insert(row).execute())

    async def update_generation(self, generation_id: UUID, **fields: Any) -> None:
        if not fields:
            return
        await self._run(
            lambda: (
                self._client.table("generations")
                .update(fields)
                .eq("id", str(generation_id))
                .execute()
            )
        )

    async def get_generation(self, generation_id: UUID) -> dict[str, Any] | None:
        result = await self._run(
            lambda: (
                self._client.table("generations")
                .select("*")
                .eq("id", str(generation_id))
                .limit(1)
                .execute()
            )
        )
        data = result.data or []
        return data[0] if data else None

    async def list_generations(self, limit: int = 20) -> list[dict[str, Any]]:
        result = await self._run(
            lambda: (
                self._client.table("generations")
                .select(_SUMMARY_COLUMNS)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        )
        return list(result.data or [])

    async def upsert_source_document(self, document: dict[str, Any]) -> dict[str, Any]:
        # content_hash is UNIQUE: re-uploading the same file returns the existing
        # row instead of duplicating storage and retrieval candidates.
        result = await self._run(
            lambda: (
                self._client.table("source_documents")
                .upsert(document, on_conflict="content_hash")
                .execute()
            )
        )
        data = result.data or []
        return data[0] if data else document

    async def get_source_documents(self, source_ids: list[UUID]) -> list[dict[str, Any]]:
        if not source_ids:
            return []
        ids = [str(s) for s in source_ids]
        result = await self._run(
            lambda: self._client.table("source_documents").select("*").in_("id", ids).execute()
        )
        return list(result.data or [])

    async def record_generation_sources(
        self, generation_id: UUID, rows: list[dict[str, Any]]
    ) -> None:
        if not rows:
            return
        payload = [{"generation_id": str(generation_id), **row} for row in rows]
        await self._run(lambda: self._client.table("generation_sources").insert(payload).execute())

    async def health(self) -> None:
        # Cheapest possible round-trip that still proves auth + network + RLS bypass.
        await self._run(lambda: self._client.table("generations").select("id").limit(1).execute())
