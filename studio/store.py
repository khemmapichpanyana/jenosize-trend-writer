"""Postgres access for the studio: threads, messages, artifacts, assets, pages."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb


class StudioStore:
    def __init__(self, conn: psycopg.AsyncConnection[dict[str, Any]]) -> None:
        self._conn = conn

    async def _one(self, sql: str, params: Any = None) -> dict[str, Any] | None:
        async with self._conn.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()

    async def _all(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        async with self._conn.cursor() as cur:
            await cur.execute(sql, params)
            return list(await cur.fetchall())

    # ---------------------------------------------------------------- threads

    async def create_thread(self, title: str = "New chat") -> dict[str, Any]:
        row = await self._one("insert into chat_threads (title) values (%s) returning *", (title,))
        assert row is not None
        return row

    async def list_threads(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._all(
            """
            select t.*, (select count(*) from artifacts a where a.thread_id = t.id) as artifact_count
            from chat_threads t order by updated_at desc limit %s
            """,
            (limit,),
        )

    async def get_thread(self, thread_id: UUID) -> dict[str, Any] | None:
        return await self._one("select * from chat_threads where id = %s", (thread_id,))

    async def touch_thread(self, thread_id: UUID, title: str | None = None) -> None:
        await self._conn.execute(
            "update chat_threads set updated_at = now(), title = coalesce(%s, title) where id = %s",
            (title, thread_id),
        )

    async def delete_thread(self, thread_id: UUID) -> None:
        await self._conn.execute("delete from chat_threads where id = %s", (thread_id,))

    # --------------------------------------------------------------- messages

    async def add_message(
        self,
        thread_id: UUID,
        role: str,
        content: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        asset_ids: list[UUID] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        row = await self._one(
            """
            insert into chat_messages (thread_id, role, content, tool_calls, asset_ids, model)
            values (%s, %s, %s, %s, %s, %s) returning *
            """,
            (thread_id, role, content, Jsonb(tool_calls or []), list(asset_ids or []), model),
        )
        assert row is not None
        return row

    async def list_messages(self, thread_id: UUID) -> list[dict[str, Any]]:
        return await self._all(
            "select * from chat_messages where thread_id = %s order by created_at, id", (thread_id,)
        )

    # -------------------------------------------------------------- artifacts

    async def create_artifact(self, thread_id: UUID, title: str) -> dict[str, Any]:
        row = await self._one(
            "insert into artifacts (thread_id, title) values (%s, %s) returning *",
            (thread_id, title),
        )
        assert row is not None
        return row

    async def add_version(
        self,
        artifact_id: UUID,
        *,
        markdown: str | None,
        html: str | None,
        meta: dict[str, Any],
        note: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Append an immutable version. The version number is taken inside one
        UPDATE, so two concurrent edits can never claim the same number."""
        async with self._conn.transaction():
            head = await self._one(
                """
                update artifacts set current_version = current_version + 1, updated_at = now(),
                       title = coalesce(%s, title)
                where id = %s returning current_version
                """,
                (title, artifact_id),
            )
            if head is None:
                raise LookupError(f"no artifact {artifact_id}")
            row = await self._one(
                """
                insert into artifact_versions (artifact_id, version, markdown, html, meta, note)
                values (%s, %s, %s, %s, %s, %s) returning *
                """,
                (artifact_id, head["current_version"], markdown, html, Jsonb(meta), note),
            )
        assert row is not None
        return row

    async def get_artifact(self, artifact_id: UUID) -> dict[str, Any] | None:
        return await self._one("select * from artifacts where id = %s", (artifact_id,))

    async def list_artifacts(self, thread_id: UUID) -> list[dict[str, Any]]:
        return await self._all(
            "select * from artifacts where thread_id = %s order by created_at", (thread_id,)
        )

    async def get_version(
        self, artifact_id: UUID, version: int | None = None
    ) -> dict[str, Any] | None:
        if version is None:
            return await self._one(
                """
                select v.* from artifact_versions v join artifacts a on a.id = v.artifact_id
                where v.artifact_id = %s and v.version = a.current_version
                """,
                (artifact_id,),
            )
        return await self._one(
            "select * from artifact_versions where artifact_id = %s and version = %s",
            (artifact_id, version),
        )

    async def list_versions(self, artifact_id: UUID) -> list[dict[str, Any]]:
        return await self._all(
            """
            select version, created_at, note, meta->>'title' as title,
                   html is not null as has_html
            from artifact_versions where artifact_id = %s order by version
            """,
            (artifact_id,),
        )

    # ----------------------------------------------------------------- assets

    async def add_asset(self, **fields: Any) -> dict[str, Any]:
        columns = ", ".join(fields)
        row = await self._one(
            f"insert into assets ({columns}) values ({', '.join(f'%({k})s' for k in fields)}) returning *",
            fields,
        )
        assert row is not None
        return row

    async def get_asset(self, asset_id: UUID) -> dict[str, Any] | None:
        return await self._one("select * from assets where id = %s", (asset_id,))

    async def list_assets(self, thread_id: UUID) -> list[dict[str, Any]]:
        return await self._all(
            "select * from assets where thread_id = %s order by created_at", (thread_id,)
        )

    async def is_asset_public(self, asset_id: UUID) -> bool:
        """Only images referenced by a currently published page are public."""
        row = await self._one(
            "select 1 as ok from published_content where status = 'published' and %s = any(asset_ids) limit 1",
            (asset_id,),
        )
        return row is not None

    # ------------------------------------------------------------- published

    async def publish(
        self,
        *,
        slug: str,
        artifact_id: UUID,
        version: int,
        title: str,
        r2_key: str,
        asset_ids: list[UUID],
    ) -> dict[str, Any]:
        """Publish or re-publish at a slug. A slug stays bound to one artifact."""
        existing = await self.get_published(slug)
        if existing and existing["artifact_id"] != artifact_id:
            raise ValueError(f"slug {slug!r} is already used by another article")
        row = await self._one(
            """
            insert into published_content (slug, artifact_id, version, title, r2_key, asset_ids)
            values (%(slug)s, %(artifact_id)s, %(version)s, %(title)s, %(r2_key)s, %(asset_ids)s)
            on conflict (slug) do update set
                version = excluded.version, title = excluded.title, r2_key = excluded.r2_key,
                asset_ids = excluded.asset_ids, status = 'published', updated_at = now()
            returning *
            """,
            {
                "slug": slug,
                "artifact_id": artifact_id,
                "version": version,
                "title": title,
                "r2_key": r2_key,
                "asset_ids": list(asset_ids),
            },
        )
        assert row is not None
        return row

    async def get_published(self, slug: str) -> dict[str, Any] | None:
        return await self._one("select * from published_content where slug = %s", (slug,))

    async def list_published(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self._all(
            "select * from published_content order by updated_at desc limit %s", (limit,)
        )

    async def published_for_artifact(self, artifact_id: UUID) -> list[dict[str, Any]]:
        return await self._all(
            "select * from published_content where artifact_id = %s order by updated_at desc",
            (artifact_id,),
        )

    async def set_published_status(self, slug: str, status: str) -> dict[str, Any] | None:
        return await self._one(
            "update published_content set status = %s, updated_at = now() where slug = %s returning *",
            (status, slug),
        )
