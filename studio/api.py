"""Studio HTTP routes: chat with the agent, artifacts, image assets, publishing.

Everything under /v1/studio needs the jobs-API key (the console proxies it
server-side). The /p/* routes are the public share links and need nothing —
they only ever serve pages a person explicitly published, and only the images
those pages reference.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import time
import unicodedata
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.errors import AppError, NotFoundError, PayloadTooLargeError, ValidationError
from app.db import build_repository
from app.services.generation import GenerationService
from app.services.llm import build_provider
from app.storage.base import Storage, put_text
from app.storage.keys import safe_filename
from pipeline.api.deps import GUARDED, DispatcherDep, SettingsDep, StorageDep
from studio import html as page
from studio.llm import AgentModels, NoAgentModelError
from studio.store import ACTIVE_RUN, ActiveRunExists, StudioStore

router = APIRouter(prefix="/v1/studio", tags=["studio"], dependencies=GUARDED)
public = APIRouter(tags=["public"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
IMAGE_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}
SLUG_MAX = 80
EVENT_POLL_S = 0.4
HEARTBEAT_S = 15.0


# ------------------------------------------------------------------ plumbing


@asynccontextmanager
async def studio_store(settings: Settings) -> AsyncIterator[StudioStore]:
    if not settings.database_url:
        raise AppError(
            "Database is not configured (DB_URL)", code="not_configured", status_code=503
        )
    conn = await psycopg.AsyncConnection.connect(
        settings.database_url, autocommit=True, row_factory=dict_row, prepare_threshold=None
    )
    try:
        yield StudioStore(conn)
    finally:
        await conn.close()


async def _store(request: Request) -> AsyncIterator[StudioStore]:
    async with studio_store(request.app.state.settings) as store:
        yield store


StoreDep = Annotated[StudioStore, Depends(_store)]


def _agent_models(request: Request) -> AgentModels:
    factory: Callable[[Settings], AgentModels] = request.app.state.agent_models_factory
    try:
        return factory(request.app.state.settings)
    except NoAgentModelError as exc:
        raise AppError(str(exc), code="not_configured", status_code=503) from exc


def _share_base(settings: Settings, request: Request) -> str:
    return (settings.public_share_base_url or str(request.base_url)).rstrip("/")


def slugify(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:SLUG_MAX].strip("-")
    return slug


# ------------------------------------------------------------------ schemas


class NewThread(BaseModel):
    title: str = Field("New chat", max_length=200)


class ChatTurn(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    asset_ids: list[UUID] = Field(default_factory=list, max_length=10)


class PublishRequest(BaseModel):
    version: int | None = Field(None, ge=1, description="Default: the current version.")
    slug: str | None = Field(None, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=SLUG_MAX)


# ------------------------------------------------------------------ threads


@router.post("/threads", status_code=201)
async def create_thread(store: StoreDep, body: NewThread | None = None) -> dict[str, Any]:
    return await store.create_thread((body or NewThread()).title)


@router.get("/threads")
async def list_threads(
    store: StoreDep, limit: int = Query(50, ge=1, le=200)
) -> list[dict[str, Any]]:
    return await store.list_threads(limit)


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: UUID, request: Request, store: StoreDep) -> dict[str, Any]:
    thread = await store.get_thread(thread_id)
    if thread is None:
        raise NotFoundError(f"no chat {thread_id}")
    artifacts = []
    for artifact in await store.list_artifacts(thread_id):
        artifacts.append(
            {
                **artifact,
                "versions": await store.list_versions(artifact["id"]),
                "published": await store.published_for_artifact(artifact["id"]),
            }
        )
    active = await store.active_run(thread_id)
    return {
        **thread,
        "messages": await store.list_messages(thread_id),
        "artifacts": artifacts,
        "assets": [_asset_view(a) for a in await store.list_assets(thread_id)],
        # A turn still running (e.g. the page was reloaded mid-turn): the console
        # re-attaches to its event stream from the start and replays it.
        "active_run": await _reconcile(request, store, active) if active else None,
    }


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: UUID, store: StoreDep) -> Response:
    await store.delete_thread(thread_id)
    return Response(status_code=204)


@router.post("/threads/{thread_id}/runs", status_code=202)
async def start_run(
    thread_id: UUID,
    turn: ChatTurn,
    request: Request,
    store: StoreDep,
    dispatcher: DispatcherDep,
    _models: Annotated[AgentModels, Depends(_agent_models)],  # 503 now, not a failed run later
) -> JSONResponse:
    """Send a message: saves it, queues the agent turn on a background worker,
    and returns at once. Follow it with GET /v1/studio/runs/{id}/events."""
    if await store.get_thread(thread_id) is None:
        raise NotFoundError(f"no chat {thread_id}")
    for asset_id in turn.asset_ids:
        asset = await store.get_asset(asset_id)
        if asset is None or asset["thread_id"] != thread_id:
            raise ValidationError(f"image {asset_id} is not part of this chat")

    active = await store.active_run(thread_id)
    if active:
        active = await _reconcile(request, store, active)
    if active and active["status"] in ACTIVE_RUN:
        return _busy(active)
    message = await store.add_message(thread_id, "user", turn.content, asset_ids=turn.asset_ids)
    try:
        run = await store.create_run(thread_id, message["id"])
    except ActiveRunExists as exc:  # lost a race with a concurrent send
        return _busy(exc.existing or {})
    call_id = await dispatcher.spawn_agent(run["id"])
    await store.set_run_call_id(run["id"], call_id)
    return JSONResponse(
        status_code=202,
        content=json.loads(json.dumps({"run": run, "message": message}, default=str)),
    )


def _busy(active: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": {
                "code": "turn_in_progress",
                "message": "The agent is still working on the previous message",
                "run_id": str(active.get("id")),
            }
        },
    )


async def _reconcile(request: Request, store: StudioStore, run: dict[str, Any]) -> dict[str, Any]:
    """A worker that died without finishing (OOM, timeout, preemption) must not
    leave the chat locked: mark it failed and close its event log."""
    if run["status"] not in ACTIVE_RUN or not run.get("modal_call_id"):
        return run
    dispatcher = request.app.state.dispatcher
    state, detail = await dispatcher.poll(run["modal_call_id"])
    if state == "running":
        return run
    fresh = await store.get_run(run["id"])
    if fresh and fresh["status"] in ACTIVE_RUN:
        message = f"the agent worker stopped unexpectedly: {detail or 'unknown'}"
        await store.append_events(
            run["id"], [("error", {"message": message}), ("done", {"status": "failed"})]
        )
        await store.finish_run(run["id"], "failed", message)
    return await store.get_run(run["id"]) or run


@router.get("/runs/{run_id}")
async def get_run(run_id: UUID, request: Request, store: StoreDep) -> dict[str, Any]:
    run = await store.get_run(run_id)
    if run is None:
        raise NotFoundError(f"no agent run {run_id}")
    return await _reconcile(request, store, run)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: UUID, request: Request, store: StoreDep) -> dict[str, Any]:
    run = await store.get_run(run_id)
    if run is None:
        raise NotFoundError(f"no agent run {run_id}")
    if run["status"] in ACTIVE_RUN:
        if run.get("modal_call_id"):
            await request.app.state.dispatcher.cancel(run["modal_call_id"])
        await store.append_events(
            run_id, [("error", {"message": "Stopped."}), ("done", {"status": "cancelled"})]
        )
        await store.finish_run(run_id, "cancelled", "cancelled via API")
    return await store.get_run(run_id) or run


@router.get("/runs/{run_id}/events")
async def run_events(
    run_id: UUID, request: Request, settings: SettingsDep, after_id: int = Query(0, ge=0)
) -> StreamingResponse:
    """The turn's event log as SSE, from `after_id` onwards.

    Every event carries `seq`; a client that reconnects with the last `seq` it
    saw gets exactly the events it missed. Polls every 0.4 s (tokens are
    coalesced by the worker, so this is a handful of rows per poll).
    """
    async with studio_store(settings) as store:
        if await store.get_run(run_id) is None:
            raise NotFoundError(f"no agent run {run_id}")

    async def stream() -> AsyncIterator[str]:
        last, beat = after_id, time.monotonic()
        # Own connection for the life of the stream (see runs.run_events).
        async with studio_store(settings) as store:
            while True:
                rows = await store.events_after(run_id, last)
                for row in rows:
                    last = row["id"]
                    payload = {**row["data"], "seq": row["id"]}
                    yield f"event: {row['type']}\ndata: {json.dumps(payload, default=str, ensure_ascii=False)}\n\n"
                    if row["type"] == "done":
                        return
                if not rows:
                    run = await store.get_run(run_id)
                    if run is not None:
                        run = await _reconcile(request, store, run)
                    if run is None or (
                        run["status"] not in ACTIVE_RUN
                        and not await store.events_after(run_id, last)
                    ):
                        yield f"event: done\ndata: {json.dumps({'status': run['status'] if run else 'missing', 'seq': last})}\n\n"
                        return
                if time.monotonic() - beat > HEARTBEAT_S:
                    beat = time.monotonic()
                    yield f"event: heartbeat\ndata: {json.dumps({'seq': last})}\n\n"
                await asyncio.sleep(EVENT_POLL_S)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


def default_writer(settings: Settings, storage: Storage) -> GenerationService:
    """The fine-tuned writer: MODEL_PROVIDER / MODEL_BASE_URL / MODEL_NAME."""
    return GenerationService(
        provider=build_provider(settings),
        repository=build_repository(settings),
        storage=storage,
        settings=settings,
    )


# ------------------------------------------------------------------ assets


def _asset_view(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        k: asset[k]
        for k in (
            "id",
            "created_at",
            "thread_id",
            "filename",
            "content_type",
            "bytes",
            "width",
            "height",
        )
    }


@router.post("/assets", status_code=201)
async def upload_asset(
    store: StoreDep,
    storage: StorageDep,
    file: Annotated[UploadFile, File(description="png, jpeg, webp or gif; max 10 MB")],
    thread_id: Annotated[UUID | None, Form()] = None,
) -> dict[str, Any]:
    """Upload an image for the agent to place on a page.

    The bytes are decoded before they are stored: a file that is not really an
    image (whatever its extension says) is rejected, and so is a decompression
    bomb.
    """
    from PIL import Image

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise PayloadTooLargeError("Image exceeds 10 MB")
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            fmt, (width, height) = img.format or "", img.size
    except Exception as exc:
        raise ValidationError(f"Not a readable image: {type(exc).__name__}") from exc
    if fmt not in IMAGE_TYPES:
        raise ValidationError(
            f"Unsupported image format {fmt or 'unknown'}; use PNG, JPEG, WEBP or GIF"
        )
    if thread_id and await store.get_thread(thread_id) is None:
        raise NotFoundError(f"no chat {thread_id}")

    asset_id = uuid4()
    filename = safe_filename(file.filename or f"image.{fmt.lower()}")
    key = f"assets/{asset_id}/{filename}"
    await storage.put_bytes(key, data, content_type=IMAGE_TYPES[fmt])
    asset = await store.add_asset(
        id=asset_id,
        thread_id=thread_id,
        filename=filename,
        content_type=IMAGE_TYPES[fmt],
        bytes=len(data),
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
        r2_key=key,
    )
    return _asset_view(asset)


@router.get("/assets/{asset_id}/raw")
async def asset_raw(asset_id: UUID, store: StoreDep, storage: StorageDep) -> Response:
    asset = await store.get_asset(asset_id)
    if asset is None:
        raise NotFoundError(f"no image {asset_id}")
    return Response(
        await storage.get_bytes(asset["r2_key"]),
        media_type=asset["content_type"],
        headers={"Cache-Control": "private, max-age=3600"},
    )


# ------------------------------------------------------------------ artifacts


@router.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: UUID, store: StoreDep) -> dict[str, Any]:
    artifact = await store.get_artifact(artifact_id)
    if artifact is None:
        raise NotFoundError(f"no artifact {artifact_id}")
    return {
        **artifact,
        "versions": await store.list_versions(artifact_id),
        "published": await store.published_for_artifact(artifact_id),
    }


@router.get("/artifacts/{artifact_id}/versions/{version}")
async def get_artifact_version(artifact_id: UUID, version: int, store: StoreDep) -> dict[str, Any]:
    row = await store.get_version(artifact_id, version)
    if row is None:
        raise NotFoundError(f"no version {version} of artifact {artifact_id}")
    return row


@router.get("/artifacts/{artifact_id}/preview", response_class=HTMLResponse)
async def preview_artifact(
    artifact_id: UUID,
    store: StoreDep,
    version: int | None = Query(None, ge=1),
    asset_base: str | None = Query(None, description="Same-origin path the images load from"),
) -> HTMLResponse:
    """The page exactly as it would publish, with images from an authenticated route."""
    artifact = await store.get_artifact(artifact_id)
    row = await store.get_version(artifact_id, version) if artifact else None
    if artifact is None or row is None:
        raise NotFoundError(f"no artifact {artifact_id} version {version or 'current'}")
    base = page.safe_asset_base(asset_base, "/v1/studio/assets")
    body = row.get("html") or page.markdown_to_html(row.get("markdown") or "")
    html = page.render_page(
        title=row["meta"].get("title") or artifact["title"],
        meta_description=row["meta"].get("meta_description"),
        body=body,
        asset_url=lambda a: f"{base}/{a}/raw",
    )
    return HTMLResponse(html, headers={"Content-Security-Policy": page.PAGE_CSP})


@router.post("/artifacts/{artifact_id}/publish")
async def publish_artifact(
    artifact_id: UUID,
    request: Request,
    store: StoreDep,
    storage: StorageDep,
    settings: SettingsDep,
    body: PublishRequest | None = None,
) -> dict[str, Any]:
    """Freeze a version as a public page at /p/{slug}. Re-publishing updates it."""
    body = body or PublishRequest()
    artifact = await store.get_artifact(artifact_id)
    row = await store.get_version(artifact_id, body.version) if artifact else None
    if artifact is None or row is None:
        raise NotFoundError(f"no artifact {artifact_id} version {body.version or 'current'}")

    html_body = row.get("html") or page.markdown_to_html(row.get("markdown") or "")
    refs = page.asset_refs(html_body)
    for asset_id in refs:
        asset = await store.get_asset(asset_id)
        if asset is None or asset["thread_id"] != artifact["thread_id"]:
            raise ValidationError(f"image {asset_id} is not part of this chat")

    title = row["meta"].get("title") or artifact["title"]
    existing = await store.published_for_artifact(artifact_id)
    slug = (
        body.slug
        or (existing[0]["slug"] if existing else slugify(title))
        or f"article-{str(artifact_id)[:8]}"
    )
    taken = await store.get_published(slug)
    if taken and taken["artifact_id"] != artifact_id:
        if body.slug:
            raise AppError(f"slug {slug!r} is taken", code="slug_taken", status_code=409)
        slug = f"{slug}-{str(artifact_id)[:6]}"

    base = _share_base(settings, request)
    document = page.render_page(
        title=title,
        meta_description=row["meta"].get("meta_description"),
        body=html_body,
        asset_url=lambda a: f"{base}/p/assets/{a}",
        canonical_url=f"{base}/p/{slug}",
    )
    key = f"published/{slug}/index.html"
    await put_text(storage, key, document, content_type="text/html; charset=utf-8")
    published = await store.publish(
        slug=slug,
        artifact_id=artifact_id,
        version=row["version"],
        title=title,
        r2_key=key,
        asset_ids=refs,
    )
    return {**published, "url": f"{base}/p/{slug}"}


@router.get("/content")
async def list_content(
    store: StoreDep, request: Request, settings: SettingsDep
) -> list[dict[str, Any]]:
    base = _share_base(settings, request)
    return [{**row, "url": f"{base}/p/{row['slug']}"} for row in await store.list_published()]


@router.post("/content/{slug}/unpublish")
async def unpublish(slug: str, store: StoreDep) -> dict[str, Any]:
    row = await store.set_published_status(slug, "unpublished")
    if row is None:
        raise NotFoundError(f"no published page {slug}")
    return row


# ------------------------------------------------------------------ public


@public.get("/p/assets/{asset_id}")
async def public_asset(asset_id: UUID, request: Request) -> Response:
    """An image, only while a published page references it."""
    settings: Settings = request.app.state.settings
    storage: Storage = request.app.state.storage_factory()
    async with studio_store(settings) as store:
        asset = await store.get_asset(asset_id)
        if asset is None or not await store.is_asset_public(asset_id):
            raise NotFoundError("not found")
    return Response(
        await storage.get_bytes(asset["r2_key"]),
        media_type=asset["content_type"],
        headers={"Cache-Control": "public, max-age=300"},
    )


@public.get("/p/{slug}", response_class=HTMLResponse)
async def public_page(slug: str, request: Request) -> HTMLResponse:
    settings: Settings = request.app.state.settings
    storage: Storage = request.app.state.storage_factory()
    async with studio_store(settings) as store:
        row = await store.get_published(slug)
    if row is None:
        raise NotFoundError("not found")
    if row["status"] != "published":
        return HTMLResponse(
            "<!doctype html><title>Unpublished</title><p>This page has been unpublished.</p>",
            status_code=410,
        )
    html = (await storage.get_bytes(row["r2_key"])).decode("utf-8")
    return HTMLResponse(
        html,
        headers={"Content-Security-Policy": page.PAGE_CSP, "Cache-Control": "public, max-age=60"},
    )
