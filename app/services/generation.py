"""The generation orchestrator: A -> F.

    A. normalize the request
    B. insert a `queued` row (so a crash is still observable)
    C. ingest + retrieve evidence (skipped, with a warning, when no sources)
    D. build the prompt and call the model (row -> `running`)
    E. quality-check; one retry with the warnings fed back as instructions
    F. persist the markdown to object storage and close the row

Every dependency is injected, so the whole flow runs with the mock provider, the
in-memory repository and filesystem storage — that is what makes `MODEL_PROVIDER=mock`
a genuine end-to-end test rather than a smoke screen.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import anyio

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db import Repository
from app.schemas.articles import (
    ArticleRequest,
    ArticleResponse,
    NormalizedParams,
    QualityReport,
    SourceRef,
)
from app.schemas.sources import SourceDocument
from app.services import ingest, normalize, prompt, quality
from app.services.llm import LLMProvider, Message, parse_article
from app.services.retrieve import Chunk, retrieve_chunks
from app.storage import Storage, put_text
from app.storage.keys import generation_key

logger = get_logger(__name__)

HEARTBEAT_INTERVAL_S = 10.0
MAX_OUTPUT_TOKENS = 3072


@dataclass(slots=True)
class _Context:
    """Mutable per-request state shared by the blocking and streaming paths."""

    generation_id: UUID
    params: NormalizedParams
    started: float
    messages: list[Message] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class GenerationService:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        repository: Repository,
        storage: Storage,
        settings: Settings,
    ) -> None:
        self._provider = provider
        self._repo = repository
        self._storage = storage
        self._settings = settings

    # ---------------------------------------------------------------- blocking

    async def generate(self, request: ArticleRequest) -> ArticleResponse:
        ctx = await self._prepare(request)
        try:
            await self._repo.update_generation(ctx.generation_id, status="running")
            result = await self._provider.complete(ctx.messages, max_tokens=MAX_OUTPUT_TOKENS)
            title, meta, body = parse_article(result.text)
            report = quality.evaluate(body, title, ctx.params)

            if not report.passed:
                # One retry only: a second failure usually means the brief itself
                # is unsatisfiable, and each retry costs a full GPU generation.
                logger.info(
                    "quality_retry",
                    extra={"generation_id": str(ctx.generation_id), "warnings": report.warnings},
                )
                retry_messages = self._messages(ctx, feedback=report.warnings)
                result = await self._provider.complete(retry_messages, max_tokens=MAX_OUTPUT_TOKENS)
                title, meta, body = parse_article(result.text)
                report = quality.evaluate(body, title, ctx.params)

            return await self._finish(
                ctx,
                title=title,
                meta=meta,
                body=body,
                report=report,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
        except Exception as exc:
            await self._fail(ctx, exc)
            raise

    # --------------------------------------------------------------- streaming

    async def stream(self, request: ArticleRequest) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield `(event_name, payload)` pairs for the SSE endpoint."""
        try:
            ctx = await self._prepare(request)
        except AppError as exc:
            yield "error", {"code": exc.code, "message": exc.message}
            return

        yield "status", {"stage": "queued", "generation_id": str(ctx.generation_id)}
        for warning in ctx.warnings:
            yield "status", {"stage": "warning", "message": warning}

        try:
            await self._repo.update_generation(ctx.generation_id, status="running")
            yield "status", {"stage": "generating", "message": "Generating article…"}

            raw = ""
            async for event, payload in self._stream_attempt(ctx.messages):
                if event == "token":
                    raw += payload["text"]
                yield event, payload

            title, meta, body = parse_article(raw)
            report = quality.evaluate(body, title, ctx.params)

            if not report.passed:
                yield "status", {"stage": "revising", "message": "Revising for quality…"}
                raw = ""
                async for event, payload in self._stream_attempt(
                    self._messages(ctx, feedback=report.warnings)
                ):
                    if event == "token":
                        raw += payload["text"]
                    yield event, payload
                title, meta, body = parse_article(raw)
                report = quality.evaluate(body, title, ctx.params)

            response = await self._finish(ctx, title=title, meta=meta, body=body, report=report)
            yield "result", json.loads(response.model_dump_json())
        except AppError as exc:
            await self._fail(ctx, exc)
            yield "error", {"code": exc.code, "message": exc.message}
        except Exception as exc:
            await self._fail(ctx, exc)
            yield "error", {"code": "internal_error", "message": "Generation failed"}

    async def _stream_attempt(
        self, messages: list[Message]
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Provider tokens, with a heartbeat whenever the model goes quiet.

        A scaled-to-zero Modal container can take 60-120 s to produce its first
        token. Without periodic bytes on the wire, Vercel's proxy and most
        corporate middleboxes will drop the connection as idle.
        """
        async for chunk in _heartbeat(
            self._provider.stream(messages, max_tokens=MAX_OUTPUT_TOKENS)
        ):
            if chunk is _HEARTBEAT:
                yield "heartbeat", {"ts": datetime.now(UTC).isoformat()}
            else:
                yield "token", {"text": chunk}

    # ------------------------------------------------------------------ stages

    async def _prepare(self, request: ArticleRequest) -> _Context:
        """Stages A-C plus prompt assembly."""
        started = time.perf_counter()
        params = normalize.normalize_request(request)  # A
        generation_id = uuid4()

        await self._repo.create_generation(  # B
            generation_id,
            json.loads(request.model_dump_json()),
            json.loads(params.model_dump_json()),
        )
        ctx = _Context(generation_id=generation_id, params=params, started=started)

        documents = await self._collect_documents(request, ctx)  # C
        if documents:
            ctx.chunks = retrieve_chunks(
                f"{params.topic} {' '.join(params.keywords)}",
                documents,
                top_k=self._settings.retrieval_top_k,
            )
            if not ctx.chunks:
                ctx.warnings.append("Sources were provided but nothing matched the topic.")
        else:
            ctx.warnings.append("No sources supplied; the article is written from the model alone.")

        ctx.messages = self._messages(ctx)
        logger.info(
            "generation_prepared",
            extra={
                "generation_id": str(generation_id),
                "chunks": len(ctx.chunks),
                "keywords": len(params.keywords),
            },
        )
        return ctx

    async def _collect_documents(
        self, request: ArticleRequest, ctx: _Context
    ) -> list[SourceDocument]:
        """Previously-uploaded sources plus an optional ad-hoc URL."""
        documents: list[SourceDocument] = []

        if request.source_ids:
            rows = await self._repo.get_source_documents(request.source_ids)
            for row in rows:
                documents.append(
                    SourceDocument(
                        id=UUID(str(row["id"])),
                        kind=row.get("kind", "upload"),
                        title=row.get("title"),
                        url=row.get("url"),
                        r2_key=row.get("r2_key"),
                        text=_row_text(row) or await self._load_text(row),
                    )
                )

        if request.source_url:
            try:
                doc = await ingest.ingest_url(str(request.source_url), self._storage)
                documents.append(doc)
                await self._repo.upsert_source_document(_source_row(doc))
            except AppError as exc:
                # A dead link should degrade the article, not fail the request.
                ctx.warnings.append(f"Could not read {request.source_url}: {exc.message}")

        return [d for d in documents if d.text.strip()]

    async def _load_text(self, row: dict[str, Any]) -> str:
        """Rehydrate text from the archived object when the row has none.

        Only the text-ish archives are worth rehydrating: an upload's archive is
        the original PDF/DOCX binary, whose extracted text already lives on the
        row (see `POST /sources/upload`).
        """
        key = str(row.get("r2_key") or "")
        if not key:
            return ""
        try:
            raw = await self._storage.get_bytes(key)
        except AppError:
            return ""
        if key.endswith(".html"):
            _, text = ingest.extract_html(raw.decode("utf-8", errors="replace"), url=row.get("url"))
            return text
        if key.endswith((".txt", ".md")):
            return raw.decode("utf-8", errors="replace")
        return ""

    def _messages(self, ctx: _Context, *, feedback: list[str] | None = None) -> list[Message]:
        return [
            {"role": "system", "content": prompt.build_system_prompt(ctx.params)},
            {
                "role": "user",
                "content": prompt.build_user_prompt(ctx.params, ctx.chunks, feedback=feedback),
            },
        ]

    async def _finish(
        self,
        ctx: _Context,
        *,
        title: str,
        meta: str,
        body: str,
        report: QualityReport,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> ArticleResponse:
        """Stage F: persist markdown, close the row, build the response."""
        latency_ms = int((time.perf_counter() - ctx.started) * 1000)
        report = report.model_copy(update={"warnings": report.warnings + ctx.warnings})

        storage_key: str | None = None
        try:
            storage_key = await put_text(
                self._storage,
                generation_key(ctx.generation_id),
                f"# {title}\n\n_{meta}_\n\n{body}\n",
                content_type="text/markdown; charset=utf-8",
            )
        except AppError as exc:
            # The article is already generated; losing the archive copy is not
            # worth failing the caller's request over.
            logger.warning("storage_write_failed", extra={"error": exc.message})
            report.warnings.append("Article was generated but could not be archived to storage.")

        sources = [
            SourceRef(
                source_id=chunk.source_id,
                url=chunk.source_url,
                title=chunk.source_title,
                score=round(chunk.score, 4),
            )
            for chunk in ctx.chunks
        ]

        await self._repo.update_generation(
            ctx.generation_id,
            status="succeeded",
            title=title,
            meta_description=meta,
            article_markdown=body,
            r2_key=storage_key,
            quality_report=json.loads(report.model_dump_json()),
            sources=[json.loads(s.model_dump_json()) for s in sources],
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completed_at=datetime.now(UTC).isoformat(),
        )
        await self._repo.record_generation_sources(
            ctx.generation_id,
            [
                {
                    "source_document_id": str(chunk.source_id) if chunk.source_id else None,
                    "chunk_index": chunk.chunk_index,
                    "score": round(chunk.score, 4),
                    "snippet": chunk.text[:500],
                }
                for chunk in ctx.chunks
            ],
        )

        return ArticleResponse(
            id=ctx.generation_id,
            status="succeeded",
            title=title,
            meta_description=meta,
            article_markdown=body,
            sources=sources,
            quality_report=report,
            model_version=self._settings.model_name,
            latency_ms=latency_ms,
            created_at=datetime.now(UTC),
        )

    async def _fail(self, ctx: _Context, exc: Exception) -> None:
        logger.exception("generation_failed", extra={"generation_id": str(ctx.generation_id)})
        message = exc.message if isinstance(exc, AppError) else "Internal error"
        try:
            await self._repo.update_generation(
                ctx.generation_id,
                status="failed",
                error=message,
                latency_ms=int((time.perf_counter() - ctx.started) * 1000),
                completed_at=datetime.now(UTC).isoformat(),
            )
        except Exception:
            logger.exception("failure_write_failed")


def _row_text(row: dict[str, Any]) -> str:
    """Extracted text is carried in `source_documents.metadata.text`.

    The table has no dedicated text column (it is an archive index, not a
    document store), and keeping the text on the row avoids an object-store
    round-trip per source on every generation.
    """
    metadata = row.get("metadata") or {}
    return str(row.get("text") or metadata.get("text") or "")


def _source_row(doc: SourceDocument) -> dict[str, Any]:
    return {
        "id": str(doc.id),
        "kind": doc.kind,
        "url": doc.url,
        "r2_key": doc.r2_key,
        "content_hash": ingest.content_hash(doc.text.encode()),
        "title": doc.title,
        "text_chars": doc.text_chars,
        "metadata": {"text": doc.text},
    }


# --------------------------------------------------------------------------- #
# Heartbeat plumbing
# --------------------------------------------------------------------------- #

_HEARTBEAT = object()
_DONE = object()


async def _heartbeat(
    source: AsyncIterator[str], interval: float = HEARTBEAT_INTERVAL_S
) -> AsyncIterator[Any]:
    """Yield items from `source`, injecting `_HEARTBEAT` during quiet periods.

    The producer runs in a task group and pushes into a zero-buffer memory
    stream, so back-pressure is preserved: we never buffer the model's output
    faster than the client reads it.
    """
    send, receive = anyio.create_memory_object_stream[Any](0)
    failure: list[BaseException] = []

    async def produce() -> None:
        try:
            async for item in source:
                await send.send(item)
        except BaseException as exc:
            failure.append(exc)
        finally:
            # Shielded: the sentinel must reach the consumer even if the task
            # group is being torn down, or the consumer would hang on receive().
            with anyio.CancelScope(shield=True):
                with contextlib.suppress(anyio.BrokenResourceError):
                    await send.send(_DONE)  # BrokenResource == consumer already gone
                send.close()

    async with anyio.create_task_group() as tg:
        tg.start_soon(produce)
        async with receive:
            while True:
                with anyio.move_on_after(interval) as scope:
                    item = await receive.receive()
                if scope.cancel_called:
                    yield _HEARTBEAT
                    continue
                if item is _DONE:
                    break
                yield item

    if failure:
        raise failure[0]
