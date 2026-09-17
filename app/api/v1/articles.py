"""Article generation endpoints (blocking, streaming, and read-back)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.core.deps import GenerationDep, RepositoryDep, require_api_key
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.schemas.articles import ArticleRequest, ArticleResponse, ArticleSummary

router = APIRouter(prefix="/articles", tags=["articles"])
logger = get_logger(__name__)

# Text/event-stream headers. `X-Accel-Buffering: no` stops nginx-class proxies
# (which Vercel sits behind) from holding the response until it completes —
# without it, "streaming" arrives as one lump at the end.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse(event: str, data: dict[str, Any]) -> str:
    """Format one SSE frame. JSON payloads keep embedded newlines from breaking it."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=ArticleResponse, dependencies=[Depends(require_api_key)])
async def create_article(request: ArticleRequest, service: GenerationDep) -> ArticleResponse:
    """Generate an article and return it in one response."""
    return await service.generate(request)


@router.post("/stream", dependencies=[Depends(require_api_key)])
async def stream_article(request: ArticleRequest, service: GenerationDep) -> StreamingResponse:
    """Generate an article, streaming `status | token | heartbeat | result | error`."""

    async def event_stream() -> AsyncIterator[str]:
        async for event, payload in service.stream(request):
            yield sse(event, payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("", response_model=list[ArticleSummary])
async def list_articles(
    repository: RepositoryDep,
    limit: int = Query(20, ge=1, le=100),
) -> list[ArticleSummary]:
    rows = await repository.list_generations(limit=limit)
    return [ArticleSummary.model_validate(_coerce(row)) for row in rows]


@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(article_id: UUID, repository: RepositoryDep) -> ArticleResponse:
    row = await repository.get_generation(article_id)
    if row is None:
        raise NotFoundError(f"No generation with id {article_id}")
    return ArticleResponse.model_validate(_coerce(row))


def _coerce(row: dict[str, Any]) -> dict[str, Any]:
    """Map a DB row onto the response schema.

    Supabase returns `model_version_id` (a FK) while the API exposes the human
    readable `model_version`, and jsonb columns come back as dicts/lists that
    pydantic can validate directly.
    """
    out = dict(row)
    out.setdefault("id", row.get("id"))
    out.setdefault("status", row.get("status", "queued"))
    out["sources"] = row.get("sources") or []
    out["quality_report"] = row.get("quality_report") or None
    out["model_version"] = row.get("model_version") or row.get("model_version_id")
    return out
