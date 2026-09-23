"""FastAPI dependencies.

Module level on purpose: with `from __future__ import annotations`, FastAPI
resolves parameter annotations from module globals, so aliases defined inside a
function would be unresolvable and silently become request-body fields.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, Request

from app.core.config import Settings
from app.core.errors import AppError, UnauthorizedError
from app.storage.base import Storage
from pipeline.api.dispatch import Dispatcher
from pipeline.store import CorpusStore, JobStore, connect_jobs


def _settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _dispatcher(request: Request) -> Dispatcher:
    return request.app.state.dispatcher  # type: ignore[no-any-return]


def _storage(request: Request) -> Storage:
    return request.app.state.storage_factory()  # type: ignore[no-any-return]


async def _stores(request: Request) -> AsyncIterator[tuple[CorpusStore, JobStore]]:
    settings: Settings = request.app.state.settings
    if not settings.database_url:
        raise AppError(
            "Database is not configured (DB_URL)", code="not_configured", status_code=503
        )
    async with connect_jobs(settings.database_url) as pair:
        yield pair


async def require_key(
    request: Request, x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
) -> None:
    """Fail closed: these endpoints spend GPU time and labelling budget."""
    expected = request.app.state.settings.jobs_api_key
    if not expected:
        raise AppError(
            "JOBS_API_KEY is not configured on the server", code="not_configured", status_code=503
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise UnauthorizedError("Missing or invalid X-API-Key header")


SettingsDep = Annotated[Settings, Depends(_settings)]
DispatcherDep = Annotated[Dispatcher, Depends(_dispatcher)]
StorageDep = Annotated[Storage, Depends(_storage)]
Stores = Annotated[tuple[CorpusStore, JobStore], Depends(_stores)]
GUARDED = [Depends(require_key)]
