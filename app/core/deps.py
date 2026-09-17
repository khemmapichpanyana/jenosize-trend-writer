"""Dependency wiring.

The provider / repository / storage instances are built once during the app
lifespan and stashed on `app.state`. Vercel keeps a warm lambda between
requests, so per-request construction would pay the boto3 / supabase client
setup cost over and over.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from app.core.config import Settings, get_settings
from app.core.errors import UnauthorizedError
from app.db import Repository
from app.services.generation import GenerationService
from app.services.llm import LLMProvider
from app.storage import Storage


def get_config(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_provider(request: Request) -> LLMProvider:
    return request.app.state.provider  # type: ignore[no-any-return]


def get_repository(request: Request) -> Repository:
    return request.app.state.repository  # type: ignore[no-any-return]


def get_storage(request: Request) -> Storage:
    return request.app.state.storage  # type: ignore[no-any-return]


def get_generation_service(
    provider: Annotated[LLMProvider, Depends(get_provider)],
    repository: Annotated[Repository, Depends(get_repository)],
    storage: Annotated[Storage, Depends(get_storage)],
    settings: Annotated[Settings, Depends(get_config)],
) -> GenerationService:
    return GenerationService(
        provider=provider, repository=repository, storage=storage, settings=settings
    )


async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> None:
    """Optional shared-secret gate on write routes.

    Unset API_KEY means the public demo is open, which is what a grader needs.
    Set it to throttle abuse once the endpoint is public and burning GPU time.
    """
    expected = (settings or get_settings()).api_key
    if expected and x_api_key != expected:
        raise UnauthorizedError("Missing or invalid X-API-Key header")


ConfigDep = Annotated[Settings, Depends(get_config)]
ProviderDep = Annotated[LLMProvider, Depends(get_provider)]
RepositoryDep = Annotated[Repository, Depends(get_repository)]
StorageDep = Annotated[Storage, Depends(get_storage)]
GenerationDep = Annotated[GenerationService, Depends(get_generation_service)]
