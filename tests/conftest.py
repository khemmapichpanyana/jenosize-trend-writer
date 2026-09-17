"""Shared fixtures.

Every test runs the app in its fully local configuration (mock model, in-memory
repository, temp-dir storage) so the suite needs no network and no accounts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.articles import NormalizedParams


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="test",
        model_provider="mock",
        persistence="none",
        storage="local",
        local_storage_dir=str(tmp_path / "data"),
        log_level="WARNING",
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    # ASGITransport drives the app in-process: no uvicorn, no sockets, and the
    # lifespan is exercised via the explicit router below.
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as ac,
    ):
        yield ac


@pytest.fixture
def params() -> NormalizedParams:
    return NormalizedParams(
        topic="Agentic AI in retail",
        industry="Retail & E-commerce",
        audience="Business leaders",
        keywords=["agentic ai", "personalization"],
        length="short",
        target_words=600,
    )
