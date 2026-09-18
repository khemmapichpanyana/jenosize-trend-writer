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

# Tests must never read a developer's real .env: it now outranks the process
# environment and holds live credentials, so a test that forgot to pass a field
# explicitly could otherwise talk to production Supabase or R2.
Settings.model_config["env_file"] = None


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


# --------------------------------------------------------------------------- #
# Pipeline integration fixtures: real Postgres + an S3-compatible R2 stand-in.
#
# The pipeline's correctness lives in SQL predicates ("which rows changed?"), so
# it is tested against a real Postgres rather than a mock. Set TEST_DATABASE_URL
# to a *disposable* database (the fixture drops and recreates its public schema);
# without it, these tests are skipped. CI provides one as a service container.
# --------------------------------------------------------------------------- #

import os  # noqa: E402
import socket  # noqa: E402

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set; skipping Postgres integration tests")
    import psycopg

    from pipeline.migrate import apply_migrations

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("drop schema if exists public cascade")
        conn.execute("create schema public")
    apply_migrations(TEST_DATABASE_URL)
    return TEST_DATABASE_URL


@pytest.fixture
async def corpus(pg_dsn: str):  # type: ignore[no-untyped-def]
    import psycopg

    from pipeline.store import connect

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("truncate training_articles, pipeline_runs, dataset_versions")
    async with connect(pg_dsn) as store:
        yield store


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def s3_endpoint():  # type: ignore[no-untyped-def]
    from moto.server import ThreadedMotoServer

    port = _free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture
def r2(s3_endpoint: str):  # type: ignore[no-untyped-def]
    """A fresh, empty bucket behind the real R2Storage class."""
    import uuid

    import boto3

    from app.storage.r2 import R2Storage

    bucket = f"test-{uuid.uuid4().hex[:12]}"
    boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    ).create_bucket(Bucket=bucket)
    return R2Storage(
        Settings(
            r2_endpoint=s3_endpoint,
            r2_access_key_id="test",
            r2_secret_access_key="test",
            r2_bucket=bucket,
        )
    )


def r2_keys(storage) -> list[str]:  # type: ignore[no-untyped-def]
    """Every object key in the test bucket."""
    response = storage._client.list_objects_v2(Bucket=storage._bucket)
    return sorted(obj["Key"] for obj in response.get("Contents", []))
