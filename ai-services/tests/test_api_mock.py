"""End-to-end API tests against the mock provider.

These are the tests that prove the Day-0 claim: the whole generation flow works
with no GPU, no database and no object store.
"""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

BRIEF = {
    "topic": "Agentic AI in Southeast Asian retail",
    "category": "Futurist",
    "industry": "ecommerce",
    "audience": "C-suite executives",
    "keywords": ["agentic ai", "retail media", "personalization"],
    "length": "short",
}


async def test_health_is_ok_locally(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database": "ok", "storage": "ok", "model": "skipped"}


async def test_root_and_openapi(client: AsyncClient) -> None:
    assert (await client.get("/")).status_code == 200
    assert (await client.get("/openapi.json")).status_code == 200


async def test_warmup_is_skipped_on_mock(client: AsyncClient) -> None:
    body = (await client.post("/api/v1/model/warmup")).json()
    assert body["status"] == "skipped"


async def test_generate_article_end_to_end(client: AsyncClient) -> None:
    response = await client.post("/api/v1/articles", json=BRIEF)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "succeeded"
    assert body["title"]
    assert body["meta_description"]
    assert "##" in body["article_markdown"]
    assert body["latency_ms"] >= 0

    report = body["quality_report"]
    assert report["word_count"] > 0
    # Every requested keyword must survive normalization into the article.
    assert report["keyword_coverage"] == 1.0
    # No sources were supplied, so the flow must say so rather than fail.
    assert any("No sources" in w for w in report["warnings"])


async def test_generation_is_readable_afterwards(client: AsyncClient) -> None:
    created = (await client.post("/api/v1/articles", json=BRIEF)).json()

    fetched = await client.get(f"/api/v1/articles/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == created["title"]

    listed = await client.get("/api/v1/articles?limit=5")
    assert listed.status_code == 200
    assert created["id"] in [row["id"] for row in listed.json()]


async def test_unknown_generation_returns_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/v1/articles/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"topic": "ab"}, id="topic-too-short"),
        pytest.param({"topic": "valid topic", "length": "enormous"}, id="bad-length"),
        pytest.param({"topic": "valid topic", "source_url": "not-a-url"}, id="bad-url"),
    ],
)
async def test_validation_errors_use_the_common_envelope(
    client: AsyncClient, payload: dict[str, object]
) -> None:
    response = await client.post("/api/v1/articles", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_keywords_accept_a_comma_separated_string(client: AsyncClient) -> None:
    body = (
        await client.post("/api/v1/articles", json={**BRIEF, "keywords": "alpha, beta ,alpha"})
    ).json()
    assert body["quality_report"]["keyword_coverage"] == 1.0


async def test_stream_emits_tokens_then_a_result(client: AsyncClient) -> None:
    events: list[tuple[str, dict]] = []
    async with client.stream("POST", "/api/v1/articles/stream", json=BRIEF) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                lines = frame.splitlines()
                name = next(ln[len("event: ") :] for ln in lines if ln.startswith("event: "))
                data = next(ln[len("data: ") :] for ln in lines if ln.startswith("data: "))
                events.append((name, json.loads(data)))

    names = [name for name, _ in events]
    assert names[0] == "status"
    assert "token" in names
    assert names[-1] == "result"

    streamed = "".join(p["text"] for n, p in events if n == "token")
    result = next(p for n, p in events if n == "result")
    assert result["status"] == "succeeded"
    # The streamed text and the final object must describe the same article.
    assert result["title"] in streamed


async def test_upload_then_generate_uses_the_source(client: AsyncClient) -> None:
    content = (
        b"Retail media networks in Thailand grew sharply last year.\n\n"
        b"Agentic AI assistants now complete checkout flows for repeat customers."
    )

    upload = await client.post(
        "/api/v1/sources/upload",
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert upload.status_code == 200, upload.text
    source_id = upload.json()["source_id"]
    assert upload.json()["text_chars"] == len(content.decode())

    body = (await client.post("/api/v1/articles", json={**BRIEF, "source_ids": [source_id]})).json()
    assert body["sources"], "retrieved chunks should be reported back as sources"
    assert body["sources"][0]["source_id"] == source_id
    assert not any("No sources" in w for w in body["quality_report"]["warnings"])


async def test_upload_rejects_unsupported_types(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/sources/upload",
        files={"file": ("image.png", b"\x89PNG\r\n", "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_markdown_is_archived_to_storage(client: AsyncClient, settings) -> None:
    from pathlib import Path

    body = (await client.post("/api/v1/articles", json=BRIEF)).json()
    archived = Path(settings.local_storage_dir) / "generations" / f"{body['id']}.md"
    assert archived.exists()
    assert body["title"] in archived.read_text()


async def test_request_id_header_is_echoed(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "trace-me"})
    assert response.headers["X-Request-ID"] == "trace-me"
