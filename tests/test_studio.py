"""Studio end to end: a real LangChain agent, driven by a scripted chat model.

Everything except the model's decisions is real: the agent graph and its
middleware, the tools, the fine-tuned-writer path (MockProvider standing in for
Modal vLLM), Postgres, the S3-compatible bucket, the sanitiser and the public
share routes.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.config import Settings
from pipeline.api import InlineDispatcher, create_jobs_app
from pipeline.store import connect_jobs
from studio import html as page
from studio.agent import history_to_messages
from studio.api import default_writer, slugify
from studio.llm import AgentModels
from studio.tools import DESIGN_SYSTEM

KEY = {"X-API-Key": "studio-key"}
STUDIO_TABLES = "chat_threads, assets, published_content, training_progress, job_runs"


class ScriptedModel(BaseChatModel):
    """A chat model whose next reply is computed from the conversation so far."""

    script: Callable[[list[BaseMessage]], AIMessage]
    model_label: str = "scripted"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:  # type: ignore[override]
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        reply = self.script(messages)
        reply.response_metadata = {"model_name": self.model_label}
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _last_tool_result(messages: list[BaseMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return dict(json.loads(str(message.content)))
    return {}


def writer_then_designer(
    image_ids: list[str], designed_html: str | None = None
) -> Callable[[list[BaseMessage]], AIMessage]:
    """Agent: write_article -> design_page(with images) -> short reply.
    Designer calls (system prompt = DESIGN_SYSTEM) get a page body back."""

    def script(messages: list[BaseMessage]) -> AIMessage:
        if isinstance(messages[0], SystemMessage) and str(messages[0].content).startswith(
            DESIGN_SYSTEM[:40]
        ):
            refs = "".join(
                f'<figure class="jz-hero"><img src="asset://{i}" alt="hero"></figure>'
                for i in image_ids
            )
            body = designed_html or (
                refs + "<h1>Agentic AI</h1>" + "<p>" + "Designed paragraph. " * 400 + "</p>"
            )
            return AIMessage(body)
        done_tools = [m for m in messages if isinstance(m, ToolMessage)]
        if not done_tools:
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "write_article",
                        "args": {
                            "topic": "Agentic AI in retail",
                            "keywords": ["agentic ai"],
                            "length": "short",
                        },
                        "id": "call-write",
                    }
                ],
            )
        if len(done_tools) == 1:
            artifact_id = _last_tool_result(messages)["artifact_id"]
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "design_page",
                        "args": {"artifact_id": artifact_id, "image_asset_ids": image_ids},
                        "id": "call-design",
                    }
                ],
            )
        return AIMessage("Drafted with the fine-tuned writer and laid it out as a page.")

    return script


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def studio_settings(pg_dsn: str, r2: Any, tmp_path: Path) -> Settings:
    return r2._settings.model_copy(
        update={
            "database_url": pg_dsn,
            "jobs_api_key": "studio-key",
            "model_provider": "mock",  # the fine-tuned writer, mocked
            "persistence": "none",
            "models_dir": str(tmp_path / "models"),
            "public_share_base_url": "https://share.example",
            "log_level": "WARNING",
        }
    )


@pytest.fixture
async def clean_db(pg_dsn: str) -> None:
    async with connect_jobs(pg_dsn) as (_, jobs):
        await jobs._conn.execute(f"truncate {STUDIO_TABLES} cascade")


def make_client(
    settings: Settings,
    r2: Any,
    script: Callable[[list[BaseMessage]], AIMessage],
    *,
    fallback: Callable[[list[BaseMessage]], AIMessage] | None = None,
) -> AsyncClient:
    def models(_: Settings) -> AgentModels:
        primary = ScriptedModel(script=script, model_label="primary")
        fallbacks: list[BaseChatModel] = (
            [ScriptedModel(script=fallback, model_label="fallback")] if fallback else []
        )
        return AgentModels(
            primary=primary,
            fallbacks=fallbacks,
            names=["primary", "fallback"][: 1 + len(fallbacks)],
        )

    app = create_jobs_app(
        InlineDispatcher(settings, r2),
        settings,
        r2,
        agent_models_factory=models,
        writer_factory=default_writer,
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://studio")


def png_bytes(size: tuple[int, int] = (64, 32)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (0, 188, 206)).save(buffer, format="PNG")
    return buffer.getvalue()


async def sse_events(response: Any) -> list[tuple[str, dict[str, Any]]]:
    events, buffer = [], ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            lines = frame.splitlines()
            name = next(ln[7:] for ln in lines if ln.startswith("event: "))
            data = next(ln[6:] for ln in lines if ln.startswith("data: "))
            events.append((name, json.loads(data)))
    return events


async def chat(
    client: AsyncClient, thread_id: str, content: str, asset_ids: list[str] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    async with client.stream(
        "POST",
        f"/v1/studio/threads/{thread_id}/messages",
        headers=KEY,
        json={"content": content, "asset_ids": asset_ids or []},
    ) as response:
        assert response.status_code == 200, await response.aread()
        return await sse_events(response)


async def new_thread_with_image(client: AsyncClient) -> tuple[str, str]:
    thread = (await client.post("/v1/studio/threads", headers=KEY)).json()
    upload = await client.post(
        "/v1/studio/assets",
        headers=KEY,
        data={"thread_id": thread["id"]},
        files={"file": ("hero shot.png", png_bytes(), "image/png")},
    )
    assert upload.status_code == 201, upload.text
    return thread["id"], upload.json()["id"]


# ------------------------------------------------------------------ the agent


async def test_a_turn_writes_designs_and_persists(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    async with make_client(studio_settings, r2, writer_then_designer([])) as client:
        thread_id, image_id = await new_thread_with_image(client)
        client_script = writer_then_designer([image_id])
    async with make_client(studio_settings, r2, client_script) as client:
        events = await chat(
            client, thread_id, "Write about agentic AI in retail with my photo", [image_id]
        )

    names = [n for n, _ in events]
    assert names[0] == "thread" and names[-2:] == ["message", "done"]
    assert "error" not in names, events
    # The article streamed into the panel from the (mock) fine-tuned writer…
    assert names.count("artifact_delta") > 20
    tools = [p["name"] for n, p in events if n == "tool_start"]
    assert tools == ["write_article", "design_page"]
    artifacts = [p for n, p in events if n == "artifact"]
    assert [a["version"] for a in artifacts] == [1, 2]
    reply = next(p for n, p in events if n == "message")
    assert "fine-tuned writer" in reply["content"] and reply["model"] == "primary"

    async with make_client(studio_settings, r2, client_script) as client:
        thread = (await client.get(f"/v1/studio/threads/{thread_id}", headers=KEY)).json()
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant"]
    assert thread["messages"][0]["asset_ids"] == [image_id]
    assert thread["title"].startswith("Write about agentic AI")
    [artifact] = thread["artifacts"]
    assert [v["version"] for v in artifact["versions"]] == [1, 2]
    assert [c["name"] for c in thread["messages"][1]["tool_calls"]] == [
        "write_article",
        "design_page",
    ]


async def test_designed_page_is_sanitised_and_keeps_only_real_images(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    hostile = (
        '<h1 onclick="steal()">Agentic AI</h1><script>alert(1)</script>'
        '<img src="asset://00000000-0000-0000-0000-000000000000">' + "<p>ok</p>" * 400
    )
    async with make_client(studio_settings, r2, writer_then_designer([])) as client:
        thread_id, _ = await new_thread_with_image(client)
    async with make_client(
        studio_settings, r2, writer_then_designer([], designed_html=hostile)
    ) as client:
        events = await chat(client, thread_id, "Write it")
        design = next(p for n, p in events if n == "tool_end" and p["name"] == "design_page")[
            "result"
        ]
        # An invented image id means the design is rejected for the standard layout.
        assert design["layout"] == "standard"
        artifact_id = design["artifact_id"]
        version = (
            await client.get(f"/v1/studio/artifacts/{artifact_id}/versions/2", headers=KEY)
        ).json()
    assert "<script" not in version["html"] and "onclick" not in version["html"]
    assert "00000000-0000-0000-0000-000000000000" not in version["html"]


async def test_primary_failure_falls_back(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    def broken(_: list[BaseMessage]) -> AIMessage:
        raise ConnectionError("modal cold start timed out")

    def answer(_: list[BaseMessage]) -> AIMessage:
        return AIMessage("Hello from the fallback.")

    async with make_client(studio_settings, r2, broken, fallback=answer) as client:
        thread = (await client.post("/v1/studio/threads", headers=KEY)).json()
        events = await chat(client, thread["id"], "hi")
    reply = next(p for n, p in events if n == "message")
    assert reply["content"] == "Hello from the fallback." and reply["model"] == "fallback"


async def test_no_agent_model_is_a_clear_503(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    app = create_jobs_app(
        InlineDispatcher(studio_settings, r2), studio_settings, r2
    )  # real factory, nothing configured
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://studio") as client:
        thread = (await client.post("/v1/studio/threads", headers=KEY)).json()
        response = await client.post(
            f"/v1/studio/threads/{thread['id']}/messages", headers=KEY, json={"content": "hi"}
        )
    assert response.status_code == 503 and "MODEL_BASE_URL" in response.json()["error"]["message"]


async def test_foreign_images_cannot_be_attached(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    async with make_client(studio_settings, r2, writer_then_designer([])) as client:
        _, other_image = await new_thread_with_image(client)
        mine = (await client.post("/v1/studio/threads", headers=KEY)).json()
        response = await client.post(
            f"/v1/studio/threads/{mine['id']}/messages",
            headers=KEY,
            json={"content": "use it", "asset_ids": [other_image]},
        )
    assert response.status_code == 422


# ------------------------------------------------------------------ assets, preview, publishing


async def test_uploads_must_really_be_images(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    async with make_client(studio_settings, r2, writer_then_designer([])) as client:
        response = await client.post(
            "/v1/studio/assets",
            headers=KEY,
            files={"file": ("fake.png", b"<?php echo 1; ?>", "image/png")},
        )
    assert response.status_code == 422 and "readable image" in response.json()["error"]["message"]


async def test_preview_publish_share_and_unpublish(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    async with make_client(studio_settings, r2, writer_then_designer([])) as client:
        thread_id, image_id = await new_thread_with_image(client)
        _, stray_image = await new_thread_with_image(client)  # never published
    script = writer_then_designer([image_id])
    async with make_client(studio_settings, r2, script) as client:
        events = await chat(client, thread_id, "Write and design it", [image_id])
        artifact_id = next(p for n, p in events if n == "artifact")["artifact_id"]

        preview = await client.get(
            f"/v1/studio/artifacts/{artifact_id}/preview",
            params={"asset_base": "/api/studio/v1/studio/assets"},
            headers=KEY,
        )
        assert preview.status_code == 200
        assert f"/api/studio/v1/studio/assets/{image_id}/raw" in preview.text
        assert (
            "script-src" not in preview.headers["content-security-policy"]
        )  # i.e. no scripts at all
        # A cross-origin asset_base is refused and falls back to the API's own path.
        evil = await client.get(
            f"/v1/studio/artifacts/{artifact_id}/preview",
            params={"asset_base": "https://evil.example"},
            headers=KEY,
        )
        assert "evil.example" not in evil.text

        # Before publishing, nothing is public.
        assert (await client.get(f"/p/assets/{image_id}")).status_code == 404

        published = (
            await client.post(f"/v1/studio/artifacts/{artifact_id}/publish", headers=KEY)
        ).json()
        assert published["url"] == f"https://share.example/p/{published['slug']}"
        assert published["version"] == 2 and published["asset_ids"] == [image_id]

        page_response = await client.get(f"/p/{published['slug']}")
        assert page_response.status_code == 200
        assert "default-src 'none'" in page_response.headers["content-security-policy"]
        assert f"https://share.example/p/assets/{image_id}" in page_response.text
        assert (await client.get(f"/p/assets/{image_id}")).status_code == 200
        assert (await client.get(f"/p/assets/{stray_image}")).status_code == 404

        # Re-publishing keeps the same slug.
        again = (
            await client.post(f"/v1/studio/artifacts/{artifact_id}/publish", headers=KEY)
        ).json()
        assert again["slug"] == published["slug"]

        listed = (await client.get("/v1/studio/content", headers=KEY)).json()
        assert [c["slug"] for c in listed] == [published["slug"]]

        await client.post(f"/v1/studio/content/{published['slug']}/unpublish", headers=KEY)
        assert (await client.get(f"/p/{published['slug']}")).status_code == 410
        assert (await client.get(f"/p/assets/{image_id}")).status_code == 404


async def test_a_slug_belongs_to_one_article(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    script = writer_then_designer([])
    async with make_client(studio_settings, r2, script) as client:
        ids = []
        for _ in range(2):
            thread = (await client.post("/v1/studio/threads", headers=KEY)).json()
            events = await chat(client, thread["id"], "Write it")
            ids.append(next(p for n, p in events if n == "artifact")["artifact_id"])
        first = (await client.post(f"/v1/studio/artifacts/{ids[0]}/publish", headers=KEY)).json()
        # Same title -> auto slug is disambiguated rather than stealing the first page.
        second = (await client.post(f"/v1/studio/artifacts/{ids[1]}/publish", headers=KEY)).json()
        assert second["slug"] != first["slug"]
        clash = await client.post(
            f"/v1/studio/artifacts/{ids[1]}/publish", headers=KEY, json={"slug": first["slug"]}
        )
    assert clash.status_code == 409


def test_slugify() -> None:
    assert slugify("Agentic AI: What Comes Next?") == "agentic-ai-what-comes-next"
    assert slugify("เทรนด์การทำ") == ""  # Thai titles fall back to article-<id> at publish time


def test_history_keeps_artifact_ids_for_revisions() -> None:
    asset_id = uuid4()
    rows: list[dict[str, Any]] = [
        {"role": "user", "content": "write it", "asset_ids": [asset_id]},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [
                {"name": "write_article", "result": '{"artifact_id": "abc", "version": 1}'}
            ],
        },
    ]
    messages = history_to_messages(rows, {asset_id: {"id": asset_id, "filename": "hero.png"}})
    assert "hero.png" in str(messages[0].content)
    assert '"artifact_id": "abc"' in str(messages[1].content)


def test_sanitizer_allowlist() -> None:
    dirty = (
        '<p style="x" onmouseover="y">t</p><script>1</script><iframe src="x"></iframe>'
        '<a href="javascript:alert(1)">j</a><img src="data:image/png;base64,AA">'
        '<img src="asset://11111111-2222-3333-4444-555555555555" alt="ok">'
    )
    clean = page.sanitize(dirty)
    for banned in ("script", "iframe", "onmouseover", "style=", "javascript:", "data:image"):
        assert banned not in clean
    assert page.asset_refs(clean) == [UUID("11111111-2222-3333-4444-555555555555")]


# ------------------------------------------------------------------ live training progress


async def test_run_events_stream_progress_then_done(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    async with connect_jobs(studio_settings.database_url or "") as (_, jobs):
        job = await jobs.create("train", {"version": "v1"})
        for step, loss in ((1, 2.1), (2, 1.7), (3, 1.4)):
            await jobs._conn.execute(
                "insert into training_progress (job_id, phase, step, total_steps, loss, gpu_util, gpu_mem_used_gb) "
                "values (%s, 'training', %s, 3, %s, 97.0, 18.5)",
                (job["id"], step, loss),
            )
        await jobs.finish(job["id"], status="succeeded", result={"train_loss": 1.4})

    async with make_client(studio_settings, r2, writer_then_designer([])) as client:
        async with client.stream("GET", f"/v1/runs/{job['id']}/events", headers=KEY) as response:
            events = await sse_events(response)
        polled = (await client.get(f"/v1/runs/{job['id']}/progress?after_id=0", headers=KEY)).json()
        resources = (await client.get("/v1/resources", headers=KEY)).json()

    names = [n for n, _ in events]
    assert names[0] == "status" and names[-1] == "done"
    losses = [p["loss"] for n, p in events if n == "progress"]
    assert losses == pytest.approx([2.1, 1.7, 1.4])
    assert len(polled) == 3 and polled[-1]["gpu_util"] == 97.0
    assert "inline_worker" in resources["functions"] and resources["active_jobs"] == []


async def test_progress_writer_is_best_effort() -> None:
    from pipeline.progress import ProgressWriter

    writer = ProgressWriter(None, None)  # no database: telemetry silently off
    writer.write(phase="training", step=1)
    assert writer.enabled is False


class StreamingScriptedModel(ScriptedModel):
    """Streams its reply in chunks, like ChatOpenAI(streaming=True) does."""

    def _stream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ):  # type: ignore[no-untyped-def]
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        reply = self.script(messages)
        if reply.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": c["name"],
                            "args": json.dumps(c["args"]),
                            "id": c["id"],
                            "index": i,
                        }
                        for i, c in enumerate(reply.tool_calls)
                    ],
                    response_metadata={"model_name": self.model_label},
                )
            )
            return
        for word in str(reply.content).split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"model_name": self.model_label})
        )


async def test_a_streaming_model_reply_is_not_doubled(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    def script(messages: list[BaseMessage]) -> AIMessage:
        return AIMessage("Here is a streamed answer in several tokens.")

    def models(_: Settings) -> AgentModels:
        return AgentModels(
            primary=StreamingScriptedModel(script=script, model_label="primary"), names=["primary"]
        )

    app = create_jobs_app(
        InlineDispatcher(studio_settings, r2), studio_settings, r2, agent_models_factory=models
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://studio") as client:
        thread = (await client.post("/v1/studio/threads", headers=KEY)).json()
        events = await chat(client, thread["id"], "hi")
    tokens = [p["text"] for n, p in events if n == "token"]
    reply = next(p for n, p in events if n == "message")["content"]
    assert len(tokens) > 3  # it really streamed
    assert reply == "Here is a streamed answer in several tokens."


async def test_a_streaming_model_can_call_tools(
    studio_settings: Settings, r2: Any, clean_db: None
) -> None:
    def models(_: Settings) -> AgentModels:
        return AgentModels(
            primary=StreamingScriptedModel(script=writer_then_designer([]), model_label="primary"),
            names=["primary"],
        )

    app = create_jobs_app(
        InlineDispatcher(studio_settings, r2),
        studio_settings,
        r2,
        agent_models_factory=models,
        writer_factory=default_writer,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://studio") as client:
        thread = (await client.post("/v1/studio/threads", headers=KEY)).json()
        events = await chat(client, thread["id"], "write it")
    assert [p["name"] for n, p in events if n == "tool_start"] == ["write_article", "design_page"]
    assert next(p for n, p in events if n == "message")["content"].startswith("Drafted")
