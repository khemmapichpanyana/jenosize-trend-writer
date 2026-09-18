"""Jobs API end to end, in-process: real Postgres, S3-compatible R2, fake site.

The production dispatcher is Modal; here `InlineDispatcher` runs the very same
`run_job` code as asyncio tasks, and GPU work goes to a fake runner that records
what it was asked to do.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from pipeline import label
from pipeline.api import InlineDispatcher, create_jobs_app
from pipeline.store import connect_jobs
from tests.conftest import r2_keys
from tests.test_pipeline_incremental import FakeLabeler, FakeSite, site  # noqa: F401

KEY = {"X-API-Key": "test-key"}


class FakeGpu:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((kind, params))
        if kind == "train":
            return {"train_loss": 1.23, "steps": 10}
        return {"summary": {"n_briefs": 2}, "results": ["large", "payload"]}


@pytest.fixture
def settings(pg_dsn: str, r2: Any) -> Settings:  # overrides the API fixture of the same name
    return r2._settings.model_copy(
        update={
            "database_url": pg_dsn,
            "jobs_api_key": "test-key",
            "labeler_base_url": "http://labeler.invalid/v1",
            "labeler_model": "fake-model",
            "log_level": "WARNING",
        }
    )


@pytest.fixture
def gpu() -> FakeGpu:
    return FakeGpu()


@pytest.fixture
def dispatcher(settings: Settings, r2: Any, gpu: FakeGpu) -> InlineDispatcher:
    return InlineDispatcher(settings, r2, gpu_runner=gpu)


@pytest.fixture
async def api(
    corpus: Any, settings: Settings, dispatcher: InlineDispatcher, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    # `corpus` truncates the tables; job_runs too, since tests share one database.
    async with connect_jobs(settings.database_url or "") as (_, _jobs):
        await _jobs._conn.execute("truncate job_runs cascade")
    monkeypatch.setattr(label, "labeler_client", lambda *_: FakeLabeler())
    monkeypatch.setattr("pipeline.scrape.asyncio.sleep", _no_sleep)
    app = create_jobs_app(dispatcher, settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://jobs") as client:
        yield client


async def _no_sleep(_: float) -> None:
    return None


async def _finish(api: AsyncClient, dispatcher: InlineDispatcher, response: Any) -> dict[str, Any]:
    assert response.status_code == 202, response.text
    await dispatcher.drain()
    run = await api.get(f"/v1/runs/{response.json()['id']}", headers=KEY)
    return dict(run.json())


# ----------------------------------------------------------------- auth


async def test_health_is_open_and_reports_configuration(api: AsyncClient) -> None:
    body = (await api.get("/health")).json()
    assert body["status"] == "ok"
    assert body["configured"]["jobs_api_key"] is True


async def test_requests_without_the_key_are_refused(api: AsyncClient) -> None:
    assert (await api.post("/v1/scrape")).status_code == 401
    assert (await api.post("/v1/scrape", headers={"X-API-Key": "wrong"})).status_code == 401


async def test_fails_closed_when_no_key_is_configured(
    settings: Settings, dispatcher: InlineDispatcher
) -> None:
    app = create_jobs_app(dispatcher, settings.model_copy(update={"jobs_api_key": None}))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://jobs") as client:
        response = await client.post("/v1/scrape", headers=KEY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_configured"


# ----------------------------------------------------------------- scrape


async def test_scrape_job_pulls_articles_into_postgres_and_r2(
    api: AsyncClient,
    dispatcher: InlineDispatcher,
    r2: Any,
    site: FakeSite,  # noqa: F811
) -> None:
    run = await _finish(api, dispatcher, await api.post("/v1/scrape", headers=KEY, json={}))

    assert run["status"] == "succeeded", run
    assert [s["stage"] for s in run["stages"]] == ["discover", "crawl", "clean"]
    assert run["result"]["crawl"]["new"] == 3
    assert run["result"]["clean"]["cleaned"] == 3
    assert len(r2_keys(r2)) == 3

    corpus = (await api.get("/v1/corpus", headers=KEY)).json()
    assert corpus["counts"]["cleaned"] == 3


async def test_a_second_scrape_writes_nothing(
    api: AsyncClient,
    dispatcher: InlineDispatcher,
    r2: Any,
    site: FakeSite,  # noqa: F811
) -> None:
    await _finish(api, dispatcher, await api.post("/v1/scrape", headers=KEY))
    before = r2_keys(r2)
    run = await _finish(api, dispatcher, await api.post("/v1/scrape", headers=KEY))
    assert run["result"]["discover"]["new"] == 0
    assert run["result"]["crawl"]["checked"] == 0  # known pages are not re-fetched
    assert r2_keys(r2) == before


async def test_scrape_params_are_validated(api: AsyncClient) -> None:
    # The delay floor keeps the API from being usable to hammer the site.
    response = await api.post("/v1/scrape", headers=KEY, json={"delay_s": 0.1})
    assert response.status_code == 422


# ----------------------------------------------------------------- one active job per kind


class _StuckDispatcher(InlineDispatcher):
    async def poll(self, call_id: str) -> tuple[Any, str | None]:
        return "running", None


async def test_a_second_job_of_the_same_kind_is_refused_while_one_runs(
    corpus: Any, settings: Settings, r2: Any
) -> None:
    async with connect_jobs(settings.database_url or "") as (_, jobs):
        await jobs._conn.execute("truncate job_runs cascade")
        row = await jobs.create("scrape", {})
        await jobs.set_call_id(row["id"], "call-still-running")

    app = create_jobs_app(_StuckDispatcher(settings, r2), settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://jobs") as client:
        response = await client.post("/v1/scrape", headers=KEY)
    assert response.status_code == 409
    assert response.json()["error"]["run_id"] == str(row["id"])


async def test_a_crashed_worker_does_not_block_new_jobs(
    api: AsyncClient,
    settings: Settings,
    site: FakeSite,  # noqa: F811
) -> None:
    # A job row left 'running' by a worker that died (its call is gone).
    async with connect_jobs(settings.database_url or "") as (_, jobs):
        dead = await jobs.create("scrape", {})
        await jobs.set_call_id(dead["id"], "call-that-no-longer-exists")
        await jobs.mark_running(dead["id"])

    response = await api.post("/v1/scrape", headers=KEY)
    assert response.status_code == 202
    old = (await api.get(f"/v1/runs/{dead['id']}", headers=KEY)).json()
    assert old["status"] == "failed"
    assert "without recording a result" in old["error"]


# ----------------------------------------------------------------- label + datasets


async def _scraped_and_labelled(api: AsyncClient, dispatcher: InlineDispatcher) -> None:
    await _finish(api, dispatcher, await api.post("/v1/scrape", headers=KEY))
    run = await _finish(api, dispatcher, await api.post("/v1/label", headers=KEY))
    assert run["status"] == "succeeded", run
    assert run["result"]["labelled"] == 3


async def test_publish_dataset_then_unchanged(
    api: AsyncClient,
    dispatcher: InlineDispatcher,
    r2: Any,
    site: FakeSite,  # noqa: F811
) -> None:
    await _scraped_and_labelled(api, dispatcher)

    first = await api.post("/v1/datasets", headers=KEY, json={"version": "v1", "eval_frac": 0.0})
    assert first.status_code == 201, first.text
    assert first.json()["status"] == "published"
    assert "datasets/v1/train.jsonl" in r2_keys(r2)

    again = await api.post("/v1/datasets", headers=KEY, json={"version": "v1", "eval_frac": 0.0})
    assert again.status_code == 200 and again.json()["status"] == "unchanged"

    listed = (await api.get("/v1/datasets", headers=KEY)).json()
    assert [d["version"] for d in listed] == ["v1"]


async def test_dataset_versions_must_look_like_v_n(api: AsyncClient) -> None:
    response = await api.post("/v1/datasets", headers=KEY, json={"version": "final-final"})
    assert response.status_code == 422


# ----------------------------------------------------------------- fine-tuning


async def test_train_requires_a_published_dataset(api: AsyncClient) -> None:
    response = await api.post("/v1/train", headers=KEY, json={"version": "v9"})
    assert response.status_code == 404


async def test_train_job_runs_on_the_gpu_with_versioned_paths(
    api: AsyncClient,
    dispatcher: InlineDispatcher,
    gpu: FakeGpu,
    site: FakeSite,  # noqa: F811
) -> None:
    await _scraped_and_labelled(api, dispatcher)
    await api.post("/v1/datasets", headers=KEY, json={"version": "v1", "eval_frac": 0.0})

    run = await _finish(
        api,
        dispatcher,
        await api.post("/v1/train", headers=KEY, json={"version": "v1", "epochs": 2}),
    )
    assert run["status"] == "succeeded", run
    assert run["result"] == {"train_loss": 1.23, "steps": 10}

    [(kind, params)] = gpu.calls
    assert kind == "train"
    assert params["dataset_uri"] == "r2://datasets/v1/train.jsonl"
    assert params["adapter_dir"] == "/models/jeno-lora-v1"  # v2 can never overwrite v1
    assert params["epochs"] == 2


async def test_eval_job_keeps_only_the_summary(
    api: AsyncClient,
    dispatcher: InlineDispatcher,
    gpu: FakeGpu,
    site: FakeSite,  # noqa: F811
) -> None:
    await _scraped_and_labelled(api, dispatcher)
    await api.post("/v1/datasets", headers=KEY, json={"version": "v1", "eval_frac": 0.0})

    run = await _finish(
        api,
        dispatcher,
        await api.post(
            "/v1/eval", headers=KEY, json={"version": "v1", "endpoint": "https://x.modal.run/v1"}
        ),
    )
    assert run["result"] == {"summary": {"n_briefs": 2}}
    assert gpu.calls[-1][1]["briefs_uri"] == "r2://datasets/v1/eval.jsonl"


async def test_gpu_jobs_fail_cleanly_without_a_gpu_runner(
    corpus: Any, settings: Settings, r2: Any
) -> None:
    async with connect_jobs(settings.database_url or "") as (store, jobs):
        await jobs._conn.execute("truncate job_runs cascade")
        await store.record_dataset_version(
            version="v1",
            fingerprint="f",
            train_count=1,
            eval_count=0,
            train_key="t",
            eval_key="e",
            card_key="c",
            params={},
        )
    no_gpu = InlineDispatcher(settings, r2, gpu_runner=None)
    app = create_jobs_app(no_gpu, settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://jobs") as client:
        response = await client.post("/v1/train", headers=KEY, json={"version": "v1"})
        await no_gpu.drain()
        run = (await client.get(f"/v1/runs/{response.json()['id']}", headers=KEY)).json()
    assert run["status"] == "failed"
    assert "run on Modal GPUs" in run["error"]


# ----------------------------------------------------------------- runs


async def test_cancel_marks_the_run_cancelled(api: AsyncClient, settings: Settings) -> None:
    async with connect_jobs(settings.database_url or "") as (_, jobs):
        row = await jobs.create("label", {})
    response = await api.post(f"/v1/runs/{row['id']}/cancel", headers=KEY)
    assert response.json()["status"] == "cancelled"


async def test_unknown_run_is_404(api: AsyncClient) -> None:
    response = await api.get("/v1/runs/00000000-0000-0000-0000-000000000000", headers=KEY)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
