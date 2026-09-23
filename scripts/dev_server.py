"""Run the local API using PORT from .env (default: 8000).

Pass ``--with-jobs`` for the assignment-friendly one-process setup. The local
Jobs API is then available below ``/jobs`` and runs scrape/label jobs in-process.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import uvicorn
from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.main import create_app


def build_app(settings: Settings, *, with_jobs: bool = False) -> FastAPI:
    """Build the article app, optionally mounting the local Jobs API."""
    application = create_app(settings)
    if with_jobs:
        from app.storage.r2 import R2Storage
        from pipeline.api.app import create_jobs_app
        from pipeline.api.dispatch import InlineDispatcher

        storage = R2Storage(settings)
        jobs_app = create_jobs_app(InlineDispatcher(settings, storage), settings, storage)
        article_schema = application.openapi()
        jobs_schema = jobs_app.openapi()
        application.mount("/jobs", jobs_app)
        # Mounted apps have their own OpenAPI document. Merge a prefixed copy
        # into the parent document so the main /docs page exposes both APIs.
        application.openapi_schema = _merge_openapi(article_schema, jobs_schema)
    return application


def _rewrite_refs(value: Any, mapping: dict[str, str]) -> Any:
    """Prefix component references from the mounted Jobs API."""
    if isinstance(value, dict):
        return {key: _rewrite_refs(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_refs(item, mapping) for item in value]
    if isinstance(value, str):
        for source, target in mapping.items():
            value = value.replace(
                f"#/components/schemas/{source}", f"#/components/schemas/{target}"
            )
    return value


def _merge_openapi(article: dict[str, Any], jobs: dict[str, Any]) -> dict[str, Any]:
    """Add Jobs API paths under /jobs without schema-name collisions."""
    merged = deepcopy(article)
    job_schemas = jobs.get("components", {}).get("schemas", {})
    mapping = {name: f"Jobs{name}" for name in job_schemas}
    merged.setdefault("components", {}).setdefault("schemas", {}).update(
        {mapping[name]: _rewrite_refs(schema, mapping) for name, schema in job_schemas.items()}
    )
    merged.setdefault("paths", {}).update(
        {
            f"/jobs{path}": _rewrite_refs(path_item, mapping)
            for path, path_item in jobs.get("paths", {}).items()
        }
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-jobs",
        action="store_true",
        help="mount the local scraping/jobs API below /jobs in this process",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.with_jobs:
        application = build_app(settings, with_jobs=True)
        print(
            f"Combined API: http://127.0.0.1:{settings.port} "
            f"(combined docs: /docs, standalone jobs docs: /jobs/docs)",
            flush=True,
        )
        # Passing an app object cannot use Uvicorn's reload supervisor. The
        # one-process assignment mode is intentionally stable until restart.
        uvicorn.run(application, host="127.0.0.1", port=settings.port, reload=False)
        return

    # Keep autoreload for the lightweight article-only development mode.
    uvicorn.run("app.main:app", host="127.0.0.1", port=settings.port, reload=True)


if __name__ == "__main__":
    main()
