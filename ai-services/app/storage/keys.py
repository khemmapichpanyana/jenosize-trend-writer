"""R2 / local object key conventions.

Keys are date- and hash-partitioned so that (a) a re-scrape of the same page is
idempotent, and (b) a whole dataset version can be listed with one prefix.
Centralised here because the pipeline, the API and the eval jobs all write into
the same bucket and must agree.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import UUID

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, *, max_len: int = 120) -> str:
    """Collapse anything that would break an S3 key or a shell into '-'."""
    cleaned = _UNSAFE.sub("-", name.strip()).strip("-._") or "file"
    return cleaned[:max_len]


def raw_scrape_key(content_hash: str, *, when: datetime | None = None) -> str:
    day = (when or datetime.now(UTC)).strftime("%Y-%m-%d")
    return f"raw/scrape/{day}/{content_hash}.html"


def upload_key(source_id: UUID | str, filename: str) -> str:
    return f"uploads/{source_id}/{safe_filename(filename)}"


def dataset_key(version: str, filename: str) -> str:
    """e.g. dataset_key("v1", "train.jsonl") -> datasets/v1/train.jsonl"""
    return f"datasets/{version}/{filename}"


def generation_key(generation_id: UUID | str) -> str:
    return f"generations/{generation_id}.md"


def eval_key(run_id: UUID | str) -> str:
    return f"eval/{run_id}/results.json"
