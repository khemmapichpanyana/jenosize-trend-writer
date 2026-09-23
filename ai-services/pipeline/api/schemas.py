"""Request/response models for the jobs API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from pipeline.jobs import VERSION_PATTERN, JobKind
from pipeline.schemas import ArticleLabels


class StageRun(BaseModel):
    stage: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JobRun(BaseModel):
    id: UUID
    kind: JobKind
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    params: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    stages: list[StageRun] | None = None


class ProgressPoint(BaseModel):
    id: int
    created_at: datetime
    phase: str
    step: int | None = None
    total_steps: int | None = None
    epoch: float | None = None
    loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    samples_per_sec: float | None = None
    gpu_util: float | None = None
    gpu_mem_used_gb: float | None = None
    gpu_mem_total_gb: float | None = None
    message: str | None = None


class CorpusStatus(BaseModel):
    counts: dict[str, int]
    recent_stages: list[StageRun]


class ArticleSummary(BaseModel):
    url: str
    category_slug: str | None = None
    title: str | None = None
    word_count: int = 0
    is_duplicate: bool = False
    labelled: bool = False
    error: str | None = None
    last_checked_at: datetime | None = None


class ArticlePage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ArticleSummary]


class ArticleDetail(ArticleSummary):
    meta_description: str | None = None
    r2_raw_key: str | None = None
    raw_url: str | None = Field(
        None, description="Temporary signed link to the archived HTML (the bucket stays private)."
    )
    content_hash: str | None = None
    clean_markdown: str | None = None
    labels: ArticleLabels | None = None
    split: str | None = None
    content_changed_at: datetime | None = None


class LabelPreviewRequest(BaseModel):
    url: str | None = Field(None, description="Article to label; default: the next unlabelled one.")


class LabelPreview(BaseModel):
    url: str
    title: str | None
    labels: ArticleLabels
    excerpt: str
    written: bool = False


class DatasetRequest(BaseModel):
    version: str = Field("v1", pattern=VERSION_PATTERN)
    eval_frac: float = Field(0.1, ge=0.0, lt=0.5)
    seed: int = 13
    overwrite: bool = Field(False, description="Replace an existing version (only if untrained).")
    quality_filter: bool = Field(
        False,
        description=(
            "Curate training rows against the production output contract; "
            "held-out eval URLs remain unchanged."
        ),
    )


class DatasetVersion(BaseModel):
    version: str
    created_at: datetime
    fingerprint: str
    train_count: int
    eval_count: int
    train_key: str
    eval_key: str
    card_key: str
    params: dict[str, Any]


class DatasetValidation(BaseModel):
    version: str
    train: int
    eval: int
    ok: bool
    problems: list[str]


class Adapter(BaseModel):
    version: str
    served_as: str
    active: bool
    metrics: dict[str, Any] = Field(default_factory=dict)


class Migration(BaseModel):
    name: str
    applied: bool


class DoctorCheck(BaseModel):
    check: str
    ok: bool
    detail: str
