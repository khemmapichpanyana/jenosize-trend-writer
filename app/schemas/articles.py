"""Request/response contract for article generation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator

Length = Literal["short", "medium", "long"]
Language = Literal["en", "th"]
GenerationStatus = Literal["queued", "running", "succeeded", "failed"]

# Target word counts per length bucket, used for prompting and quality checks.
TARGET_WORDS: dict[str, int] = {"short": 600, "medium": 1000, "long": 1500}


class ArticleRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=200)
    category: str | None = None
    industry: str | None = None
    audience: str | None = None
    keywords: list[str] = Field(default_factory=list)
    source_url: HttpUrl | None = None
    source_ids: list[UUID] = Field(default_factory=list)
    length: Length = "medium"
    language: Language = "en"

    @field_validator("keywords", mode="before")
    @classmethod
    def _accept_csv(cls, v: object) -> object:
        """Form posts and quick curl calls send "a, b, c" rather than a JSON list."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "Agentic AI in Southeast Asian retail",
                "category": "Futurist",
                "industry": "ecommerce",
                "audience": "C-suite executives",
                "keywords": ["agentic ai", "retail", "personalization"],
                "length": "medium",
                "language": "en",
            }
        }
    }


class NormalizedParams(BaseModel):
    """`ArticleRequest` after cleaning; this is what the prompt builder consumes."""

    topic: str
    category: str | None = None
    industry: str | None = None
    audience: str = "Business leaders"
    keywords: list[str] = Field(default_factory=list)
    length: Length = "medium"
    language: Language = "en"
    target_words: int = TARGET_WORDS["medium"]


class SourceRef(BaseModel):
    source_id: UUID | None = None
    url: str | None = None
    title: str | None = None
    score: float = 0.0


class QualityReport(BaseModel):
    keyword_coverage: float = Field(ge=0.0, le=1.0)
    word_count: int
    heading_count: int
    title_length: int
    passed: bool
    warnings: list[str] = Field(default_factory=list)


class ArticleResponse(BaseModel):
    id: UUID
    status: GenerationStatus
    title: str | None = None
    meta_description: str | None = None
    article_markdown: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    quality_report: QualityReport | None = None
    model_version: str | None = None
    latency_ms: int | None = None
    created_at: datetime | None = None
    error: str | None = None


class ArticleSummary(BaseModel):
    """Row shape for `GET /articles` — no markdown body, keeps the list cheap."""

    id: UUID
    status: GenerationStatus
    title: str | None = None
    created_at: datetime | None = None
    latency_ms: int | None = None
