"""Corpus records shared by every pipeline stage.

One row per article in `training_articles`. Each stage writes its own columns
and records the content fingerprint it worked from, which is how the next run
knows whether there is anything left to do:

    discover -> url, category_slug
    crawl    -> content_hash (fingerprint), r2_raw_key, title, meta_description
    clean    -> clean_markdown, word_count, cleaned_hash, clean_version
    label    -> labels, labelled_hash, label_version, labeler_model
    build    -> split
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.articles import Language, Length

Split = Literal["train", "eval"]


class ArticleLabels(BaseModel):
    """The brief a reader would have written to commission this article.

    `category` is not inferred: it comes free from the URL path, so the model is
    conditioned on the publisher's own taxonomy rather than an LLM's guess.
    """

    topic: str
    category: str | None = None
    industry: str | None = None
    audience: str | None = None
    keywords: list[str] = Field(default_factory=list)
    length: Length = "medium"
    language: Language = "en"


class TrainingArticle(BaseModel):
    url: str
    category_slug: str | None = None
    title: str | None = None
    meta_description: str | None = None
    r2_raw_key: str | None = None
    content_hash: str | None = None
    clean_markdown: str | None = None
    word_count: int = 0
    is_duplicate: bool = False
    labels: ArticleLabels | None = None
    split: Split = "train"
    error: str | None = None
    first_seen_at: datetime | None = None
    last_checked_at: datetime | None = None
    content_changed_at: datetime | None = None
    cleaned_hash: str | None = None
    clean_version: int | None = None
    labelled_hash: str | None = None
    label_version: int | None = None
    labeler_model: str | None = None
