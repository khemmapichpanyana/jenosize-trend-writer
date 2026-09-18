"""Corpus records shared by every pipeline stage.

One model flows through all four stages, gaining fields as it goes:

    scrape  -> url, category_slug, title, meta_description, raw_key, content_hash
    clean   -> clean_markdown, word_count, is_duplicate
    label   -> labels (the reverse-engineered brief)
    build   -> split
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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
    raw_key: str | None = None
    content_hash: str | None = None
    clean_markdown: str | None = None
    word_count: int = 0
    is_duplicate: bool = False
    labels: ArticleLabels | None = None
    split: Split = "train"
    error: str | None = None
    fetched_at: datetime | None = None

    def merged(self, **fields: Any) -> TrainingArticle:
        """Copy with `fields` applied — stages never mutate a record in place."""
        return self.model_copy(update=fields)
