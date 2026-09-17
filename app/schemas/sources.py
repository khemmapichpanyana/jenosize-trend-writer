"""Source-document schemas (uploads and fetched URLs)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

SourceKind = Literal["url", "upload", "scrape"]


class SourceDocument(BaseModel):
    """A retrievable document: extracted plain text plus provenance."""

    id: UUID
    kind: SourceKind
    title: str | None = None
    url: str | None = None
    r2_key: str | None = None
    text: str = ""
    text_chars: int = 0


class UploadResponse(BaseModel):
    source_id: UUID
    title: str
    text_chars: int
