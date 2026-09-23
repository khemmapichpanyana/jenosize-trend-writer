"""Upload a reference document to ground a later generation."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.deps import ConfigDep, RepositoryDep, StorageDep, require_api_key
from app.core.errors import PayloadTooLargeError, ValidationError
from app.core.logging import get_logger
from app.schemas.sources import UploadResponse
from app.services import ingest

router = APIRouter(prefix="/sources", tags=["sources"])
logger = get_logger(__name__)


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
async def upload_source(
    storage: StorageDep,
    repository: RepositoryDep,
    settings: ConfigDep,
    file: Annotated[UploadFile, File(description="pdf, docx, txt or md, max 10 MB")],
) -> UploadResponse:
    data = await file.read()
    # Read-then-check rather than streaming: Vercel already caps the request body
    # well below this, so the memory exposure is bounded either way.
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit"
        )
    if not data:
        raise ValidationError("Uploaded file is empty")

    document = await ingest.ingest_upload(file.filename or "upload.txt", data, storage)
    if not document.text.strip():
        raise ValidationError("No text could be extracted from this file")

    await repository.upsert_source_document(
        {
            "id": str(document.id),
            "kind": "upload",
            "r2_key": document.r2_key,
            "content_hash": ingest.content_hash(data),
            "title": document.title,
            "text_chars": document.text_chars,
            # The extracted text is kept on the row so retrieval does not need a
            # second object-store round-trip on every generation.
            "metadata": {"text": document.text},
        }
    )

    logger.info(
        "source_uploaded", extra={"source_id": str(document.id), "chars": document.text_chars}
    )
    return UploadResponse(
        source_id=document.id,
        title=document.title or "upload",
        text_chars=document.text_chars,
    )
