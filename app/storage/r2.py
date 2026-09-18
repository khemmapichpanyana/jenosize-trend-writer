"""Cloudflare R2 via boto3 (S3-compatible).

boto3 is synchronous and does its own connection pooling, so every call is
pushed onto a worker thread rather than blocking the event loop. The client is
built once per process: constructing one costs ~100 ms of botocore session
setup, which is painful on a cold Vercel lambda.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import anyio

from app.core.config import Settings
from app.core.errors import NotFoundError, UpstreamError


class R2Storage:
    name = "r2"

    def __init__(self, settings: Settings) -> None:
        missing = [
            key
            for key, value in {
                "R2_ACCOUNT_ID or R2_ENDPOINT": settings.r2_account_id or settings.r2_endpoint,
                "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
                "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"STORAGE=r2 requires {', '.join(missing)}")
        self._settings = settings
        self._bucket = settings.r2_bucket

    @cached_property
    def _client(self) -> Any:
        import boto3
        from botocore.config import Config

        return boto3.client(
            "s3",
            endpoint_url=self._settings.r2_endpoint_url,
            aws_access_key_id=self._settings.r2_access_key_id,
            aws_secret_access_key=self._settings.r2_secret_access_key,
            # R2 ignores the region but botocore insists on one; "auto" is what
            # Cloudflare documents.
            region_name="auto",
            config=Config(
                signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}
            ),
        )

    async def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        def _put() -> None:
            self._client.put_object(
                Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
            )

        try:
            await anyio.to_thread.run_sync(_put)
        except Exception as exc:
            raise UpstreamError(f"R2 put failed for {key}: {exc}") from exc
        return key

    async def get_bytes(self, key: str) -> bytes:
        def _get() -> bytes:
            obj = self._client.get_object(Bucket=self._bucket, Key=key)
            return bytes(obj["Body"].read())

        try:
            return await anyio.to_thread.run_sync(_get)
        except self._client.exceptions.NoSuchKey as exc:
            raise NotFoundError(f"object not found: {key}") from exc
        except Exception as exc:
            raise UpstreamError(f"R2 get failed for {key}: {exc}") from exc

    async def exists(self, key: str) -> bool:
        def _head() -> bool:
            try:
                self._client.head_object(Bucket=self._bucket, Key=key)
                return True
            except Exception:
                return False

        return await anyio.to_thread.run_sync(_head)

    async def health(self) -> None:
        def _head_bucket() -> None:
            self._client.head_bucket(Bucket=self._bucket)

        try:
            await anyio.to_thread.run_sync(_head_bucket)
        except Exception as exc:
            raise UpstreamError(f"R2 unreachable: {exc}") from exc
