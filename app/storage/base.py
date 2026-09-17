"""Object-storage seam.

Same reason as `LLMProvider`: the services write markdown and read documents
without knowing whether they land in `./.data/` or Cloudflare R2.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    name: str

    async def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        """Store `data` at `key` and return the key."""
        ...

    async def get_bytes(self, key: str) -> bytes: ...

    async def exists(self, key: str) -> bool: ...

    async def health(self) -> None:
        """Raise if the backend is unreachable."""
        ...


async def put_text(
    storage: Storage, key: str, text: str, *, content_type: str = "text/plain; charset=utf-8"
) -> str:
    return await storage.put_bytes(key, text.encode("utf-8"), content_type=content_type)
