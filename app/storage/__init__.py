from __future__ import annotations

from app.core.config import Settings
from app.storage.base import Storage, put_text
from app.storage.local import LocalStorage

__all__ = ["Storage", "build_storage", "put_text"]


def build_storage(settings: Settings) -> Storage:
    if settings.storage == "r2":
        from app.storage.r2 import R2Storage  # imported lazily so boto3 stays off the hot path

        return R2Storage(settings)
    return LocalStorage(settings.local_storage_dir)
