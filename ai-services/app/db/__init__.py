from __future__ import annotations

from app.core.config import Settings
from app.db.base import Repository
from app.db.null import NullRepository

__all__ = ["Repository", "build_repository"]


def build_repository(settings: Settings) -> Repository:
    if settings.persistence == "supabase":
        from app.db.supabase_repo import SupabaseRepository  # lazy: keeps supabase off cold start

        return SupabaseRepository(settings)
    return NullRepository()
