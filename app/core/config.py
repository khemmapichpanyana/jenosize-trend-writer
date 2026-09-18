"""Application settings.

Every deployment target (laptop, Vercel preview, Vercel production) is configured
through the same env vars; the only thing that changes is which *backend* each
pluggable subsystem resolves to. That keeps "runs locally with zero cloud
accounts" a first-class supported mode rather than an afterthought.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelProvider = Literal["mock", "openai_compatible"]
Persistence = Literal["none", "supabase"]
StorageBackend = Literal["local", "r2"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_*` is a protected namespace in pydantic v2; we deliberately use
        # MODEL_PROVIDER / MODEL_BASE_URL because those names read better in the
        # deployment dashboards, so the protection is disabled here.
        protected_namespaces=(),
    )

    # --- app -----------------------------------------------------------------
    app_env: str = "local"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    public_base_url: str = "http://localhost:8000"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])
    api_key: str | None = None

    # --- model ---------------------------------------------------------------
    model_provider: ModelProvider = "mock"
    model_base_url: str | None = None
    model_api_key: str | None = None
    model_name: str = "jeno-lora"
    model_timeout_s: float = 240.0

    # --- persistence ---------------------------------------------------------
    persistence: Persistence = "none"
    # Direct Postgres connection used by the data pipeline (psycopg). The API
    # keeps using SUPABASE_URL + key; both can point at the same database.
    database_url: str | None = None
    supabase_url: str | None = None
    supabase_secret_key: str | None = None

    # --- storage -------------------------------------------------------------
    storage: StorageBackend = "local"
    r2_account_id: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket: str = "jenosize-trend-writer"
    # Optional override: point the S3 client at MinIO/moto for tests, or at a
    # jurisdiction-specific R2 endpoint. Normally derived from R2_ACCOUNT_ID.
    r2_endpoint: str | None = None
    local_storage_dir: str = ".data"

    # --- offline labeling (pipeline only, unused by the API) -----------------
    labeler_base_url: str | None = None
    labeler_api_key: str | None = None
    labeler_model: str | None = None

    # --- limits --------------------------------------------------------------
    max_upload_bytes: int = 10 * 1024 * 1024
    retrieval_top_k: int = 6

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        """Accept the comma-separated form that dashboards/`.env` files use."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def r2_endpoint_url(self) -> str | None:
        if self.r2_endpoint:
            return self.r2_endpoint
        if not self.r2_account_id:
            return None
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"


@lru_cache
def get_settings() -> Settings:
    """Cached so the env is parsed once per process (Vercel reuses warm lambdas)."""
    return Settings()
