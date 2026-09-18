"""Application settings.

Every deployment target (laptop, Vercel preview, Vercel production) is configured
through the same env vars; the only thing that changes is which *backend* each
pluggable subsystem resolves to. That keeps "runs locally with zero cloud
accounts" a first-class supported mode rather than an afterthought.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import quote, urlparse

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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
        # Fields with env aliases (below) can still be set by field name in code.
        populate_by_name=True,
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
    supabase_url: str | None = None
    # Server-side key (`sb_secret_…`); bypasses RLS, never sent to a browser.
    supabase_secret_key: str | None = None

    # Postgres connection used by the data pipeline (psycopg). Set DB_URL (or
    # DATABASE_URL) outright, or let it be built from SUPABASE_URL + DB_PASSWORD
    # (+ DB_HOST) — see `_derive_database_url`. An explicit URL always wins.
    database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DB_URL", "DATABASE_URL", "database_url"),
    )
    db_password: str | None = None
    # Supabase's direct host (db.<ref>.supabase.co) is IPv6-only on the free
    # plan; Modal containers and many networks are IPv4-only. Set DB_HOST to the
    # project's *session pooler* (Dashboard -> Connect -> Session pooler), e.g.
    # aws-0-ap-northeast-2.pooler.supabase.com, to work everywhere.
    db_host: str | None = None
    db_port: int = 5432
    db_name: str = "postgres"
    db_user: str | None = None

    # --- storage -------------------------------------------------------------
    storage: StorageBackend = "local"
    r2_account_id: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket: str = Field(
        default="jenosize-trend-writer",
        validation_alias=AliasChoices("R2_JENOSIZE_BUCKET", "R2_BUCKET", "r2_bucket"),
    )
    # The S3 API endpoint. Cloudflare shows it as "S3 API" on the bucket page;
    # derived from R2_ACCOUNT_ID when unset. Also how tests point at moto.
    r2_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("R2_API_ENDPOINT", "R2_ENDPOINT", "r2_endpoint"),
    )
    local_storage_dir: str = ".data"

    # --- jobs API (modal/jobs.py) ------------------------------------------
    # Shared secret for the scrape/train endpoints. They spend money (GPU time,
    # labelling calls), so the jobs API refuses every request when this is unset.
    jobs_api_key: str | None = None
    # Where trained adapters are mounted (the Modal volume `jeno-models`).
    models_dir: str = "/models"

    # --- studio agent (studio/, served by the jobs API on Modal) --------------
    # The agent's *orchestrating* model. It runs on the same Modal vLLM server as
    # the fine-tuned writer (MODEL_BASE_URL) but uses the base weights: the LoRA
    # is trained on the article output format, not on tool calling. The writing
    # itself goes through the fine-tuned model (MODEL_NAME) via write_article.
    agent_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    # Optional fallback when the GPU is cold or down (any OpenAI-compatible
    # endpoint). Defaults to the labelling LLM when unset.
    agent_fallback_base_url: str | None = None
    agent_fallback_api_key: str | None = None
    agent_fallback_model: str | None = None
    # Base URL for shared links, e.g. the console's domain (which proxies /p/*).
    # Unset: links point at this API itself.
    public_share_base_url: str | None = None

    # --- offline labeling (pipeline only, unused by the API) -----------------
    labeler_base_url: str | None = None
    labeler_api_key: str | None = None
    labeler_model: str | None = None

    # --- limits --------------------------------------------------------------
    max_upload_bytes: int = 10 * 1024 * 1024
    retrieval_top_k: int = 6

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """This project's `.env` beats variables exported in the shell.

        The pydantic default is the reverse, which is right for deployments but
        dangerous on a laptop: generic names such as R2_BUCKET or
        R2_ACCESS_KEY_ID exported globally for another project would silently
        redirect this service's writes into that project's bucket. (Observed:
        a shell-exported R2_BUCKET for a different app won over this repo's
        .env.) Vercel and Modal have no `.env` file, so there the real
        environment still applies.
        """
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        """Accept the comma-separated form that dashboards/`.env` files use."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def _derive_database_url(self) -> Settings:
        """Build DATABASE_URL from the Supabase pieces when it isn't given.

        The password is percent-encoded: a raw '@' or '/' in it would otherwise
        silently change the host the URL points at. The pooler requires the
        user to be `postgres.<project-ref>`; the direct host wants `postgres`.
        """
        if self.database_url or not (self.supabase_url and self.db_password):
            return self
        ref = (urlparse(self.supabase_url).hostname or "").split(".")[0]
        if not ref:
            return self
        host = self.db_host or f"db.{ref}.supabase.co"
        user = self.db_user or (f"postgres.{ref}" if "pooler.supabase.com" in host else "postgres")
        self.database_url = (
            f"postgresql://{quote(user, safe='')}:{quote(self.db_password, safe='')}"
            f"@{host}:{self.db_port}/{self.db_name}?sslmode=require"
        )
        return self

    @property
    def r2_endpoint_url(self) -> str | None:
        if self.r2_endpoint:
            # Cloudflare's dashboard sometimes shows the endpoint with the
            # bucket appended; boto3 wants the bare origin.
            parsed = urlparse(self.r2_endpoint)
            return f"{parsed.scheme}://{parsed.netloc}"
        if not self.r2_account_id:
            return None
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"


@lru_cache
def get_settings() -> Settings:
    """Cached so the env is parsed once per process (Vercel reuses warm lambdas)."""
    return Settings()
