"""Settings resolution: env names, DATABASE_URL derivation, and precedence."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from app.core.config import Settings

REF = "abcdefghijklmnopqrst"
SUPABASE_URL = f"https://{REF}.supabase.co"


def _settings(tmp_path: Path, dotenv: str = "") -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(dotenv, encoding="utf-8")
    return Settings(_env_file=env_file)  # type: ignore[call-arg]


def test_database_url_is_built_for_the_session_pooler(tmp_path: Path) -> None:
    s = _settings(
        tmp_path,
        f"SUPABASE_URL={SUPABASE_URL}\nDB_PASSWORD=pw\nDB_HOST=aws-0-ap-northeast-2.pooler.supabase.com\n",
    )
    url = urlparse(s.database_url or "")
    assert url.hostname == "aws-0-ap-northeast-2.pooler.supabase.com"
    assert url.username == f"postgres.{REF}"  # the pooler routes on this
    assert url.port == 5432 and url.path == "/postgres"
    assert "sslmode=require" in (s.database_url or "")


def test_database_url_defaults_to_the_direct_host(tmp_path: Path) -> None:
    s = _settings(tmp_path, f"SUPABASE_URL={SUPABASE_URL}\nDB_PASSWORD=pw\n")
    url = urlparse(s.database_url or "")
    assert url.hostname == f"db.{REF}.supabase.co"
    assert url.username == "postgres"


def test_password_special_characters_cannot_change_the_host(tmp_path: Path) -> None:
    tricky = "p@ss:w/rd#?"
    s = _settings(tmp_path, f"SUPABASE_URL={SUPABASE_URL}\nDB_PASSWORD='{tricky}'\n")
    url = urlparse(s.database_url or "")
    assert url.hostname == f"db.{REF}.supabase.co"
    assert unquote(url.password or "") == tricky


def test_explicit_database_url_wins(tmp_path: Path) -> None:
    s = _settings(
        tmp_path,
        f"SUPABASE_URL={SUPABASE_URL}\nDB_PASSWORD=pw\nDATABASE_URL=postgresql://u:p@h:5432/d\n",
    )
    assert s.database_url == "postgresql://u:p@h:5432/d"


def test_no_database_url_without_a_password(tmp_path: Path) -> None:
    assert _settings(tmp_path, f"SUPABASE_URL={SUPABASE_URL}\n").database_url is None


def test_r2_names_from_the_project_env_are_understood(tmp_path: Path) -> None:
    s = _settings(
        tmp_path,
        "R2_JENOSIZE_BUCKET=jenosize-ai-content\n"
        "R2_API_ENDPOINT=https://acct.r2.cloudflarestorage.com\n",
    )
    assert s.r2_bucket == "jenosize-ai-content"
    assert s.r2_endpoint_url == "https://acct.r2.cloudflarestorage.com"


def test_r2_endpoint_with_a_pasted_bucket_path_is_trimmed(tmp_path: Path) -> None:
    s = _settings(
        tmp_path, "R2_API_ENDPOINT=https://acct.r2.cloudflarestorage.com/jenosize-ai-content\n"
    )
    assert s.r2_endpoint_url == "https://acct.r2.cloudflarestorage.com"


def test_generic_names_still_work(tmp_path: Path) -> None:
    s = _settings(tmp_path, "R2_BUCKET=b\nR2_ENDPOINT=https://e.example\n")
    assert s.r2_bucket == "b" and s.r2_endpoint_url == "https://e.example"


def test_project_env_file_beats_a_shell_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Observed on a real machine: ~/.zshrc exported R2_BUCKET / R2_ACCESS_KEY_ID
    # for another app, and they silently won over this repo's .env.
    monkeypatch.setenv("R2_BUCKET", "some-other-project")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "other-project-key")
    s = _settings(
        tmp_path, "R2_JENOSIZE_BUCKET=jenosize-ai-content\nR2_ACCESS_KEY_ID=this-project-key\n"
    )
    assert s.r2_bucket == "jenosize-ai-content"
    assert s.r2_access_key_id == "this-project-key"


def test_real_environment_applies_when_there_is_no_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Vercel / Modal: no .env, configuration comes from the environment.
    monkeypatch.setenv("R2_JENOSIZE_BUCKET", "from-deployment")
    assert Settings(_env_file=None).r2_bucket == "from-deployment"  # type: ignore[call-arg]


def test_fields_can_still_be_set_by_name_in_code() -> None:
    s = Settings(_env_file=None, r2_bucket="x", r2_endpoint="https://y.example")  # type: ignore[call-arg]
    assert s.r2_bucket == "x" and s.r2_endpoint_url == "https://y.example"
