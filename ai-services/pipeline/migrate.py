"""Apply `supabase/migrations/*.sql` to any Postgres via DATABASE_URL.

    uv run python -m pipeline.migrate up
    uv run python -m pipeline.migrate status

A tiny runner rather than the Supabase CLI so the same command works against
Supabase, Neon, RDS or a local Postgres, with nothing to install. Applied files
are recorded in `schema_migrations`; each file runs in its own transaction, so a
failure leaves the database at the last good migration.

Use either this or `supabase db push` for a given database, not both: they keep
separate bookkeeping and would each try to re-apply the other's work.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import typer

from app.core.config import get_settings

app = typer.Typer(help="Apply SQL migrations to DATABASE_URL.")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "supabase" / "migrations"


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def apply_migrations(dsn: str) -> list[str]:
    """Apply every pending migration in filename order; return what was applied."""
    applied_now: list[str] = []
    with psycopg.connect(dsn, prepare_threshold=None) as conn:
        conn.execute(
            "create table if not exists schema_migrations ("
            " version text primary key, applied_at timestamptz not null default now())"
        )
        # Same posture as every other table: on Supabase, a public table without
        # RLS is readable *and writable* by the anon key through the REST API.
        conn.execute("alter table schema_migrations enable row level security")
        conn.commit()
        done = {row[0] for row in conn.execute("select version from schema_migrations")}
        for path in migration_files():
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
                conn.execute("insert into schema_migrations (version) values (%s)", (path.name,))
            applied_now.append(path.name)
    return applied_now


def _dsn() -> str:
    dsn = get_settings().database_url
    if not dsn:
        typer.echo("DATABASE_URL is not set (see .env.example)")
        raise typer.Exit(code=1)
    return dsn


@app.command()
def up() -> None:
    """Apply all pending migrations."""
    applied = apply_migrations(_dsn())
    if not applied:
        typer.echo("database is up to date")
    for name in applied:
        typer.echo(f"  applied {name}")


@app.command()
def status() -> None:
    """List migrations and whether each has been applied."""
    with psycopg.connect(_dsn(), prepare_threshold=None) as conn:
        exists = conn.execute("select to_regclass('schema_migrations')").fetchone()
        done = (
            {row[0] for row in conn.execute("select version from schema_migrations")}
            if exists and exists[0]
            else set()
        )
    for path in migration_files():
        typer.echo(f"  {'✓' if path.name in done else '·'} {path.name}")


if __name__ == "__main__":
    app()
