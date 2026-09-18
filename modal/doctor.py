"""Pre-flight check: does everything the GPU and pipeline jobs need work on Modal?

    make modal-doctor          # = modal run modal/doctor.py

Runs in one small CPU container (fractions of a cent) and checks, from inside
Modal: this repo's modules import, the secrets are set, Postgres is reachable
and migrated, and the R2 bucket is reachable. Run it before the first training
job, so a missing secret costs seconds rather than a GPU cold start.
"""

from __future__ import annotations

from common import app, app_image, pipeline_secret, r2_secret


@app.function(image=app_image, secrets=[r2_secret, pipeline_secret], timeout=120)
def doctor() -> list[tuple[str, bool, str]]:
    import importlib
    import os

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:  # type: ignore[no-untyped-def]
        try:
            checks.append((name, True, str(fn() or "ok")))
        except Exception as exc:
            checks.append((name, False, f"{type(exc).__name__}: {exc}"[:200]))

    for module in ("common", "app.main", "pipeline.api", "pipeline.jobs"):
        check(f"import {module}", lambda m=module: importlib.import_module(m) and "ok")

    from app.core.config import Settings

    settings = Settings()
    resolved = {
        "database (DB_URL / DATABASE_URL, or SUPABASE_URL + DB_PASSWORD)": settings.database_url,
        "JOBS_API_KEY": settings.jobs_api_key,
        "R2 credentials": settings.r2_access_key_id and settings.r2_secret_access_key,
        "R2 endpoint (R2_API_ENDPOINT or R2_ACCOUNT_ID)": settings.r2_endpoint_url,
    }
    for label, value in resolved.items():
        checks.append((f"config {label}", bool(value), "set" if value else "missing"))
    for var in ("LABELER_BASE_URL", "LABELER_MODEL"):
        present = bool(os.environ.get(var))
        checks.append(
            (f"env {var} (needed for labelling)", present, "set" if present else "missing")
        )

    def postgres() -> str:
        import psycopg

        with psycopg.connect(
            settings.database_url or "", prepare_threshold=None, connect_timeout=10
        ) as conn:
            applied = [
                r[0] for r in conn.execute("select version from schema_migrations order by 1")
            ]
        return f"migrations applied: {', '.join(applied)}"

    def r2() -> str:
        from app.storage.r2 import R2Storage

        storage = R2Storage(settings)
        storage._client.head_bucket(Bucket=settings.r2_bucket)
        return f"bucket {settings.r2_bucket} reachable"

    check("postgres", postgres)
    check("r2", r2)
    return checks


@app.local_entrypoint()
def doctor_main() -> None:
    results = doctor.remote()
    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name:<{width}}  {detail}")
    if not all(ok for name, ok, _ in results if "needed for labelling" not in name):
        raise SystemExit(1)
