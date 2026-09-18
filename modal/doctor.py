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

    for var in (
        "DATABASE_URL",
        "JOBS_API_KEY",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    ):
        check(
            f"env {var}",
            lambda v=var: "set" if os.environ.get(v) else (_ for _ in ()).throw(KeyError(v)),
        )
    for var in ("LABELER_BASE_URL", "LABELER_MODEL"):
        checks.append(
            (
                f"env {var} (needed for labelling)",
                bool(os.environ.get(var)),
                "set" if os.environ.get(var) else "missing",
            )
        )

    def postgres() -> str:
        import psycopg

        with psycopg.connect(
            os.environ["DATABASE_URL"], prepare_threshold=None, connect_timeout=10
        ) as conn:
            applied = [
                r[0] for r in conn.execute("select version from schema_migrations order by 1")
            ]
        return f"migrations applied: {', '.join(applied)}"

    def r2() -> str:
        from app.core.config import Settings
        from app.storage.r2 import R2Storage

        storage = R2Storage(Settings())
        storage._client.head_bucket(Bucket=storage._bucket)
        return f"bucket {storage._bucket} reachable"

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
