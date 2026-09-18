"""Create/update the Modal secrets from this repo's .env — no copy-pasting keys.

    make modal-secrets            # = uv run python scripts/modal_secrets.py

Each secret gets only the keys its functions need (least privilege): the R2
secret never sees the database password, and nothing on Modal ever receives
SUPABASE_PUBLISHABLE_KEY or the Cloudflare account API token, which no job uses.
Values are written to a private temp file for `modal secret create
--from-dotenv` and never printed or passed on the command line.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# secret name -> (required keys, optional keys). "a|b" = at least one of a, b.
SECRETS: dict[str, tuple[list[str], list[str]]] = {
    "jeno-r2": (
        [
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_JENOSIZE_BUCKET|R2_BUCKET",
            "R2_API_ENDPOINT|R2_ENDPOINT|R2_ACCOUNT_ID",
        ],
        ["R2_ACCOUNT_ID", "R2_API_ENDPOINT", "R2_ENDPOINT", "R2_JENOSIZE_BUCKET", "R2_BUCKET"],
    ),
    "jeno-pipeline": (
        ["DATABASE_URL|DB_PASSWORD", "JOBS_API_KEY"],
        [
            "DATABASE_URL",
            "SUPABASE_URL",
            "DB_PASSWORD",
            "DB_HOST",
            "DB_PORT",
            "DB_USER",
            "LABELER_BASE_URL",
            "LABELER_API_KEY",
            "LABELER_MODEL",
        ],
    ),
    "jeno-hf": (["HF_TOKEN"], []),
    "jeno-vllm": (["VLLM_API_KEY"], []),
}


def main() -> int:
    if not ENV_FILE.exists():
        print(f"no {ENV_FILE}")
        return 1
    env = {k: v for k, v in dotenv_values(ENV_FILE).items() if v}
    failed = False
    for name, (required, optional) in SECRETS.items():
        missing = [r for r in required if not any(k in env for k in r.split("|"))]
        if (
            "DB_PASSWORD" in env
            and "DATABASE_URL" not in env
            and "SUPABASE_URL" not in env
            and name == "jeno-pipeline"
        ):
            missing.append("SUPABASE_URL (needed with DB_PASSWORD)")
        if missing:
            print(f"  · {name}: skipped, .env is missing {', '.join(missing)}")
            failed = failed or name in ("jeno-r2", "jeno-pipeline")
            continue
        keys = sorted(
            {k for r in required for k in r.split("|") if k in env}
            | {k for k in optional if k in env}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secret.env"
            path.touch(mode=0o600)
            path.write_text("".join(f"{k}={env[k]}\n" for k in keys), encoding="utf-8")
            result = subprocess.run(
                ["modal", "secret", "create", name, "--from-dotenv", str(path), "--force"],
                capture_output=True,
                text=True,
                env=os.environ,
            )
        if result.returncode == 0:
            print(f"  ✓ {name}: {', '.join(keys)}")
        else:
            failed = True
            print(
                f"  ✗ {name}: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'failed'}"
            )
    if failed:
        print("\nFix the .env entries above and re-run. JOBS_API_KEY: openssl rand -hex 24")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
