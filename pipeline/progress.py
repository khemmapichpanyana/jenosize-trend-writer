"""Live training telemetry: written by the GPU job, read by the console.

The training container calls `ProgressWriter.write()` from a Hugging Face
TrainerCallback on every logged step. Rows land in `training_progress`, keyed by
the jobs-API run id, and `GET /v1/runs/{id}/events` streams them to the browser.

Telemetry must never be able to fail a training run: every write is best-effort,
and the writer disables itself after repeated errors instead of retrying forever
on the GPU's clock.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FIELDS = (
    "phase",
    "step",
    "total_steps",
    "epoch",
    "loss",
    "learning_rate",
    "grad_norm",
    "samples_per_sec",
    "gpu_util",
    "gpu_mem_used_gb",
    "gpu_mem_total_gb",
    "message",
)
MAX_ERRORS = 5


class ProgressWriter:
    def __init__(self, dsn: str | None, job_id: str | None) -> None:
        self._job_id = job_id
        self._conn: Any = None
        self._errors = 0
        if not (dsn and job_id):
            return  # a CLI run without a job: nothing to report to
        try:
            import psycopg

            self._conn = psycopg.connect(
                dsn, autocommit=True, prepare_threshold=None, connect_timeout=10
            )
        except Exception as exc:  # telemetry is optional
            logger.warning("progress writer disabled: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def write(self, **fields: Any) -> None:
        if self._conn is None:
            return
        row = {k: fields.get(k) for k in FIELDS}
        try:
            self._conn.execute(
                f"insert into training_progress (job_id, {', '.join(FIELDS)}) "
                f"values (%(job_id)s, {', '.join(f'%({k})s' for k in FIELDS)})",
                {"job_id": self._job_id, **row},
            )
        except Exception as exc:
            self._errors += 1
            logger.warning("progress write failed (%s/%s): %s", self._errors, MAX_ERRORS, exc)
            if self._errors >= MAX_ERRORS:
                self.close()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


def gpu_snapshot() -> dict[str, float]:
    """Utilisation and memory of GPU 0, or {} off-GPU. Uses NVML (nvidia-ml-py)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": float(util.gpu),
            "gpu_mem_used_gb": round(mem.used / 1024**3, 2),
            "gpu_mem_total_gb": round(mem.total / 1024**3, 2),
        }
    except Exception:
        return {}
