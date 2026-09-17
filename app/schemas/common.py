"""Shared response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CheckStatus = Literal["ok", "error", "skipped"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, CheckStatus] = Field(
        description="Per-subsystem probe result: database, storage, model.",
    )


class WarmupResponse(BaseModel):
    status: Literal["ok", "error", "skipped"]
    latency_ms: int
    detail: str | None = None
