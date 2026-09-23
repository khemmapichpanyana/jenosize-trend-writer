"""One error shape for the whole API: {"error": {code, message, request_id}}.

Clients (the demo page, graders' curl, future SDKs) should never have to branch
on FastAPI's default `{"detail": ...}` for validation errors and something else
for ours.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for errors we raise deliberately and can map to a status code."""

    code = "internal_error"
    status_code = 500

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class ValidationError(AppError):
    code = "validation_error"
    status_code = 422


class NotFoundError(AppError):
    code = "not_found"
    status_code = 404


class UnauthorizedError(AppError):
    code = "unauthorized"
    status_code = 401


class UpstreamError(AppError):
    """The model endpoint, database or object store failed us."""

    code = "upstream_error"
    status_code = 502


class PayloadTooLargeError(AppError):
    code = "payload_too_large"
    status_code = 413


def error_body(code: str, message: str) -> dict[str, dict[str, str | None]]:
    return {"error": {"code": code, "message": message, "request_id": request_id_ctx.get()}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        # 5xx is our fault and deserves a stack trace; 4xx is the caller's.
        log = logger.exception if exc.status_code >= 500 else logger.warning
        log("app_error", extra={"code": exc.code, "status": exc.status_code, "detail": exc.message})
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
        message = f"{loc}: {first.get('msg', 'invalid request')}"
        return JSONResponse(status_code=422, content=error_body("validation_error", message))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "http_error"
        )
        return JSONResponse(status_code=exc.status_code, content=error_body(code, str(exc.detail)))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=500, content=error_body("internal_error", "Internal server error")
        )
