"""Consistent error response helpers for the FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from netlift.api.models import APIErrorDetail, APIErrorResponse
from netlift.utils.errors import ContractError


def _error_payload(
    code: str, message: str, details: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return APIErrorResponse(
        error=APIErrorDetail(code=code, message=message, details=details)
    ).model_dump(exclude_none=True)


async def request_validation_handler(
    _request: Request, error: Exception
) -> JSONResponse:
    if not isinstance(error, RequestValidationError):
        raise error
    details = [
        {
            "location": [str(part) for part in item.get("loc", ())],
            "message": str(item.get("msg", "Invalid value.")),
            "type": str(item.get("type", "value_error")),
        }
        for item in error.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            "request_validation_error", "Request validation failed.", details
        ),
    )


async def http_exception_handler(_request: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, StarletteHTTPException):
        raise error
    message = str(error.detail)
    code = {
        400: "bad_request",
        404: "not_found",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "invalid_request",
        503: "service_not_ready",
    }.get(error.status_code, "http_error")
    details = error.detail if isinstance(error.detail, list) else None
    if isinstance(error.detail, dict):
        message = str(error.detail.get("message", message))
        details = error.detail.get("details")
    return JSONResponse(
        status_code=error.status_code,
        content=_error_payload(code, message, details),
        headers=error.headers,
    )


async def contract_error_handler(_request: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, ContractError):
        raise error
    return JSONResponse(
        status_code=422,
        content=_error_payload("contract_error", str(error)),
    )
