"""FastAPI application factory for NetLift's six-endpoint API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from netlift.api.champion_service import service_from_environment
from netlift.api.errors import (
    contract_error_handler,
    http_exception_handler,
    request_validation_handler,
)
from netlift.api.routes.health import router as health_router
from netlift.api.routes.model import router as model_router
from netlift.api.routes.monitoring import router as monitoring_router
from netlift.api.routes.policy import router as policy_router
from netlift.api.routes.scoring import router as scoring_router
from netlift.api.service import NetLiftService
from netlift.utils.errors import ContractError


def create_app(service: NetLiftService | None = None) -> FastAPI:
    """Create an app with an injectable adapter for tests and production wiring.

    Without an explicit adapter, the champion is loaded from ``NETLIFT_MODEL_DIR``
    when it is set (T59), and the T45 mock is used otherwise.
    """
    application = FastAPI(
        title="NetLift API",
        version="0.1.0",
        description=(
            "API-first interface for NetLift. T45 ships deterministic mock responses; "
            "they are not predictions, measured results, or drift calculations."
        ),
        contact={"name": "NetLift project team"},
    )
    application.state.netlift_service = service or service_from_environment()
    application.include_router(scoring_router)
    application.include_router(policy_router)
    application.include_router(model_router)
    application.include_router(monitoring_router)
    application.include_router(health_router)
    application.add_exception_handler(
        RequestValidationError, request_validation_handler
    )
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(ContractError, contract_error_handler)
    return application


app = create_app()
