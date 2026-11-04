"""Policy simulation endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from netlift.api.dependencies import require_ready
from netlift.api.models import (
    APIErrorResponse,
    PolicySimulationRequest,
    PolicySimulationResponse,
)
from netlift.api.service import NetLiftService

router = APIRouter(prefix="/policy", tags=["policy"])


@router.post(
    "/simulate",
    response_model=PolicySimulationResponse,
    responses={
        422: {"model": APIErrorResponse, "description": "Invalid simulation inputs."},
        503: {
            "model": APIErrorResponse,
            "description": "Simulation service is not ready.",
        },
    },
    summary="Run a what-if policy simulation",
    description=(
        "Accepts the inputs for a shared-replay what-if request and returns the T45 mock "
        "response shape. The production adapter must call the shared replay code. Until "
        "Q1 is decided, data_slice must be "
        "validation. This endpoint is a deterministic mock: it does not open the locked "
        "outcomes, or create an MLflow recorded run."
    ),
)
async def simulate_policy(
    payload: PolicySimulationRequest,
    service: Annotated[NetLiftService, Depends(require_ready)],
) -> PolicySimulationResponse:
    return service.simulate(payload)
