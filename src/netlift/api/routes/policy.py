"""Policy simulation endpoint."""

from __future__ import annotations

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
        "validation. The mock compares uplift, response-model and random policies at "
        "equal k (the same budget-derived contact count); blanket contact is shown as "
        "a full-population reference. Measured production differences require a paired "
        "bootstrap on shared within-arm resamples. The current deterministic mock does "
        "not calculate a real bootstrap interval, open locked outcomes, or create an "
        "MLflow recorded run."
    ),
)
async def simulate_policy(
    payload: PolicySimulationRequest,
    service: NetLiftService = Depends(require_ready),
) -> PolicySimulationResponse:
    return service.simulate(payload)
