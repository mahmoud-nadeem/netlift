"""Liveness/readiness endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from netlift.api.dependencies import get_service
from netlift.api.models import HealthResponse
from netlift.api.service import NetLiftService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API liveness and readiness",
)
async def health(
    service: Annotated[NetLiftService, Depends(get_service)],
) -> HealthResponse:
    return service.health()
