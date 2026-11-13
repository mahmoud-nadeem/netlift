"""Drift report endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from netlift.api.dependencies import require_ready
from netlift.api.models import APIErrorResponse, DriftReportResponse
from netlift.api.service import NetLiftService

router = APIRouter(prefix="/monitor", tags=["monitoring"])


@router.get(
    "/drift",
    response_model=DriftReportResponse,
    responses={
        503: {
            "model": APIErrorResponse,
            "description": "Monitoring service is not ready.",
        }
    },
    summary="Get the latest drift report",
    description=(
        "Returns the latest available report. When no report exists, returns HTTP 200 "
        "with status=not_available and empty report fields so the dashboard can render "
        "explicit empty state. T45 values are synthetic mocks."
    ),
)
async def drift_report(
    service: Annotated[NetLiftService, Depends(require_ready)],
) -> DriftReportResponse:
    return service.drift_report()
