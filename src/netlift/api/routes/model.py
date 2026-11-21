"""Model metadata endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from netlift.api.dependencies import require_ready
from netlift.api.models import APIErrorResponse, ModelInfoResponse
from netlift.api.service import NetLiftService

router = APIRouter(prefix="/model", tags=["model"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    responses={
        503: {"model": APIErrorResponse, "description": "Model service is not ready."}
    },
    summary="Get active model metadata and metrics",
)
async def model_info(
    service: Annotated[NetLiftService, Depends(require_ready)],
) -> ModelInfoResponse:
    return service.model_info()
