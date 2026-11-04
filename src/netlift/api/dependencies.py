"""FastAPI dependencies shared by the route modules."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from netlift.api.service import NetLiftService


def get_service(request: Request) -> NetLiftService:
    """Return the service adapter attached to the application instance."""
    return request.app.state.netlift_service


def require_ready(
    service: Annotated[NetLiftService, Depends(get_service)],
) -> NetLiftService:
    """Block business endpoints when the service adapter is not ready."""
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "The scoring service is not ready; no score or simulation was produced."
            ),
        )
    return service
