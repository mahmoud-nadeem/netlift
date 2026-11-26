"""Single-customer and batch scoring endpoints."""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections import Counter
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import ValidationError

from netlift.api.dependencies import require_ready
from netlift.api.models import APIErrorResponse, ScoreRequest
from netlift.api.schema import FIELDS, ScoringResponse
from netlift.api.service import NetLiftService

router = APIRouter(tags=["scoring"])

SCORE_HEADERS: dict[int | str, dict[str, Any]] = {
    422: {
        "model": APIErrorResponse,
        "description": "Invalid or forbidden scoring input.",
    },
    503: {"model": APIErrorResponse, "description": "Scoring service is not ready."},
}
BATCH_RESERVED_COLUMNS = {"customer_id", "contact_cost", "conversion_margin"}


@router.post(
    "/score",
    response_model=ScoringResponse,
    responses=SCORE_HEADERS,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            {"$ref": "#/components/schemas/ScoreRequest"},
                            {
                                "type": "object",
                                "description": (
                                    "Flat form: reserved fields plus numeric feature "
                                    "columns."
                                ),
                                "required": [
                                    "customer_id",
                                    "contact_cost",
                                    "conversion_margin",
                                ],
                                "properties": {
                                    "customer_id": {"type": "string"},
                                    "contact_cost": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                    },
                                    "conversion_margin": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                    },
                                },
                                "additionalProperties": {"type": "number"},
                            },
                        ]
                    }
                }
            },
        }
    },
    summary="Score one customer",
    description=(
        "Returns the frozen T36 eleven-field scoring response. The expected profit is "
        "calculated using the request's contact_cost and conversion_margin; those "
        "assumptions are echoed in response headers. T45 currently returns "
        "deterministic mocks."
    ),
)
async def score_customer(
    payload: ScoreRequest,
    response: Response,
    service: Annotated[NetLiftService, Depends(require_ready)],
) -> ScoringResponse:
    result = service.score(payload)
    response.headers["X-NetLift-Contact-Cost"] = _number_header(payload.contact_cost)
    response.headers["X-NetLift-Conversion-Margin"] = _number_header(
        payload.conversion_margin
    )
    response.headers["X-NetLift-Mock"] = "true"
    return result


@router.post(
    "/score/batch",
    response_class=Response,
    responses={
        200: {
            "description": "ZIP archive containing scores.csv and summary.json.",
            "content": {
                "application/zip": {"schema": {"type": "string", "format": "binary"}}
            },
        },
        400: {"model": APIErrorResponse, "description": "Malformed CSV or JSON input."},
        415: {"model": APIErrorResponse, "description": "Unsupported content type."},
        422: {
            "model": APIErrorResponse,
            "description": "Invalid row or inconsistent economics.",
        },
        503: {
            "model": APIErrorResponse,
            "description": "Scoring service is not ready.",
        },
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "oneOf": [
                                {"$ref": "#/components/schemas/ScoreRequest"},
                                {
                                    "type": "object",
                                    "description": (
                                        "Flat row: reserved fields plus numeric feature "
                                        "columns."
                                    ),
                                    "required": [
                                        "customer_id",
                                        "contact_cost",
                                        "conversion_margin",
                                    ],
                                    "properties": {
                                        "customer_id": {"type": "string"},
                                        "contact_cost": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                        },
                                        "conversion_margin": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                        },
                                    },
                                    "additionalProperties": {"type": "number"},
                                },
                            ]
                        },
                    }
                },
                "text/csv": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "CSV with customer_id, contact_cost, conversion_margin and "
                            "one or more numeric feature columns."
                        ),
                    }
                },
            },
        }
    },
    summary="Score a batch of customers",
    description=(
        "Accepts a JSON array or raw text/csv. Every row must carry the same positive "
        "contact_cost and conversion_margin. Returns a ZIP with scores.csv and "
        "summary.json. "
        "Validation is atomic: one invalid row rejects the entire batch."
    ),
    tags=["scoring"],
)
async def score_batch(
    request: Request,
    service: Annotated[NetLiftService, Depends(require_ready)],
) -> Response:
    content_type = (
        request.headers.get("content-type", "")
        .split(";", maxsplit=1)[0]
        .strip()
        .lower()
    )
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Request body must not be empty.")

    if content_type == "application/json":
        rows = _parse_json_rows(body)
    elif content_type == "text/csv":
        rows = _parse_csv_rows(body)
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/json or text/csv.",
        )

    requests = [_validate_row(row, index) for index, row in enumerate(rows)]
    _require_shared_economics(requests)

    # Do not score any row until every input row has passed validation.
    results = [service.score(row) for row in requests]
    contact_cost = requests[0].contact_cost
    conversion_margin = requests[0].conversion_margin
    summary = {
        "row_count": len(results),
        "model_version": results[0].model_version,
        "verdict_counts": dict(Counter(item.verdict for item in results)),
        "campaign_economics": {
            "contact_cost": contact_cost,
            "conversion_margin": conversion_margin,
            "profit_formula": "conversion_margin * uplift_score - contact_cost",
        },
        "total_expected_incremental_profit": sum(
            item.expected_incremental_profit for item in results
        ),
        "is_mock": True,
        "message": "Synthetic mock scores; not measured model output.",
    }
    archive = _build_result_archive(results, summary)
    return Response(
        content=archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="netlift-scores.zip"',
            "X-NetLift-Mock": "true",
        },
    )


def _parse_json_rows(body: bytes) -> list[dict[str, Any]]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON: {error}."
        ) from error
    if not isinstance(payload, list) or not payload:
        raise HTTPException(
            status_code=400, detail="JSON batch input must be a non-empty array."
        )
    if any(not isinstance(row, dict) for row in payload):
        raise HTTPException(
            status_code=400, detail="Every JSON batch row must be an object."
        )
    return payload


def _parse_csv_rows(body: bytes) -> list[dict[str, Any]]:
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400, detail="CSV input must be UTF-8 encoded."
        ) from error

    reader = csv.reader(io.StringIO(text))
    try:
        headers = next(reader)
    except StopIteration as error:
        raise HTTPException(
            status_code=400, detail="CSV input must include a header row."
        ) from error
    if not headers or any(not name.strip() for name in headers):
        raise HTTPException(status_code=400, detail="CSV headers must be non-empty.")
    if len(headers) != len(set(headers)):
        raise HTTPException(
            status_code=400, detail="CSV contains duplicate column names."
        )
    missing = sorted(BATCH_RESERVED_COLUMNS - set(headers))
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {missing}.",
        )
    if not (set(headers) - BATCH_RESERVED_COLUMNS):
        raise HTTPException(
            status_code=400, detail="CSV must contain at least one feature column."
        )

    rows: list[dict[str, Any]] = []
    for row_number, values in enumerate(reader, start=2):
        if len(values) != len(headers):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"CSV row {row_number} has {len(values)} cells; "
                    f"expected {len(headers)}."
                ),
            )
        raw = dict(zip(headers, values, strict=True))
        converted: dict[str, Any] = {
            "customer_id": raw["customer_id"],
            "contact_cost": _parse_csv_number(
                raw["contact_cost"], "contact_cost", row_number
            ),
            "conversion_margin": _parse_csv_number(
                raw["conversion_margin"], "conversion_margin", row_number
            ),
            "features": {
                name: _parse_csv_feature(value)
                for name, value in raw.items()
                if name not in BATCH_RESERVED_COLUMNS
            },
        }
        rows.append(converted)
    if not rows:
        raise HTTPException(
            status_code=400, detail="CSV batch input must contain at least one row."
        )
    return rows


def _parse_csv_number(value: str, field: str, row_number: int) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=f"CSV row {row_number}: {field} must be numeric.",
        ) from error


def _parse_csv_feature(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _validate_row(row: dict[str, Any], index: int) -> ScoreRequest:
    try:
        return ScoreRequest.model_validate(row)
    except ValidationError as error:
        details = [
            {
                "location": [
                    f"rows[{index}]",
                    *(str(part) for part in item.get("loc", ())),
                ],
                "message": str(item.get("msg", "Invalid value.")),
                "type": str(item.get("type", "value_error")),
            }
            for item in error.errors()
        ]
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Batch row {index} failed validation; no rows were scored.",
                "details": details,
            },
        ) from error


def _require_shared_economics(requests: list[ScoreRequest]) -> None:
    costs = {row.contact_cost for row in requests}
    margins = {row.conversion_margin for row in requests}
    if len(costs) != 1 or len(margins) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "All rows in a batch must use the same contact_cost and "
                "conversion_margin campaign assumptions. No rows were scored."
            ),
        )


def _build_result_archive(
    results: list[ScoringResponse], summary: dict[str, Any]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    for result in results:
        row = result.to_dict()
        row["uplift_ci_95"] = json.dumps(row["uplift_ci_95"], separators=(",", ":"))
        row["top_drivers"] = json.dumps(row["top_drivers"], separators=(",", ":"))
        writer.writerow(row)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("scores.csv", output.getvalue())
        bundle.writestr("summary.json", json.dumps(summary, indent=2, sort_keys=True))
    return archive.getvalue()


def _number_header(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Campaign economics must be finite.")
    return format(value, ".12g")
