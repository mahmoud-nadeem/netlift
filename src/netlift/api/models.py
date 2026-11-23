"""Pydantic transport models for the T45 HTTP contract.

The scoring response itself remains the frozen T36 dataclass in ``schema.py``.
These models describe API requests and non-scoring endpoint payloads only.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from netlift.data.contracts import validate_features
from netlift.utils.errors import ContractError


class StrictAPIModel(BaseModel):
    """Base for transport models that reject undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class ScoreRequest(StrictAPIModel):
    """Single-customer scoring request.

    ``features`` contains preprocessed numeric covariates only. The campaign
    economics are request inputs (T45 resolution of T33 Q4).
    """

    customer_id: str = Field(
        min_length=1, description="Stable row/customer identifier."
    )
    features: dict[str, float] = Field(
        min_length=1,
        description=(
            "Preprocessed numeric features; treatment/outcome columns are forbidden."
        ),
    )
    contact_cost: float = Field(gt=0, allow_inf_nan=False)
    conversion_margin: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def collect_inline_features(cls, value: object) -> object:
        """Accept either a nested ``features`` map or flat feature columns."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        reserved = {"customer_id", "features", "contact_cost", "conversion_margin"}
        inline = {name: data.pop(name) for name in tuple(data) if name not in reserved}
        if inline:
            existing = data.get("features")
            if existing is None:
                data["features"] = inline
            elif isinstance(existing, dict):
                overlap = sorted(set(existing) & set(inline))
                if overlap:
                    raise ValueError(f"Feature columns were supplied twice: {overlap}.")
                data["features"] = {**existing, **inline}
            else:
                raise ValueError(
                    "features must be an object when supplied with flat columns."
                )
        return data

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("customer_id must not be blank.")
        return value

    @field_validator("features", mode="before")
    @classmethod
    def validate_feature_values(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise ValueError("features must be a non-empty object of numeric values.")
        converted: dict[str, float] = {}
        for name, raw in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Feature names must be non-empty strings.")
            if not isinstance(raw, bool | int | float):
                raise ValueError(f"Feature '{name}' must be numeric.")  # noqa: TRY004
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"Feature '{name}' must be finite.")
            converted[name] = number
        return converted

    @model_validator(mode="after")
    def validate_feature_contract(self) -> ScoreRequest:
        generic_outcome = [
            name for name in self.features if name.strip().lower() == "outcome"
        ]
        if generic_outcome:
            raise ValueError(
                f"Forbidden column(s) in the feature matrix: {generic_outcome}."
            )
        try:
            validate_features(pd.DataFrame([self.features]))
        except ContractError as error:
            raise ValueError(str(error)) from error
        return self


class PolicySimulationRequest(StrictAPIModel):
    """Inputs that uniquely identify a validation-split what-if simulation."""

    scenario_id: str = Field(
        min_length=1, description="Scenario identifier for this what-if request."
    )
    data_slice: Literal["validation"] = Field(
        default="validation",
        description=(
            "T45 interim rule: replays use validation only while Q1 remains open."
        ),
    )
    model_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    budget: float = Field(gt=0, allow_inf_nan=False)
    contact_cost: float = Field(gt=0, allow_inf_nan=False)
    conversion_margin: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_one_contact(self) -> PolicySimulationRequest:
        if self.budget < self.contact_cost:
            raise ValueError("budget must cover at least one contact_cost.")
        return self


class MetricEstimate(StrictAPIModel):
    """One metric and its confidence interval; values are illustrative in mock mode."""

    name: str
    value: float | None
    ci_95: tuple[float, float] | None
    status: Literal["illustrative_mock", "unavailable"] = "illustrative_mock"

    @model_validator(mode="after")
    def validate_interval(self) -> MetricEstimate:
        if self.ci_95 is not None and self.ci_95[0] > self.ci_95[1]:
            raise ValueError("ci_95 lower bound must not exceed upper bound.")
        return self


class ModelInfoResponse(StrictAPIModel):
    model_version: str
    training_date: date | None
    metrics: list[MetricEstimate]
    source: Literal["mock"] = "mock"
    is_mock: Literal[True] = True
    message: str


class FeatureDrift(StrictAPIModel):
    feature: str
    psi: float | None
    ks_statistic: float | None
    status: Literal["illustrative_mock", "unavailable"] = "illustrative_mock"


class EffectDrift(StrictAPIModel):
    metric: str
    current_estimate: float
    reference_estimate: float
    difference: float
    ci_95: tuple[float, float]
    source: Literal["mock_replay_output"] = "mock_replay_output"


class ScoreDistribution(StrictAPIModel):
    mean: float
    standard_deviation: float
    p05: float
    median: float
    p95: float
    status: Literal["illustrative_mock"] = "illustrative_mock"


class DataQualityStatus(StrictAPIModel):
    status: Literal["illustrative_mock", "unavailable"]
    max_null_rate: float | None
    range_violations: int | None
    schema_changes: int | None


class DriftReportResponse(StrictAPIModel):
    status: Literal["available", "not_available"]
    generated_at: datetime | None
    model_version: str | None
    feature_drift: list[FeatureDrift]
    score_drift: ScoreDistribution | None
    effect_drift: EffectDrift | None
    data_quality: DataQualityStatus | None
    is_mock: Literal[True] = True
    message: str


class CurvePoint(StrictAPIModel):
    fraction_targeted: float = Field(ge=0, le=1)
    cumulative_incremental_conversions: float
    ci_95: tuple[float, float]


class PolicyResult(StrictAPIModel):
    policy: Literal["uplift", "response_model", "random", "blanket"]
    contacted: int = Field(ge=0)
    total_cost: float
    incremental_conversions: float
    incremental_conversions_ci_95: tuple[float, float]
    incremental_profit: float
    profit_per_unit_spent: float | None
    curve: list[CurvePoint]


class PolicyDifference(StrictAPIModel):
    comparison: Literal["uplift_vs_response_model"] = "uplift_vs_response_model"
    incremental_conversions_difference: float
    ci_95: tuple[float, float]
    conclusion: Literal["uplift_better", "response_model_better", "not_distinguishable"]


class PolicySimulationResponse(StrictAPIModel):
    simulation_id: str
    scenario_id: str
    data_slice: Literal["validation"]
    model_version: str
    seed: int
    budget: float
    contact_cost: float
    conversion_margin: float
    policies: list[PolicyResult]
    comparison: PolicyDifference
    recorded_run_created: Literal[False] = False
    source: Literal["mock"] = "mock"
    is_mock: Literal[True] = True
    message: str


class HealthResponse(StrictAPIModel):
    status: Literal["ready", "not_ready"]
    ready: bool
    mode: Literal["mock"] = "mock"
    model_loaded: Literal[False] = False
    message: str


class APIErrorDetail(StrictAPIModel):
    code: str
    message: str
    details: list[dict[str, object]] | None = None


class APIErrorResponse(StrictAPIModel):
    error: APIErrorDetail
