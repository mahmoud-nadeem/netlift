"""Deterministic mock service used by the T45 FastAPI skeleton.

No model inference, MLflow writes, recorded replay, or real drift calculation
happens here. The class is intentionally replaceable by production adapters.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Literal, Protocol, TypedDict

from netlift.api.models import (
    CurvePoint,
    DataQualityStatus,
    DriftReportResponse,
    EffectDrift,
    FeatureDrift,
    HealthResponse,
    MetricEstimate,
    ModelInfoResponse,
    PolicyDifference,
    PolicyResult,
    PolicySimulationRequest,
    PolicySimulationResponse,
    ScoreDistribution,
    ScoreRequest,
)
from netlift.api.schema import (
    Driver,
    Recommendation,
    ScoringResponse,
    SegmentLabel,
    Verdict,
)

logger = logging.getLogger(__name__)

MOCK_MODEL_VERSION = "netlift-mock-2026.11.04"
MOCK_TIMESTAMP = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


class NetLiftService(Protocol):
    """Adapter contract required by the FastAPI route layer."""

    @property
    def ready(self) -> bool: ...

    def health(self) -> HealthResponse: ...

    def score(self, request: ScoreRequest) -> ScoringResponse: ...

    def model_info(self) -> ModelInfoResponse: ...

    def drift_report(self) -> DriftReportResponse: ...

    def simulate(
        self, request: PolicySimulationRequest
    ) -> PolicySimulationResponse: ...


class MockScoreScenario(TypedDict):
    uplift: float
    interval: tuple[float, float]
    decile: int
    segment: SegmentLabel
    recommendation: Recommendation
    verdict: Verdict


class MockNetLiftService:
    """Return stable synthetic payloads while the real scoring stack is built."""

    def __init__(self, *, ready: bool = True, drift_available: bool = True) -> None:
        self.ready = ready
        self.drift_available = drift_available

    def health(self) -> HealthResponse:
        status: Literal["ready", "not_ready"] = "ready" if self.ready else "not_ready"
        message = (
            "The API mock adapter is ready; no champion model is loaded."
            if self.ready
            else "The scoring service is not ready. No score can be returned."
        )
        return HealthResponse(status=status, ready=self.ready, message=message)

    def score(self, request: ScoreRequest) -> ScoringResponse:
        """Return a mock scenario selected by a reserved customer-id prefix."""
        case = request.customer_id.split(":", maxsplit=1)[0]
        scenarios: dict[str, MockScoreScenario] = {
            "mock-insufficient-evidence": {
                "uplift": 0.0001,
                "interval": (-0.0012, 0.0015),
                "decile": 5,
                "segment": "unknown",
                "recommendation": "no_action",
                "verdict": "insufficient_evidence",
            },
            "mock-insufficient-control-data": {
                "uplift": 0.0041,
                "interval": (0.0009, 0.0074),
                "decile": 1,
                "segment": "unknown",
                "recommendation": "no_action",
                "verdict": "insufficient_control_data",
            },
            "mock-sleeping-dog": {
                "uplift": -0.006,
                "interval": (-0.009, -0.002),
                "decile": 10,
                "segment": "sleeping_dog",
                "recommendation": "suppress",
                "verdict": "confident",
            },
        }
        default_scenario: MockScoreScenario = {
            "uplift": 0.0041,
            "interval": (0.0009, 0.0074),
            "decile": 1,
            "segment": "persuadable",
            "recommendation": "contact",
            "verdict": "confident",
        }
        scenario = scenarios.get(case, default_scenario)
        profit = request.conversion_margin * scenario["uplift"] - request.contact_cost
        response = ScoringResponse(
            customer_id=request.customer_id,
            model_version=MOCK_MODEL_VERSION,
            uplift_score=scenario["uplift"],
            uplift_ci_95=scenario["interval"],
            decile=scenario["decile"],
            segment_label=scenario["segment"],
            expected_incremental_profit=profit,
            recommendation=scenario["recommendation"],
            verdict=scenario["verdict"],
            top_drivers=(Driver("f3", 0.0012), Driver("f7", -0.0005)),
            scored_at=MOCK_TIMESTAMP,
        )
        logger.info(
            "Mock score created",
            extra={
                "customer_id": request.customer_id,
                "model_version": response.model_version,
                "verdict": response.verdict,
                # The mock has no fitted decision-layer segment threshold.
                "control_event_threshold": "not supplied by mock adapter",
            },
        )
        return response

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_version=MOCK_MODEL_VERSION,
            training_date=None,
            metrics=[
                MetricEstimate(
                    name="qini_coefficient",
                    value=0.31,
                    ci_95=(0.24, 0.38),
                ),
                MetricEstimate(name="auuc", value=0.12, ci_95=(0.08, 0.16)),
                MetricEstimate(
                    name="uplift_at_10pct",
                    value=0.0025,
                    ci_95=(0.001, 0.004),
                ),
            ],
            message=(
                "Illustrative mock metadata only; no registered champion, "
                "training date, or measured evaluation result is connected."
            ),
        )

    def drift_report(self) -> DriftReportResponse:
        if not self.drift_available:
            return DriftReportResponse(
                status="not_available",
                generated_at=None,
                model_version=None,
                feature_drift=[],
                score_drift=None,
                effect_drift=None,
                data_quality=None,
                message="No drift report is available yet.",
            )
        return DriftReportResponse(
            status="available",
            generated_at=MOCK_TIMESTAMP,
            model_version=MOCK_MODEL_VERSION,
            feature_drift=[
                FeatureDrift(feature="f0", psi=0.08, ks_statistic=0.04),
                FeatureDrift(feature="f3", psi=0.17, ks_statistic=0.09),
                FeatureDrift(feature="f7", psi=0.11, ks_statistic=0.06),
            ],
            score_drift=ScoreDistribution(
                mean=0.0018,
                standard_deviation=0.0042,
                p05=-0.0041,
                median=0.0012,
                p95=0.0097,
            ),
            effect_drift=EffectDrift(
                metric="incremental_conversions_rate",
                current_estimate=0.0012,
                reference_estimate=0.001,
                difference=0.0002,
                ci_95=(-0.0003, 0.0007),
            ),
            data_quality=DataQualityStatus(
                status="illustrative_mock",
                max_null_rate=0.0,
                range_violations=0,
                schema_changes=0,
            ),
            message=(
                "Illustrative mock drift values only. Alert thresholds and live "
                "reference-distribution calculations are not connected."
            ),
        )

    def simulate(self, request: PolicySimulationRequest) -> PolicySimulationResponse:
        """Produce a deterministic, explicitly synthetic four-policy comparison."""
        max_contacts = int(request.budget // request.contact_cost)
        budget_contacts = min(max_contacts, 4_000)
        blanket_contacts = 5_000  # fixed synthetic cohort size for the blanket policy
        definitions: tuple[
            tuple[
                Literal["uplift", "response_model", "random", "blanket"],
                int,
                float,
            ],
            ...,
        ] = (
            ("uplift", budget_contacts, 0.04),
            ("response_model", budget_contacts, 0.02),
            ("random", budget_contacts, 0.01),
            ("blanket", blanket_contacts, 0.015),
        )
        policies: list[PolicyResult] = []
        uplift_conversions = 0.0
        response_conversions = 0.0
        for policy_name, contacted, incremental_rate in definitions:
            conversions = contacted * incremental_rate
            low = conversions * 0.5
            high = conversions * 1.5
            total_cost = contacted * request.contact_cost
            profit = conversions * request.conversion_margin - total_cost
            policies.append(
                PolicyResult(
                    policy=policy_name,
                    contacted=contacted,
                    total_cost=total_cost,
                    incremental_conversions=conversions,
                    incremental_conversions_ci_95=(low, high),
                    incremental_profit=profit,
                    profit_per_unit_spent=profit / total_cost if total_cost else None,
                    curve=[
                        CurvePoint(
                            fraction_targeted=fraction,
                            cumulative_incremental_conversions=conversions * fraction,
                            ci_95=(low * fraction, high * fraction),
                        )
                        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
                    ],
                )
            )
            if policy_name == "uplift":
                uplift_conversions = conversions
            elif policy_name == "response_model":
                response_conversions = conversions

        difference = uplift_conversions - response_conversions
        half_width = max(1.0, abs(difference) * 0.5)
        interval = (difference - half_width, difference + half_width)
        conclusion: Literal[
            "uplift_better", "response_model_better", "not_distinguishable"
        ] = (
            "uplift_better"
            if interval[0] > 0
            else "response_model_better"
            if interval[1] < 0
            else "not_distinguishable"
        )
        fingerprint_source = "|".join(
            (
                request.scenario_id,
                request.data_slice,
                request.model_version,
                str(request.seed),
                format(request.budget, ".12g"),
                format(request.contact_cost, ".12g"),
                format(request.conversion_margin, ".12g"),
            )
        )
        simulation_fingerprint = hashlib.sha256(
            fingerprint_source.encode("utf-8")
        ).hexdigest()[:16]
        return PolicySimulationResponse(
            simulation_id=f"mock-{simulation_fingerprint}",
            scenario_id=request.scenario_id,
            data_slice="validation",
            model_version=request.model_version,
            seed=request.seed,
            budget=request.budget,
            contact_cost=request.contact_cost,
            conversion_margin=request.conversion_margin,
            policies=policies,
            comparison=PolicyDifference(
                incremental_conversions_difference=difference,
                ci_95=interval,
                conclusion=conclusion,
            ),
            message=(
                "Synthetic demonstration only. This endpoint does not execute "
                "the replay engine, inspect outcomes, or create an MLflow recorded run."
            ),
        )
