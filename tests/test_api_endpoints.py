"""T45 API contract tests: six routes, mock responses, validation and OpenAPI."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Any

import pytest
from fastapi.testclient import TestClient

from netlift.api.app import create_app
from netlift.api.service import MOCK_MODEL_VERSION, MockNetLiftService


def client(*, ready: bool = True, drift_available: bool = True) -> TestClient:
    return TestClient(
        create_app(MockNetLiftService(ready=ready, drift_available=drift_available))
    )


def score_payload(customer_id: str = "customer-001") -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "features": {"f0": 0.2, "f3": 1.5},
        "contact_cost": 5.0,
        "conversion_margin": 200.0,
    }


def test_api_exposes_exactly_the_six_specification_endpoints() -> None:
    response = client().get("/openapi.json")
    assert response.status_code == 200
    paths = set(response.json()["paths"])
    assert paths == {
        "/score",
        "/score/batch",
        "/policy/simulate",
        "/model/info",
        "/monitor/drift",
        "/health",
    }


def test_score_returns_frozen_eleven_field_schema_and_economics_headers() -> None:
    response = client().post("/score", json=score_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "customer_id",
        "model_version",
        "uplift_score",
        "uplift_ci_95",
        "decile",
        "segment_label",
        "expected_incremental_profit",
        "recommendation",
        "verdict",
        "top_drivers",
        "scored_at",
    }
    assert body["model_version"] == MOCK_MODEL_VERSION
    assert body["uplift_score"] == pytest.approx(0.03)
    assert body["uplift_ci_95"] == pytest.approx([0.02, 0.04])
    assert body["decile"] == 1
    assert body["expected_incremental_profit"] == pytest.approx(1.0)  # 200 * 0.03 - 5
    assert body["scored_at"].endswith("Z")
    assert response.headers["x-netlift-contact-cost"] == "5"
    assert response.headers["x-netlift-conversion-margin"] == "200"
    assert response.headers["x-netlift-mock"] == "true"


def test_score_flat_feature_columns_are_accepted() -> None:
    payload = {
        "customer_id": "customer-flat",
        "f0": 0.1,
        "f1": 1,
        "contact_cost": 5,
        "conversion_margin": 200,
    }
    response = client().post("/score", json=payload)
    assert response.status_code == 200
    assert response.json()["customer_id"] == "customer-flat"


def test_mock_covers_both_abstention_verdicts_without_directional_action() -> None:
    test_cases = {
        "mock-insufficient-evidence:001": "insufficient_evidence",
        "mock-insufficient-control-data:001": "insufficient_control_data",
    }
    for customer_id, verdict in test_cases.items():
        response = client().post("/score", json=score_payload(customer_id))
        assert response.status_code == 200
        body = response.json()
        assert body["verdict"] == verdict
        assert body["recommendation"] == "no_action"
        assert body["segment_label"] == "unknown"
        assert len(body["uplift_ci_95"]) == 2


def test_mock_sleeping_dog_is_suppressed() -> None:
    response = client().post("/score", json=score_payload("mock-sleeping-dog:001"))
    assert response.status_code == 200
    assert response.json()["segment_label"] == "sleeping_dog"
    assert response.json()["recommendation"] == "suppress"
    assert response.json()["verdict"] == "confident"


def test_score_rejects_forbidden_columns_and_names_them() -> None:
    payload = score_payload()
    payload["features"]["treatment"] = 1
    response = client().post("/score", json=payload)
    assert response.status_code == 422
    assert "treatment" in response.text


def test_score_rejects_generic_outcome_column() -> None:
    payload = score_payload()
    payload["features"]["outcome"] = 1
    response = client().post("/score", json=payload)
    assert response.status_code == 422
    assert "outcome" in response.text


def test_score_rejects_non_finite_feature_values() -> None:
    body = (
        '{"customer_id":"customer-001","features":{"f0":1e999,"f3":1.5},'
        '"contact_cost":5,"conversion_margin":200}'
    )
    response = client().post(
        "/score", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert "finite" in response.text


def _read_batch_archive(content: bytes) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(content)) as bundle:
        assert set(bundle.namelist()) == {"scores.csv", "summary.json"}
        scores_csv = bundle.read("scores.csv").decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(scores_csv)))
        summary = json.loads(bundle.read("summary.json"))
    return rows, summary


def test_batch_json_returns_zip_and_includes_both_abstention_cases() -> None:
    rows = [
        score_payload("customer-positive"),
        score_payload("mock-insufficient-evidence:001"),
        score_payload("mock-insufficient-control-data:001"),
    ]
    response = client().post("/score/batch", json=rows)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]
    scores, summary = _read_batch_archive(response.content)
    assert len(scores) == 3
    assert summary["row_count"] == 3
    assert summary["verdict_counts"] == {
        "confident": 1,
        "insufficient_evidence": 1,
        "insufficient_control_data": 1,
    }
    assert summary["is_mock"] is True


def test_batch_json_accepts_flat_feature_columns() -> None:
    rows = [
        {
            "customer_id": "flat-json-001",
            "contact_cost": 5,
            "conversion_margin": 200,
            "f0": 0.1,
            "f1": 3,
        }
    ]
    response = client().post("/score/batch", json=rows)
    assert response.status_code == 200
    scores, summary = _read_batch_archive(response.content)
    assert scores[0]["customer_id"] == "flat-json-001"
    assert summary["row_count"] == 1


def test_batch_csv_is_supported() -> None:
    csv_body = (
        "customer_id,contact_cost,conversion_margin,f0,f3\n"
        "customer-001,5,200,0.2,1.5\n"
        "mock-insufficient-evidence:002,5,200,0.4,2.5\n"
    )
    response = client().post(
        "/score/batch",
        content=csv_body,
        headers={"content-type": "text/csv"},
    )
    assert response.status_code == 200
    scores, summary = _read_batch_archive(response.content)
    assert len(scores) == 2
    assert summary["campaign_economics"] == {
        "contact_cost": 5.0,
        "conversion_margin": 200.0,
        "profit_formula": "conversion_margin * uplift_score - contact_cost",
    }


def test_batch_validation_is_atomic_when_any_row_is_invalid() -> None:
    rows = [score_payload("valid-first"), score_payload("invalid-second")]
    rows[1]["features"]["exposure"] = 1
    response = client().post("/score/batch", json=rows)
    assert response.status_code == 422
    assert "rows[1]" in response.text
    assert "exposure" in response.text
    assert response.headers["content-type"].startswith("application/json")


def test_batch_rejects_different_campaign_economics() -> None:
    rows = [score_payload("first"), score_payload("second")]
    rows[1]["contact_cost"] = 7.0
    response = client().post("/score/batch", json=rows)
    assert response.status_code == 422
    assert "same contact_cost" in response.text


def test_batch_rejects_unsupported_content_type() -> None:
    response = client().post(
        "/score/batch", content="hello", headers={"content-type": "text/plain"}
    )
    assert response.status_code == 415


def test_policy_simulation_returns_four_mock_policies_without_recording() -> None:
    payload = {
        "scenario_id": "campaign-demo",
        "data_slice": "validation",
        "model_version": MOCK_MODEL_VERSION,
        "seed": 42,
        "budget": 20_000,
        "contact_cost": 5,
        "conversion_margin": 200,
    }
    response = client().post("/policy/simulate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert {item["policy"] for item in body["policies"]} == {
        "uplift",
        "response_model",
        "random",
        "blanket",
    }
    assert body["recorded_run_created"] is False
    assert body["data_slice"] == "validation"
    assert body["is_mock"] is True
    assert all(len(item["curve"]) == 5 for item in body["policies"])


def test_policy_simulation_is_deterministic_for_identical_inputs() -> None:
    payload = {
        "scenario_id": "campaign-demo",
        "data_slice": "validation",
        "model_version": MOCK_MODEL_VERSION,
        "seed": 314,
        "budget": 20_000,
        "contact_cost": 5,
        "conversion_margin": 200,
    }
    first = client().post("/policy/simulate", json=payload)
    second = client().post("/policy/simulate", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


@pytest.mark.parametrize("invalid_slice", ["test", "locked_test", "Test", "train"])
def test_policy_simulation_refuses_non_validation_splits(
    invalid_slice: str,
) -> None:
    payload = {
        "scenario_id": "campaign-demo",
        "data_slice": invalid_slice,
        "model_version": MOCK_MODEL_VERSION,
        "seed": 42,
        "budget": 20_000,
        "contact_cost": 5,
        "conversion_margin": 200,
    }
    response = client().post("/policy/simulate", json=payload)
    assert response.status_code == 422

    error = response.json()["error"]
    assert error["code"] == "request_validation_error"
    assert any("data_slice" in detail["location"] for detail in error["details"])
    assert any(
        "The locked test split is opened once at T70; "
        "this endpoint accepts 'validation' only." in detail["message"]
        for detail in error["details"]
    )


def test_policy_simulation_refuses_budget_below_one_contact() -> None:
    payload = {
        "scenario_id": "campaign-demo",
        "data_slice": "validation",
        "model_version": MOCK_MODEL_VERSION,
        "seed": 42,
        "budget": 2,
        "contact_cost": 5,
        "conversion_margin": 200,
    }
    response = client().post("/policy/simulate", json=payload)
    assert response.status_code == 422
    assert "at least one contact_cost" in response.text


def test_model_info_identifies_all_values_as_illustrative_mocks() -> None:
    response = client().get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["is_mock"] is True
    assert body["source"] == "mock"
    assert body["training_date"] is None
    assert {item["name"] for item in body["metrics"]} == {
        "qini_coefficient",
        "auuc",
        "uplift_at_10pct",
    }
    assert all(item["status"] == "illustrative_mock" for item in body["metrics"])


def test_drift_endpoint_returns_mock_report() -> None:
    response = client().get("/monitor/drift")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["is_mock"] is True
    assert body["feature_drift"]
    assert body["score_drift"] is not None
    assert body["effect_drift"]["source"] == "mock_replay_output"
    assert body["data_quality"] is not None


def test_drift_endpoint_has_explicit_empty_state() -> None:
    response = client(drift_available=False).get("/monitor/drift")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_available"
    assert body["feature_drift"] == []
    assert body["score_drift"] is None
    assert body["effect_drift"] is None
    assert body["data_quality"] is None
    assert "No drift report" in body["message"]


def test_health_reports_mock_readiness_and_unready_service_blocks_scoring() -> None:
    ready_client = client()
    health = ready_client.get("/health")
    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert health.json()["model_loaded"] is False
    assert health.json()["mode"] == "mock"

    unready_client = client(ready=False)
    health = unready_client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "not_ready"
    assert health.json()["ready"] is False
    score = unready_client.post("/score", json=score_payload())
    assert score.status_code == 503
    assert score.json()["error"]["code"] == "service_not_ready"


def test_transport_models_accept_production_adapter_values() -> None:
    """Mock markers must not make the public response types mock-only."""
    from datetime import UTC, datetime

    from netlift.api.models import (
        DataQualityStatus,
        DriftReportResponse,
        EffectDrift,
        FeatureDrift,
        HealthResponse,
        MetricEstimate,
        ModelInfoResponse,
        PolicySimulationResponse,
        ScoreDistribution,
    )

    metric = MetricEstimate(
        name="qini_coefficient", value=0.31, ci_95=(0.24, 0.38), status="available"
    )
    model_info = ModelInfoResponse(
        model_version="netlift-xlearner-1.0.0",
        training_date=None,
        metrics=[metric],
        source="registry",
        is_mock=False,
        message="Registered model metadata.",
    )
    feature = FeatureDrift(
        feature="f0", psi=0.08, ks_statistic=0.04, status="available"
    )
    effect = EffectDrift(
        metric="incremental_conversions_rate",
        current_estimate=0.0012,
        reference_estimate=0.001,
        difference=0.0002,
        ci_95=(-0.0003, 0.0007),
        source="replay_output",
    )
    drift = DriftReportResponse(
        status="available",
        generated_at=datetime(2026, 11, 4, 12, 0, tzinfo=UTC),
        model_version="netlift-xlearner-1.0.0",
        feature_drift=[feature],
        score_drift=ScoreDistribution(
            mean=0.0018,
            standard_deviation=0.0042,
            p05=-0.0041,
            median=0.0012,
            p95=0.0097,
            status="available",
        ),
        effect_drift=effect,
        data_quality=DataQualityStatus(
            status="available",
            max_null_rate=0.0,
            range_violations=0,
            schema_changes=0,
        ),
        is_mock=False,
        message="Measured drift report.",
    )
    health = HealthResponse(
        status="ready", ready=True, mode="live", model_loaded=True, message="Ready."
    )
    simulation = PolicySimulationResponse(
        simulation_id="sim-001",
        scenario_id="scenario-001",
        data_slice="validation",
        model_version="netlift-xlearner-1.0.0",
        seed=42,
        budget=20_000,
        contact_cost=5,
        conversion_margin=200,
        policies=[],
        comparison={
            "incremental_conversions_difference": 1.0,
            "ci_95": (0.2, 1.8),
            "conclusion": "uplift_better",
        },
        recorded_run_created=False,
        source="live",
        is_mock=False,
        message="Production what-if result.",
    )

    assert model_info.source == "registry" and model_info.is_mock is False
    assert drift.effect_drift is not None
    assert drift.effect_drift.source == "replay_output" and drift.is_mock is False
    assert health.mode == "live" and health.model_loaded is True
    assert simulation.source == "live" and simulation.is_mock is False
    assert simulation.recorded_run_created is False
    assert simulation.data_slice == "validation"
