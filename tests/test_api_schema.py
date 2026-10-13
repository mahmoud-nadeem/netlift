"""Tests for interface 3: the scoring response."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from netlift.api.schema import Driver, ScoringResponse
from netlift.utils.errors import ContractError

# The example in the specification's Model Output Schema section.
SPEC_EXAMPLE: dict[str, Any] = {
    "customer_id": "c_00193472",
    "model_version": "netlift-xlearner-2026.11.03",
    "uplift_score": 0.0041,
    "uplift_ci_95": [0.0009, 0.0074],
    "decile": 1,
    "segment_label": "persuadable",
    "expected_incremental_profit": 1.87,
    "recommendation": "contact",
    "verdict": "confident",
    "top_drivers": [
        {"feature": "f3", "contribution": 0.0012},
        {"feature": "f7", "contribution": -0.0005},
    ],
    "scored_at": "2026-11-03T14:22:10Z",
}


def _response(**overrides: object) -> ScoringResponse:
    fields: dict[str, object] = {
        "customer_id": "c_1",
        "model_version": "netlift-test",
        "uplift_score": 0.004,
        "uplift_ci_95": (0.001, 0.007),
        "decile": 1,
        "segment_label": "persuadable",
        "expected_incremental_profit": 1.5,
        "recommendation": "contact",
        "verdict": "confident",
        "top_drivers": (Driver("f3", 0.0012),),
        "scored_at": datetime(2026, 11, 3, 14, 22, 10, tzinfo=UTC),
    }
    fields.update(overrides)
    return ScoringResponse(**fields)  # type: ignore[arg-type]


def test_specification_example_round_trips_through_json() -> None:
    response = ScoringResponse.from_dict(json.loads(json.dumps(SPEC_EXAMPLE)))
    assert response.to_dict() == SPEC_EXAMPLE
    assert ScoringResponse.from_dict(response.to_dict()) == response


def test_object_round_trips_through_dict() -> None:
    response = _response(verdict="insufficient_evidence", recommendation="no_action")
    assert ScoringResponse.from_dict(response.to_dict()) == response


@pytest.mark.parametrize(
    "verdict", ["insufficient_evidence", "insufficient_control_data"]
)
@pytest.mark.parametrize("recommendation", ["contact", "suppress"])
def test_abstention_never_carries_a_directional_recommendation(
    verdict: str, recommendation: str
) -> None:
    with pytest.raises(ContractError, match="no_action"):
        _response(verdict=verdict, recommendation=recommendation)


@pytest.mark.parametrize(
    "verdict", ["insufficient_evidence", "insufficient_control_data"]
)
def test_abstention_with_no_action_is_accepted(verdict: str) -> None:
    assert _response(verdict=verdict, recommendation="no_action").verdict == verdict


def test_reversed_interval_is_rejected() -> None:
    with pytest.raises(ContractError, match="above the upper bound"):
        _response(uplift_ci_95=(0.007, 0.001))


def test_estimate_outside_its_interval_is_allowed() -> None:
    assert _response(uplift_score=0.02).uplift_score == 0.02


@pytest.mark.parametrize("decile", [0, 11, 1.0, True])
def test_decile_must_be_an_int_from_1_to_10(decile: object) -> None:
    with pytest.raises(ContractError, match="decile"):
        _response(decile=decile)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("segment_label", "whale"),
        ("recommendation", "maybe"),
        ("verdict", "unsure"),
    ],
)
def test_values_outside_the_enumerations_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ContractError, match=field):
        _response(**{field: value})


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ContractError, match="timezone-aware"):
        _response(scored_at=datetime(2026, 11, 3, 14, 22, 10))  # noqa: DTZ001 (deliberately naive)


def test_non_finite_score_is_rejected() -> None:
    with pytest.raises(ContractError, match="finite"):
        _response(uplift_score=float("inf"))


def test_missing_and_unknown_keys_are_named() -> None:
    data = dict(SPEC_EXAMPLE)
    del data["verdict"]
    data["confidence"] = 0.9
    with pytest.raises(
        ContractError, match=r"Missing: \['verdict'\]; unknown: \['confidence'\]"
    ):
        ScoringResponse.from_dict(data)


def test_malformed_driver_is_rejected() -> None:
    data = dict(SPEC_EXAMPLE)
    data["top_drivers"] = [{"feature": "f3"}]
    with pytest.raises(ContractError, match="driver"):
        ScoringResponse.from_dict(data)


def test_bad_timestamp_is_rejected() -> None:
    data = dict(SPEC_EXAMPLE)
    data["scored_at"] = "yesterday"
    with pytest.raises(ContractError, match="ISO 8601"):
        ScoringResponse.from_dict(data)
