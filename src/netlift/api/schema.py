"""Interface 3: the scoring response.

Produced by the decision layer and the model registry (Ali, Mahmoud) and consumed
by the API and the dashboard (Mazen, Mohammed). Frozen by T36;
``docs/design/interfaces.md`` describes it in full. After it merges, any change
needs a record in ``docs/decisions/``.

Mirrors the specification's Model Output Schema field by field. This is a
standard-library dataclass because pydantic and FastAPI arrive in M5; the API
layer wraps this type then.

The rule from the specification's Uncertainty and Abstention section that this
type encodes: "Where a segment's interval straddles zero, the system returns
verdict = "insufficient_evidence" rather than a directional recommendation." The
same applies to ``insufficient_control_data``. So any verdict other than
``confident`` carries ``recommendation = "no_action"``.

``segment_label`` is presentational only. No dataset labels customers as
persuadables, so nothing may treat the label as ground truth.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, get_args

from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)

SegmentLabel = Literal[
    "persuadable", "sure_thing", "lost_cause", "sleeping_dog", "unknown"
]
Recommendation = Literal["contact", "suppress", "no_action"]
Verdict = Literal["confident", "insufficient_evidence", "insufficient_control_data"]

DIRECTIONAL_RECOMMENDATIONS: frozenset[str] = frozenset({"contact", "suppress"})

FIELDS = (
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
)


@dataclass(frozen=True)
class Driver:
    """One feature's contribution to a customer's uplift score."""

    feature: str
    contribution: float

    def __post_init__(self) -> None:
        """Require a feature name and a finite contribution."""
        if not isinstance(self.feature, str) or not self.feature:
            raise ContractError("A driver needs a feature name.")
        _require_number("top_drivers.contribution", self.contribution)


@dataclass(frozen=True)
class ScoringResponse:
    """The response for one scored customer."""

    customer_id: str
    model_version: str
    uplift_score: float
    uplift_ci_95: tuple[float, float]
    decile: int
    segment_label: SegmentLabel
    expected_incremental_profit: float
    recommendation: Recommendation
    verdict: Verdict
    top_drivers: tuple[Driver, ...]
    scored_at: datetime

    def __post_init__(self) -> None:
        """Validate every field and the rules that tie fields together."""
        for name in ("customer_id", "model_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be a non-empty string.")
        _require_number("uplift_score", self.uplift_score)
        _require_number("expected_incremental_profit", self.expected_incremental_profit)
        if len(self.uplift_ci_95) != 2:
            raise ContractError("uplift_ci_95 must have exactly two bounds.")
        low, high = self.uplift_ci_95
        _require_number("uplift_ci_95 lower bound", low)
        _require_number("uplift_ci_95 upper bound", high)
        if low > high:
            raise ContractError(
                f"uplift_ci_95 lower bound ({low}) is above the upper bound ({high})."
            )
        if isinstance(self.decile, bool) or not isinstance(self.decile, int):
            raise ContractError(f"decile must be an int, got {self.decile!r}.")
        if not 1 <= self.decile <= 10:
            raise ContractError(f"decile must be between 1 and 10, got {self.decile}.")
        _require_member("segment_label", self.segment_label, get_args(SegmentLabel))
        _require_member("recommendation", self.recommendation, get_args(Recommendation))
        _require_member("verdict", self.verdict, get_args(Verdict))
        if (
            self.verdict != "confident"
            and self.recommendation in DIRECTIONAL_RECOMMENDATIONS
        ):
            raise ContractError(
                f"verdict '{self.verdict}' cannot carry the directional recommendation "
                f"'{self.recommendation}'; it must be 'no_action' "
                "(Specification, Uncertainty and Abstention)."
            )
        if not all(isinstance(driver, Driver) for driver in self.top_drivers):
            raise ContractError("top_drivers must contain Driver objects.")
        if (
            not isinstance(self.scored_at, datetime)
            or self.scored_at.utcoffset() is None
        ):
            raise ContractError("scored_at must be a timezone-aware datetime.")

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict with the field names and shapes of the specification."""
        return {
            "customer_id": self.customer_id,
            "model_version": self.model_version,
            "uplift_score": self.uplift_score,
            "uplift_ci_95": [self.uplift_ci_95[0], self.uplift_ci_95[1]],
            "decile": self.decile,
            "segment_label": self.segment_label,
            "expected_incremental_profit": self.expected_incremental_profit,
            "recommendation": self.recommendation,
            "verdict": self.verdict,
            "top_drivers": [
                {"feature": driver.feature, "contribution": driver.contribution}
                for driver in self.top_drivers
            ],
            "scored_at": _format_timestamp(self.scored_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScoringResponse:
        """Build a response from a dict shaped like :meth:`to_dict` output."""
        missing = [name for name in FIELDS if name not in data]
        unknown = sorted(set(data) - set(FIELDS))
        if missing or unknown:
            raise ContractError(
                f"Scoring response keys do not match the schema. Missing: {missing}; "
                f"unknown: {unknown}."
            )
        interval = data["uplift_ci_95"]
        if not isinstance(interval, list | tuple) or len(interval) != 2:
            raise ContractError("uplift_ci_95 must be a list of two numbers.")
        drivers = data["top_drivers"]
        if not isinstance(drivers, list | tuple):
            raise ContractError("top_drivers must be a list.")
        parsed_drivers = []
        for item in drivers:
            if not isinstance(item, Mapping) or set(item) != {
                "feature",
                "contribution",
            }:
                raise ContractError(
                    "Each driver must have exactly the keys 'feature' and 'contribution'."
                )
            parsed_drivers.append(Driver(item["feature"], item["contribution"]))
        stamp = data["scored_at"]
        if not isinstance(stamp, str):
            raise ContractError("scored_at must be an ISO 8601 string.")
        try:
            scored_at = datetime.fromisoformat(stamp)
        except ValueError as error:
            raise ContractError(f"scored_at is not ISO 8601: {stamp!r}.") from error
        return cls(
            customer_id=data["customer_id"],
            model_version=data["model_version"],
            uplift_score=data["uplift_score"],
            uplift_ci_95=(interval[0], interval[1]),
            decile=data["decile"],
            segment_label=data["segment_label"],
            expected_incremental_profit=data["expected_incremental_profit"],
            recommendation=data["recommendation"],
            verdict=data["verdict"],
            top_drivers=tuple(parsed_drivers),
            scored_at=scored_at,
        )


def _require_number(name: str, value: object) -> None:
    """Raise unless ``value`` is a finite int or float (bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be a number, got {value!r}.")
    if not math.isfinite(value):
        raise ContractError(f"{name} must be finite, got {value}.")


def _require_member(name: str, value: object, allowed: tuple[object, ...]) -> None:
    """Raise if ``value`` is not one of the allowed literal values."""
    if value not in allowed:
        raise ContractError(f"{name} must be one of {list(allowed)}, got {value!r}.")


def _format_timestamp(moment: datetime) -> str:
    """ISO 8601 with a 'Z' suffix for UTC, as in the specification's example."""
    text = moment.isoformat()
    return text[:-6] + "Z" if text.endswith("+00:00") else text
