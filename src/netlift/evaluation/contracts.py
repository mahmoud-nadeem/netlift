"""Interface 2: the metric signature.

Produced by the evaluation harness (T39) and consumed by everyone who trains a
model. Frozen by T36; ``docs/design/interfaces.md`` describes it in full. After
it merges, any change needs a record in ``docs/decisions/``.

Every metric has the shape ``metric(y, uplift, t, *, n_resamples,
confidence_level, seed) -> MetricResult``. The argument order is
scikit-uplift's ``(y_true, uplift, treatment)``. The interval is a percentile
bootstrap whose resamples are drawn within each treatment arm, preserving the
arm proportions (Specification, Statistical Significance).

scikit-uplift is the single source of truth for every reported Qini, AUUC and
uplift@k (``docs/decisions/0001-uplift-library.md``). A result with one of those
names and any other library is rejected.

A percentile bootstrap does not guarantee that the point estimate lies inside
its own interval, so ``MetricResult`` does not require it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy.typing as npt

from netlift.utils.errors import ContractError

if TYPE_CHECKING:
    from netlift.utils.tracking import Run

logger = logging.getLogger(__name__)

SOURCE_OF_TRUTH_LIBRARY = "scikit-uplift"
REPORTED_METRICS: frozenset[str] = frozenset(
    {"qini_coefficient", "auuc", "uplift_at_k"}
)
# Specification, Statistical Significance: "Use at least 1,000 resamples".
MIN_REPORTED_RESAMPLES = 1000

SettingValue = str | int | float


@dataclass(frozen=True)
class MetricResult:
    """A metric value with its bootstrap interval and everything needed to rebuild it."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_resamples: int
    seed: int
    library: str
    library_version: str
    settings: dict[str, SettingValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the result; the value is not required to sit inside the interval."""
        if not self.name:
            raise ContractError("MetricResult needs a metric name.")
        for label, number in (
            ("value", self.value),
            ("ci_low", self.ci_low),
            ("ci_high", self.ci_high),
        ):
            if isinstance(number, bool) or not isinstance(number, int | float):
                raise ContractError(f"{self.name}: {label} must be a number.")
            if not math.isfinite(number):
                raise ContractError(f"{self.name}: {label} is not finite ({number}).")
        if self.ci_low > self.ci_high:
            raise ContractError(
                f"{self.name}: ci_low ({self.ci_low}) is above ci_high ({self.ci_high})."
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ContractError(
                f"{self.name}: confidence_level must be between 0 and 1, "
                f"got {self.confidence_level}."
            )
        for label, count in (("n_resamples", self.n_resamples), ("seed", self.seed)):
            if isinstance(count, bool) or not isinstance(count, int):
                raise ContractError(f"{self.name}: {label} must be an int.")
        if self.n_resamples < 1:
            raise ContractError(f"{self.name}: n_resamples must be at least 1.")
        if not self.library or not self.library_version:
            raise ContractError(
                f"{self.name}: library and library_version are required."
            )
        if self.name in REPORTED_METRICS and self.library != SOURCE_OF_TRUTH_LIBRARY:
            raise ContractError(
                f"{self.name} is reported from {SOURCE_OF_TRUTH_LIBRARY} only "
                f"(ADR 0001); got library '{self.library}'."
            )

    @property
    def value_inside_interval(self) -> bool:
        """Whether the point estimate lies inside its own interval."""
        return self.ci_low <= self.value <= self.ci_high


class UpliftMetric(Protocol):
    """Signature every uplift metric in the evaluation harness implements.

    Metric-specific options (for example ``k`` and ``strategy`` for uplift@k)
    are extra keyword arguments with defaults, and are echoed in
    ``MetricResult.settings``.
    """

    def __call__(
        self,
        y: npt.ArrayLike,
        uplift: npt.ArrayLike,
        t: npt.ArrayLike,
        *,
        n_resamples: int,
        confidence_level: float,
        seed: int,
    ) -> MetricResult:
        """Compute the metric and its within-arm bootstrap interval."""
        ...


def log_metric_result(run: Run, kpi_id: str, result: MetricResult) -> None:
    """Log a metric result against a KPI through the tracking wrapper."""
    if result.n_resamples < MIN_REPORTED_RESAMPLES:
        raise ContractError(
            f"{result.name}: {result.n_resamples} bootstrap resamples. A reported "
            f"result needs at least {MIN_REPORTED_RESAMPLES}."
        )
    run.log_kpi(
        kpi_id,
        value=result.value,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        confidence=result.confidence_level,
    )
    params: dict[str, object] = {
        f"{kpi_id}_metric": result.name,
        f"{kpi_id}_library": f"{result.library}=={result.library_version}",
        f"{kpi_id}_n_resamples": result.n_resamples,
        f"{kpi_id}_seed": result.seed,
    }
    params.update({f"{kpi_id}_{key}": value for key, value in result.settings.items()})
    run.log_params(params)
