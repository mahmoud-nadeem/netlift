"""Interface 2: the metric signature.

Produced by the evaluation harness (T39) and consumed by everyone who trains a
model. Frozen by T36; ``docs/design/interfaces.md`` describes it in full. After
it merges, any change needs a record in ``docs/decisions/``.

Every metric has the shape ``metric(y, uplift, t, *, n_resamples,
confidence_level, seed) -> MetricResult``. The argument order is
scikit-uplift's ``(y_true, uplift, treatment)``. The interval is a percentile
bootstrap whose resamples are drawn within each treatment arm, preserving the
arm proportions (Specification, Statistical Significance).

The resample indices depend only on ``(t, n_resamples, seed)``, never on ``y``
or on a model's scores, so two models scored with the same ``t`` and seed are
compared on the same resamples (Statistical Significance: "Compare champion and
challenger on the same bootstrap resamples"). ``within_arm_bootstrap_indices``
draws them, ``fingerprint_indices`` hashes them, and ``paired_difference``
refuses to compare two results whose fingerprints differ.

scikit-uplift is the single source of truth for every reported Qini, AUUC and
uplift@k (``docs/decisions/0001-uplift-library.md``). A result with one of those
names and any other library is rejected.

A percentile bootstrap does not guarantee that the point estimate lies inside
its own interval, so ``MetricResult`` does not require it.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
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

# Specification, Evaluation Metrics: the two uplift@k computation strategies.
UPLIFT_AT_K = "uplift_at_k"
UPLIFT_AT_K_STRATEGIES: frozenset[str] = frozenset({"overall", "by_group"})

# T41: log_metric_result attaches each full result here, one JSON file per key.
RESAMPLES_ARTIFACT_DIR = "resamples"

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

SettingValue = str | int | float


@dataclass(frozen=True)
class MetricResult:
    """A metric value with its bootstrap interval and everything needed to rebuild it.

    ``resamples`` holds the metric's value on each bootstrap resample, in order,
    and ``resample_fingerprint`` identifies the resample indices they were
    computed on (see :func:`fingerprint_indices`).
    """

    name: str
    value: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_resamples: int
    seed: int
    library: str
    library_version: str
    resamples: tuple[float, ...]
    resample_fingerprint: str
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
            _require_finite(f"{self.name}: {label}", number)
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
        if self.name.startswith(UPLIFT_AT_K):
            _require_uplift_at_k_settings(self.name, self.settings)
        resamples = tuple(self.resamples)
        if len(resamples) != self.n_resamples:
            raise ContractError(
                f"{self.name}: {len(resamples)} resample values for "
                f"n_resamples={self.n_resamples}; they must match."
            )
        for number in resamples:
            _require_finite(f"{self.name}: a resample value", number)
        object.__setattr__(self, "resamples", tuple(float(v) for v in resamples))
        if not isinstance(self.resample_fingerprint, str) or not _SHA256_HEX.match(
            self.resample_fingerprint
        ):
            raise ContractError(
                f"{self.name}: resample_fingerprint must be a sha256 hex digest "
                "from fingerprint_indices()."
            )

    @property
    def value_inside_interval(self) -> bool:
        """Whether the point estimate lies inside its own interval."""
        return self.ci_low <= self.value <= self.ci_high

    @property
    def metric_key(self) -> str:
        """The name used in MLflow keys; uplift@k carries its k and strategy.

        ``uplift_at_k`` with k=0.1 and strategy "overall" becomes
        ``uplift_at_10pct_overall``, so two k values never share a key.
        """
        if not self.name.startswith(UPLIFT_AT_K):
            return self.name
        percent = f"{round(float(self.settings['k']) * 100, 6):g}"
        suffix = self.name[len(UPLIFT_AT_K) :]
        return f"uplift_at_{percent}pct_{self.settings['strategy']}{suffix}"


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


def within_arm_bootstrap_indices(
    t: npt.ArrayLike, n_resamples: int, seed: int
) -> Iterator[npt.NDArray[np.int64]]:
    """Yield one array of row indices per resample, drawn with replacement within each arm.

    Depends only on ``t``, ``n_resamples`` and ``seed``. Yields one resample at a
    time, so 1,000 resamples of Criteo's 14M rows never sit in memory together.
    """
    arms = np.asarray(t)
    if arms.ndim != 1:
        raise ContractError(f"t must be one-dimensional, got shape {arms.shape}.")
    if arms.dtype.kind not in "biuf" or not np.isin(arms, (0, 1)).all():
        unexpected = sorted({str(v) for v in arms[~np.isin(arms, (0, 1))].tolist()})
        raise ContractError(
            f"t must be binary 0/1 (bool is fine); found {unexpected[:5]}. A raw arm "
            "column with more than two arms must be mapped by a TreatmentDefinition first."
        )
    control = np.flatnonzero(arms == 0).astype(np.int64)
    treated = np.flatnonzero(arms == 1).astype(np.int64)
    if control.size == 0 or treated.size == 0:
        raise ContractError("Bootstrapping within arms needs both arms in t.")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < 1
    ):
        raise ContractError(f"n_resamples must be a positive int, got {n_resamples!r}.")
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        yield np.concatenate(
            [
                control[rng.integers(0, control.size, size=control.size)],
                treated[rng.integers(0, treated.size, size=treated.size)],
            ]
        )


def fingerprint_indices(indices: Iterable[npt.ArrayLike]) -> str:
    """SHA-256 over the resample index arrays, in order: the resample fingerprint."""
    digest = hashlib.sha256()
    count = 0
    for array in indices:
        values = np.ascontiguousarray(np.asarray(array, dtype=np.int64))
        digest.update(values.size.to_bytes(8, "little"))
        digest.update(values.tobytes())
        count += 1
    if count == 0:
        raise ContractError("Cannot fingerprint an empty set of resamples.")
    return digest.hexdigest()


def paired_difference(a: MetricResult, b: MetricResult) -> MetricResult:
    """The difference a - b, with its interval from the per-resample differences.

    Refuses unless both results were computed on the same resamples and the same
    way: same name, settings, number of resamples, seed, resample fingerprint,
    confidence level and library version.
    """
    mismatched = [
        label
        for label, left, right in (
            ("name", a.name, b.name),
            ("settings", a.settings, b.settings),
            ("n_resamples", a.n_resamples, b.n_resamples),
            ("seed", a.seed, b.seed),
            ("resample_fingerprint", a.resample_fingerprint, b.resample_fingerprint),
            ("confidence_level", a.confidence_level, b.confidence_level),
            ("library_version", a.library_version, b.library_version),
        )
        if left != right
    ]
    if mismatched:
        raise ContractError(
            f"paired_difference needs both results on the same resamples; "
            f"these differ: {mismatched}."
        )
    differences = np.asarray(a.resamples) - np.asarray(b.resamples)
    tail = (1.0 - a.confidence_level) / 2.0 * 100.0
    low, high = np.percentile(differences, [tail, 100.0 - tail])
    return MetricResult(
        name=f"{a.name}_difference",
        value=a.value - b.value,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=a.confidence_level,
        n_resamples=a.n_resamples,
        seed=a.seed,
        library=a.library,
        library_version=a.library_version,
        resamples=tuple(float(v) for v in differences),
        resample_fingerprint=a.resample_fingerprint,
        settings=dict(a.settings),
    )


def log_metric_result(run: Run, kpi_id: str, result: MetricResult) -> str:
    """Log a metric result under ``<kpi_id>_<metric key>`` and return that key.

    Every parameter is prefixed with the full key, so two metrics under one KPI
    (Qini and AUUC under K2, or uplift@k at two values of k) can share a run:
    MLflow refuses to change a parameter's value within a run.

    The whole result, per-resample values included, is also attached as
    ``resamples/<key>.json``. ``MetricResult(**json.loads(...))`` rebuilds it, so
    a champion and a challenger logged in different runs can still be compared
    with :func:`paired_difference`.
    """
    if result.n_resamples < MIN_REPORTED_RESAMPLES:
        raise ContractError(
            f"{result.name}: {result.n_resamples} bootstrap resamples. A reported "
            f"result needs at least {MIN_REPORTED_RESAMPLES}."
        )
    key = run.log_kpi(
        kpi_id,
        value=result.value,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        confidence=result.confidence_level,
        metric=result.metric_key,
    )
    params: dict[str, object] = {
        f"{key}_library": f"{result.library}=={result.library_version}",
        f"{key}_n_resamples": result.n_resamples,
        f"{key}_seed": result.seed,
        f"{key}_resample_fingerprint": result.resample_fingerprint,
    }
    params.update({f"{key}_{name}": value for name, value in result.settings.items()})
    run.log_params(params)
    run.log_dict(asdict(result), resamples_artifact_file(key))
    return key


def resamples_artifact_file(key: str) -> str:
    """Where ``log_metric_result`` stores the full result for ``key`` in a run's artifacts."""
    return f"{RESAMPLES_ARTIFACT_DIR}/{key}.json"


def _require_finite(label: str, number: object) -> None:
    """Raise unless ``number`` is a finite int or float (bool is not a number here)."""
    if isinstance(number, bool) or not isinstance(number, int | float):
        raise ContractError(f"{label} must be a number.")
    if not math.isfinite(number):
        raise ContractError(f"{label} is not finite ({number}).")


def _require_uplift_at_k_settings(name: str, settings: dict[str, SettingValue]) -> None:
    """Raise unless an uplift@k result records k in (0, 1] and a known strategy."""
    k = settings.get("k")
    if isinstance(k, bool) or not isinstance(k, int | float) or not 0.0 < k <= 1.0:
        raise ContractError(
            f"{name}: settings['k'] must be a number in (0, 1], got {k!r}."
        )
    strategy = settings.get("strategy")
    if strategy not in UPLIFT_AT_K_STRATEGIES:
        raise ContractError(
            f"{name}: settings['strategy'] must be one of "
            f"{sorted(UPLIFT_AT_K_STRATEGIES)}, got {strategy!r}."
        )
