"""Interface 2 implementations: Qini coefficient, AUUC and uplift@k with bootstrap intervals.

The Qini coefficient and AUUC are computed by scikit-uplift (ADR 0001, decision 3).
uplift@k is computed here, tie-aware (ADR 0007): scikit-uplift takes the first n rows
after a stable sort, so when the cut falls inside a block of tied scores its value
depends on the row order. Here the tied block at the cut shares the remaining places
by fractional weight, which is the expected result when tied rows are picked at
random. When the cut does not fall inside a tie, the value is exactly scikit-uplift's.
This module also adds what neither library provides (ADR 0001, decision 5): the
percentile bootstrap resampled within treatment arms, using the frozen
``within_arm_bootstrap_indices`` of ``contracts.py``.

Each public metric has the frozen signature ``metric(y, uplift, t, *, n_resamples,
confidence_level, seed, split_name, row_ids, dataset)`` (ADR 0004). ``split_name``,
``row_ids`` and ``dataset`` have no default: they fill the result's split and its
evaluation fingerprint, which binds the dataset and the ordered row ids to ``y`` and
``t``. Metric-specific options (``k`` and ``strategy`` for uplift@k) are extra
keywords with defaults, echoed in the ``MetricResult``.

A resample whose metric is undefined (NaN or infinite) is never turned into a
number: the call raises ``ContractError`` naming the resample. Dropping it would
break the resample-by-resample pairing that ``paired_difference`` relies on.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable, Generator
from functools import lru_cache
from importlib import metadata
from typing import get_args

import numpy as np
import numpy.typing as npt
import sklift.metrics as sk_metrics  # type: ignore[import-untyped]

from netlift.data.contracts import DatasetName, SplitName
from netlift.evaluation.contracts import (
    SOURCE_OF_TRUTH_LIBRARY,
    UPLIFT_AT_K,
    UPLIFT_AT_K_STRATEGIES,
    MetricResult,
    SettingValue,
    fingerprint_evaluation_set,
    fingerprint_indices,
    within_arm_bootstrap_indices,
)
from netlift.utils.errors import ContractError

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

# scikit-uplift's own default; the primary k comes from config (spec, K1).
DEFAULT_K = 0.3
DEFAULT_STRATEGY = "overall"

# Parameters this module passes by keyword. Checked once per process so that a
# renamed argument in another scikit-uplift release fails loudly, not silently.
_REQUIRED_PARAMETERS: dict[str, tuple[str, ...]] = {
    "qini_auc_score": ("y_true", "uplift", "treatment", "negative_effect"),
    "uplift_auc_score": ("y_true", "uplift", "treatment"),
    "uplift_at_k": ("y_true", "uplift", "treatment", "strategy", "k"),
    "qini_curve": ("y_true", "uplift", "treatment"),
    "uplift_curve": ("y_true", "uplift", "treatment"),
}


# --------------------------------------------------------------------------- #
# Shared helpers (also used by curves.py)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def library_version() -> str:
    """The installed scikit-uplift version, read from the package metadata."""
    try:
        return metadata.version(SOURCE_OF_TRUTH_LIBRARY)
    except metadata.PackageNotFoundError as error:
        raise ContractError(
            f"{SOURCE_OF_TRUTH_LIBRARY} is not installed; it is the source of "
            "truth for every reported metric (ADR 0001)."
        ) from error


@lru_cache(maxsize=1)
def ensure_sklift_api() -> None:
    """Raise ``ContractError`` unless the installed scikit-uplift has the signatures used here."""
    for function_name, expected in _REQUIRED_PARAMETERS.items():
        function = getattr(sk_metrics, function_name, None)
        if function is None:
            raise ContractError(f"sklift.metrics has no '{function_name}'.")
        missing = [
            name
            for name in expected
            if name not in inspect.signature(function).parameters
        ]
        if missing:
            raise ContractError(
                f"sklift.metrics.{function_name} lacks parameters {missing}; this "
                f"harness was written for scikit-uplift 0.5.1 (found {library_version()})."
            )


def _as_binary(label: str, values: npt.ArrayLike) -> IntArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ContractError(
            f"{label} must be one-dimensional, got shape {array.shape}."
        )
    if array.dtype.kind not in "biuf" or not np.isin(array, (0, 1)).all():
        raise ContractError(
            f"{label} must contain only 0 and 1 (no NaN, no other values)."
        )
    return array.astype(np.int64)


def prepare_inputs(
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    require_positive_outcome: bool,
) -> tuple[IntArray, FloatArray, IntArray]:
    """Validate ``(y, uplift, t)`` once and return them as int64, float64, int64 arrays.

    Rejects empty or mismatched inputs, values outside {0, 1} in ``y`` or ``t``,
    non-finite uplift scores, and a ``t`` with a single arm. Qini and AUUC are
    also undefined when no row has a positive outcome (the perfect curve has no
    area), so ``require_positive_outcome`` rejects that case up front.
    """
    y_array = _as_binary("y", y)
    t_array = _as_binary("t", t)
    score = np.asarray(uplift)
    if score.ndim != 1:
        raise ContractError(f"uplift must be one-dimensional, got shape {score.shape}.")
    if score.dtype.kind not in "biuf":
        raise ContractError("uplift must be numeric.")
    score = score.astype(np.float64)
    if not (y_array.size == score.size == t_array.size):
        raise ContractError(
            f"y, uplift and t must have the same length, got {y_array.size}, "
            f"{score.size} and {t_array.size}."
        )
    if y_array.size == 0:
        raise ContractError("Cannot evaluate an empty set of rows.")
    bad = int((~np.isfinite(score)).sum())
    if bad:
        raise ContractError(f"uplift has {bad} NaN or infinite value(s).")
    if not ((t_array == 0).any() and (t_array == 1).any()):
        raise ContractError("t must contain both arms (control 0 and treatment 1).")
    if require_positive_outcome and not y_array.any():
        raise ContractError(
            "y has no positive outcome, so Qini and AUUC are undefined (the "
            "perfect curve has no area)."
        )
    return y_array, score, t_array


def check_bootstrap_arguments(
    n_resamples: int, confidence_level: float, seed: int, split_name: str
) -> None:
    """Raise ``ContractError`` for arguments that would produce a meaningless interval."""
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < 1
    ):
        raise ContractError(f"n_resamples must be a positive int, got {n_resamples!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError(f"seed must be an int, got {seed!r}.")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, int | float)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ContractError(
            f"confidence_level must be between 0 and 1, got {confidence_level!r}."
        )
    if split_name not in get_args(SplitName):
        raise ContractError(
            f"split_name must be one of {get_args(SplitName)}, got {split_name!r}."
        )


def evaluate_on_resamples[T](
    t: IntArray,
    n_resamples: int,
    seed: int,
    evaluate: Callable[[int, IntArray], T],
) -> tuple[list[T], str]:
    """Run ``evaluate(resample_number, indices)`` on every within-arm resample.

    Returns the results in order and the resample fingerprint. The indices are
    generated once: ``fingerprint_indices`` consumes a generator that evaluates
    each resample as it yields it, so the hashing logic stays the frozen
    contract's and no second pass over 1,000 x 2.8M indices is needed.
    """
    results: list[T] = []

    def generator() -> Generator[IntArray, None, None]:
        for number, indices in enumerate(
            within_arm_bootstrap_indices(t, n_resamples, seed)
        ):
            results.append(evaluate(number, indices))
            yield indices

    fingerprint = fingerprint_indices(generator())
    return results, fingerprint


def quiet_call(
    function: Callable[..., object], *args: object, **kwargs: object
) -> float:
    """Call a scikit-uplift function, hiding empty-slice warnings; the caller checks finiteness."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(function(*args, **kwargs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Point estimates: the only places scikit-uplift's metric functions are called
# --------------------------------------------------------------------------- #


def point_qini_coefficient(y: IntArray, uplift: FloatArray, t: IntArray) -> float:
    """Qini coefficient (Radcliffe 2007), perfect-curve normalised, negative effects included.

    Inputs must already be validated by :func:`prepare_inputs`.
    """
    return quiet_call(sk_metrics.qini_auc_score, y, uplift, t, negative_effect=True)


def point_auuc(y: IntArray, uplift: FloatArray, t: IntArray) -> float:
    """Area under the uplift curve as scikit-uplift defines it (``uplift_auc_score``)."""
    return quiet_call(sk_metrics.uplift_auc_score, y, uplift, t)


def top_weights(score: FloatArray, n_size: int) -> FloatArray:
    """Selection weight of each row in the top ``n_size`` rows by score, ties shared.

    Rows scored above the cut weigh 1 and rows below it 0. The rows tied at the cut
    share the remaining places: each weighs (n_size - rows above) / rows tied, the
    probability of being picked when tied rows are drawn at random (ADR 0007). When
    no tie crosses the cut, the weights are exactly the first ``n_size`` rows of a
    sort, as in scikit-uplift.
    """
    weights = np.zeros(len(score), dtype=np.float64)
    if n_size <= 0:
        return weights
    cut = np.partition(score, len(score) - n_size)[len(score) - n_size]
    above = score > cut
    tied = score == cut
    weights[above] = 1.0
    weights[tied] = (n_size - int(above.sum())) / int(tied.sum())
    return weights


def _weighted_rate(y: IntArray, weights: FloatArray) -> float:
    """Weighted outcome rate; NaN when the weights sum to zero, as an empty mean is."""
    total = float(weights.sum())
    if total == 0.0:
        return math.nan
    return float(np.dot(weights, y)) / total


def point_uplift_at_k(
    y: IntArray,
    uplift: FloatArray,
    t: IntArray,
    *,
    k: float,
    strategy: str,
) -> float:
    """Treated rate minus control rate in the top ``k`` fraction, by ``strategy``, ties shared.

    ``overall`` takes the top ``int(n * k)`` rows of both arms together;
    ``by_group`` takes the top ``int(n_arm * k)`` rows of each arm. The tied block at
    a cut gets fractional weight (:func:`top_weights`, ADR 0007), so the value does
    not depend on the row order. Without a tie at the cut it equals scikit-uplift's
    ``uplift_at_k``. ``k`` must already be a float strictly between 0 and 1.
    """
    treated = t == 1
    if strategy == "overall":
        weights = top_weights(uplift, int(len(y) * k))
        return _weighted_rate(y[treated], weights[treated]) - _weighted_rate(
            y[~treated], weights[~treated]
        )
    rates = []
    for arm in (treated, ~treated):
        score = uplift[arm]
        rates.append(_weighted_rate(y[arm], top_weights(score, int(len(score) * k))))
    return rates[0] - rates[1]


def _finite(value: float, what: str) -> float:
    if not math.isfinite(value):
        raise ContractError(
            f"{what} is undefined ({value}). A metric that cannot be computed is "
            "never reported as zero; check that every arm has rows in the "
            "evaluated slice and that the outcome has positives."
        )
    return value


# --------------------------------------------------------------------------- #
# Bootstrap wrapper
# --------------------------------------------------------------------------- #


def _bootstrap_metric(
    name: str,
    point: Callable[[IntArray, FloatArray, IntArray], float],
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    split_name: SplitName,
    row_ids: npt.ArrayLike,
    dataset: DatasetName,
    settings: dict[str, SettingValue],
    require_positive_outcome: bool,
) -> MetricResult:
    check_bootstrap_arguments(n_resamples, confidence_level, seed, split_name)
    ensure_sklift_api()
    y_array, score, t_array = prepare_inputs(
        y, uplift, t, require_positive_outcome=require_positive_outcome
    )
    # Cheap, and it validates row_ids and dataset, so a bad one fails before the
    # bootstrap rather than after 1,000 resamples.
    evaluation_fingerprint = fingerprint_evaluation_set(
        y_array, t_array, row_ids=row_ids, dataset=dataset
    )
    value = _finite(point(y_array, score, t_array), f"{name}: the point estimate")

    def evaluate(number: int, indices: IntArray) -> float:
        return _finite(
            point(y_array[indices], score[indices], t_array[indices]),
            f"{name}: bootstrap resample {number}",
        )

    resamples, resample_fingerprint = evaluate_on_resamples(
        t_array, n_resamples, seed, evaluate
    )
    tail = (1.0 - confidence_level) / 2.0 * 100.0
    low, high = np.percentile(resamples, [tail, 100.0 - tail])
    return MetricResult(
        name=name,
        value=value,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=float(confidence_level),
        n_resamples=n_resamples,
        seed=seed,
        library=SOURCE_OF_TRUTH_LIBRARY,
        library_version=library_version(),
        resamples=tuple(resamples),
        resample_fingerprint=resample_fingerprint,
        evaluation_fingerprint=evaluation_fingerprint,
        split_name=split_name,
        settings=settings,
    )


def qini_coefficient(
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    split_name: SplitName,
    row_ids: npt.ArrayLike,
    dataset: DatasetName,
) -> MetricResult:
    """Qini coefficient with a within-arm percentile bootstrap interval (K2).

    ``split_name``, ``row_ids`` and ``dataset`` are required (ADR 0004). Pass
    "train" for out-of-fold predictions and "test" only inside the T70
    procedure. ``row_ids`` are interface 1's ``row_id`` values, in the same order
    as ``y``, ``uplift`` and ``t``.
    """
    return _bootstrap_metric(
        "qini_coefficient",
        point_qini_coefficient,
        y,
        uplift,
        t,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        split_name=split_name,
        row_ids=row_ids,
        dataset=dataset,
        settings={"negative_effect": "true"},
        require_positive_outcome=True,
    )


def auuc(
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    split_name: SplitName,
    row_ids: npt.ArrayLike,
    dataset: DatasetName,
) -> MetricResult:
    """AUUC (scikit-uplift ``uplift_auc_score``) with a within-arm bootstrap interval (K2).

    ``split_name``, ``row_ids`` and ``dataset`` are required (ADR 0004).
    """
    return _bootstrap_metric(
        "auuc",
        point_auuc,
        y,
        uplift,
        t,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        split_name=split_name,
        row_ids=row_ids,
        dataset=dataset,
        settings={},
        require_positive_outcome=True,
    )


def uplift_at_k(
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    split_name: SplitName,
    row_ids: npt.ArrayLike,
    dataset: DatasetName,
    k: float = DEFAULT_K,
    strategy: str = DEFAULT_STRATEGY,
) -> MetricResult:
    """uplift@k with a within-arm bootstrap interval; ``k`` and ``strategy`` are recorded.

    ``k`` is a fraction strictly between 0 and 1. scikit-uplift 0.5.1 rejects a
    float ``k`` of 1.0 and reads an int as a row count, so both are refused here
    instead of being translated. ``split_name``, ``row_ids`` and ``dataset`` are
    required (ADR 0004).
    """
    if isinstance(k, bool) or not isinstance(k, int | float) or not 0.0 < k < 1.0:
        raise ContractError(
            f"k must be a fraction strictly between 0 and 1, got {k!r}."
        )
    if strategy not in UPLIFT_AT_K_STRATEGIES:
        raise ContractError(
            f"strategy must be one of {sorted(UPLIFT_AT_K_STRATEGIES)}, got {strategy!r}."
        )
    fraction = float(k)

    def point(y_: IntArray, u_: FloatArray, t_: IntArray) -> float:
        return point_uplift_at_k(y_, u_, t_, k=fraction, strategy=strategy)

    return _bootstrap_metric(
        UPLIFT_AT_K,
        point,
        y,
        uplift,
        t,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        split_name=split_name,
        row_ids=row_ids,
        dataset=dataset,
        settings={"k": fraction, "strategy": strategy, "ties": "fractional"},
        require_positive_outcome=False,
    )
