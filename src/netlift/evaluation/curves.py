"""Qini, uplift and cumulative-gain curves with pointwise bootstrap confidence bands.

The curves come from scikit-uplift (``qini_curve``, ``uplift_curve``). The bands
use the same within-arm resamples as the metrics (``within_arm_bootstrap_indices``),
so a curve band and a metric interval computed with the same ``t`` and ``seed``
share one ``resample_fingerprint``.

How the resample curves are aligned. scikit-uplift returns a piecewise-linear
curve with a knot at every distinct score, x counted in targeted rows. Resampling
within arms keeps both arm sizes, so every resample has the same N rows and every
curve runs from x = 0 to x = N: the grid needs no extrapolation and no endpoint
rule. Each curve is evaluated on the fractions ``0, 1/G, ..., 1`` (x = fraction * N)
by linear interpolation between its knots, which is the interpolation the
trapezoidal area (the Qini and AUUC integrals) already assumes. At each grid
point the band is the percentile interval of the resample values.

The band is pointwise: each grid point has its own 95% coverage, not the curve as
a whole. The point curve (on the original data) can lie outside the band, as a
percentile bootstrap does not guarantee otherwise.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import sklift.metrics as sk_metrics  # type: ignore[import-untyped]

from netlift.data.contracts import SplitName
from netlift.evaluation.contracts import (
    SOURCE_OF_TRUTH_LIBRARY,
    fingerprint_evaluation_set,
)
from netlift.evaluation.metrics import (
    DEFAULT_SPLIT,
    FloatArray,
    IntArray,
    check_bootstrap_arguments,
    ensure_sklift_api,
    evaluate_on_resamples,
    library_version,
    prepare_inputs,
)
from netlift.utils.errors import ContractError

DEFAULT_GRID_POINTS = 100

CurveFunction = Callable[
    [IntArray, FloatArray, IntArray], tuple[FloatArray, FloatArray]
]


@dataclass(frozen=True)
class CurveBand:
    """A curve on a common fraction grid, with its pointwise bootstrap band.

    ``point``, ``low``, ``high`` and ``random_baseline`` are in the curve's own
    unit (incremental conversions for Qini, scikit-uplift's uplift-curve unit
    for the other two) at each fraction of the population targeted.
    ``random_baseline`` is the straight line from (0, 0) to the point curve's end.
    """

    name: str
    fraction: FloatArray
    point: FloatArray
    low: FloatArray
    high: FloatArray
    random_baseline: FloatArray
    confidence_level: float
    n_resamples: int
    seed: int
    library: str
    library_version: str
    resample_fingerprint: str
    evaluation_fingerprint: str
    split_name: SplitName
    band_kind: str = field(default="pointwise")

    def __post_init__(self) -> None:
        """Check that every array sits on the same grid."""
        shape = self.fraction.shape
        for label, array in (
            ("point", self.point),
            ("low", self.low),
            ("high", self.high),
            ("random_baseline", self.random_baseline),
        ):
            if array.shape != shape:
                raise ContractError(f"{self.name}: {label} does not match the grid.")
        if not (self.low <= self.high).all():
            raise ContractError(f"{self.name}: the band has low above high.")


def _curve_on_grid(
    function: CurveFunction,
    y: IntArray,
    uplift: FloatArray,
    t: IntArray,
    grid: FloatArray,
    what: str,
) -> FloatArray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, values = function(y, uplift, t)
    x = np.asarray(x, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n_rows = float(y.size)
    if x.size < 2 or x[0] != 0.0 or x[-1] != n_rows:
        raise ContractError(
            f"{what}: the curve must run from 0 to {int(n_rows)} targeted rows "
            f"(got {x[:1]} to {x[-1:]}); the grid alignment relies on it."
        )
    if not np.isfinite(values).all():
        raise ContractError(f"{what}: the curve has non-finite values.")
    return np.interp(grid * n_rows, x, values)  # type: ignore[no-any-return]


def _curve_band(
    name: str,
    function: CurveFunction,
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    split_name: str,
    n_grid: int,
) -> CurveBand:
    check_bootstrap_arguments(n_resamples, confidence_level, seed, split_name)
    if isinstance(n_grid, bool) or not isinstance(n_grid, int) or n_grid < 2:
        raise ContractError(f"n_grid must be an int of at least 2, got {n_grid!r}.")
    ensure_sklift_api()
    y_array, score, t_array = prepare_inputs(
        y, uplift, t, require_positive_outcome=True
    )
    grid = np.linspace(0.0, 1.0, n_grid + 1)
    point = _curve_on_grid(
        function, y_array, score, t_array, grid, f"{name}: point curve"
    )

    def evaluate(number: int, indices: IntArray) -> FloatArray:
        return _curve_on_grid(
            function,
            y_array[indices],
            score[indices],
            t_array[indices],
            grid,
            f"{name}: bootstrap resample {number}",
        )

    curves, fingerprint = evaluate_on_resamples(t_array, n_resamples, seed, evaluate)
    tail = (1.0 - confidence_level) / 2.0 * 100.0
    low, high = np.percentile(np.vstack(curves), [tail, 100.0 - tail], axis=0)
    return CurveBand(
        name=name,
        fraction=grid,
        point=point,
        low=low,
        high=high,
        random_baseline=grid * point[-1],
        confidence_level=float(confidence_level),
        n_resamples=n_resamples,
        seed=seed,
        library=SOURCE_OF_TRUTH_LIBRARY,
        library_version=library_version(),
        resample_fingerprint=fingerprint,
        evaluation_fingerprint=fingerprint_evaluation_set(y_array, t_array),
        split_name=split_name,  # type: ignore[arg-type]
    )


def qini_curve(
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    split_name: SplitName = DEFAULT_SPLIT,
    n_grid: int = DEFAULT_GRID_POINTS,
) -> CurveBand:
    """Qini curve (cumulative incremental conversions) with a bootstrap band."""
    return _curve_band(
        "qini_curve",
        sk_metrics.qini_curve,
        y,
        uplift,
        t,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        split_name=split_name,
        n_grid=n_grid,
    )


def uplift_curve(
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    split_name: SplitName = DEFAULT_SPLIT,
    n_grid: int = DEFAULT_GRID_POINTS,
) -> CurveBand:
    """scikit-uplift's uplift curve, (treated rate - control rate) x rows targeted, with a band."""
    return _curve_band(
        "uplift_curve",
        sk_metrics.uplift_curve,
        y,
        uplift,
        t,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        split_name=split_name,
        n_grid=n_grid,
    )


def cumulative_gain_curve(
    y: npt.ArrayLike,
    uplift: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    split_name: SplitName = DEFAULT_SPLIT,
    n_grid: int = DEFAULT_GRID_POINTS,
) -> CurveBand:
    """Cumulative gain for the dashboard.

    scikit-uplift's uplift curve is the cumulative gain (lift x rows targeted),
    so this is the same quantity under its dashboard name; scikit-uplift has no
    separate cumulative-gain function and none is invented here.
    """
    return _curve_band(
        "cumulative_gain_curve",
        sk_metrics.uplift_curve,
        y,
        uplift,
        t,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        split_name=split_name,
        n_grid=n_grid,
    )
