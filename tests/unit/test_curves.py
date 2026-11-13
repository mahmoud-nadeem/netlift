"""Tests for the curves and their bootstrap bands (T39).

Hand-computed values use fixture A of ``test_metrics.py`` (the curve tables are
derived in its docstring): the Qini curve at m = 0..8 targeted rows is
0, 1, 1, 2, 1, 1/2, 1, 2/3, 1 and the uplift curve is 0, 1, 2, 3, 2, 5/6, 2, 7/6, 2.
With ``n_grid=8`` the fractions are m/8, so the grid points are the knots and
no interpolation is involved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import sklift.metrics as sk_metrics  # type: ignore[import-untyped]

from netlift.evaluation.contracts import (
    fingerprint_evaluation_set,
    fingerprint_indices,
    within_arm_bootstrap_indices,
)
from netlift.evaluation.curves import (
    CurveBand,
    cumulative_gain_curve,
    qini_curve,
    uplift_curve,
)
from netlift.evaluation.metrics import qini_coefficient
from netlift.utils.errors import ContractError

N_RESAMPLES = 50  # unit tests only; a reported band needs 1,000 resamples.

# Fixture A of test_metrics.py, not in score order.
_ORDER = [3, 6, 0, 7, 2, 5, 1, 4]
A_UPLIFT = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])[_ORDER]
A_T = np.array([1, 0, 1, 0, 1, 0, 1, 0])[_ORDER]
A_Y = np.array([1, 0, 1, 1, 0, 0, 0, 0])[_ORDER]


def _data(
    n: int = 600, seed: int = 0
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.15 + 0.15 * t * (x > 0)).astype(np.int64)
    return y, x + rng.normal(0.0, 0.5, n), t


CURVES = [qini_curve, uplift_curve, cumulative_gain_curve]
QINI_A = [0, 1, 1, 2, 1, 0.5, 1, 2 / 3, 1]
UPLIFT_A = [0, 1, 2, 3, 2, 5 / 6, 2, 7 / 6, 2]


def _band(curve: Any, **overrides: Any) -> CurveBand:
    y, uplift, t = _data()
    arguments: dict[str, Any] = {
        "n_resamples": N_RESAMPLES,
        "confidence_level": 0.95,
        "seed": 7,
    }
    arguments.update(overrides)
    band: CurveBand = curve(y, uplift, t, **arguments)
    return band


def test_hand_computed_qini_curve_on_the_grid() -> None:
    band = qini_curve(
        A_Y, A_UPLIFT, A_T, n_resamples=10, confidence_level=0.95, seed=0, n_grid=8
    )
    assert band.fraction == pytest.approx(np.arange(9) / 8)
    assert band.point == pytest.approx(QINI_A)
    assert band.random_baseline == pytest.approx(np.arange(9) / 8 * 1.0)


@pytest.mark.parametrize("curve", [uplift_curve, cumulative_gain_curve])
def test_hand_computed_uplift_curve_on_the_grid(curve: Any) -> None:
    band = curve(
        A_Y, A_UPLIFT, A_T, n_resamples=10, confidence_level=0.95, seed=0, n_grid=8
    )
    assert band.point == pytest.approx(UPLIFT_A)


def test_interpolation_between_knots_is_linear() -> None:
    """Grid fraction 1/16 is 0.5 rows: half way between knots m=0 (0) and m=1 (1) -> 0.5.

    Fraction 3/16 is 1.5 rows: half way between m=1 (1) and m=2 (1) -> 1.0.
    """
    band = qini_curve(
        A_Y, A_UPLIFT, A_T, n_resamples=10, confidence_level=0.95, seed=0, n_grid=16
    )
    assert band.point[1] == pytest.approx(0.5)
    assert band.point[3] == pytest.approx(1.0)


def test_point_curve_matches_scikit_uplift_knots() -> None:
    y, uplift, t = _data()
    band = _band(qini_curve, n_grid=10)
    x, values = sk_metrics.qini_curve(y, uplift, t)
    assert band.point == pytest.approx(np.interp(band.fraction * y.size, x, values))


@pytest.mark.parametrize("curve", CURVES)
def test_band_shape_and_ordering(curve: Any) -> None:
    band = _band(curve, n_grid=20)
    assert band.fraction.shape == band.point.shape == band.low.shape == (21,)
    assert (band.low <= band.high).all()
    assert band.fraction[0] == 0.0
    assert band.fraction[-1] == 1.0
    assert band.band_kind == "pointwise"
    # Every curve starts at (0, 0): the band has zero width there.
    assert band.low[0] == band.high[0] == band.point[0] == 0.0


@pytest.mark.parametrize("curve", CURVES)
def test_band_is_deterministic_and_seed_dependent(curve: Any) -> None:
    first, again, other = _band(curve), _band(curve), _band(curve, seed=8)
    assert np.array_equal(first.low, again.low)
    assert np.array_equal(first.high, again.high)
    assert first.resample_fingerprint == again.resample_fingerprint
    assert first.resample_fingerprint != other.resample_fingerprint
    assert not np.array_equal(first.high, other.high)


@pytest.mark.parametrize("curve", CURVES)
def test_band_fingerprints_match_the_contract(curve: Any) -> None:
    y, _, t = _data()
    band = _band(curve)
    assert band.resample_fingerprint == fingerprint_indices(
        within_arm_bootstrap_indices(t, N_RESAMPLES, 7)
    )
    assert band.evaluation_fingerprint == fingerprint_evaluation_set(y, t)
    assert band.split_name == "validation"
    assert band.library == "scikit-uplift"


def test_curve_band_shares_resamples_with_the_metric() -> None:
    y, uplift, t = _data()
    metric = qini_coefficient(
        y, uplift, t, n_resamples=N_RESAMPLES, confidence_level=0.95, seed=7
    )
    assert metric.resample_fingerprint == _band(qini_curve).resample_fingerprint


def test_band_equals_pointwise_percentiles_of_the_resample_curves() -> None:
    y, uplift, t = _data()
    band = _band(qini_curve, n_grid=10)
    grid = band.fraction * y.size
    stack = []
    for indices in within_arm_bootstrap_indices(t, N_RESAMPLES, 7):
        x, values = sk_metrics.qini_curve(y[indices], uplift[indices], t[indices])
        stack.append(np.interp(grid, x, values))
    low, high = np.percentile(np.vstack(stack), [2.5, 97.5], axis=0)
    assert band.low == pytest.approx(low)
    assert band.high == pytest.approx(high)


def test_lower_confidence_level_gives_a_nested_band() -> None:
    wide, narrow = _band(qini_curve), _band(qini_curve, confidence_level=0.8)
    assert (wide.low <= narrow.low + 1e-12).all()
    assert (narrow.high <= wide.high + 1e-12).all()


@pytest.mark.parametrize("curve", CURVES)
@pytest.mark.parametrize(
    "override",
    [
        {"n_grid": 1},
        {"n_grid": True},
        {"n_resamples": 0},
        {"confidence_level": 1.0},
        {"split_name": "holdout"},
    ],
)
def test_invalid_arguments_are_rejected(curve: Any, override: dict[str, Any]) -> None:
    with pytest.raises(ContractError):
        _band(curve, **override)


@pytest.mark.parametrize("curve", CURVES)
def test_invalid_data_is_rejected(curve: Any) -> None:
    kwargs = {"n_resamples": 5, "confidence_level": 0.95, "seed": 0}
    with pytest.raises(ContractError, match="both arms"):
        curve([0, 1, 1], [0.1, 0.2, 0.3], [1, 1, 1], **kwargs)
    with pytest.raises(ContractError, match="no positive outcome"):
        curve([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1], **kwargs)
    with pytest.raises(ContractError, match="NaN or infinite"):
        curve([0, 1, 1, 0], [0.1, np.nan, 0.3, 0.4], [0, 1, 0, 1], **kwargs)
