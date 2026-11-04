"""Tests for the evaluation metrics (T39): hand-computed values first, then the bootstrap.

Hand-computed fixtures
----------------------
Fixture A (Qini, AUUC). Eight rows, listed here in score order (best first)::

    row  uplift  t  y
     0    0.8    1  1
     1    0.7    0  0
     2    0.6    1  1
     3    0.5    0  1
     4    0.4    1  0
     5    0.3    0  0
     6    0.2    1  0
     7    0.1    0  0

Cumulative counts after the top m rows (T treated, C control, Y1 treated
conversions, Y0 control conversions)::

    m   T  C  Y1  Y0   Qini = Y1 - Y0*T/C      Uplift = (Y1/T - Y0/C)*m
    1   1  0  1   0    1                        1
    2   1  1  1   0    1                        2
    3   2  1  2   0    2                        3
    4   2  2  2   1    2 - 1*1     = 1          (1 - 1/2)*4 = 2
    5   3  2  2   1    2 - 3/2     = 1/2        (2/3 - 1/2)*5 = 5/6
    6   3  3  2   1    2 - 1       = 1          (2/3 - 1/3)*6 = 2
    7   4  3  2   1    2 - 4/3     = 2/3        (1/2 - 1/3)*7 = 7/6
    8   4  4  2   1    2 - 1       = 1          (1/2 - 1/4)*8 = 2

Qini curve points (0..8): 0, 1, 1, 2, 1, 1/2, 1, 2/3, 1. Trapezoid area = 23/3.
Random line: (0, 0) to (8, 1), area 4.
Perfect Qini curve: score +1 for a treated converter, -1 for a control converter,
0 otherwise. Blocks: rows {0, 2} (+1) end at m=2 with value 2; rows {1,4,5,6,7} (0)
end at m=7 with T=4, C=3, Y1=2, Y0=0, value 2; row {3} (-1) ends at m=8 with
value 1. Points (0,0), (2,2), (7,2), (8,1): area 2 + 10 + 3/2 = 13.5.
Qini coefficient = (23/3 - 4) / (13.5 - 4) = (11/3) / (19/2) = 22/57.

Uplift curve points (0..8): 0, 1, 2, 3, 2, 5/6, 2, 7/6, 2. Trapezoid area = 13.
Random line: (0, 0) to (8, 2), area 8.
Perfect uplift curve (scikit-uplift): control responders (y=1, t=0) = 1 (row 3),
treated non-responders (y=0, t=1) = 2 (rows 4, 6). Since 1 > 2 is false, the
ideal score is 2*(y == t) + t: rows {0, 2} -> 3, rows {1, 5, 7} -> 2,
rows {4, 6} -> 1, row {3} -> 0. Block ends: m=2 -> (2/2)*2 = 2;
m=5 (T=2, C=3, Y1=2, Y0=0) -> 5; m=7 (T=4, C=3, Y1=2, Y0=0) -> (2/4)*7 = 3.5;
m=8 -> (1/2 - 1/4)*8 = 2. Points (0,0), (2,2), (5,5), (7,3.5), (8,2):
area 2 + 10.5 + 8.5 + 2.75 = 23.75.
AUUC = (13 - 8) / (23.75 - 8) = 5 / 15.75 = 20/63.

The perfect-curve rule above is scikit-uplift 0.5.1's; it was confirmed because the
hand derivation reproduces the library's value, not assumed.

Fixture B (uplift@k). Eight rows in score order, t/y: (1,1) (1,1) (1,0) (0,1)
(0,0) (1,0) (0,0) (0,0). Four treated (y = 1, 1, 0, 0) and four control
(y = 1, 0, 0, 0).

  overall,  k=0.5: top 4 rows = (1,1) (1,1) (1,0) (0,1): treated 2/3, control 1/1 -> -1/3.
  by_group, k=0.5: top 2 treated = 1, 1 -> 1; top 2 control = 1, 0 -> 1/2     -> +1/2.
  k=1.0, both:     treated 2/4, control 1/4                                   -> 1/4.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import sklift.metrics as sk_metrics  # type: ignore[import-untyped]

from netlift.evaluation import metrics as metrics_module
from netlift.evaluation.contracts import (
    MetricResult,
    fingerprint_evaluation_set,
    fingerprint_indices,
    paired_difference,
    within_arm_bootstrap_indices,
)
from netlift.evaluation.metrics import (
    auuc,
    point_auuc,
    point_qini_coefficient,
    point_uplift_at_k,
    qini_coefficient,
    uplift_at_k,
)
from netlift.utils.errors import ContractError

# Fixture A, deliberately NOT in score order so the sort is exercised.
_A_ORDER = [3, 6, 0, 7, 2, 5, 1, 4]
_A_UPLIFT = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
_A_T = np.array([1, 0, 1, 0, 1, 0, 1, 0])
_A_Y = np.array([1, 0, 1, 1, 0, 0, 0, 0])
A_UPLIFT = _A_UPLIFT[_A_ORDER]
A_T = _A_T[_A_ORDER]
A_Y = _A_Y[_A_ORDER]

_B_ORDER = [5, 2, 7, 0, 3, 6, 1, 4]
_B_UPLIFT = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
_B_T = np.array([1, 1, 1, 0, 0, 1, 0, 0])
_B_Y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
B_UPLIFT = _B_UPLIFT[_B_ORDER]
B_T = _B_T[_B_ORDER]
B_Y = _B_Y[_B_ORDER]

N_RESAMPLES = 50  # unit tests only; a reported result needs 1,000 (log_metric_result).


def _data(
    n: int = 600, seed: int = 0
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.15 + 0.15 * t * (x > 0)).astype(np.int64)
    uplift = x + rng.normal(0.0, 0.5, n)
    return y, uplift, t


def _run(metric: Any, **overrides: Any) -> MetricResult:
    y, uplift, t = _data()
    arguments: dict[str, Any] = {
        "n_resamples": N_RESAMPLES,
        "confidence_level": 0.95,
        "seed": 7,
    }
    arguments.update(overrides)
    result: MetricResult = metric(y, uplift, t, **arguments)
    return result


METRICS = [qini_coefficient, auuc, uplift_at_k]


# --------------------------------------------------------------------------- #
# Hand-computed point estimates (independent of scikit-uplift's own output)
# --------------------------------------------------------------------------- #


def test_hand_computed_qini_coefficient() -> None:
    assert point_qini_coefficient(A_Y, A_UPLIFT, A_T) == pytest.approx(22 / 57)


def test_hand_computed_auuc() -> None:
    """20/63 is what scikit-uplift actually returns for this fixture."""
    assert point_auuc(A_Y, A_UPLIFT, A_T) == pytest.approx(20 / 63)


@pytest.mark.parametrize(
    ("k", "strategy", "expected"),
    [
        (0.5, "overall", -1 / 3),
        (0.5, "by_group", 1 / 2),
        (1.0, "overall", 1 / 4),
        (1.0, "by_group", 1 / 4),
    ],
)
def test_hand_computed_uplift_at_k(k: float, strategy: str, expected: float) -> None:
    value = point_uplift_at_k(B_Y, B_UPLIFT, B_T, k=k, strategy=strategy)
    assert value == pytest.approx(expected)


def test_the_two_uplift_at_k_strategies_differ_on_fixture_b() -> None:
    overall = point_uplift_at_k(B_Y, B_UPLIFT, B_T, k=0.5, strategy="overall")
    by_group = point_uplift_at_k(B_Y, B_UPLIFT, B_T, k=0.5, strategy="by_group")
    assert overall != pytest.approx(by_group)


# --------------------------------------------------------------------------- #
# Cross-check of the wrapper's point estimate against scikit-uplift itself
# --------------------------------------------------------------------------- #


def test_point_estimates_match_scikit_uplift() -> None:
    y, uplift, t = _data()
    assert _run(qini_coefficient).value == pytest.approx(
        sk_metrics.qini_auc_score(y, uplift, t, negative_effect=True)
    )
    assert _run(auuc).value == pytest.approx(sk_metrics.uplift_auc_score(y, uplift, t))
    for strategy in ("overall", "by_group"):
        wrapped = _run(uplift_at_k, k=0.2, strategy=strategy).value
        direct = sk_metrics.uplift_at_k(y, uplift, t, strategy=strategy, k=0.2)
        assert wrapped == pytest.approx(direct)


def test_library_fields_come_from_the_installed_package() -> None:
    result = _run(qini_coefficient)
    assert result.library == "scikit-uplift"
    assert result.library_version == metadata.version("scikit-uplift")
    assert result.settings == {"negative_effect": "true"}


# --------------------------------------------------------------------------- #
# Determinism, interval, fingerprints, split name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", METRICS)
def test_same_inputs_and_seed_give_identical_results(metric: Any) -> None:
    assert _run(metric) == _run(metric)


@pytest.mark.parametrize("metric", METRICS)
def test_a_different_seed_changes_the_resamples_and_fingerprint(metric: Any) -> None:
    first, second = _run(metric, seed=1), _run(metric, seed=2)
    assert first.value == second.value
    assert first.resample_fingerprint != second.resample_fingerprint
    assert first.resamples != second.resamples


@pytest.mark.parametrize("metric", METRICS)
def test_interval_is_the_percentile_interval_of_the_resamples(metric: Any) -> None:
    result = _run(metric)
    low, high = np.percentile(result.resamples, [2.5, 97.5])
    assert result.n_resamples == N_RESAMPLES == len(result.resamples)
    assert result.ci_low == pytest.approx(low)
    assert result.ci_high == pytest.approx(high)
    assert result.ci_low <= result.ci_high
    assert result.confidence_level == 0.95


@pytest.mark.parametrize("metric", METRICS)
def test_a_lower_confidence_level_gives_a_nested_interval(metric: Any) -> None:
    wide = _run(metric, confidence_level=0.95)
    narrow = _run(metric, confidence_level=0.80)
    assert wide.ci_low <= narrow.ci_low <= narrow.ci_high <= wide.ci_high


@pytest.mark.parametrize("metric", METRICS)
def test_fingerprints_match_the_frozen_contract(metric: Any) -> None:
    y, _, t = _data()
    result = _run(metric)
    assert result.resample_fingerprint == fingerprint_indices(
        within_arm_bootstrap_indices(t, N_RESAMPLES, 7)
    )
    assert result.evaluation_fingerprint == fingerprint_evaluation_set(y, t)


@pytest.mark.parametrize("metric", METRICS)
def test_split_name_defaults_to_validation_and_can_be_set(metric: Any) -> None:
    assert _run(metric).split_name == "validation"
    assert _run(metric, split_name="train").split_name == "train"
    with pytest.raises(ContractError, match="split_name"):
        _run(metric, split_name="holdout")


def test_resamples_are_the_metric_on_the_contract_indices() -> None:
    """The first resample value is the point metric on the first contract resample."""
    y, uplift, t = _data()
    first = next(within_arm_bootstrap_indices(t, N_RESAMPLES, 7))
    expected = point_qini_coefficient(y[first], uplift[first], t[first])
    assert _run(qini_coefficient).resamples[0] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# uplift@k keys and settings
# --------------------------------------------------------------------------- #


def test_metric_keys_differ_for_different_k_and_strategy() -> None:
    keys = {
        _run(uplift_at_k, k=0.1).metric_key,
        _run(uplift_at_k, k=0.3).metric_key,
        _run(uplift_at_k, k=0.3, strategy="by_group").metric_key,
    }
    assert keys == {
        "uplift_at_10pct_overall",
        "uplift_at_30pct_overall",
        "uplift_at_30pct_by_group",
    }


# --------------------------------------------------------------------------- #
# Paired difference
# --------------------------------------------------------------------------- #


def test_paired_difference_of_two_scorers_on_the_same_resamples() -> None:
    y, uplift, t = _data()
    noise = np.random.default_rng(3).normal(size=uplift.size)
    kwargs = {"n_resamples": N_RESAMPLES, "confidence_level": 0.95, "seed": 7}
    good = qini_coefficient(y, uplift, t, **kwargs)
    random_scores = qini_coefficient(y, noise, t, **kwargs)
    difference = paired_difference(good, random_scores)
    per_resample = np.asarray(good.resamples) - np.asarray(random_scores.resamples)
    assert difference.name == "qini_coefficient_difference"
    assert difference.value == pytest.approx(good.value - random_scores.value)
    assert difference.ci_low == pytest.approx(np.percentile(per_resample, 2.5))
    assert good.resample_fingerprint == random_scores.resample_fingerprint


def test_paired_difference_refuses_different_splits_k_or_outcomes() -> None:
    with pytest.raises(ContractError, match="split_name"):
        paired_difference(_run(auuc), _run(auuc, split_name="train"))
    with pytest.raises(ContractError, match="settings"):
        paired_difference(_run(uplift_at_k, k=0.1), _run(uplift_at_k, k=0.3))
    y, uplift, t = _data()
    other_y = np.where(np.arange(y.size) % 7 == 0, 1 - y, y)
    kwargs = {"n_resamples": N_RESAMPLES, "confidence_level": 0.95, "seed": 7}
    with pytest.raises(ContractError, match="evaluation_fingerprint"):
        paired_difference(
            qini_coefficient(y, uplift, t, **kwargs),
            qini_coefficient(other_y, uplift, t, **kwargs),
        )


# --------------------------------------------------------------------------- #
# Invalid input and undefined results
# --------------------------------------------------------------------------- #


def _call(metric: Any, y: Any, uplift: Any, t: Any, **extra: Any) -> MetricResult:
    result: MetricResult = metric(
        y, uplift, t, n_resamples=5, confidence_level=0.95, seed=0, **extra
    )
    return result


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    ("y", "uplift", "t", "match"),
    [
        ([], [], [], "empty|both arms"),
        ([0, 1, 1], [0.1, 0.2], [0, 1, 1], "same length"),
        ([0, 2, 1, 1], [0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1], "only 0 and 1"),
        ([0, 1, 1, 0.5], [0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1], "only 0 and 1"),
        ([0, 1, 1, 0], [0.1, 0.2, 0.3, 0.4], [0, 1, 2, 1], "only 0 and 1"),
        ([0, 1, 1, 0], [0.1, 0.2, 0.3, 0.4], [0, 1, np.nan, 1], "only 0 and 1"),
        ([0, 1, 1, 0], [0.1, 0.2, 0.3, 0.4], [1, 1, 1, 1], "both arms"),
        ([0, 1, 1, 0], [0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0], "both arms"),
        ([0, 1, 1, 0], [0.1, np.nan, 0.3, 0.4], [0, 1, 0, 1], "NaN or infinite"),
        ([0, 1, 1, 0], [0.1, np.inf, 0.3, 0.4], [0, 1, 0, 1], "NaN or infinite"),
        ([[0, 1]], [[0.1, 0.2]], [[0, 1]], "one-dimensional"),
        (["a", "b"], [0.1, 0.2], [0, 1], "only 0 and 1"),
    ],
)
def test_invalid_inputs_are_rejected(
    metric: Any, y: Any, uplift: Any, t: Any, match: str
) -> None:
    with pytest.raises(ContractError, match=match):
        _call(metric, y, uplift, t)


@pytest.mark.parametrize("metric", [qini_coefficient, auuc])
def test_no_positive_outcome_is_rejected_not_zeroed(metric: Any) -> None:
    with pytest.raises(ContractError, match="no positive outcome"):
        _call(metric, [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])


@pytest.mark.parametrize("k", [0, 0.0, -0.1, 1.5, True, "0.3", None])
def test_invalid_k_is_rejected(k: Any) -> None:
    with pytest.raises(ContractError, match="k must be"):
        _run(uplift_at_k, k=k)


def test_unsupported_strategy_is_rejected() -> None:
    with pytest.raises(ContractError, match="strategy"):
        _run(uplift_at_k, strategy="top")


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "override",
    [
        {"n_resamples": 0},
        {"n_resamples": True},
        {"n_resamples": 10.0},
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"confidence_level": 95},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_invalid_bootstrap_arguments_are_rejected(
    metric: Any, override: dict[str, Any]
) -> None:
    with pytest.raises(ContractError):
        _run(metric, **override)


def test_a_slice_without_one_arm_is_undefined_not_zero() -> None:
    """k so small that the top slice holds no control row: uplift@k is undefined."""
    with pytest.raises(ContractError, match="undefined"):
        _run(uplift_at_k, k=0.001)


def test_an_undefined_resample_raises_and_names_the_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = metrics_module.sk_metrics.qini_auc_score
    calls = {"n": 0}

    def sometimes_nan(*args: Any, **kwargs: Any) -> float:
        calls["n"] += 1
        return float("nan") if calls["n"] == 4 else float(real(*args, **kwargs))

    monkeypatch.setattr(metrics_module.sk_metrics, "qini_auc_score", sometimes_nan)
    with pytest.raises(ContractError, match=r"resample 2.*undefined"):
        _run(qini_coefficient)
        
def test_an_int_k_of_one_means_one_hundred_percent() -> None:
    """scikit-uplift reads an int k as a row count; the wrapper never passes one."""
    result = _run(uplift_at_k, k=1)
    assert result.settings["k"] == 1.0
    assert result.metric_key == "uplift_at_100pct_overall"
