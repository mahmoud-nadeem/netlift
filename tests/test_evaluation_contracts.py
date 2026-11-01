"""Tests for interface 2: the metric result and how it reaches the tracking wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from netlift.evaluation.contracts import (
    MetricResult,
    UpliftMetric,
    fingerprint_indices,
    log_metric_result,
    paired_difference,
    within_arm_bootstrap_indices,
)
from netlift.utils.errors import ContractError
from netlift.utils.tracking import start_run

FINGERPRINT = "ab" * 32


def _result(**overrides: object) -> MetricResult:
    fields: dict[str, object] = {
        "name": "qini_coefficient",
        "value": 0.0847,
        "ci_low": 0.0612,
        "ci_high": 0.1103,
        "confidence_level": 0.95,
        "n_resamples": 1000,
        "seed": 42,
        "library": "scikit-uplift",
        "library_version": "0.5.1",
        "resample_fingerprint": FINGERPRINT,
    }
    fields.update(overrides)
    if "resamples" not in fields:
        count = fields["n_resamples"]
        fields["resamples"] = (0.08,) * (count if isinstance(count, int) else 1)
    return MetricResult(**fields)  # type: ignore[arg-type]


def _uplift_at_k(k: float, strategy: str = "overall") -> MetricResult:
    return _result(name="uplift_at_k", settings={"k": k, "strategy": strategy})


def test_valid_result_is_accepted() -> None:
    result = _uplift_at_k(0.3)
    assert result.value_inside_interval
    assert result.settings["strategy"] == "overall"


def test_value_outside_the_interval_is_allowed() -> None:
    """A percentile bootstrap does not guarantee the estimate is inside."""
    result = _result(value=0.2)
    assert not result.value_inside_interval


def test_reversed_interval_is_rejected() -> None:
    with pytest.raises(ContractError, match="above ci_high"):
        _result(ci_low=0.2, ci_high=0.1)


@pytest.mark.parametrize("level", [0.0, 1.0, 95])
def test_confidence_level_must_be_a_fraction(level: float) -> None:
    with pytest.raises(ContractError, match="confidence_level"):
        _result(confidence_level=level)


@pytest.mark.parametrize("field", ["value", "ci_low", "ci_high"])
def test_non_finite_numbers_are_rejected(field: str) -> None:
    with pytest.raises(ContractError, match="not finite"):
        _result(**{field: float("nan")})


def test_reported_metric_must_come_from_scikit_uplift() -> None:
    with pytest.raises(ContractError, match="ADR 0001"):
        _result(library="causalml", library_version="0.17.0")


def test_other_metrics_may_use_another_library() -> None:
    result = _result(
        name="cumulative_gain", library="causalml", library_version="0.17.0"
    )
    assert result.library == "causalml"


def test_missing_provenance_is_rejected() -> None:
    with pytest.raises(ContractError, match="library"):
        _result(library_version="")


def test_a_conforming_function_satisfies_the_protocol() -> None:
    def constant_metric(
        y: npt.ArrayLike,
        uplift: npt.ArrayLike,
        t: npt.ArrayLike,
        *,
        n_resamples: int,
        confidence_level: float,
        seed: int,
    ) -> MetricResult:
        return _result(
            n_resamples=n_resamples, confidence_level=confidence_level, seed=seed
        )

    metric: UpliftMetric = constant_metric
    result = metric(
        [0, 1], [0.1, 0.2], [0, 1], n_resamples=1000, confidence_level=0.9, seed=3
    )
    assert result.seed == 3


# --------------------------------------------------------------------------- #
# uplift@k carries k and strategy
# --------------------------------------------------------------------------- #


def test_uplift_at_k_key_carries_k_and_strategy() -> None:
    assert _uplift_at_k(0.1).metric_key == "uplift_at_10pct_overall"
    assert _uplift_at_k(0.125, "by_group").metric_key == "uplift_at_12.5pct_by_group"
    assert _result().metric_key == "qini_coefficient"


@pytest.mark.parametrize(
    "settings",
    [
        {"strategy": "overall"},
        {"k": 0.0, "strategy": "overall"},
        {"k": 1.5, "strategy": "overall"},
        {"k": True, "strategy": "overall"},
        {"k": 0.1},
        {"k": 0.1, "strategy": "top"},
    ],
)
def test_uplift_at_k_requires_valid_k_and_strategy(settings: dict[str, object]) -> None:
    with pytest.raises(ContractError, match="settings"):
        _result(name="uplift_at_k", settings=settings)


# --------------------------------------------------------------------------- #
# resamples, fingerprint and the paired comparison
# --------------------------------------------------------------------------- #


def test_resample_count_must_match_n_resamples() -> None:
    with pytest.raises(ContractError, match="must match"):
        _result(n_resamples=1000, resamples=(0.1, 0.2))


def test_fingerprint_must_be_a_sha256_digest() -> None:
    with pytest.raises(ContractError, match="sha256"):
        _result(resample_fingerprint="not-a-digest")


def test_indices_stay_within_arms() -> None:
    t = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    for indices in within_arm_bootstrap_indices(t, n_resamples=20, seed=5):
        assert len(indices) == len(t)
        assert int((t[indices] == 0).sum()) == 3
        assert int((t[indices] == 1).sum()) == 5


def _mean_uplift(
    uplift: npt.NDArray[np.float64], t: npt.NDArray[np.int64], *, n: int, seed: int
) -> MetricResult:
    """A toy metric built the way T39 is meant to build real ones."""
    values = tuple(
        float(uplift[indices].mean())
        for indices in within_arm_bootstrap_indices(t, n_resamples=n, seed=seed)
    )
    low, high = np.percentile(values, [2.5, 97.5])
    return MetricResult(
        name="mean_uplift",
        value=float(uplift.mean()),
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=0.95,
        n_resamples=n,
        seed=seed,
        library="numpy",
        library_version=np.__version__,
        resamples=values,
        resample_fingerprint=fingerprint_indices(
            within_arm_bootstrap_indices(t, n_resamples=n, seed=seed)
        ),
    )


def test_same_t_and_seed_give_the_same_fingerprint_and_a_paired_difference() -> None:
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, size=300)
    champion = _mean_uplift(rng.normal(0.05, 0.02, size=300), t, n=200, seed=11)
    challenger = _mean_uplift(rng.normal(0.03, 0.02, size=300), t, n=200, seed=11)
    assert champion.resample_fingerprint == challenger.resample_fingerprint

    difference = paired_difference(champion, challenger)
    per_resample = np.asarray(champion.resamples) - np.asarray(challenger.resamples)
    assert difference.name == "mean_uplift_difference"
    assert difference.value == pytest.approx(champion.value - challenger.value)
    assert difference.ci_low == pytest.approx(np.percentile(per_resample, 2.5))
    assert difference.ci_high == pytest.approx(np.percentile(per_resample, 97.5))
    assert difference.resample_fingerprint == champion.resample_fingerprint


def test_a_different_seed_cannot_be_paired() -> None:
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, size=200)
    uplift = rng.normal(0.05, 0.02, size=200)
    first = _mean_uplift(uplift, t, n=100, seed=1)
    second = _mean_uplift(uplift, t, n=100, seed=2)
    assert first.resample_fingerprint != second.resample_fingerprint
    with pytest.raises(ContractError, match="seed"):
        paired_difference(first, second)


@pytest.mark.parametrize("t", [[0, 0, 1, 1, 2, 2], [0, 1, -1], [0.0, 1.0, 0.5]])
def test_indices_reject_t_outside_zero_and_one(t: list[float]) -> None:
    """A raw three-arm column would otherwise lose an arm without an error."""
    with pytest.raises(ContractError, match="binary 0/1"):
        next(within_arm_bootstrap_indices(np.array(t), n_resamples=3, seed=0))


def test_indices_accept_bool_t() -> None:
    t = np.array([False, False, True, True])
    assert len(next(within_arm_bootstrap_indices(t, n_resamples=1, seed=0))) == 4


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"confidence_level": 0.9}, "confidence_level"),
        ({"library_version": "0.5.2"}, "library_version"),
    ],
)
def test_different_level_or_library_version_cannot_be_paired(
    override: dict[str, object], field: str
) -> None:
    with pytest.raises(ContractError, match=field):
        paired_difference(_result(), _result(**override))


def test_different_settings_cannot_be_paired() -> None:
    with pytest.raises(ContractError, match="settings"):
        paired_difference(_uplift_at_k(0.1), _uplift_at_k(0.3))


# --------------------------------------------------------------------------- #
# logging through the tracking wrapper
# --------------------------------------------------------------------------- #


@pytest.fixture
def isolated_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the test at its own throwaway tracking database."""
    database = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{database.as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")


def _last_run() -> pd.Series:
    import mlflow

    experiment = mlflow.get_experiment_by_name("test")
    assert experiment is not None
    runs = mlflow.search_runs([experiment.experiment_id], max_results=1)
    assert isinstance(runs, pd.DataFrame)
    return runs.iloc[0]


@pytest.mark.usefixtures("isolated_tracking")
def test_result_is_logged_under_kpi_and_metric_key() -> None:
    with start_run("metric result", seed=1) as run:
        key = log_metric_result(run, "K2", _uplift_at_k(0.3))

    assert key == "K2_uplift_at_30pct_overall"
    row = _last_run()
    assert row["metrics.K2_uplift_at_30pct_overall"] == pytest.approx(0.0847)
    assert row["metrics.K2_uplift_at_30pct_overall_ci_low"] == pytest.approx(0.0612)
    assert row["params.K2_uplift_at_30pct_overall_library"] == "scikit-uplift==0.5.1"
    assert row["params.K2_uplift_at_30pct_overall_strategy"] == "overall"
    assert row["params.K2_uplift_at_30pct_overall_resample_fingerprint"] == FINGERPRINT
    assert row["tags.K2_uplift_at_30pct_overall_confidence"] == "0.95"


@pytest.mark.usefixtures("isolated_tracking")
def test_two_k_values_share_one_run() -> None:
    """The collision Ali found: k=0.1 and k=0.3 used to share one key."""
    with start_run("two k", seed=1) as run:
        first = log_metric_result(run, "K2", _uplift_at_k(0.1))
        second = log_metric_result(run, "K2", _uplift_at_k(0.3))

    assert (first, second) == (
        "K2_uplift_at_10pct_overall",
        "K2_uplift_at_30pct_overall",
    )
    row = _last_run()
    assert row["params.K2_uplift_at_10pct_overall_k"] == "0.1"
    assert row["params.K2_uplift_at_30pct_overall_k"] == "0.3"


@pytest.mark.usefixtures("isolated_tracking")
def test_qini_and_auuc_share_one_run_under_k2() -> None:
    qini = _result(name="qini_coefficient", seed=7)
    auuc = _result(
        name="auuc", value=0.031, ci_low=0.020, ci_high=0.045, n_resamples=2000, seed=8
    )
    with start_run("two metrics", seed=1) as run:
        log_metric_result(run, "K2", qini)
        log_metric_result(run, "K2", auuc)

    row = _last_run()
    assert row["metrics.K2_qini_coefficient"] == pytest.approx(0.0847)
    assert row["metrics.K2_auuc"] == pytest.approx(0.031)
    assert row["metrics.K2_auuc_ci_high"] == pytest.approx(0.045)
    assert row["params.K2_qini_coefficient_n_resamples"] == "1000"
    assert row["params.K2_auuc_n_resamples"] == "2000"
    assert row["params.K2_qini_coefficient_seed"] == "7"
    assert row["params.K2_auuc_seed"] == "8"


@pytest.mark.usefixtures("isolated_tracking")
def test_outside_interval_result_is_tagged() -> None:
    with start_run("outside", seed=1) as run:
        log_metric_result(run, "K2", _result(name="auuc", value=0.2))

    row = _last_run()
    assert row["tags.K2_auuc_value_outside_interval"] == "true"
    assert "tags.K2_qini_coefficient_value_outside_interval" not in row.index


@pytest.mark.usefixtures("isolated_tracking")
def test_too_few_resamples_cannot_be_reported() -> None:
    with (
        start_run("few resamples", seed=1) as run,
        pytest.raises(ContractError, match="at least 1000"),
    ):
        log_metric_result(run, "K2", _result(n_resamples=200))
