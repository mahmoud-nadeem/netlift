"""Tests for interface 2: the metric result and how it reaches the tracking wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy.typing as npt
import pytest

from netlift.evaluation.contracts import (
    MetricResult,
    UpliftMetric,
    log_metric_result,
)
from netlift.utils.errors import ContractError
from netlift.utils.tracking import start_run


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
    }
    fields.update(overrides)
    return MetricResult(**fields)  # type: ignore[arg-type]


def test_valid_result_is_accepted() -> None:
    result = _result(name="uplift_at_k", settings={"k": 0.3, "strategy": "overall"})
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
# logging through the tracking wrapper
# --------------------------------------------------------------------------- #


@pytest.fixture
def isolated_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the test at its own throwaway tracking database."""
    database = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{database.as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")


@pytest.mark.usefixtures("isolated_tracking")
def test_result_is_logged_against_a_kpi() -> None:
    import mlflow

    result = _result(name="uplift_at_k", settings={"k": 0.3, "strategy": "overall"})
    with start_run("metric result", seed=1) as run:
        log_metric_result(run, "K2", result)

    experiment = mlflow.get_experiment_by_name("test")
    assert experiment is not None
    row = mlflow.search_runs([experiment.experiment_id], max_results=1).iloc[0]
    assert row["metrics.K2"] == pytest.approx(0.0847)
    assert row["metrics.K2_ci_low"] == pytest.approx(0.0612)
    assert row["params.K2_library"] == "scikit-uplift==0.5.1"
    assert row["params.K2_strategy"] == "overall"
    assert row["tags.K2_confidence"] == "0.95"


@pytest.mark.usefixtures("isolated_tracking")
def test_too_few_resamples_cannot_be_reported() -> None:
    with (
        start_run("few resamples", seed=1) as run,
        pytest.raises(ContractError, match="at least 1000"),
    ):
        log_metric_result(run, "K2", _result(n_resamples=200))
