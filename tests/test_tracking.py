"""Tests for the experiment tracking wrapper.

These check the rules the wrapper exists to enforce, not MLflow itself. If one
of these fails, a result could be logged that nobody can reproduce or defend.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from netlift.utils.tracking import (
    KPI_CONFIG,
    TrackingError,
    load_kpi_ids,
    start_run,
)


@pytest.fixture(autouse=True)
def isolated_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every test at its own throwaway tracking database."""
    database = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{database.as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")


# --------------------------------------------------------------------------- #
# KPI definitions
# --------------------------------------------------------------------------- #


def test_kpi_config_exists() -> None:
    """The KPI file is the source of metric names, so it has to be there."""
    assert KPI_CONFIG.exists(), f"config/kpis.yaml missing at {KPI_CONFIG}"


def test_kpi_ids_load() -> None:
    ids = load_kpi_ids()
    assert "K1" in ids
    assert "K2" in ids
    assert len(ids) >= 11


def test_missing_kpi_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TrackingError, match="not found"):
        load_kpi_ids(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------- #
# the rules this module exists to enforce
# --------------------------------------------------------------------------- #


def test_unknown_kpi_is_rejected() -> None:
    """A metric name that is not a defined KPI cannot be logged as a result."""
    with (
        start_run("unknown kpi", seed=1) as run,
        pytest.raises(TrackingError, match="K99"),
    ):
        run.log_kpi("K99", value=0.1, ci_low=0.0, ci_high=0.2)


def test_reversed_interval_is_rejected() -> None:
    with (
        start_run("reversed", seed=1) as run,
        pytest.raises(TrackingError, match="wrong way"),
    ):
        run.log_kpi("K2", value=0.1, ci_low=0.5, ci_high=0.0)


def test_value_outside_its_interval_is_rejected() -> None:
    with (
        start_run("outside", seed=1) as run,
        pytest.raises(TrackingError, match="outside"),
    ):
        run.log_kpi("K2", value=0.9, ci_low=0.0, ci_high=0.2)


def test_directory_tracking_store_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLflow 3.x rejects the ./mlruns store. Fail early, with the reason."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "./mlruns")
    with pytest.raises(TrackingError, match="directory store"), start_run("nope"):
        pass


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def test_seed_is_applied_not_merely_recorded() -> None:
    """Two runs with the same seed must draw the same numbers.

    If the wrapper logged the seed without applying it, the tag would be a
    claim rather than a fact, and every run would be unreproducible while
    appearing otherwise.
    """
    with start_run("seed a", seed=123):
        first = (random.random(), float(np.random.rand()))
    with start_run("seed b", seed=123):
        second = (random.random(), float(np.random.rand()))
    assert first == second


def test_provenance_tags_are_recorded(tmp_path: Path) -> None:
    """Commit, dirty flag, seed and dataset checksum land on the run."""
    import mlflow

    dataset = tmp_path / "tiny.csv"
    dataset.write_text("a,b\n1,2\n", encoding="utf-8")

    with start_run("provenance", dataset=dataset, seed=7) as run:
        run.log_metrics({"rows": 1})

    experiment = mlflow.get_experiment_by_name("test")
    assert experiment is not None
    runs = mlflow.search_runs([experiment.experiment_id], max_results=1)
    tags = {
        c[len("tags.") :]: runs.iloc[0][c]
        for c in runs.columns
        if c.startswith("tags.")
    }

    assert tags["seed"] == "7"
    assert tags["dataset_file"] == "tiny.csv"
    assert len(tags["dataset_sha256"]) == 64
    assert tags["git_dirty"] in {"true", "false"}
    assert tags["git_commit"]


def test_kpi_logs_value_and_both_bounds(tmp_path: Path) -> None:
    import mlflow

    with start_run("kpi", seed=1) as run:
        run.log_kpi("K2", value=0.0847, ci_low=0.0612, ci_high=0.1103)

    experiment = mlflow.get_experiment_by_name("test")
    assert experiment is not None
    runs = mlflow.search_runs([experiment.experiment_id], max_results=1)
    row = runs.iloc[0]

    assert row["metrics.K2"] == pytest.approx(0.0847)
    assert row["metrics.K2_ci_low"] == pytest.approx(0.0612)
    assert row["metrics.K2_ci_high"] == pytest.approx(0.1103)
