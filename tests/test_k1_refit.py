"""T57 to T44: the K1 model choice and the full-train refit.

The data here is SYNTHETIC. Never use it for a reported result.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from netlift.data.synthetic import make_synthetic_splits
from netlift.models.k1_refit import (
    SearchedModel,
    choose_k1_models,
    read_search_report,
    refit_and_score,
)
from netlift.utils import tracking
from netlift.utils.errors import ContractError
from netlift.utils.tracking import TrackingError

SMALL_LGBM = {"n_estimators": 20, "num_leaves": 4, "verbosity": -1, "n_jobs": 1}


def search_report(
    t_qini: float = 0.02, x_qini: float = 0.03, **overrides: Any
) -> dict[str, Any]:
    """A search summary shaped like reports/t57-criteo-conversion-search.json."""

    def family(name: str, value: float) -> dict[str, Any]:
        return {
            "family": name,
            "best_value": value,
            "best_params": SMALL_LGBM,
            "best_trial": 7,
            "n_trials_completed": 30,
            "elapsed_s": 1.0,
        }

    report: dict[str, Any] = {
        "task": "T57",
        "dataset": "criteo",
        "target": "conversion",
        "n_trials_budget": 30,
        "seed": 42,
        "families": [
            family("t_learner", t_qini),
            family("x_learner", x_qini),
            family("uplift_forest", 0.5),
        ],
    }
    report.update(overrides)
    return report


def test_the_higher_out_of_fold_qini_is_k1s_model() -> None:
    choice = choose_k1_models(search_report(t_qini=0.02, x_qini=0.03))
    assert choice.primary.family == "x_learner"
    assert choice.sensitivity.family == "t_learner"
    choice = choose_k1_models(search_report(t_qini=0.04, x_qini=0.03))
    assert choice.primary.family == "t_learner"
    assert choice.n_trials_budget == 30


def test_the_forest_is_never_a_k1_candidate() -> None:
    choice = choose_k1_models(search_report())
    assert {choice.primary.family, choice.sensitivity.family} == {
        "t_learner",
        "x_learner",
    }


def test_a_tie_goes_to_the_t_learner() -> None:
    assert (
        choose_k1_models(search_report(t_qini=0.03, x_qini=0.03)).primary.family
        == "t_learner"
    )


@pytest.mark.parametrize(
    "overrides",
    [{"dataset": "hillstrom"}, {"target": "visit"}],
)
def test_only_a_criteo_conversion_search_feeds_k1(overrides: dict[str, Any]) -> None:
    with pytest.raises(ContractError, match="Criteo conversion"):
        choose_k1_models(search_report(**overrides))


def test_a_missing_family_or_a_non_finite_qini_is_refused() -> None:
    report = search_report()
    report["families"] = report["families"][:1]
    with pytest.raises(ContractError, match="x_learner"):
        choose_k1_models(report)
    with pytest.raises(ContractError, match="non-finite"):
        choose_k1_models(search_report(x_qini=float("nan")))


def test_a_missing_report_is_named(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not found"):
        read_search_report(tmp_path / "absent.json")


@pytest.mark.parametrize("family", ["t_learner", "x_learner"])
def test_refit_scores_every_validation_row_and_leaves_test_locked(family: str) -> None:
    splits = make_synthetic_splits("visit", n_rows=2000, seed=0)
    model = SearchedModel(family, 0.02, SMALL_LGBM, 0, 30)  # type: ignore[arg-type]
    scores = refit_and_score(splits, model, seed=42)
    assert list(scores.columns) == ["row_id", "score"]
    assert np.array_equal(scores["row_id"].to_numpy(), splits.validation.row_id)
    assert np.isfinite(scores["score"]).all()
    assert not splits.test.is_open


def load_script() -> ModuleType:
    """Import scripts/refit_k1_models.py, which is not a package module."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "refit_k1_models.py"
    spec = importlib.util.spec_from_file_location("refit_k1_models", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, dirty: bool
) -> tuple[ModuleType, Path]:
    """The script wired to synthetic splits, a temporary tracking store and a report."""
    module = load_script()
    splits = make_synthetic_splits("visit", n_rows=2000, seed=0)
    monkeypatch.setattr(module, "load_splits", lambda *a, **kw: splits)
    monkeypatch.setattr(module, "git_dirty", lambda: dirty)
    monkeypatch.setattr(tracking, "git_dirty", lambda: dirty)
    data_file = tmp_path / "criteo.parquet"
    data_file.write_bytes(b"synthetic stand-in")
    monkeypatch.setattr(module, "CRITEO_PARQUET", data_file)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.chdir(tmp_path)
    report = tmp_path / "search.json"
    report.write_text(json.dumps(search_report()), encoding="utf-8")
    return module, report


def test_the_script_logs_both_runs_and_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    module, report = prepared_script(tmp_path, monkeypatch, dirty=False)
    out = tmp_path / "k1"
    module.main(["--report", str(report), "--out-dir", str(out)])

    summary = json.loads((out / "k1-input.json").read_text(encoding="utf-8"))
    assert summary["test_split_opened"] is False
    assert summary["runs"]["k1_uplift_model"]["family"] == "x_learner"
    assert summary["runs"]["k1_sensitivity"]["family"] == "t_learner"

    runs = mlflow.search_runs(experiment_names=["netlift-criteo-conversion"])
    assert len(runs) == 2
    assert set(runs["tags.k1_role"]) == {"k1_uplift_model", "k1_sensitivity"}
    assert set(runs["tags.task"]) == {"T57"}
    assert set(runs["tags.test_split_opened"]) == {"false"}
    assert set(runs["tags.git_dirty"]) == {"false"}
    assert set(runs["tags.mlflow.runName"]) == {
        "T57-x-learner-refit",
        "T57-t-learner-refit",
    }

    for family in ("x_learner", "t_learner"):
        manifest = json.loads(
            (out / family / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["tuned"] is True
        assert manifest["test_split_opened"] is False
        assert manifest["n_trials_completed"] == 30
        scores = pd.read_parquet(out / family / "validation_scores.parquet")
        outcomes = pd.read_parquet(out / family / "validation_outcomes.parquet")
        assert scores["row_id"].tolist() == outcomes["row_id"].tolist()
        run_id = summary["runs"][manifest["role"]]["run_id"]
        logged = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)}
        assert logged == {
            "manifest.json",
            "validation_outcomes.parquet",
            "validation_scores.parquet",
        }


def test_the_script_refuses_a_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, report = prepared_script(tmp_path, monkeypatch, dirty=True)
    with pytest.raises(TrackingError, match="uncommitted"):
        module.main(["--report", str(report), "--out-dir", str(tmp_path / "k1")])
    assert not (tmp_path / "k1").exists()
