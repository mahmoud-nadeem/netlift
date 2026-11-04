"""T38 regression tests on T36 synthetic data; no real-data tests or results."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netlift.data.contracts import validate_features
from netlift.models.baselines import BASELINE_NAMES, build_baseline
from netlift.utils.errors import ContractError

PARAMS = {"n_estimators": 10, "min_child_samples": 5, "n_jobs": 1, "verbosity": -1}


def test_training_membership_scores_and_locked_test(synthetic_splits, monkeypatch):
    """Fit exactly treated train rows, keep row IDs aligned, never open test."""
    from netlift.models import baselines

    original = baselines.LGBMClassifier.fit
    calls = []

    def fit(self, X, y, **kwargs):
        calls.append((X.copy(), y.copy()))
        return original(self, X, y, **kwargs)

    def forbidden_open(self, reason):
        pytest.fail("T38 opened the locked test split")

    monkeypatch.setattr(baselines.LGBMClassifier, "fit", fit)
    monkeypatch.setattr(type(synthetic_splits.test), "open", forbidden_open)
    results = [
        build_baseline(
            synthetic_splits, name, seed=42, contact_cost=2, model_params=PARAMS
        )
        for name in BASELINE_NAMES
    ]
    train, val = synthetic_splits.train, synthetic_splits.validation
    assert len(calls) == 1
    pd.testing.assert_frame_equal(calls[0][0], train.X.loc[train.t == 1])
    np.testing.assert_array_equal(calls[0][1], train.y[train.t == 1])
    for result in results:
        np.testing.assert_array_equal(result.scores.row_id, val.row_id)
        assert len(result.scores) == val.n_rows
        assert np.isfinite(result.scores.score).all()
    validate_features(train.X)
    validate_features(val.X)
    blanket = results[1]
    assert blanket.metrics["average_effect"] == pytest.approx(
        val.y[val.t == 1].mean() - val.y[val.t == 0].mean()
    )
    assert (
        blanket.metrics["uplift_at_100pct_overall"] == blanket.metrics["average_effect"]
    )
    assert blanket.metrics["number_contacted"] == val.n_rows
    assert blanket.metrics["total_cost"] == val.n_rows * 2
    assert "qini_coefficient" not in blanket.metrics
    assert "auuc" not in blanket.metrics
    assert not synthetic_splits.test.is_open


def test_random_reproducible_and_seed_sensitive(synthetic_splits):
    """Seed controls random scores without altering row alignment."""

    def scores(seed):
        return build_baseline(
            synthetic_splits, "random", seed=seed, contact_cost=1, model_params=PARAMS
        ).scores

    pd.testing.assert_frame_equal(scores(42), scores(42))
    assert not np.array_equal(scores(42).score, scores(43).score)


def test_response_scores_ignore_validation_outcomes(synthetic_splits):
    """Validation outcomes may evaluate the ranking but never determine it."""
    changed = replace(
        synthetic_splits,
        validation=replace(
            synthetic_splits.validation, y=1 - synthetic_splits.validation.y
        ),
    )
    scores = [
        build_baseline(
            s, "response-model", seed=42, contact_cost=1, model_params=PARAMS
        ).scores
        for s in (synthetic_splits, changed)
    ]
    pd.testing.assert_frame_equal(*scores)


@pytest.mark.parametrize("cost", [0, -1, float("nan"), float("inf")])
def test_bad_contact_cost(synthetic_splits, cost):
    """Invalid economics must fail before logging cost figures."""
    with pytest.raises(ContractError, match="contact_cost"):
        build_baseline(
            synthetic_splits, "blanket", seed=42, contact_cost=cost, model_params=PARAMS
        )


def test_rechecks_mutated_features(synthetic_splits):
    """The frozen FeatureMatrix does not make its DataFrame immutable."""
    synthetic_splits.train.X["exposure"] = 0
    with pytest.raises(ContractError):
        build_baseline(
            synthetic_splits,
            "response-model",
            seed=42,
            contact_cost=1,
            model_params=PARAMS,
        )


def test_rejects_single_class_treated_training(synthetic_splits):
    """A degenerate response target must not silently become a classifier."""
    y = synthetic_splits.train.y.copy()
    y[synthetic_splits.train.t == 1] = 0
    changed = replace(synthetic_splits, train=replace(synthetic_splits.train, y=y))
    with pytest.raises(ContractError, match="treated train"):
        build_baseline(
            changed, "response-model", seed=42, contact_cost=1, model_params=PARAMS
        )


def test_real_mlflow_artifact_roundtrip(synthetic_splits, tmp_path, monkeypatch):
    """Three local MLflow runs retain aligned scores and reproducible models."""
    import joblib
    import mlflow

    from netlift.models import baselines

    spec = importlib.util.spec_from_file_location(
        "t38_runner", Path(__file__).resolve().parents[1] / "scripts/run_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_splits", lambda *a, **kw: synthetic_splits)
    raw = tmp_path / "synthetic.csv"
    synthetic_splits.train.X.to_csv(raw, index=False)
    monkeypatch.setattr(module, "HILLSTROM_CSV", raw)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.chdir(tmp_path)
    original_fit = baselines.LGBMClassifier.fit

    def fit_in_run(self, X, y, **kwargs):
        assert mlflow.active_run() is not None
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(baselines.LGBMClassifier, "fit", fit_in_run)
    rows = module.run_target("visit", {"seed": 42, "response_model": PARAMS}, 2)
    assert len(rows) == 3
    runs = mlflow.search_runs(experiment_names=["netlift-hillstrom-visit"])
    assert len(runs) == 3
    assert set(runs["tags.task"]) == {"T38"}
    assert set(runs["tags.result_status"]) == {"preliminary"}
    assert not any(c.startswith("metrics.K") for c in runs.columns)
    assert runs["tags.dataset_sha256"].str.len().eq(64).all()
    for run_id in runs.run_id:

        def artifact(name, run_id=run_id):
            return Path(
                mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name)
            )

        saved = pd.read_csv(artifact("validation_scores.csv"))
        np.testing.assert_array_equal(saved.row_id, synthetic_splits.validation.row_id)
        outcomes = pd.read_csv(artifact("validation_outcomes.csv"))
        np.testing.assert_array_equal(outcomes.row_id, saved.row_id)
        np.testing.assert_array_equal(outcomes.y, synthetic_splits.validation.y)
        manifest = json.loads(artifact("manifest.json").read_text())
        assert manifest["n_resamples"] == 0
        assert manifest["library"] == "scikit-uplift"
        if manifest["baseline"] == "response-model":
            model = joblib.load(artifact("response_model.joblib"))
            np.testing.assert_allclose(
                model.predict_proba(synthetic_splits.validation.X)[:, 1], saved.score
            )
    assert not synthetic_splits.test.is_open
