"""#107 runner test: a real local MLflow round trip on T36 synthetic data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from netlift.data.synthetic import make_synthetic_splits
from netlift.models import response_tuning as rt

ROOT = Path(__file__).resolve().parents[1]
SPACE = {
    "num_leaves": {"type": "int", "low": 4, "high": 8, "log": False},
    "min_child_samples": {"type": "int", "low": 5, "high": 20, "log": False},
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.2, "log": True},
    "n_estimators": {"type": "int", "low": 5, "high": 15, "log": False},
    "reg_lambda": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "t38_tuning_runner", ROOT / "scripts/tune_response_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset", ["hillstrom", "criteo"])
def test_run_one_logs_tuned_run_and_artifacts(
    synthetic_splits, tmp_path, monkeypatch, dataset
):
    """One tuned run keeps aligned scores, every trial and the required fields."""
    module = _load_script()
    if dataset == "criteo":
        synthetic_splits = make_synthetic_splits("conversion", n_rows=6000, seed=42)
    loads = []

    def load(*args, **kwargs):
        loads.append(kwargs)
        return synthetic_splits

    raw = tmp_path / "synthetic.csv"
    synthetic_splits.train.X.to_csv(raw, index=False)
    monkeypatch.setitem(module.DATA_FILES, dataset, raw)
    monkeypatch.setattr(module, "load_splits", load)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.chdir(tmp_path)
    config = rt.read_config(ROOT / "config/response_tuning.yaml")
    config["search_space"] = SPACE
    config["fixed"] = {"n_jobs": 1, "verbosity": -1, "deterministic": True}
    target = "conversion" if dataset == "criteo" else "visit"

    row = module.run_one(dataset, target, "log_loss", config, n_trials=2)

    assert loads == [{"target": target}]
    assert row["n_trials"] == 2
    runs = mlflow.search_runs(experiment_names=[f"netlift-{dataset}-{target}"])
    assert len(runs) == 1
    tags = runs.iloc[0]
    assert tags["tags.mlflow.runName"] == "T38-response-model-tuned"
    assert tags["tags.task"] == "T38"
    assert tags["tags.result_status"] == "preliminary"
    assert tags["tags.tuned"] == "true"
    assert tags["tags.n_trials"] == "2"
    assert tags["tags.test_split_opened"] == "false"
    assert not any(c.startswith("metrics.K") for c in runs.columns)

    run_id = runs.run_id.iloc[0]

    def artifact(name):
        return Path(
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name)
        )

    saved = pd.read_csv(artifact("validation_scores.csv"))
    np.testing.assert_array_equal(saved.row_id, synthetic_splits.validation.row_id)
    outcomes = pd.read_csv(artifact("validation_outcomes.csv"))
    np.testing.assert_array_equal(outcomes.row_id, saved.row_id)
    np.testing.assert_array_equal(outcomes.y, synthetic_splits.validation.y)
    assert len(pd.read_csv(artifact("trials.csv"))) == 2
    manifest = json.loads(artifact("manifest.json").read_text())
    assert manifest["tuned"] is True
    assert manifest["n_trials"] == 2
    assert manifest["test_split_opened"] is False
    assert manifest["n_resamples"] == 0
    assert manifest["metrics"]["auc_ci_low"] <= manifest["metrics"]["auc"]
    assert manifest["metrics"]["auc"] <= manifest["metrics"]["auc_ci_high"]
    if dataset == "criteo":
        assert manifest["tuning_rows"] < manifest["train_rows"]
    model = joblib.load(artifact("response_model.joblib"))
    columns = synthetic_splits.train.X.columns
    np.testing.assert_allclose(
        model.predict_proba(synthetic_splits.validation.X[columns])[:, 1], saved.score
    )
    assert not synthetic_splits.test.is_open


def test_run_one_rejects_unsupported_dataset(synthetic_splits):
    """Criteo has no visit target in this task, and unknown datasets fail early."""
    module = _load_script()
    config = rt.read_config(ROOT / "config/response_tuning.yaml")
    with pytest.raises(Exception, match="Unsupported"):
        module.run_one("criteo", "visit", "log_loss", config, n_trials=1)
    with pytest.raises(Exception, match="Unsupported"):
        module.run_one("other", "visit", "log_loss", config, n_trials=1)
