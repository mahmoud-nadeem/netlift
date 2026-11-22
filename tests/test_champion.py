"""T57 champion: choice, refit and registration, and the champion–challenger table.

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
from netlift.models.champion import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    choose_champion,
)
from netlift.models.uplift_search import fit_model, fit_predict
from netlift.utils import tracking
from netlift.utils.errors import ContractError

SMALL_FOREST = {"n_estimators": 3, "max_depth": 2, "min_samples_leaf": 50, "n_jobs": 1}
SMALL_LGBM = {"n_estimators": 20, "num_leaves": 4, "verbosity": -1, "n_jobs": 1}


def search_report(**values: float) -> dict[str, Any]:
    """A Criteo conversion search summary with the given out-of-fold Qini values."""
    qini = {"t_learner": 0.18, "x_learner": 0.17, "uplift_forest": 0.19, **values}
    return {
        "task": "T57",
        "dataset": "criteo",
        "target": "conversion",
        "n_trials_budget": 30,
        "seed": 42,
        "families": [
            {
                "family": family,
                "best_value": value,
                "best_params": SMALL_FOREST
                if family == "uplift_forest"
                else SMALL_LGBM,
                "best_trial": 7,
                "n_trials_completed": 30,
                "elapsed_s": 1.0,
            }
            for family, value in qini.items()
        ],
    }


def test_the_champion_is_the_best_out_of_fold_qini_of_any_family() -> None:
    assert choose_champion(search_report()).family == "uplift_forest"
    assert choose_champion(search_report(t_learner=0.3)).family == "t_learner"


def test_the_champion_needs_a_criteo_conversion_search() -> None:
    report = search_report()
    report["dataset"] = "hillstrom"
    with pytest.raises(ContractError, match="Criteo conversion"):
        choose_champion(report)
    with pytest.raises(ContractError, match="non-finite"):
        choose_champion(search_report(uplift_forest=float("nan")))


@pytest.mark.parametrize("family", ["t_learner", "x_learner", "uplift_forest"])
def test_a_fitted_model_predicts_what_fit_predict_returns(family: str) -> None:
    splits = make_synthetic_splits("visit", n_rows=2000, seed=0)
    train, validation = splits.train, splits.validation
    params = SMALL_FOREST if family == "uplift_forest" else SMALL_LGBM
    model = fit_model(family, params, train.X, train.t, train.y, seed=42)  # type: ignore[arg-type]
    direct = fit_predict(
        family,  # type: ignore[arg-type]
        params,
        train.X,
        train.t,
        train.y,
        validation.X,
        seed=42,
    )
    assert np.allclose(model.predict_uplift(validation.X), direct)
    with pytest.raises(ContractError):
        model.predict_uplift(validation.X.assign(exposure=0))


def load_script(name: str) -> ModuleType:
    """Import a script from scripts/, which is not a package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_champion_is_refitted_registered_and_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    module = load_script("refit_champion")
    splits = make_synthetic_splits("visit", n_rows=2000, seed=0)
    monkeypatch.setattr(module, "load_splits", lambda *a, **kw: splits)
    monkeypatch.setattr(module, "git_dirty", lambda: False)
    monkeypatch.setattr(tracking, "git_dirty", lambda: False)
    data_file = tmp_path / "criteo.parquet"
    data_file.write_bytes(b"synthetic stand-in")
    monkeypatch.setattr(module, "CRITEO_PARQUET", data_file)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    monkeypatch.chdir(tmp_path)
    report = tmp_path / "search.json"
    report.write_text(json.dumps(search_report()), encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy\n", encoding="utf-8")
    out = tmp_path / "champion"

    module.main(
        [
            "--report",
            str(report),
            "--out-dir",
            str(out),
            "--requirements",
            str(requirements),
        ]
    )

    summary = json.loads((out / "champion.json").read_text(encoding="utf-8"))
    assert summary["champion"]["family"] == "uplift_forest"
    assert summary["test_split_opened"] is False
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    assert version.tags["run_id"] == summary["mlflow_run_id"]
    assert version.tags["search_report_sha256"] == tracking.file_sha256(report)
    assert version.tags["git_commit"] == summary["git_commit"]
    run = client.get_run(summary["mlflow_run_id"])
    assert run.data.tags["task"] == "T57"
    assert run.data.tags["git_dirty"] == "false"
    assert run.data.tags["mlflow.runName"] == "T57-uplift-forest-champion"

    loaded = mlflow.pyfunc.load_model(
        f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    )
    predicted = np.asarray(loaded.predict(splits.validation.X), dtype=np.float64)
    stored = pd.read_parquet(out / "validation_scores.parquet")
    assert stored["row_id"].tolist() == splits.validation.row_id.tolist()
    assert np.allclose(predicted, stored["score"].to_numpy())


def write_scores(path: Path, row_id: np.ndarray, score: np.ndarray) -> Path:
    """Write a row_id + score parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": row_id, "score": score}).to_parquet(path, index=False)
    return path


def table_inputs(tmp_path: Path) -> list[str]:
    """Synthetic outcomes and three score files in shuffled row order."""
    rng = np.random.default_rng(0)
    n = 3000
    row_id = np.arange(n, dtype=np.int64) * 3
    t = rng.integers(0, 2, n)
    effect = rng.normal(size=n)
    y = (rng.random(n) < 0.2 + 0.1 * t * (effect > 0)).astype(int)
    pd.DataFrame({"row_id": row_id, "y": y, "t": t}).to_parquet(
        tmp_path / "outcomes.parquet", index=False
    )
    order = rng.permutation(n)
    files = {
        "forest": effect,
        "t": effect + rng.normal(scale=0.5, size=n),
        "x": rng.normal(size=n),
    }
    paths = {
        name: write_scores(tmp_path / f"{name}.parquet", row_id[order], score[order])
        for name, score in files.items()
    }
    return [
        "--outcomes",
        str(tmp_path / "outcomes.parquet"),
        "--forest",
        str(paths["forest"]),
        "--t-learner",
        str(paths["t"]),
        "--x-learner",
        str(paths["x"]),
        "--n-resamples",
        "20",
        "--out",
        str(tmp_path / "table"),
        "--results",
        str(tmp_path / "results.json"),
    ]


def test_the_table_has_every_metric_and_paired_differences(tmp_path: Path) -> None:
    module = load_script("champion_challenger_table")
    module.main(table_inputs(tmp_path))
    report = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    expected = {"qini", "auuc", "uplift_at_10", "uplift_at_20", "uplift_at_30"}
    for model in ("uplift_forest", "t_learner", "x_learner"):
        assert set(report["metrics"][model]) == expected
    assert set(report["differences_vs_forest"]) == {"t_learner", "x_learner"}
    assert report["pending"] == ["response_model"]
    difference = report["differences_vs_forest"]["x_learner"]["qini"]
    forest = report["metrics"]["uplift_forest"]["qini"]
    x = report["metrics"]["x_learner"]["qini"]
    assert difference["value"] == pytest.approx(x["value"] - forest["value"])
    text = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert "pending" in text and "63e214a5" in text
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(saved["uplift_forest"]["qini"]["resamples"]) == 20


def test_a_score_file_with_other_rows_is_refused(tmp_path: Path) -> None:
    module = load_script("champion_challenger_table")
    args = table_inputs(tmp_path)
    write_scores(tmp_path / "bad.parquet", np.arange(10, dtype=np.int64), np.zeros(10))
    args[args.index("--x-learner") + 1] = str(tmp_path / "bad.parquet")
    with pytest.raises(ContractError, match="exactly the validation rows"):
        module.main(args)
