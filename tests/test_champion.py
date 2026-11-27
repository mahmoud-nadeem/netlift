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


def table_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The table script, with no response-model bundle at its default path."""
    module = load_script("champion_challenger_table")
    monkeypatch.setattr(module, "RESPONSE_BUNDLE", tmp_path / "absent.zip")
    return module


def write_bundle(
    tmp_path: Path,
    module: ModuleType,
    manifest_changes: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> Path:
    """A response-model bundle shaped like Mohammed's, scoring the synthetic rows."""
    import zipfile

    outcomes = pd.read_parquet(tmp_path / "outcomes.parquet")
    rng = np.random.default_rng(1)
    scores = pd.DataFrame(
        {"row_id": outcomes["row_id"], "score": rng.random(len(outcomes))}
    )
    manifest = {**module.RESPONSE_MANIFEST, **(manifest_changes or {})}
    metadata = {"info": {"run_id": run_id or module.RESPONSE_MODEL_RUN}}
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle/manifest.json", json.dumps(manifest))
        archive.writestr("bundle/run-metadata.json", json.dumps(metadata))
        archive.writestr("bundle/validation_scores.csv", scores.to_csv(index=False))
    return path


def with_bundle(args: list[str], path: Path) -> list[str]:
    """The table's arguments with the bundle in the response-model slot."""
    return [
        *args,
        "--response-model",
        str(path),
        "--response-bundle-sha256",
        tracking.file_sha256(path),
    ]


def test_the_table_has_every_metric_and_paired_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = table_script(tmp_path, monkeypatch)
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
    assert len(saved["results"]["uplift_forest"]["qini"]["resamples"]) == 20
    assert set(saved["provenance"]) == expected
    assert "Code (commit that computed each column)" in text


def test_only_the_named_metrics_are_recomputed_with_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = table_script(tmp_path, monkeypatch)
    args = table_inputs(tmp_path)
    module.main(args)
    first = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    saved["provenance"]["qini"] = {"git_commit": "earlier", "git_dirty": False}
    (tmp_path / "results.json").write_text(json.dumps(saved), encoding="utf-8")
    module.main([*args, "--metrics", "uplift_at_10", "--keep-other-metrics"])
    second = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    assert set(second["metrics"]["t_learner"]) == set(first["metrics"]["t_learner"])
    assert (
        second["metrics"]["t_learner"]["qini"] == first["metrics"]["t_learner"]["qini"]
    )
    assert second["provenance"]["qini"]["git_commit"] == "earlier"
    assert second["provenance"]["uplift_at_10"]["git_commit"] != "earlier"
    assert second["seconds"].keys() == {
        f"{model}/uplift_at_10" for model in ("uplift_forest", "t_learner", "x_learner")
    }


def test_keeping_metrics_from_other_settings_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = table_script(tmp_path, monkeypatch)
    args = table_inputs(tmp_path)
    module.main(args)
    changed = [
        *args,
        "--seed",
        "7",
        "--metrics",
        "uplift_at_10",
        "--keep-other-metrics",
    ]
    with pytest.raises(ContractError, match="other inputs or bootstrap settings"):
        module.main(changed)


def test_the_response_bundle_fills_the_fourth_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = table_script(tmp_path, monkeypatch)
    args = table_inputs(tmp_path)
    bundle = write_bundle(tmp_path, module)
    module.main(with_bundle(args, bundle))
    report = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    assert report["pending"] == []
    assert set(report["differences_vs_forest"]) == {
        "t_learner",
        "x_learner",
        "response_model",
    }
    assert report["inputs"]["response_model"]["sha256"] == tracking.file_sha256(bundle)
    text = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert "pending" not in text and tracking.file_sha256(bundle) in text


def test_a_bundle_with_another_sha256_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = table_script(tmp_path, monkeypatch)
    args = table_inputs(tmp_path)
    bundle = write_bundle(tmp_path, module)
    args = [*args, "--response-model", str(bundle)]
    with pytest.raises(ContractError, match="SHA-256"):
        module.main(args)


@pytest.mark.parametrize(
    ("changes", "run_id", "message"),
    [
        ({"tuned": False}, None, "manifest"),
        ({"objective": "qini"}, None, "manifest"),
        ({"n_trials": 10}, None, "manifest"),
        ({"test_split_opened": True}, None, "manifest"),
        ({"split_seed": 7}, None, "manifest"),
        ({}, "0" * 32, "not the response model"),
    ],
)
def test_a_bundle_that_is_not_k1s_response_model_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    run_id: str | None,
    message: str,
) -> None:
    module = table_script(tmp_path, monkeypatch)
    args = table_inputs(tmp_path)
    bundle = write_bundle(tmp_path, module, changes, run_id)
    with pytest.raises(ContractError, match=message):
        module.main(with_bundle(args, bundle))


def test_a_score_file_with_other_rows_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = table_script(tmp_path, monkeypatch)
    args = table_inputs(tmp_path)
    write_scores(tmp_path / "bad.parquet", np.arange(10, dtype=np.int64), np.zeros(10))
    args[args.index("--x-learner") + 1] = str(tmp_path / "bad.parquet")
    with pytest.raises(ContractError, match="exactly the validation rows"):
        module.main(args)
