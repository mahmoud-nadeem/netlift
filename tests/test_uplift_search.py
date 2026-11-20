"""T57: the out-of-fold Qini search (ADR 0006).

The data here is SYNTHETIC, with a known uplift, so the tests can tell a search that
finds the effect from one that does not. Never use it for a reported result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from netlift.data.contracts import (
    DatasetSplits,
    FeatureMatrix,
    FeatureMatrixMeta,
    TreatmentDefinition,
)
from netlift.models.uplift_search import (
    FAMILIES,
    FamilyResult,
    SearchSettings,
    fit_predict,
    load_search_config,
    oof_fold_ids,
    out_of_fold_uplift,
    qini,
    run_study,
    search_families,
    select_champion,
)
from netlift.utils.errors import ContractError

TREATMENT = TreatmentDefinition("treatment", (1,), (0,))

# Small, fast spaces: enough to show the search works, not the real ones.
FAST_LGBM = {
    "num_leaves": {"type": "int", "low": 4, "high": 8},
    "n_estimators": {"type": "int", "low": 20, "high": 40},
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.2, "log": True},
}
FAST_FOREST = {
    "evaluationFunction": {"type": "categorical", "choices": ["KL", "ED", "Chi"]},
    "n_estimators": {"type": "int", "low": 5, "high": 10},
    "max_depth": {"type": "int", "low": 2, "high": 3},
    "min_samples_leaf": {"type": "int", "low": 50, "high": 100},
}


def known_uplift_matrix(
    split_name: str = "train", n: int = 3000, seed: int = 0
) -> tuple[FeatureMatrix, np.ndarray]:
    """Synthetic rows whose true uplift is +0.25 where x0 > 0 and -0.05 elsewhere."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["x0", "x1", "x2", "x3"])
    t = rng.integers(0, 2, n).astype(np.int8)
    base = 0.2 + 0.05 * np.tanh(X["x1"].to_numpy())
    tau = np.where(X["x0"].to_numpy() > 0, 0.25, -0.05)
    y = (rng.random(n) < base + t * tau).astype(np.int8)
    matrix = FeatureMatrix(
        X=X,
        t=t,
        y=y,
        fold=np.zeros(n, dtype=np.int64),
        row_id=np.arange(n, dtype=np.int64) + (0 if split_name == "train" else n),
        meta=FeatureMatrixMeta("hillstrom", "visit", TREATMENT, 42, split_name),  # type: ignore[arg-type]
    )
    return matrix, tau


def settings(n_trials: int = 3) -> SearchSettings:
    """Small settings for the tests."""
    return SearchSettings(
        n_trials=n_trials,
        n_folds=3,
        seed=42,
        lightgbm_fixed={"n_jobs": 1, "verbosity": -1, "deterministic": True},
        lightgbm_space=FAST_LGBM,
        forest_fixed={"n_jobs": 1},
        forest_space=FAST_FOREST,
    )


def test_folds_are_stratified_on_the_joint_key() -> None:
    matrix, _ = known_uplift_matrix()
    folds = oof_fold_ids(matrix.t, matrix.y, n_folds=3, seed=42)
    assert set(np.unique(folds)) == {0, 1, 2}
    key = 2 * matrix.t.astype(int) + matrix.y.astype(int)
    for cell in np.unique(key):
        counts = np.bincount(folds[key == cell], minlength=3)
        assert counts.max() - counts.min() <= 1


def test_folds_are_reproducible_from_the_seed() -> None:
    matrix, _ = known_uplift_matrix()
    a = oof_fold_ids(matrix.t, matrix.y, n_folds=3, seed=42)
    b = oof_fold_ids(matrix.t, matrix.y, n_folds=3, seed=42)
    c = oof_fold_ids(matrix.t, matrix.y, n_folds=3, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_ranks_the_true_effect_above_random(family: str) -> None:
    matrix, tau = known_uplift_matrix()
    folds = oof_fold_ids(matrix.t, matrix.y, n_folds=3, seed=42)
    params: dict[str, Any] = (
        {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 50, "n_jobs": 1}
        if family == "uplift_forest"
        else {"n_estimators": 40, "num_leaves": 4, "verbosity": -1, "n_jobs": 1}
    )
    uplift = out_of_fold_uplift(matrix, family, params, folds=folds, seed=42)  # type: ignore[arg-type]
    # The estimate separates the two true-effect groups in the right direction.
    assert uplift[tau > 0].mean() > uplift[tau < 0].mean()
    random_scores = np.random.default_rng(1).random(matrix.n_rows)
    assert qini(matrix.y, uplift, matrix.t, negative_effect=True) > qini(
        matrix.y, random_scores, matrix.t, negative_effect=True
    )


def test_the_true_uplift_scores_a_higher_qini_than_its_reverse() -> None:
    matrix, tau = known_uplift_matrix()
    right = qini(matrix.y, tau, matrix.t, negative_effect=True)
    reversed_ = qini(matrix.y, -tau, matrix.t, negative_effect=True)
    assert right > 0 > reversed_


@pytest.mark.parametrize("family", FAMILIES)
def test_a_study_runs_n_trials_and_returns_its_best(family: str) -> None:
    matrix, _ = known_uplift_matrix()
    logged: list[tuple[str, int, dict[str, Any], dict[str, float]]] = []
    result = run_study(
        matrix,
        family,  # type: ignore[arg-type]
        settings(n_trials=3),
        log_trial=lambda f, n, p, m: logged.append((f, n, p, m)),
    )
    assert result.n_trials_completed == 3
    assert [entry[1] for entry in logged] == [0, 1, 2]
    assert result.best_value == max(entry[3]["oof_qini"] for entry in logged)
    assert {"oof_qini", "fit_seconds", "fold0_qini", "fold1_qini", "fold2_qini"} <= set(
        logged[0][3]
    )
    assert result.best_value > 0


def test_the_same_seed_gives_the_same_trials() -> None:
    matrix, _ = known_uplift_matrix()
    first: list[float] = []
    second: list[float] = []
    run_study(
        matrix,
        "t_learner",
        settings(),
        log_trial=lambda f, n, p, m: first.append(m["oof_qini"]),
    )
    run_study(
        matrix,
        "t_learner",
        settings(),
        log_trial=lambda f, n, p, m: second.append(m["oof_qini"]),
    )
    assert first == second


@pytest.mark.parametrize("split_name", ["validation", "test"])
def test_the_objective_refuses_any_split_but_train(split_name: str) -> None:
    matrix, _ = known_uplift_matrix(split_name)
    with pytest.raises(ContractError, match="training split only"):
        run_study(matrix, "t_learner", settings())
    with pytest.raises(ContractError, match="training split only"):
        out_of_fold_uplift(
            matrix,
            "t_learner",
            {},
            folds=np.zeros(matrix.n_rows, dtype=np.int64),
            seed=0,
        )


class TrainOnlySplits:
    """A DatasetSplits stand-in whose validation and test fail the test if read."""

    def __init__(self, train: FeatureMatrix) -> None:
        self.train = train

    @property
    def validation(self) -> FeatureMatrix:
        pytest.fail("the search read the validation split")

    @property
    def test(self) -> FeatureMatrix:
        pytest.fail("the search read the test split")


def test_search_families_never_touches_validation_or_test() -> None:
    matrix, _ = known_uplift_matrix()
    splits: DatasetSplits = TrainOnlySplits(matrix)  # type: ignore[assignment]
    results = search_families(
        splits, ("t_learner", "uplift_forest"), settings(n_trials=2)
    )
    assert [r.family for r in results] == ["t_learner", "uplift_forest"]


def test_the_champion_is_the_best_out_of_fold_qini() -> None:
    results = [
        FamilyResult("t_learner", 0.02, {}, 1, 3, 1.0),
        FamilyResult("uplift_forest", 0.05, {}, 0, 3, 1.0),
        FamilyResult("x_learner", 0.04, {}, 2, 3, 1.0),
    ]
    assert select_champion(results).family == "uplift_forest"
    with pytest.raises(ContractError):
        select_champion([])


def test_forest_max_features_is_capped_at_the_column_count() -> None:
    matrix, _ = known_uplift_matrix()
    params = {
        "n_estimators": 3,
        "max_depth": 2,
        "min_samples_leaf": 50,
        "max_features": 12,
        "n_jobs": 1,
    }
    scores = fit_predict(
        "uplift_forest", params, matrix.X, matrix.t, matrix.y, matrix.X.head(10), seed=0
    )
    assert scores.shape == (10,)


def test_forbidden_columns_are_refused() -> None:
    matrix, _ = known_uplift_matrix()
    leaked = matrix.X.assign(exposure=0)
    with pytest.raises(ContractError):
        fit_predict("t_learner", {}, leaked, matrix.t, matrix.y, leaked, seed=0)


def test_settings_reject_a_bad_budget() -> None:
    with pytest.raises(ContractError):
        SearchSettings(n_trials=0)
    with pytest.raises(ContractError):
        SearchSettings(n_trials=True)  # type: ignore[arg-type]
    with pytest.raises(ContractError):
        SearchSettings(n_trials=3, timeout_s=0)


def test_the_config_file_needs_n_trials_before_a_study() -> None:
    config = load_search_config()
    assert config["n_folds"] == 3
    assert config["families"] == list(FAMILIES)
    built = SearchSettings.from_config(config, n_trials=2)
    assert built.n_trials == 2
