"""#107 tests on T36 synthetic data; no real-data tests or results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from netlift.models import response_tuning as rt
from netlift.utils.errors import ContractError

SPACE = {
    "num_leaves": {"type": "int", "low": 4, "high": 8, "log": False},
    "min_child_samples": {"type": "int", "low": 5, "high": 20, "log": False},
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.2, "log": True},
    "n_estimators": {"type": "int", "low": 5, "high": 15, "log": False},
    "reg_lambda": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
}
FIXED = {"n_jobs": 1, "verbosity": -1, "deterministic": True, "force_col_wise": True}
PARAMS = {
    "num_leaves": 4,
    "min_child_samples": 5,
    "learning_rate": 0.1,
    "n_estimators": 8,
    "reg_lambda": 1.0,
    "colsample_bytree": 1.0,
}


def test_oof_uses_only_treated_rows_outside_the_fold(synthetic_splits, monkeypatch):
    """Each fold's model sees treated rows of the other folds and predicts its own."""
    train = synthetic_splits.train
    calls = []
    original = rt.LGBMClassifier.fit

    def fit(self, X, y, **kwargs):
        calls.append(X.index.to_numpy())
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(rt.LGBMClassifier, "fit", fit)
    oof = rt.oof_predictions(train, PARAMS, seed=42, fixed=FIXED)
    assert len(calls) == len(train.fold_ids)
    for fold_id, rows in zip(train.fold_ids, calls, strict=True):
        expected = np.flatnonzero((train.fold != fold_id) & (train.t == 1))
        np.testing.assert_array_equal(rows, expected)
    assert oof.shape == (train.n_rows,)
    assert np.isfinite(oof).all()
    assert ((oof > 0) & (oof < 1)).all()


def test_tune_is_reproducible_and_logs_every_trial(synthetic_splits):
    """The same seed gives the same search; the table has one row per trial."""
    kwargs = {
        "objective": "log_loss",
        "n_trials": 3,
        "seed": 42,
        "space": SPACE,
        "fixed": FIXED,
    }
    a = rt.tune(synthetic_splits.train, **kwargs)
    b = rt.tune(synthetic_splits.train, **kwargs)
    assert a.best_params == b.best_params
    assert a.best_value == b.best_value
    assert a.n_trials == len(a.trials) == 3
    for column in ("trial", *rt.SEARCH_KEYS, "oof_log_loss", "oof_qini", "seconds"):
        assert column in a.trials.columns
    assert a.best_value == a.trials["oof_log_loss"].min()
    for name in rt.SEARCH_KEYS:
        spec = SPACE[name]
        assert a.trials[name].between(spec["low"], spec["high"]).all()


def test_qini_objective_maximises(synthetic_splits):
    """The sensitivity objective picks the trial with the highest OOF Qini."""
    result = rt.tune(
        synthetic_splits.train,
        objective="qini",
        n_trials=3,
        seed=42,
        space=SPACE,
        fixed=FIXED,
    )
    assert result.best_value == result.trials["oof_qini"].max()


def test_tune_rejects_bad_inputs(synthetic_splits):
    """Only the train split, a known objective and a positive budget are accepted."""
    base = {"seed": 42, "space": SPACE, "fixed": FIXED}
    with pytest.raises(ContractError, match="objective"):
        rt.tune(synthetic_splits.train, objective="auc", n_trials=1, **base)
    with pytest.raises(ContractError, match="n_trials"):
        rt.tune(synthetic_splits.train, objective="log_loss", n_trials=0, **base)
    with pytest.raises(ContractError, match="train split"):
        rt.tune(synthetic_splits.validation, objective="log_loss", n_trials=1, **base)


def test_protected_fixed_params_are_rejected(synthetic_splits):
    """Seed, objective and imbalance settings cannot be overridden through config."""
    with pytest.raises(ContractError, match="not allowed"):
        rt.oof_predictions(
            synthetic_splits.train, PARAMS, seed=42, fixed={"scale_pos_weight": 5}
        )


def test_fit_final_uses_treated_train_rows_only(synthetic_splits, monkeypatch):
    """The refit sees exactly the treated rows of the train split."""
    train = synthetic_splits.train
    seen = []
    original = rt.LGBMClassifier.fit

    def fit(self, X, y, **kwargs):
        seen.append((X.copy(), y.copy()))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(rt.LGBMClassifier, "fit", fit)
    model = rt.fit_final(train, PARAMS, seed=42, fixed=FIXED)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0].index, np.flatnonzero(train.t == 1))
    proba = model.predict_proba(synthetic_splits.validation.X[train.X.columns])
    assert proba.shape == (synthetic_splits.validation.n_rows, 2)


def test_base_rate_reference_hand_computed():
    """A constant model equals the constant predictor; a worse one is positive."""
    y = np.array([0, 0, 0, 1, 0, 0, 0, 1])
    constant = np.full(8, 0.25)
    same = rt.base_rate_reference(y, constant, n_resamples=50, seed=1)
    assert same["base_rate"] == 0.25
    assert same["constant_log_loss"] == pytest.approx(
        -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    )
    assert same["log_loss_diff"] == pytest.approx(0.0, abs=1e-12)
    assert same["auc"] == 0.5
    assert same["auc_null_se"] == pytest.approx(np.sqrt(9 / (12 * 2 * 6)))
    worse = rt.base_rate_reference(y, np.full(8, 0.6), n_resamples=50, seed=1)
    assert worse["log_loss_diff"] > 0


def test_base_rate_reference_needs_both_outcomes():
    """A single-class sample has no AUC and no meaningful base rate."""
    with pytest.raises(ContractError, match="both outcomes"):
        rt.base_rate_reference(np.zeros(5), np.full(5, 0.1))


def test_read_config_accepts_the_committed_file_and_rejects_bad_ones(tmp_path):
    """The shipped config validates; unknown keys and bad ranges do not."""
    root = Path(__file__).resolve().parents[1]
    config = rt.read_config(root / "config/response_tuning.yaml")
    assert config["n_trials"] == 30
    assert set(config["search_space"]) == set(rt.SEARCH_KEYS)
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\nextra: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        rt.read_config(bad)
    text = (root / "config/response_tuning.yaml").read_text(encoding="utf-8")
    bad.write_text(text.replace("n_trials: 30", "n_trials: 0"), encoding="utf-8")
    with pytest.raises(ValueError, match="n_trials"):
        rt.read_config(bad)
