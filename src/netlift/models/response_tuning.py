"""Tune the T38 response-model baseline inside the train folds (#107, ADR 0004).

Only the train matrix is passed to the search, so validation and test rows cannot
influence a trial. Each trial fits LightGBM on the treated rows outside a fold and
predicts every row of that fold, giving one out-of-fold (OOF) prediction per train
row. The primary objective is OOF log loss on treated rows; the sensitivity
objective is the OOF Qini coefficient over all rows. All numbers stay preliminary
until T39 supplies intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from scipy.stats import norm, rankdata
from sklearn.metrics import log_loss, roc_auc_score
from sklift.metrics import qini_auc_score

from netlift.data.contracts import FeatureMatrix, validate_features
from netlift.data.split import subsample_part
from netlift.utils.errors import ContractError

OBJECTIVES = ("log_loss", "qini")
SEARCH_KEYS = (
    "num_leaves",
    "min_child_samples",
    "learning_rate",
    "n_estimators",
    "reg_lambda",
    "colsample_bytree",
)
PROTECTED = {
    "random_state",
    "seed",
    "objective",
    "class_weight",
    "is_unbalance",
    "scale_pos_weight",
    *SEARCH_KEYS,
}
CONFIG_KEYS = {"seed", "n_trials", "subsample_frac", "fixed", "search_space"}
DATASETS = ("hillstrom", "criteo")
CLIP = 1e-15


def read_config(path: Path) -> dict[str, Any]:
    """Validate the tuning configuration before any experiment starts."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"Config must contain exactly {sorted(CONFIG_KEYS)}.")
    seed, n_trials = config["seed"], config["n_trials"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    frac = config["subsample_frac"]
    if not isinstance(frac, dict) or set(frac) != set(DATASETS):
        raise ValueError(f"subsample_frac must have exactly {list(DATASETS)}.")
    for dataset, value in frac.items():
        if value is not None and not (isinstance(value, float) and 0.0 < value <= 1.0):
            raise ValueError(f"subsample_frac[{dataset}] must be null or in (0, 1].")
    if not isinstance(config["fixed"], dict) or PROTECTED & set(config["fixed"]):
        raise ValueError("fixed must be a mapping without searched or protected keys.")
    space = config["search_space"]
    if not isinstance(space, dict) or set(space) != set(SEARCH_KEYS):
        raise ValueError(f"search_space must have exactly {list(SEARCH_KEYS)}.")
    for name, spec in space.items():
        if (
            not isinstance(spec, dict)
            or spec.get("type") not in {"int", "float"}
            or not spec["low"] < spec["high"]
        ):
            raise ValueError(f"search_space[{name}] needs type, low < high.")
    return config


def suggest_params(
    trial: optuna.Trial, space: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Draw one point from the search space defined in the config."""
    params: dict[str, Any] = {}
    for name in SEARCH_KEYS:
        spec = space[name]
        log = bool(spec.get("log", False))
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=log
            )
        else:
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=log
            )
    return params


def _model(seed: int, fixed: dict[str, Any], params: dict[str, Any]) -> LGBMClassifier:
    if PROTECTED & set(fixed):
        raise ContractError("Seed, objective and imbalance overrides are not allowed.")
    return LGBMClassifier(objective="binary", random_state=seed, **fixed, **params)


def oof_predictions(
    train: FeatureMatrix,
    params: dict[str, Any],
    *,
    seed: int,
    fixed: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """P(Y=1 | X, T=1) for every train row, from a model that never saw its fold."""
    validate_features(train.X)
    oof = np.full(train.n_rows, np.nan)
    for fold_id in train.fold_ids:
        held = train.fold == fold_id
        fit = ~held & (train.t == 1)
        if len(np.unique(train.y[fit])) != 2:
            raise ContractError(
                f"Fold {fold_id}: treated rows outside it need both outcomes."
            )
        model = _model(seed, fixed, params)
        model.fit(train.X.loc[fit], train.y[fit])
        oof[held] = np.asarray(
            model.predict_proba(train.X.loc[held]), dtype=np.float64
        )[:, 1]
    if not np.isfinite(oof).all():
        raise ContractError("Out-of-fold predictions are missing for some train rows.")
    return oof


def oof_metrics(train: FeatureMatrix, oof: npt.NDArray[np.float64]) -> dict[str, float]:
    """OOF log loss on treated rows and OOF Qini over all rows (scikit-uplift)."""
    treated = train.t == 1
    metrics = {
        "oof_log_loss": float(log_loss(train.y[treated], oof[treated], labels=[0, 1])),
        "oof_qini": float(qini_auc_score(train.y, oof, train.t, negative_effect=True)),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ContractError("Undefined out-of-fold metric; inspect outcome counts.")
    return metrics


@dataclass(frozen=True)
class TuningResult:
    """The outcome of one search: the best point and every trial."""

    objective: str
    best_params: dict[str, Any]
    best_value: float
    best_oof: dict[str, float]
    trials: pd.DataFrame
    n_trials: int


def tune(
    train: FeatureMatrix,
    *,
    objective: str,
    n_trials: int,
    seed: int,
    space: dict[str, dict[str, Any]],
    fixed: dict[str, Any],
) -> TuningResult:
    """Search inside the train folds only; validation and test are never passed in."""
    if objective not in OBJECTIVES:
        raise ContractError(
            f"objective must be one of {OBJECTIVES}, got {objective!r}."
        )
    if train.meta.split_name != "train":
        raise ContractError("Tuning uses the train split only.")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ContractError("n_trials must be a positive integer.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize" if objective == "log_loss" else "maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def run_trial(trial: optuna.Trial) -> float:
        params = suggest_params(trial, space)
        started = perf_counter()
        metrics = oof_metrics(
            train, oof_predictions(train, params, seed=seed, fixed=fixed)
        )
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("seconds", perf_counter() - started)
        return metrics["oof_log_loss" if objective == "log_loss" else "oof_qini"]

    study.optimize(run_trial, n_trials=n_trials)
    rows = [
        {"trial": t.number, **t.params, **t.user_attrs, "value": t.value}
        for t in study.trials
    ]
    best = study.best_trial
    if best.value is None:
        raise ContractError("The best trial has no value.")
    return TuningResult(
        objective=objective,
        best_params=dict(best.params),
        best_value=float(best.value),
        best_oof={k: float(best.user_attrs[k]) for k in ("oof_log_loss", "oof_qini")},
        trials=pd.DataFrame(rows),
        n_trials=len(study.trials),
    )


def fit_final(
    train: FeatureMatrix,
    params: dict[str, Any],
    *,
    seed: int,
    fixed: dict[str, Any],
) -> LGBMClassifier:
    """Refit the chosen parameters on every treated row of the train split."""
    validate_features(train.X)
    treated = train.t == 1
    if len(np.unique(train.y[treated])) != 2:
        raise ContractError("Response model needs both outcomes in treated train.")
    model = _model(seed, fixed, params)
    model.fit(train.X.loc[treated], train.y[treated])
    return model


def tuning_subset(
    train: FeatureMatrix, frac: float | None, *, seed: int
) -> FeatureMatrix:
    """Sample inside train on (treatment, conversion, fold), keeping canonical IDs.

    This runner uses conversion for Criteo, matching T37's stratification key.
    Hillstrom is not subsampled. No validation or test outcome enters selection.
    """
    if train.meta.split_name != "train":
        raise ContractError("Subsampling uses the train split only.")
    if frac is None or frac == 1.0:
        return train
    if train.meta.target_outcome != "conversion":
        raise ContractError("Subsampling currently requires the conversion target.")
    if not 0.0 < frac < 1.0:
        raise ContractError("Subsample fraction must be strictly between zero and one.")
    n_folds = int(train.fold.max()) + 1
    key = (train.t.astype(np.int64) * 2 + train.y) * n_folds + train.fold
    idx = subsample_part(np.arange(train.n_rows), key, frac, seed)
    return FeatureMatrix(
        X=train.X.iloc[idx].reset_index(drop=True),
        t=train.t[idx],
        y=train.y[idx],
        fold=train.fold[idx],
        row_id=train.row_id[idx],
        meta=train.meta,
    )


def auc_interval(
    y: npt.NDArray[Any], score: npt.NDArray[Any]
) -> tuple[float, float, float]:
    """AUC and a clipped 95% normal DeLong interval for one fixed model.

    Midranks include ties. Placement variances avoid a positive-by-negative
    matrix and repeated sorting of millions of rows in an AUC bootstrap.
    This interval describes response discrimination, not an uplift metric.
    """
    y = np.asarray(y)
    p = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ContractError("AUC needs aligned binary outcomes and scores.")
    if not np.isfinite(p).all():
        raise ContractError("AUC scores must be finite.")
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        raise ContractError("AUC interval needs at least two observations per outcome.")
    both_ranks = rankdata(np.concatenate((pos, neg)), method="average")
    pos_placements = (both_ranks[: len(pos)] - rankdata(pos)) / len(neg)
    neg_placements = 1.0 - (both_ranks[len(pos) :] - rankdata(neg)) / len(pos)
    auc = float(pos_placements.mean())
    variance = float(
        np.var(pos_placements, ddof=1) / len(pos)
        + np.var(neg_placements, ddof=1) / len(neg)
    )
    half_width = float(norm.ppf(0.975) * np.sqrt(max(variance, 0.0)))
    return auc, max(0.0, auc - half_width), min(1.0, auc + half_width)


def base_rate_reference(
    y: npt.NDArray[Any],
    score: npt.NDArray[Any],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Compare a model with the best constant predictor on treated validation rows.

    The constant is the rows' own positive rate, so it is the strongest possible
    constant and a deliberately hard reference. The interval is a percentile
    bootstrap of the mean per-row log-loss difference (model minus constant);
    positive means the model is worse. The model's AUC has a DeLong interval.
    A constant score's AUC is exactly 0.5, with the degenerate interval [0.5, 0.5].
    The null SE remains a separate diagnostic and is not an AUC confidence interval.
    The validation-rate constant is an oracle diagnostic, not a deployable model.
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(score, dtype=np.float64), CLIP, 1.0 - CLIP)
    n, positives = len(y), int(y.sum())
    negatives = n - positives
    if positives == 0 or negatives == 0:
        raise ContractError("Reference needs both outcomes among treated rows.")
    rate = positives / n
    per_row_const = -(y * np.log(rate) + (1 - y) * np.log(1 - rate))
    per_row_model = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    diff = per_row_model - per_row_const
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])
    low, high = np.percentile(boot, [2.5, 97.5])
    auc = float(roc_auc_score(y, p))
    _, auc_low, auc_high = auc_interval(y, p)
    null_se = float(np.sqrt((n + 1) / (12 * positives * negatives)))
    return {
        "treated_rows": float(n),
        "treated_positives": float(positives),
        "base_rate": float(rate),
        "constant_log_loss": float(per_row_const.mean()),
        "model_log_loss": float(per_row_model.mean()),
        "log_loss_diff": float(diff.mean()),
        "log_loss_diff_ci_low": float(low),
        "log_loss_diff_ci_high": float(high),
        "auc": auc,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "constant_auc": 0.5,
        "constant_auc_ci_low": 0.5,
        "constant_auc_ci_high": 0.5,
        "auc_null_se": null_se,
        "auc_z": (auc - 0.5) / null_se,
    }
