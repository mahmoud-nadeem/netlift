"""T57: Optuna search over uplift-model families under an out-of-fold Qini objective.

The protocol is fixed in ADR 0006 before the first trial:

- the objective is scikit-uplift's Qini coefficient on out-of-fold predictions inside
  the training split, with stratified K-fold on the joint (t, y) key. Validation and
  test are never read here; :func:`search_families` touches ``splits.train`` only;
- the families are the T-learner and X-learner with LightGBM base learners, and
  causalml's ``UpliftRandomForestClassifier`` with the split criterion as a
  categorical hyperparameter;
- the search is Optuna TPE with a fixed seed and the same ``n_trials`` for every
  family, with a timeout as a cap; every trial can be logged to MLflow;
- the champion is the family and parameters with the best out-of-fold Qini.

The T- and X-learners here follow the definitions in #117 (T56) exactly: two
outcome classifiers for the T-learner; for the X-learner, imputed effects regressed
within each arm and combined with the treated share as the weight. They are
replaced by #117's classes once it merges (ADR 0006).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import yaml
from causalml.inference.tree import UpliftRandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from sklift.metrics import qini_auc_score

from netlift.data.contracts import DatasetSplits, FeatureMatrix, validate_features
from netlift.data.split import joint_key
from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)

Family = Literal["t_learner", "x_learner", "uplift_forest"]
FAMILIES: tuple[Family, ...] = get_args(Family)

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "uplift_search.yaml"

_CONTROL = "control"
_TREATMENT = "treatment"

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def load_search_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read ``config/uplift_search.yaml``."""
    with path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    return config


@dataclass(frozen=True)
class SearchSettings:
    """What every family's study shares: folds, seed, trial budget and the Qini option."""

    n_trials: int
    timeout_s: float | None = None
    n_folds: int = 3
    seed: int = 42
    negative_effect: bool = True
    lightgbm_fixed: Mapping[str, Any] = field(default_factory=dict)
    lightgbm_space: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    forest_fixed: Mapping[str, Any] = field(default_factory=dict)
    forest_space: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.n_trials, bool) or not isinstance(self.n_trials, int):
            raise ContractError(f"n_trials must be an int, got {self.n_trials!r}.")
        if self.n_trials < 1:
            raise ContractError(f"n_trials must be at least 1, got {self.n_trials}.")
        if self.n_folds < 2:
            raise ContractError(f"n_folds must be at least 2, got {self.n_folds}.")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ContractError(f"timeout_s must be positive, got {self.timeout_s}.")

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, n_trials: int | None = None
    ) -> SearchSettings:
        """Build the settings from the YAML; ``n_trials`` overrides the file's value."""
        trials = n_trials if n_trials is not None else config.get("n_trials")
        if trials is None:
            raise ContractError(
                "n_trials is not set. ADR 0006 fixes it from the measured fit time."
            )
        hours = config.get("timeout_hours")
        return cls(
            n_trials=int(trials),
            timeout_s=None if hours is None else float(hours) * 3600.0,
            n_folds=int(config["n_folds"]),
            seed=int(config["seed"]),
            negative_effect=bool(config["qini"]["negative_effect"]),
            lightgbm_fixed=dict(config["lightgbm"]["fixed"]),
            lightgbm_space=dict(config["lightgbm"]["search_space"]),
            forest_fixed=dict(config["uplift_forest"]["fixed"]),
            forest_space=dict(config["uplift_forest"]["search_space"]),
        )


def _require_train(matrix: FeatureMatrix) -> None:
    """Refuse any split other than train: the objective must never see validation or test."""
    if matrix.meta.split_name != "train":
        raise ContractError(
            "The search objective runs on the training split only, got "
            f"{matrix.meta.split_name!r}. Validation is for reporting (spec section 26)."
        )


def oof_fold_ids(
    t: npt.ArrayLike, y: npt.ArrayLike, *, n_folds: int, seed: int
) -> IntArray:
    """Fold id per row: stratified K-fold on the joint (t, y) key, shuffled with ``seed``."""
    key = joint_key(t, y)
    folds = np.full(len(key), -1, dtype=np.int64)
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for number, (_, held) in enumerate(splitter.split(np.zeros(len(key)), key)):
        folds[held] = number
    return folds


def suggest_params(
    trial: optuna.Trial, space: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Draw one value per hyperparameter from a YAML search space."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ContractError(f"Unknown search-space type {kind!r} for {name!r}.")
    return params


def _positive_probability(model: LGBMClassifier, X: pd.DataFrame) -> FloatArray:
    """P(y = 1) from a fitted LightGBM classifier."""
    return np.asarray(model.predict_proba(X), dtype=np.float64)[:, 1]


def _fit_outcome_models(
    X: pd.DataFrame,
    t: npt.NDArray[np.int8],
    y: npt.NDArray[np.int8],
    params: Mapping[str, Any],
) -> tuple[LGBMClassifier, LGBMClassifier]:
    """One outcome classifier per arm (the T-learner of #117)."""
    control, treated = t == 0, t == 1
    if np.unique(y[control]).size != 2 or np.unique(y[treated]).size != 2:
        raise ContractError(
            "Each arm needs both outcome classes to fit an outcome model."
        )
    control_model = LGBMClassifier(**dict(params))
    treated_model = LGBMClassifier(**dict(params))
    control_model.fit(X.loc[control], y[control])
    treated_model.fit(X.loc[treated], y[treated])
    return control_model, treated_model


@dataclass(frozen=True)
class UpliftModel:
    """A fitted model of one family that predicts uplift for new rows.

    T-learner: ``parts = (control_model, treated_model)``. X-learner:
    ``parts = (control_effect, treated_effect)`` with ``treated_share`` as the weight.
    Uplift forest: ``parts = (forest,)``.
    """

    family: Family
    parts: tuple[Any, ...]
    feature_columns: tuple[str, ...]
    treated_share: float = 0.0

    def predict_uplift(self, X: pd.DataFrame) -> FloatArray:
        """Uplift estimate for each row of ``X``, after the forbidden-column check."""
        validate_features(X)
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ContractError(f"Missing feature columns: {missing}.")
        inputs = X.loc[:, list(self.feature_columns)]
        if self.family == "t_learner":
            control_model, treated_model = self.parts
            return _positive_probability(treated_model, inputs) - _positive_probability(
                control_model, inputs
            )
        if self.family == "x_learner":
            control_effect, treated_effect = self.parts
            share = self.treated_share
            return np.asarray(
                share * np.asarray(control_effect.predict(inputs), dtype=np.float64)
                + (1.0 - share)
                * np.asarray(treated_effect.predict(inputs), dtype=np.float64),
                dtype=np.float64,
            )
        (forest,) = self.parts
        predicted = np.asarray(forest.predict(inputs.to_numpy(dtype=np.float64)))
        return np.asarray(predicted.reshape(len(inputs), -1)[:, 0], dtype=np.float64)


def fit_model(
    family: Family,
    params: Mapping[str, Any],
    X_fit: pd.DataFrame,
    t_fit: npt.NDArray[np.int8],
    y_fit: npt.NDArray[np.int8],
    *,
    seed: int,
) -> UpliftModel:
    """Fit one family on the given rows and return the fitted model."""
    validate_features(X_fit)
    columns = tuple(str(c) for c in X_fit.columns)
    if family in ("t_learner", "x_learner"):
        lgbm = {**params, "random_state": seed}
        control_model, treated_model = _fit_outcome_models(X_fit, t_fit, y_fit, lgbm)
        if family == "t_learner":
            return UpliftModel(family, (control_model, treated_model), columns)
        control, treated = t_fit == 0, t_fit == 1
        control_target = _positive_probability(
            treated_model, X_fit.loc[control]
        ) - y_fit[control].astype(np.float64)
        treated_target = y_fit[treated].astype(np.float64) - _positive_probability(
            control_model, X_fit.loc[treated]
        )
        control_effect = LGBMRegressor(**lgbm)
        treated_effect = LGBMRegressor(**lgbm)
        control_effect.fit(X_fit.loc[control], control_target)
        treated_effect.fit(X_fit.loc[treated], treated_target)
        return UpliftModel(
            family,
            (control_effect, treated_effect),
            columns,
            treated_share=float(treated.mean()),
        )
    if family == "uplift_forest":
        forest_params = dict(params)
        forest_params["max_features"] = min(
            int(forest_params.get("max_features", X_fit.shape[1])), X_fit.shape[1]
        )
        labels = np.where(t_fit == 1, _TREATMENT, _CONTROL)
        forest = UpliftRandomForestClassifier(
            control_name=_CONTROL, random_state=seed, **forest_params
        )
        forest.fit(X_fit.to_numpy(dtype=np.float64), labels, y_fit)
        return UpliftModel(family, (forest,), columns)
    raise ContractError(f"Unknown family {family!r}; expected one of {FAMILIES}.")


def fit_predict(
    family: Family,
    params: Mapping[str, Any],
    X_fit: pd.DataFrame,
    t_fit: npt.NDArray[np.int8],
    y_fit: npt.NDArray[np.int8],
    X_pred: pd.DataFrame,
    *,
    seed: int,
) -> FloatArray:
    """Fit one family on the fit rows and return its uplift estimate for the predict rows."""
    validate_features(X_pred)
    model = fit_model(family, params, X_fit, t_fit, y_fit, seed=seed)
    return model.predict_uplift(X_pred)


def out_of_fold_uplift(
    train: FeatureMatrix,
    family: Family,
    params: Mapping[str, Any],
    *,
    folds: IntArray,
    seed: int,
) -> FloatArray:
    """Uplift for every training row from a model that did not see that row."""
    _require_train(train)
    if len(folds) != train.n_rows:
        raise ContractError(f"folds has {len(folds)} rows, train has {train.n_rows}.")
    uplift = np.full(train.n_rows, np.nan, dtype=np.float64)
    for number in np.unique(folds):
        held = folds == number
        uplift[held] = fit_predict(
            family,
            params,
            train.X.loc[~held],
            train.t[~held],
            train.y[~held],
            train.X.loc[held],
            seed=seed,
        )
    if not np.all(np.isfinite(uplift)):
        raise ContractError(f"{family} produced non-finite out-of-fold uplift.")
    return uplift


def qini(
    y: npt.ArrayLike, uplift: npt.ArrayLike, t: npt.ArrayLike, *, negative_effect: bool
) -> float:
    """scikit-uplift's Qini coefficient (``qini_auc_score``), the single source of truth."""
    return float(qini_auc_score(y, uplift, t, negative_effect=negative_effect))


TrialLogger = Callable[[Family, int, dict[str, Any], dict[str, float]], None]


@dataclass(frozen=True)
class FamilyResult:
    """The outcome of one family's study."""

    family: Family
    best_value: float
    best_params: dict[str, Any]
    best_trial: int
    n_trials_completed: int
    elapsed_s: float


def run_study(
    train: FeatureMatrix,
    family: Family,
    settings: SearchSettings,
    *,
    log_trial: TrialLogger | None = None,
) -> FamilyResult:
    """Run one family's TPE study on the training split and return its best trial."""
    _require_train(train)
    if family not in FAMILIES:
        raise ContractError(f"Unknown family {family!r}; expected one of {FAMILIES}.")
    folds = oof_fold_ids(train.t, train.y, n_folds=settings.n_folds, seed=settings.seed)
    space = (
        settings.forest_space if family == "uplift_forest" else settings.lightgbm_space
    )
    fixed = (
        settings.forest_fixed if family == "uplift_forest" else settings.lightgbm_fixed
    )

    def objective(trial: optuna.Trial) -> float:
        params = {**dict(fixed), **suggest_params(trial, space)}
        started = time.perf_counter()
        uplift = out_of_fold_uplift(
            train, family, params, folds=folds, seed=settings.seed
        )
        value = qini(train.y, uplift, train.t, negative_effect=settings.negative_effect)
        metrics = {"oof_qini": value, "fit_seconds": time.perf_counter() - started}
        for number in range(settings.n_folds):
            held = folds == number
            metrics[f"fold{number}_qini"] = qini(
                train.y[held],
                uplift[held],
                train.t[held],
                negative_effect=settings.negative_effect,
            )
        logger.info(
            "%s trial %d: oof_qini=%.6f in %.0f s",
            family,
            trial.number,
            value,
            metrics["fit_seconds"],
        )
        if log_trial is not None:
            log_trial(family, trial.number, params, metrics)
        return value

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.seed),
        study_name=f"T57-{family}",
    )
    started = time.perf_counter()
    study.optimize(objective, n_trials=settings.n_trials, timeout=settings.timeout_s)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ContractError(f"No {family} trial completed.")
    best = study.best_trial
    return FamilyResult(
        family=family,
        best_value=float(best.value) if best.value is not None else float("nan"),
        best_params={**dict(fixed), **best.params},
        best_trial=best.number,
        n_trials_completed=len(completed),
        elapsed_s=time.perf_counter() - started,
    )


def search_families(
    splits: DatasetSplits,
    families: tuple[Family, ...],
    settings: SearchSettings,
    *,
    log_trial: TrialLogger | None = None,
) -> list[FamilyResult]:
    """Run every family's study on ``splits.train``. Validation and test are never read."""
    train = splits.train
    return [
        run_study(train, family, settings, log_trial=log_trial) for family in families
    ]


def select_champion(results: list[FamilyResult]) -> FamilyResult:
    """The family and parameters with the best out-of-fold Qini (ADR 0006)."""
    if not results:
        raise ContractError("No study results to choose a champion from.")
    return max(results, key=lambda result: result.best_value)
