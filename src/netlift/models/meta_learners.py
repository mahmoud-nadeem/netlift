"""S-learner and class-transformation models for T56."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from netlift.data.contracts import FeatureMatrix, validate_features

_TREATMENT_COLUMN = "__netlift_treatment__"


@dataclass(frozen=True)
class SLearner:
    """Predict uplift by toggling treatment in one outcome model."""

    model: LGBMClassifier
    feature_columns: tuple[str, ...]

    def predict_uplift(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        validate_features(features)
        inputs = features.loc[:, list(self.feature_columns)].copy()
        inputs[_TREATMENT_COLUMN] = 0
        control = np.asarray(self.model.predict_proba(inputs), dtype=np.float64)[:, 1]
        inputs[_TREATMENT_COLUMN] = 1
        treated = np.asarray(self.model.predict_proba(inputs), dtype=np.float64)[:, 1]
        return np.asarray(treated - control, dtype=np.float64)


def fit_s_learner(train: FeatureMatrix, params: Mapping[str, Any]) -> SLearner:
    """Fit one outcome classifier on both arms of the training split."""
    validate_features(train.X)
    if set(np.unique(train.t)) != {0, 1}:
        raise ValueError("S-learner training requires both treatment arms.")
    if _TREATMENT_COLUMN in train.X.columns:
        raise ValueError("Input features contain the reserved treatment column.")

    inputs = train.X.assign(**{_TREATMENT_COLUMN: train.t})
    model = LGBMClassifier(**dict(params))
    model.fit(inputs, train.y)
    return SLearner(model=model, feature_columns=tuple(train.X.columns))


@dataclass(frozen=True)
class ClassTransformation:
    """Predict uplift from the transformed target's probability."""

    model: LGBMClassifier
    feature_columns: tuple[str, ...]

    def predict_uplift(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        validate_features(features)
        inputs = features.loc[:, list(self.feature_columns)]
        probability = np.asarray(self.model.predict_proba(inputs), dtype=np.float64)[
            :, 1
        ]
        return np.asarray(2.0 * probability - 1.0, dtype=np.float64)


def fit_class_transformation(
    train: FeatureMatrix,
    params: Mapping[str, Any],
    *,
    propensity_weighted: bool,
) -> ClassTransformation:
    """Fit the naive or arm-weighted classifier on training rows."""
    validate_features(train.X)
    treated_share = float(np.mean(train.t))
    if not 0.0 < treated_share < 1.0:
        raise ValueError("Class transformation requires both treatment arms.")

    transformed = (train.t == train.y).astype(np.int8)
    if np.unique(transformed).size != 2:
        raise ValueError("The transformed target requires both classes.")

    weights = None
    if propensity_weighted:
        weights = np.where(
            train.t == 1,
            0.5 / treated_share,
            0.5 / (1.0 - treated_share),
        )

    model = LGBMClassifier(**dict(params))
    model.fit(train.X, transformed, sample_weight=weights)
    return ClassTransformation(model=model, feature_columns=tuple(train.X.columns))


@dataclass(frozen=True)
class TLearner:
    """Estimate uplift as the difference between two outcome models."""

    control_model: LGBMClassifier
    treated_model: LGBMClassifier
    feature_columns: tuple[str, ...]

    def predict_uplift(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        validate_features(features)
        inputs = features.loc[:, list(self.feature_columns)]
        treated = np.asarray(
            self.treated_model.predict_proba(inputs), dtype=np.float64
        )[:, 1]
        control = np.asarray(
            self.control_model.predict_proba(inputs), dtype=np.float64
        )[:, 1]
        return np.asarray(treated - control, dtype=np.float64)


def fit_t_learner(train: FeatureMatrix, params: Mapping[str, Any]) -> TLearner:
    """Fit one binary outcome classifier per treatment arm."""
    validate_features(train.X)
    control = train.t == 0
    treated = train.t == 1
    if np.unique(train.y[control]).size != 2 or np.unique(train.y[treated]).size != 2:
        raise ValueError("T-learner requires both outcome classes in each arm.")

    control_model = LGBMClassifier(**dict(params))
    treated_model = LGBMClassifier(**dict(params))
    control_model.fit(train.X.loc[control], train.y[control])
    treated_model.fit(train.X.loc[treated], train.y[treated])
    return TLearner(control_model, treated_model, tuple(train.X.columns))


@dataclass(frozen=True)
class XLearner:
    """Combine arm-specific predictions of the imputed treatment effect."""

    control_effect: LGBMRegressor
    treated_effect: LGBMRegressor
    treated_share: float
    feature_columns: tuple[str, ...]

    def predict_uplift(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        validate_features(features)
        inputs = features.loc[:, list(self.feature_columns)]
        score = self.treated_share * np.asarray(
            self.control_effect.predict(inputs), dtype=np.float64
        ) + (1.0 - self.treated_share) * np.asarray(
            self.treated_effect.predict(inputs), dtype=np.float64
        )
        return np.asarray(score, dtype=np.float64)


def fit_x_learner(
    train: FeatureMatrix,
    outcome_params: Mapping[str, Any],
    effect_params: Mapping[str, Any],
) -> XLearner:
    """Fit outcome models, then regress imputed effects within each arm."""
    outcomes = fit_t_learner(train, outcome_params)
    treated = train.t == 1
    control = ~treated
    control_x = train.X.loc[control]
    treated_x = train.X.loc[treated]

    control_target = (
        np.asarray(outcomes.treated_model.predict_proba(control_x), dtype=np.float64)[
            :, 1
        ]
        - train.y[control]
    )
    treated_target = (
        train.y[treated]
        - np.asarray(outcomes.control_model.predict_proba(treated_x), dtype=np.float64)[
            :, 1
        ]
    )

    control_effect = LGBMRegressor(**dict(effect_params))
    treated_effect = LGBMRegressor(**dict(effect_params))
    control_effect.fit(control_x, control_target)
    treated_effect.fit(treated_x, treated_target)
    return XLearner(
        control_effect, treated_effect, float(treated.mean()), outcomes.feature_columns
    )
