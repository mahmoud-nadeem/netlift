"""S-learner and class-transformation models for T56."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier

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
        control = self.model.predict_proba(inputs)[:, 1]
        inputs[_TREATMENT_COLUMN] = 1
        treated = self.model.predict_proba(inputs)[:, 1]
        return np.asarray(treated - control, dtype=np.float64)


def fit_s_learner(train: FeatureMatrix, params: Mapping[str, object]) -> SLearner:
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
        probability = self.model.predict_proba(inputs)[:, 1]
        return np.asarray(2.0 * probability - 1.0, dtype=np.float64)


def fit_class_transformation(
    train: FeatureMatrix,
    params: Mapping[str, object],
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
