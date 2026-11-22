"""Regression tests for the T56 meta-learners and class transformation."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from netlift.data.contracts import (
    SYNTHETIC_TREATMENT,
    FeatureMatrix,
    FeatureMatrixMeta,
)
from netlift.models.meta_learners import (
    XLearner,
    fit_class_transformation,
    fit_s_learner,
    fit_t_learner,
    fit_x_learner,
)
from netlift.utils.errors import ContractError


def test_propensity_weighting_corrects_unequal_arm_bias() -> None:
    """Equal arm response rates imply zero uplift even with an 85/15 split."""
    treatment = np.repeat([0, 1], [150, 850]).astype(np.int8)
    row_ids = np.arange(len(treatment), dtype=np.int64)
    outcomes = (row_ids % 10 == 0).astype(np.int8)
    features = pd.DataFrame({"feature_1": np.zeros(len(treatment))})
    train = FeatureMatrix(
        X=features,
        t=treatment,
        y=outcomes,
        fold=np.zeros_like(row_ids),
        row_id=row_ids,
        meta=FeatureMatrixMeta(
            "synthetic", "conversion", SYNTHETIC_TREATMENT, 42, "train"
        ),
    )
    params = {"n_estimators": 10, "random_state": 42, "verbosity": -1, "n_jobs": 2}
    naive = fit_class_transformation(train, params, propensity_weighted=False)
    weighted = fit_class_transformation(train, params, propensity_weighted=True)

    probe = features.iloc[:10]
    np.testing.assert_allclose(naive.predict_uplift(probe), -0.56, atol=1e-6)
    np.testing.assert_allclose(weighted.predict_uplift(probe), 0.0, atol=1e-6)


def test_meta_learners_recover_positive_effect() -> None:
    """All three learners recover a 20-point arm difference."""
    treatment = np.repeat([0, 1], [150, 850]).astype(np.int8)
    row_ids = np.arange(len(treatment), dtype=np.int64)
    outcomes = (row_ids % 10 < np.where(treatment == 1, 3, 1)).astype(np.int8)
    features = pd.DataFrame({"feature_1": np.zeros(len(treatment))})

    train = FeatureMatrix(
        X=features,
        t=treatment,
        y=outcomes,
        fold=np.zeros_like(row_ids),
        row_id=row_ids,
        meta=FeatureMatrixMeta(
            "synthetic", "conversion", SYNTHETIC_TREATMENT, 42, "train"
        ),
    )
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": 2,
    }
    models = {
        "S": fit_s_learner(train, params),
        "T": fit_t_learner(train, params),
        "X": fit_x_learner(train, params, params),
    }

    probe = features.iloc[:10].copy()
    original_probe = probe.copy()
    for name, model in models.items():
        scores = model.predict_uplift(probe)
        assert scores.shape == (10,), name
        np.testing.assert_allclose(scores, 0.20, atol=0.01, err_msg=name)
        pd.testing.assert_frame_equal(probe, original_probe)


@pytest.fixture
def heterogeneous_training_matrix() -> FeatureMatrix:
    """Build three groups whose treatment effects are -0.2, +0.2, and +0.5."""
    groups = np.repeat(np.arange(3), 500)
    treatment = np.tile(np.repeat([0, 1], [100, 400]), 3).astype(np.int8)
    row_ids = np.arange(len(treatment), dtype=np.int64)
    control_rates = np.array([4, 2, 2])
    treated_rates = np.array([2, 4, 7])
    rates = np.where(treatment == 1, treated_rates[groups], control_rates[groups])
    return FeatureMatrix(
        X=pd.DataFrame({"feature_1": groups}),
        t=treatment,
        y=(row_ids % 10 < rates).astype(np.int8),
        fold=np.zeros_like(row_ids),
        row_id=row_ids,
        meta=FeatureMatrixMeta(
            "synthetic", "conversion", SYNTHETIC_TREATMENT, 42, "train"
        ),
    )


def test_meta_learners_rank_heterogeneous_effects(
    heterogeneous_training_matrix: FeatureMatrix,
) -> None:
    train = heterogeneous_training_matrix
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": 2,
    }
    models = {
        "S": fit_s_learner(train, params),
        "T": fit_t_learner(train, params),
        "X": fit_x_learner(train, params, params),
        "weighted CT": fit_class_transformation(
            train, params, propensity_weighted=True
        ),
    }

    probe = pd.DataFrame({"feature_1": [2, 0, 1]})
    for name, model in models.items():
        scores = model.predict_uplift(probe)
        np.testing.assert_array_equal(np.argsort(scores), [1, 2, 0], err_msg=name)
        np.testing.assert_allclose(scores, [0.5, -0.2, 0.2], atol=0.02, err_msg=name)


def test_x_learner_weights_the_control_effect_by_treated_share() -> None:
    control_effect = Mock(spec=LGBMRegressor)
    control_effect.predict.return_value = np.array([0.5, -0.2])
    treated_effect = Mock(spec=LGBMRegressor)
    treated_effect.predict.return_value = np.array([0.0, 0.3])
    model = XLearner(control_effect, treated_effect, 0.8, ("feature_1",))

    scores = model.predict_uplift(pd.DataFrame({"feature_1": [0, 1]}))
    np.testing.assert_allclose(scores, [0.4, -0.1])


@pytest.mark.parametrize("column", ["exposure", "visit", "conversion"])
def test_meta_learners_reject_forbidden_features_at_fit_and_predict(
    heterogeneous_training_matrix: FeatureMatrix, column: str
) -> None:
    train = heterogeneous_training_matrix
    params = {"n_estimators": 5, "random_state": 42, "verbosity": -1, "n_jobs": 2}
    fitters = (
        lambda: fit_s_learner(train, params),
        lambda: fit_t_learner(train, params),
        lambda: fit_x_learner(train, params, params),
        lambda: fit_class_transformation(train, params, propensity_weighted=True),
    )
    models = [fit() for fit in fitters]

    # Add the column after construction to exercise the fit entry points.
    train.X[column] = 0
    for fit, model in zip(fitters, models, strict=True):
        with pytest.raises(ContractError, match=f"Forbidden column.*{column}"):
            fit()
        with pytest.raises(ContractError, match=f"Forbidden column.*{column}"):
            model.predict_uplift(train.X.iloc[:3])


def test_meta_learners_reject_a_single_treatment_arm(
    heterogeneous_training_matrix: FeatureMatrix,
) -> None:
    train = heterogeneous_training_matrix
    # Bypass the matrix constructor's both-arms check to exercise the fit guards.
    train.t[:] = 1
    with pytest.raises(ValueError, match="both treatment arms"):
        fit_s_learner(train, {})
    with pytest.raises(ValueError, match="both treatment arms"):
        fit_class_transformation(train, {}, propensity_weighted=True)
    with pytest.raises(ValueError, match="both outcome classes in each arm"):
        fit_t_learner(train, {})
    with pytest.raises(ValueError, match="both outcome classes in each arm"):
        fit_x_learner(train, {}, {})


def test_meta_learners_reject_single_class_targets(
    heterogeneous_training_matrix: FeatureMatrix,
) -> None:
    train = heterogeneous_training_matrix
    train.y[:] = train.t
    with pytest.raises(ValueError, match="transformed target requires both classes"):
        fit_class_transformation(train, {}, propensity_weighted=True)
    with pytest.raises(ValueError, match="both outcome classes in each arm"):
        fit_t_learner(train, {})
    with pytest.raises(ValueError, match="both outcome classes in each arm"):
        fit_x_learner(train, {}, {})
