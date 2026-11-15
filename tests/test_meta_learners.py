"""Regression test for class transformation with unequal treatment shares."""

import numpy as np
import pandas as pd

from netlift.data.contracts import (
    SYNTHETIC_TREATMENT,
    FeatureMatrix,
    FeatureMatrixMeta,
)
from netlift.models.meta_learners import fit_class_transformation


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
    params = {"n_estimators": 10, "verbosity": -1, "n_jobs": 2}
    naive = fit_class_transformation(train, params, propensity_weighted=False)
    weighted = fit_class_transformation(train, params, propensity_weighted=True)

    probe = features.iloc[:10]
    np.testing.assert_allclose(naive.predict_uplift(probe), -0.56, atol=1e-6)
    np.testing.assert_allclose(weighted.predict_uplift(probe), 0.0, atol=1e-6)


def test_meta_learners_recover_positive_effect() -> None:
    """All three learners recover a 20-point arm difference."""
    from netlift.models.meta_learners import (
        fit_s_learner,
        fit_t_learner,
        fit_x_learner,
    )

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
