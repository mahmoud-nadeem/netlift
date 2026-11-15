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
