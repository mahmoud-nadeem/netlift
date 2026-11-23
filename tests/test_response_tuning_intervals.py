"""Additional #107 checks: AUC intervals and canonical train subsampling."""

import numpy as np
import pytest

from netlift.data.split import subsample_part
from netlift.data.synthetic import make_synthetic_splits
from netlift.models.response_tuning import auc_interval, tuning_subset


def test_auc_variance_matches_pairwise_placements():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.2, 0.5, 0.9, 0.1, 0.2, 0.5, 0.8])
    pair = (p[y == 1, None] > p[y == 0]).astype(float)
    pair += 0.5 * (p[y == 1, None] == p[y == 0])
    auc, low, high = auc_interval(y, p)
    variance = pair.mean(axis=1).var(ddof=1) / 3 + pair.mean(axis=0).var(ddof=1) / 4
    from scipy.stats import norm

    half = norm.ppf(0.975) * np.sqrt(variance)
    assert auc == pytest.approx(pair.mean())
    assert low == pytest.approx(max(0, auc - half))
    assert high == pytest.approx(min(1, auc + half))


def test_constant_auc_interval_is_a_point():
    assert auc_interval(np.array([0, 0, 1, 1]), np.full(4, 0.3)) == (0.5, 0.5, 0.5)


def test_sample_matches_canonical_train_key_and_keeps_ids():
    splits = make_synthetic_splits("conversion", n_rows=6000, seed=42)
    train = splits.train
    small = tuning_subset(train, 0.1, seed=42)
    key = (train.t.astype(np.int64) * 2 + train.y) * 5 + train.fold
    idx = subsample_part(np.arange(train.n_rows), key, 0.1, 42)
    np.testing.assert_array_equal(small.row_id, train.row_id[idx])
    np.testing.assert_array_equal(small.fold, train.fold[idx])
    assert not set(small.row_id) & set(splits.validation.row_id)
    assert not splits.test.is_open
    assert tuning_subset(train, None, seed=42) is train
