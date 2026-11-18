"""T43 / K4: the permuted-treatment negative control, on the synthetic fixture."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from netlift.data.contracts import DatasetSplits
from netlift.evaluation.contracts import MIN_REPORTED_RESAMPLES
from netlift.utils.errors import ContractError

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_negative_control.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_negative_control", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_negative_control"] = module
    spec.loader.exec_module(module)
    return module


nc = _load_script()


def test_permuted_treatment_ci_contains_zero(synthetic_splits: DatasetSplits) -> None:
    """K4: under a permuted treatment the Qini 95% interval contains zero."""
    result = nc.run_negative_control(
        synthetic_splits, seed=42, n_resamples=MIN_REPORTED_RESAMPLES
    )
    metric = result.metric
    assert metric.n_resamples >= MIN_REPORTED_RESAMPLES
    assert metric.split_name == "validation"
    assert metric.library == "scikit-uplift"
    assert metric.ci_low <= 0.0 <= metric.ci_high, (
        f"Qini CI [{metric.ci_low:.4f}, {metric.ci_high:.4f}] excludes zero "
        "under permuted treatment: look for a leak or a bug."
    )
    assert result.passed


def test_negative_control_is_deterministic(synthetic_splits: DatasetSplits) -> None:
    """Same seed, same interval and same resamples."""
    first = nc.run_negative_control(synthetic_splits, seed=7, n_resamples=200)
    second = nc.run_negative_control(synthetic_splits, seed=7, n_resamples=200)
    assert first.metric.value == second.metric.value
    assert first.metric.ci_low == second.metric.ci_low
    assert first.metric.ci_high == second.metric.ci_high
    assert first.metric.resample_fingerprint == second.metric.resample_fingerprint


def test_permutation_keeps_arm_sizes_and_changes_assignment(
    synthetic_splits: DatasetSplits,
) -> None:
    """The shuffle preserves the arm proportions and really changes who is treated."""
    t = synthetic_splits.train.t
    permuted = nc.permute_treatment(t, np.random.default_rng(0))
    assert permuted.sum() == t.sum()
    assert (permuted != t).any()


def test_negative_control_never_opens_the_locked_test_split(
    synthetic_splits: DatasetSplits,
) -> None:
    """Only train and validation are read."""
    nc.run_negative_control(synthetic_splits, seed=42, n_resamples=50)
    assert not synthetic_splits.test.is_open


def test_rejects_bad_subsample_fraction(synthetic_splits: DatasetSplits) -> None:
    """A subsample fraction outside (0, 1] is refused, not clipped."""
    with pytest.raises(ContractError):
        nc.run_negative_control(synthetic_splits, subsample_fraction=0.0)
