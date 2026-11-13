"""Tests for scripts/randomization_integrity.py (T40, K6 gate)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _load_script() -> ModuleType:
    """Import scripts/randomization_integrity.py by path (scripts/ is not a package)."""
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "randomization_integrity.py"
    )
    spec = importlib.util.spec_from_file_location("randomization_integrity", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ri = _load_script()

N_ROWS = 4000


def _cfg() -> dict[str, Any]:
    """Pre-registered thresholds, with a fast bootstrap and no AUC subsampling."""
    cfg: dict[str, Any] = dict(ri.DEFAULTS)
    cfg["n_boot"] = 50
    cfg["auc_sample"] = None
    return cfg


def _synthetic(shift_f3: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Covariates f0..f4 and a randomly assigned treatment flag.

    `shift_f3` moves f3 by that many standard deviations for treated rows only.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(N_ROWS, 5)), columns=[f"f{i}" for i in range(5)]
    )
    treated = rng.permutation(np.arange(N_ROWS) % 2 == 0)
    X.loc[treated, "f3"] += shift_f3
    return X, treated


def test_smd_hand_computed() -> None:
    """Treated [1,2,3] vs control [3,4,5]: means 2 and 4, variances 1, SMD = -2."""
    value = ri.smd(np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0]))
    assert value == pytest.approx(-2.0)


def test_smd_zero_variance_is_zero() -> None:
    """A constant covariate has no spread, so its SMD is defined as 0."""
    value = ri.smd(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    assert value == 0.0


def test_randomized_data_passes() -> None:
    """Randomly assigned treatment must give status PASS."""
    X, treated = _synthetic()
    result = ri.run_comparison("synthetic randomized", X, treated, _cfg())
    assert result["status"] == "PASS"
    assert result["max_smd"] < 0.1
    assert result["overlap_ok"]


def test_planted_imbalance_fails_and_names_covariate() -> None:
    """A 1 SD shift of f3 between arms must FAIL and the reason must name f3."""
    X, treated = _synthetic(shift_f3=1.0)
    result = ri.run_comparison("synthetic imbalanced", X, treated, _cfg())
    assert result["status"] == "FAIL"
    assert any("f3" in reason for reason in result["reasons"])


def test_empty_arm_is_rejected() -> None:
    """A comparison with no control rows must raise instead of returning numbers."""
    X, _ = _synthetic()
    with pytest.raises(ValueError, match="empty"):
        ri.run_comparison("all treated", X, np.ones(N_ROWS, dtype=bool), _cfg())


def test_forbidden_columns_are_not_covariates() -> None:
    """Outcomes and post-treatment columns must never enter the check."""
    forbidden = {"exposure", "visit", "conversion", "spend", "treatment", "segment"}
    assert not forbidden & set(ri.CRITEO_FEATURES)
    assert not forbidden & set(ri.HILLSTROM_FEATURES)
