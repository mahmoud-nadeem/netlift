"""Tests for the complier-baseline estimator in scripts/criteo_complier_baseline.py."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "criteo_complier_baseline.py"


def load_script() -> ModuleType:
    """Import the script as a module without running main()."""
    spec = importlib.util.spec_from_file_location("criteo_complier_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def estimate(
    n_control: int,
    k_control: int,
    n_exposed: int,
    k_exposed: int,
    n_unexposed: int,
    k_unexposed: int,
) -> dict[str, float]:
    """Run point_estimates on plain counts and return scalar results."""
    module = load_script()
    result = module.point_estimates(
        n_control=n_control,
        k_control=np.array([float(k_control)]),
        n_exposed=np.array([float(n_exposed)]),
        k_exposed=np.array([float(k_exposed)]),
        n_unexposed=np.array([float(n_unexposed)]),
        k_unexposed=np.array([float(k_unexposed)]),
    )
    return {key: float(value[0]) for key, value in result.items()}


def test_hand_computed_case() -> None:
    """Check every quantity against values worked out by hand.

    Control: 20 of 1,000 convert, so y_control = 0.02.
    Treated: 100 exposed (25 convert), 900 unexposed (9 convert).
    pi = 0.1, y_treated = 34 / 1,000 = 0.034, ITT = 0.014, LATE = 0.14.
    Baseline = (0.02 - 0.9 * 0.01) / 0.1 = 0.11, and 0.25 - 0.11 = 0.14.
    """
    result = estimate(1_000, 20, 100, 25, 900, 9)

    assert result["pi"] == pytest.approx(0.1)
    assert result["y_control"] == pytest.approx(0.02)
    assert result["y_exposed"] == pytest.approx(0.25)
    assert result["y_unexposed"] == pytest.approx(0.01)
    assert result["itt"] == pytest.approx(0.014)
    assert result["late"] == pytest.approx(0.14)
    assert result["baseline"] == pytest.approx(0.11)
    assert result["lift_vs_complier"] == pytest.approx(14 / 11)
    assert result["lift_vs_population"] == pytest.approx(7.0)


def test_negative_baseline_gives_no_lift() -> None:
    """A baseline below zero must not produce a relative lift."""
    result = estimate(1_000, 5, 100, 25, 900, 9)

    assert result["baseline"] < 0
    assert np.isnan(result["lift_vs_complier"])


def test_criteo_counts_match_specification() -> None:
    """Reproduce the Section 6 figures from the counted Criteo cells."""
    result = estimate(2_096_937, 4_063, 428_212, 23_031, 11_454_443, 13_680)

    assert result["pi"] == pytest.approx(0.036037, abs=5e-7)
    assert result["y_control"] == pytest.approx(0.001938, abs=5e-7)
    assert result["itt"] == pytest.approx(0.001152, abs=5e-7)
    assert result["late"] == pytest.approx(0.031964, abs=5e-7)
    assert result["baseline"] == pytest.approx(0.021820, abs=5e-7)


def test_relative_lift_format() -> None:
    """The lift is shown as a relative increase with the multiple of baseline."""
    module = load_script()

    assert module.relative(14 / 11) == "+127% (2.27x)"
