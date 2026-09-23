"""Verify that this machine's environment matches the NetLift specification.

Every team member runs this once after setting up, and pastes the output into
the environment task. It is the evidence that the environment is identical on
all five machines, which is what milestone 1 requires.

Usage:
    python scripts/verify_environment.py
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

REQUIRED_PYTHON: tuple[int, int] = (3, 12)

# MLflow prints an unrelated hint on first use. Not wanted in a report.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# package name on PyPI -> module name to import
PACKAGES: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "scikit-uplift": "sklift",
    "causalml": "causalml.inference.meta",
    "statsmodels": "statsmodels",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "shap": "shap",
    "mlflow": "mlflow",
    "optuna": "optuna",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "PyYAML": "yaml",
    "python-dotenv": "dotenv",
    "ipykernel": "ipykernel",
}

# PACKAGES must mirror requirements.txt exactly. check_pin_coverage below
# enforces that, so a package added to the pins but forgotten here fails the
# run instead of going quietly unchecked.
#
# requirements-dev.txt is deliberately not verified here. Those tools are
# verified by being used: CI runs ruff, mypy and pytest on every pull request.

# No ANSI colour codes anywhere in this file on purpose. The output of this
# script is pasted into GitHub and Discord, and escape sequences paste as
# garbage.


@dataclass(frozen=True)
class Check:
    """One verification result."""

    name: str
    passed: bool
    detail: str


def _read_pins(requirements: Path) -> dict[str, str]:
    """Parse `name==version` lines out of a requirements file."""
    pins: dict[str, str] = {}
    if not requirements.exists():
        return pins
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if "==" in line:
            name, _, version = line.partition("==")
            pins[name.strip().lower()] = version.strip()
    return pins


def check_python() -> Check:
    """Confirm the interpreter is the required Python version."""
    actual = sys.version_info[:2]
    ok = actual == REQUIRED_PYTHON
    detail = (
        f"{actual[0]}.{actual[1]} (required {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]})"
    )
    return Check("python", ok, detail)


def check_virtualenv() -> Check:
    """Confirm we are inside a virtual environment, not the system Python."""
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    detail = sys.prefix if in_venv else "NOT in a virtual environment"
    return Check("virtualenv", in_venv, detail)


def check_package(pypi_name: str, module_name: str, pins: dict[str, str]) -> Check:
    """Import a package and compare its installed version against the pin."""
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
        return Check(pypi_name, False, f"import failed: {type(exc).__name__}: {exc}")

    try:
        installed = metadata.version(pypi_name)
    except metadata.PackageNotFoundError:
        return Check(pypi_name, False, "installed but version not found")

    expected = pins.get(pypi_name.lower())
    if expected is None:
        return Check(pypi_name, True, f"{installed} (not pinned)")
    if installed != expected:
        return Check(pypi_name, False, f"{installed}, expected {expected}")
    return Check(pypi_name, True, installed)


def check_pin_coverage(pins: dict[str, str]) -> Check:
    """Every package pinned in requirements.txt must be checked by this script.

    Without this, adding a dependency to the pins and forgetting to add it to
    PACKAGES leaves it silently unverified, and the report claims a coverage it
    does not have.
    """
    checked = {name.lower() for name in PACKAGES}
    missing = sorted(name for name in pins if name not in checked)
    if not pins:
        return Check("pin coverage", False, "requirements.txt not found or has no pins")
    if missing:
        return Check(
            "pin coverage",
            False,
            f"pinned but not checked: {', '.join(missing)}",
        )
    return Check("pin coverage", True, f"all {len(pins)} pinned packages are checked")


def check_mlflow_backend() -> Check:
    """MLflow 3.x refuses the plain ./mlruns file store. Confirm SQLite works."""
    import logging
    import shutil
    import tempfile

    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001
        return Check("mlflow backend", False, f"mlflow not importable: {exc}")

    # Keep MLflow's own INFO logging out of the report.
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    # The temporary directory is created and removed by hand rather than with
    # tempfile.TemporaryDirectory. SQLAlchemy holds the SQLite file open, and
    # Windows refuses to delete an open file, so the context manager's cleanup
    # raises PermissionError on exit even though the check itself succeeded.
    tmp = tempfile.mkdtemp(prefix="netlift-verify-")
    database = Path(tmp) / "verify.db"
    try:
        mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
        mlflow.set_experiment("environment-verification")
        with mlflow.start_run(run_name="verify"):
            mlflow.log_param("check", "environment")
            mlflow.log_metric("ok", 1.0)
    except Exception as exc:  # noqa: BLE001
        return Check("mlflow backend", False, f"{type(exc).__name__}: {exc}")
    finally:
        # ignore_errors because the open SQLite handle may still block the
        # delete on Windows. A stray file in the system temp directory is not
        # a reason to fail an environment check.
        shutil.rmtree(tmp, ignore_errors=True)

    return Check("mlflow backend", True, "sqlite tracking store writable")


def check_uplift_runtime() -> Check:
    """Fit one uplift model end to end. Imports succeeding is not enough."""
    import warnings

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklift.metrics import qini_auc_score
        from sklift.models import TwoModels
    except Exception as exc:  # noqa: BLE001
        return Check("uplift runtime", False, f"import failed: {exc}")

    try:
        rng = np.random.default_rng(0)
        n = 2_000
        x = rng.normal(size=(n, 4))
        treatment = rng.integers(0, 2, n)
        base = 1.0 / (1.0 + np.exp(-x[:, 0]))
        probability = np.clip(base + treatment * 0.25 * (x[:, 1] > 0), 0.01, 0.99)
        y = (rng.random(n) < probability).astype(int)

        model = TwoModels(
            estimator_trmnt=LogisticRegression(max_iter=1_000),
            estimator_ctrl=LogisticRegression(max_iter=1_000),
            method="vanilla",
        )
        with warnings.catch_warnings():
            # scikit-uplift still calls a scikit-learn helper that is on its way
            # out. Harmless here, and noisy in a verification report.
            warnings.simplefilter("ignore", FutureWarning)
            model.fit(x, y, treatment)
            qini = qini_auc_score(y, model.predict(x), treatment)
    except Exception as exc:  # noqa: BLE001
        return Check("uplift runtime", False, f"{type(exc).__name__}: {exc}")

    return Check("uplift runtime", True, f"two-model qini {qini:.4f} on synthetic data")


def main() -> int:
    """Run every check and print a report. Returns a process exit code."""
    root = Path(__file__).resolve().parent.parent
    pins = _read_pins(root / "requirements.txt")

    print()
    print("NetLift environment verification")
    print(f"  machine   {platform.platform()}")
    print(f"  python    {sys.executable}")
    print()

    checks: list[Check] = [check_python(), check_virtualenv()]
    checks += [check_package(p, m, pins) for p, m in PACKAGES.items()]
    checks += [check_pin_coverage(pins), check_mlflow_backend(), check_uplift_runtime()]

    width = max(len(c.name) for c in checks)
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  {mark}  {check.name:<{width}}  {check.detail}")

    failed = [c for c in checks if not c.passed]
    print()
    if failed:
        print(f"{len(failed)} of {len(checks)} checks failed.")
        print(
            "Post the full output in #help. Do not start your task until this is green."
        )
        return 1

    print(f"All {len(checks)} checks passed.")
    print("Paste this output into the environment task as your verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
