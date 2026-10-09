"""Experiment tracking.

A thin wrapper over MLflow, so that every run in this project is recorded the
same way and the project's rules are enforced by code instead of by five people
remembering them.

Three rules live here.

**Every run records what produced it.** The git commit, whether the working tree
was clean at the time, the checksum of the dataset, and the random seed. A
metric without those is a number nobody can reproduce, including whoever
produced it.

**Every headline number carries a confidence interval.** ``log_kpi`` will not
accept a value without one, and will not accept a KPI id that is not defined in
``config/kpis.yaml``. That keeps MLflow's metric names and the KPI framework
from drifting apart.

**The seed is applied, not just recorded.** ``start_run`` seeds Python's
``random`` and NumPy itself, so the logged seed is true by construction rather
than true if the caller remembered.

Usage::

    from pathlib import Path
    from netlift.utils.tracking import start_run

    with start_run(
        "x-learner, lightgbm base",
        dataset=Path("data/raw/hillstrom.csv"),
        seed=42,
    ) as run:
        run.log_params({"learner": "x", "base_model": "lightgbm"})
        run.log_metrics({"train_rows": 44_800})
        run.log_kpi("K2", value=0.0847, ci_low=0.0612, ci_high=0.1103)
        run.log_artifact(Path("reports/figures/qini.png"))

Nobody calls ``mlflow`` directly. If something is needed that this module does
not expose, add it here rather than reaching around it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
KPI_CONFIG = REPO_ROOT / "config" / "kpis.yaml"

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "netlift"

# MLflow 3.x no longer accepts a plain directory as the tracking backend. Rather
# than guessing which strings are directories, only these are allowed through.
ACCEPTED_URI_SCHEMES = (
    "sqlite:",
    "postgresql:",
    "mysql:",
    "mssql:",
    "http:",
    "https:",
    "databricks",
)

_CHUNK = 1024 * 1024


class TrackingError(RuntimeError):
    """Raised when a run is configured or logged in a way the project forbids."""


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str | None:
    """Run a git command in the repository, or return None if that is not possible."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_commit() -> str:
    """The commit the working tree is on, or 'unknown' outside a git checkout."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    """Whether uncommitted changes exist.

    This matters more than it looks. If the tree is dirty, the commit hash does
    not describe the code that actually ran, so a run tagged with it cannot be
    rebuilt from that commit. Recording the fact is the difference between
    tracking and the appearance of tracking.
    """
    status = _git("status", "--porcelain")
    return bool(status)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so large datasets do not enter memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed every source of randomness this project uses."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        return
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# KPI definitions
# --------------------------------------------------------------------------- #


def load_kpi_ids(config_path: Path | None = None) -> set[str]:
    """Read the KPI ids defined in config/kpis.yaml.

    The ids are the vocabulary for metric names, so that the same quantity is
    not logged as 'qini', 'Qini' and 'qini_score' by three different people.
    """
    path = config_path or KPI_CONFIG
    if not path.exists():
        raise TrackingError(
            f"KPI definitions not found at {path}. "
            "Metric names come from that file, so it has to exist."
        )
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = document.get("kpis") or []
    ids = {str(entry["id"]) for entry in entries if "id" in entry}
    if not ids:
        raise TrackingError(f"No KPI ids found in {path}.")
    return ids


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


class Run:
    """A single tracked experiment. Obtained from :func:`start_run`."""

    def __init__(self, kpi_ids: set[str]) -> None:
        self._kpi_ids = kpi_ids

    def log_params(self, params: dict[str, Any]) -> None:
        """Log the things that were chosen: learner, hyperparameters, split rules."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log diagnostics measured during the run.

        Headline results do not go here. Use :meth:`log_kpi`, which requires a
        confidence interval.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_kpi(
        self,
        kpi_id: str,
        value: float,
        ci_low: float,
        ci_high: float,
        confidence: float = 0.95,
    ) -> None:
        """Log a headline result against a KPI defined in config/kpis.yaml.

        The confidence interval is required, not optional. Every headline number
        in this project is reported with one, and a rule that depends on five
        people remembering it will be broken at least once.
        """
        if kpi_id not in self._kpi_ids:
            known = ", ".join(sorted(self._kpi_ids))
            raise TrackingError(
                f"'{kpi_id}' is not a KPI defined in config/kpis.yaml. Known ids: {known}. "
                "Add the KPI to that file before reporting against it."
            )
        if ci_low > ci_high:
            raise TrackingError(
                f"{kpi_id}: ci_low ({ci_low}) is above ci_high ({ci_high}). "
                "The bounds are the wrong way round."
            )
        if not ci_low <= value <= ci_high:
            raise TrackingError(
                f"{kpi_id}: value {value} lies outside its interval "
                f"[{ci_low}, {ci_high}]. One of the three is wrong."
            )
        mlflow.log_metrics(
            {
                kpi_id: value,
                f"{kpi_id}_ci_low": ci_low,
                f"{kpi_id}_ci_high": ci_high,
            }
        )
        mlflow.set_tag(f"{kpi_id}_confidence", str(confidence))

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Attach a file to this run: a figure, a decile table, a model."""
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Add provenance tags beyond the ones recorded automatically."""
        mlflow.set_tags(tags)


def _resolve_tracking_uri() -> str:
    """Read the tracking URI from the environment, rejecting directory stores.

    MLflow 3.x refuses the plain './mlruns' directory store outright. Failing
    here, naming the reason and the fix, beats failing inside MLflow with a
    stack trace that never mentions the word directory.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI).strip()
    if not uri:
        uri = DEFAULT_TRACKING_URI
    if not uri.startswith(ACCEPTED_URI_SCHEMES):
        raise TrackingError(
            f"MLFLOW_TRACKING_URI is set to '{uri}', which is a directory store. "
            "MLflow 3.x rejects those. Use a database URI, for example "
            f"'{DEFAULT_TRACKING_URI}'. See docs/experiment-tracking.md."
        )
    return uri


@contextmanager
def start_run(
    run_name: str,
    *,
    dataset: Path | None = None,
    seed: int = 42,
    experiment: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Run]:
    """Start a tracked run, seeding randomness and recording provenance.

    Args:
        run_name: A short description of what this run is, shown in the UI.
        dataset: The data file this run used. Its SHA-256 is recorded, so the
            run names the exact bytes it was computed from.
        seed: Applied to ``random`` and NumPy, then recorded.
        experiment: Overrides ``MLFLOW_EXPERIMENT_NAME``.
        tags: Extra provenance tags.

    Yields:
        A :class:`Run` to log against.
    """
    seed_everything(seed)

    mlflow.set_tracking_uri(_resolve_tracking_uri())
    mlflow.set_experiment(
        experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    )

    provenance: dict[str, str] = {
        "git_commit": git_commit(),
        "git_dirty": str(git_dirty()).lower(),
        "seed": str(seed),
    }
    if dataset is not None:
        provenance["dataset_file"] = dataset.name
        provenance["dataset_sha256"] = file_sha256(dataset)
    if tags:
        provenance.update(tags)

    kpi_ids = load_kpi_ids()

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(provenance)
        if provenance["git_dirty"] == "true":
            logger.warning(
                "Working tree has uncommitted changes. This run is tagged "
                "git_dirty=true, because commit %s does not describe the code "
                "that just ran and the result cannot be rebuilt from it.",
                provenance["git_commit"][:8],
            )
        yield Run(kpi_ids)
