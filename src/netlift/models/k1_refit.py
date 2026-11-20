"""K1 input from T57: choose the T- or X-learner, refit on the full training split, score validation.

KPI K1 (``config/kpis.yaml``) compares the uplift policy with the response model on
Criteo conversion. At M4 the uplift policy is the T-learner or X-learner with the
higher out-of-fold Qini from the T57 search (ADR 0006); the other is reported as
sensitivity. This module reads the search summary, refits both with their best
parameters on the full training split (no subsample), and scores the full validation
split. The locked test split is never opened.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from netlift.data.contracts import DatasetSplits
from netlift.models.uplift_search import Family, fit_predict
from netlift.utils.errors import ContractError

K1_FAMILIES: tuple[Family, Family] = ("t_learner", "x_learner")


@dataclass(frozen=True)
class SearchedModel:
    """One family's best trial, as the T57 search summary records it."""

    family: Family
    oof_qini: float
    params: dict[str, Any]
    best_trial: int
    n_trials_completed: int


@dataclass(frozen=True)
class K1Choice:
    """The K1 uplift model and its sensitivity model, with the search they came from."""

    primary: SearchedModel
    sensitivity: SearchedModel
    n_trials_budget: int
    seed: int


def read_search_report(path: Path) -> dict[str, Any]:
    """Read a ``reports/t57-<dataset>-<target>-search.json`` summary."""
    if not path.exists():
        raise ContractError(f"{path} not found. Run the T57 search first.")
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


def choose_k1_models(report: Mapping[str, Any]) -> K1Choice:
    """The T/X learner with the higher out-of-fold Qini is K1's; the other is sensitivity.

    The search must be Criteo conversion, as K1 is. A tie goes to the T-learner,
    the simpler of the two.
    """
    if report.get("dataset") != "criteo" or report.get("target") != "conversion":
        raise ContractError(
            "K1 is computed on Criteo conversion; this search is "
            f"{report.get('dataset')!r} / {report.get('target')!r}."
        )
    found: dict[str, SearchedModel] = {}
    for entry in report.get("families", []):
        family = entry.get("family")
        if family in K1_FAMILIES:
            value = float(entry["best_value"])
            if not np.isfinite(value):
                raise ContractError(f"{family} has a non-finite out-of-fold Qini.")
            found[family] = SearchedModel(
                family=cast(Family, family),
                oof_qini=value,
                params=dict(entry["best_params"]),
                best_trial=int(entry["best_trial"]),
                n_trials_completed=int(entry["n_trials_completed"]),
            )
    missing = [family for family in K1_FAMILIES if family not in found]
    if missing:
        raise ContractError(f"The search summary has no result for {missing}.")
    t_model, x_model = found["t_learner"], found["x_learner"]
    primary, sensitivity = (
        (x_model, t_model)
        if x_model.oof_qini > t_model.oof_qini
        else (t_model, x_model)
    )
    return K1Choice(
        primary=primary,
        sensitivity=sensitivity,
        n_trials_budget=int(report["n_trials_budget"]),
        seed=int(report["seed"]),
    )


def refit_and_score(
    splits: DatasetSplits, model: SearchedModel, *, seed: int
) -> pd.DataFrame:
    """Fit on all of ``splits.train`` and return validation scores keyed by ``row_id``.

    Only ``train`` and ``validation`` are read. The test split stays locked; the
    caller checks ``splits.test.is_open`` afterwards.
    """
    train, validation = splits.train, splits.validation
    if train.meta.split_name != "train" or validation.meta.split_name != "validation":
        raise ContractError("refit_and_score needs the train and validation splits.")
    score = fit_predict(
        model.family,
        model.params,
        train.X,
        train.t,
        train.y,
        validation.X,
        seed=seed,
    )
    if score.shape != (validation.n_rows,) or not np.isfinite(score).all():
        raise ContractError("Expected one finite score per validation row.")
    return pd.DataFrame({"row_id": validation.row_id, "score": score})
