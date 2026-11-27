"""T57 champion: the best out-of-fold Qini across all families, refitted and registered.

ADR 0006 makes the family and parameters with the best out-of-fold Qini in the
Criteo conversion search T57's champion. This module picks it from the search
summary and wraps the fitted model as an MLflow ``pyfunc`` model, so T59 can load
it from the Model Registry with ``mlflow.pyfunc.load_model`` and get uplift scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from netlift.models.k1_refit import SearchedModel
from netlift.models.uplift_search import FAMILIES, Family, UpliftModel
from netlift.utils.errors import ContractError

REGISTERED_MODEL_NAME = "netlift-criteo-conversion-champion"
CHAMPION_ALIAS = "champion"
MODEL_ARTIFACT = "uplift_model.joblib"


def choose_champion(report: Mapping[str, Any]) -> SearchedModel:
    """The family with the best out-of-fold Qini in a Criteo conversion search."""
    if report.get("dataset") != "criteo" or report.get("target") != "conversion":
        raise ContractError(
            "The champion comes from the Criteo conversion search; this search is "
            f"{report.get('dataset')!r} / {report.get('target')!r}."
        )
    candidates: list[SearchedModel] = []
    for entry in report.get("families", []):
        if entry.get("family") not in FAMILIES:
            continue
        value = float(entry["best_value"])
        if not np.isfinite(value):
            raise ContractError(f"{entry['family']} has a non-finite out-of-fold Qini.")
        candidates.append(
            SearchedModel(
                family=cast(Family, entry["family"]),
                oof_qini=value,
                params=dict(entry["best_params"]),
                best_trial=int(entry["best_trial"]),
                n_trials_completed=int(entry["n_trials_completed"]),
            )
        )
    if not candidates:
        raise ContractError("The search summary has no family results.")
    return max(candidates, key=lambda model: model.oof_qini)


class UpliftPyfunc(PythonModel):
    """MLflow wrapper: ``predict`` returns one uplift score per input row."""

    def load_context(self, context: Any) -> None:
        """Load the fitted :class:`UpliftModel` saved next to the MLflow model."""
        self.model: UpliftModel = joblib.load(context.artifacts["uplift_model"])

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: Any = None
    ) -> np.ndarray:
        """Uplift for each row; forbidden columns are rejected by the model."""
        return self.model.predict_uplift(model_input)


def save_uplift_model(model: UpliftModel, folder: Path) -> Path:
    """Write the fitted model with joblib and return the file path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MODEL_ARTIFACT
    joblib.dump(model, path)
    return path
