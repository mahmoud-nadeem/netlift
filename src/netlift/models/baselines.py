"""T38 baselines: specification §§9, 25, 27 and ADR 0001.

Only the response baseline learns a model. Random is a seeded ranking and
blanket is an all-contact policy. All metric values here are preliminary;
T39 supplies the bootstrap intervals before they can become headline results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

from netlift.data.contracts import DatasetSplits, validate_features
from netlift.utils.errors import ContractError

BaselineName = Literal["random", "blanket", "response-model"]
BASELINE_NAMES: tuple[BaselineName, ...] = ("random", "blanket", "response-model")
UPLIFT_FRACTIONS = (0.1, 0.2, 0.3)  # Required by T38, not tuned.


@dataclass(frozen=True)
class BaselineResult:
    """Validation scores in input row order and preliminary diagnostics."""

    name: BaselineName
    scores: pd.DataFrame
    metrics: dict[str, float]
    model: LGBMClassifier | None = None


def ranking_metrics(
    y: npt.NDArray[np.int8], score: npt.NDArray[np.float64], t: npt.NDArray[np.int8]
) -> dict[str, float]:
    """Compute scikit-uplift metrics with explicit settings and finite results."""
    result = {
        "qini_coefficient": float(qini_auc_score(y, score, t, negative_effect=True)),
        "auuc": float(uplift_auc_score(y, score, t)),
    }
    for k in UPLIFT_FRACTIONS:
        # Match sklift's stable tie order and integer truncation exactly.
        selected = np.argsort(score, kind="mergesort")[::-1][: int(len(y) * k)]
        if len(np.unique(t[selected])) != 2:
            raise ContractError(f"uplift@{k}: selected validation rows lack an arm.")
        result[f"uplift_at_{int(k * 100)}pct_overall"] = float(
            uplift_at_k(y, score, t, strategy="overall", k=k)
        )
    if not all(np.isfinite(v) for v in result.values()):
        raise ContractError(
            "Undefined preliminary ranking metric; inspect outcome counts."
        )
    return result


def build_baseline(
    splits: DatasetSplits,
    name: BaselineName,
    *,
    seed: int,
    contact_cost: float,
    model_params: dict[str, Any],
) -> BaselineResult:
    """Build one policy on train and evaluate on validation; never access test.

    Cost is an explicit scenario assumption. Fixed model parameters come from
    config; this task performs no tuning or selection on validation.
    Blanket's saved score of one is a selection indicator, not predicted uplift.
    """
    if name not in BASELINE_NAMES:
        raise ContractError(f"Unknown baseline: {name!r}.")
    if not np.isfinite(contact_cost) or contact_cost <= 0:
        raise ContractError("contact_cost must be finite and greater than zero.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("seed must be a non-negative integer.")
    train, val = splits.train, splits.validation
    validate_features(train.X)
    validate_features(val.X)

    def frame(score: npt.NDArray[np.float64]) -> pd.DataFrame:
        if score.shape != (val.n_rows,) or not np.isfinite(score).all():
            raise ContractError("Expected one finite score per validation row.")
        return pd.DataFrame({"row_id": val.row_id, "score": score})

    if name == "blanket":
        # sklift 0.5.1 excludes k=1.0, so use the full-population definition.
        ate = float(val.y[val.t == 1].mean() - val.y[val.t == 0].mean())
        total_cost = val.n_rows * contact_cost
        if not np.isfinite(total_cost):
            raise ContractError("total_cost overflowed; check contact_cost.")
        return BaselineResult(
            name,
            frame(np.ones(val.n_rows)),
            {
                "average_effect": ate,
                "uplift_at_100pct_overall": ate,
                "number_contacted": float(val.n_rows),
                "total_cost": total_cost,
            },
        )
    if name == "random":
        score = np.random.default_rng(seed).uniform(size=val.n_rows)
        return BaselineResult(name, frame(score), ranking_metrics(val.y, score, val.t))

    treated = train.t == 1
    if len(np.unique(train.y[treated])) != 2:
        raise ContractError("Response model needs both outcomes in treated train.")
    if len(np.unique(val.y[val.t == 1])) != 2:
        raise ContractError(
            "Response ROC-AUC needs both outcomes in treated validation."
        )
    protected = {
        "random_state",
        "seed",
        "objective",
        "class_weight",
        "is_unbalance",
        "scale_pos_weight",
    }
    if protected.intersection(model_params):
        raise ContractError(
            "Seed/objective and class-imbalance overrides are not allowed."
        )
    model = LGBMClassifier(objective="binary", random_state=seed, **model_params)
    model.fit(train.X.loc[treated], train.y[treated])
    score = np.asarray(model.predict_proba(val.X), dtype=np.float64)[:, 1]
    metrics = ranking_metrics(val.y, score, val.t)
    tv = val.t == 1
    metrics["treated_roc_auc"] = float(roc_auc_score(val.y[tv], score[tv]))
    metrics["treated_log_loss"] = float(log_loss(val.y[tv], score[tv], labels=[0, 1]))
    return BaselineResult(name, frame(score), metrics, model)
