"""T43 / K4: permuted-treatment negative control.

Specification: Statistical Significance (protocol item 4) and Behavioural and
Invariance Tests (item 1). The whole pipeline is re-run with the treatment
indicator randomly permuted, within each split, so that ``t`` carries no
information about the potential outcomes. A sound pipeline must then measure
no uplift: the 95% within-arm bootstrap interval of the Qini coefficient has
to contain zero. If it does not, there is a bug or a leak.

Every reported number comes from ``netlift.evaluation.metrics`` (scikit-uplift,
ADR 0001). Only the validation split is read; the locked test split is never
opened here.

Usage:
    python scripts/run_negative_control.py --dataset hillstrom --seed 42
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier

from netlift.data.contracts import (
    DatasetName,
    DatasetSplits,
    FeatureMatrix,
    TargetOutcome,
    validate_features,
)
from netlift.evaluation.contracts import MIN_REPORTED_RESAMPLES, MetricResult
from netlift.evaluation.metrics import qini_coefficient
from netlift.utils.errors import ContractError

DEFAULT_SEED = 42
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


@dataclass(frozen=True)
class NegativeControlResult:
    """The Qini coefficient under permuted treatment, and whether K4 passes."""

    metric: MetricResult
    n_train: int
    n_validation: int
    seed: int
    passed: bool


def permute_treatment(
    t: npt.NDArray[np.int8], rng: np.random.Generator
) -> npt.NDArray[np.int8]:
    """Shuffle ``t`` at random; arm sizes are preserved and ``y`` is untouched."""
    permuted = rng.permutation(t)
    if permuted.shape != t.shape or not (permuted != t).any():
        raise ContractError("The permutation left the treatment indicator unchanged.")
    return np.asarray(permuted, dtype=np.int8)


def fit_t_learner(
    X: pd.DataFrame,
    y: npt.NDArray[np.int8],
    t: npt.NDArray[np.int8],
    *,
    seed: int,
    model_params: dict[str, Any],
) -> tuple[LGBMClassifier, LGBMClassifier]:
    """Fit one LightGBM per arm: a plain T-learner, enough to exercise the pipeline."""
    models: list[LGBMClassifier] = []
    for arm in (0, 1):
        mask = t == arm
        if len(np.unique(y[mask])) != 2:
            raise ContractError(
                f"Arm {arm} has a single outcome value after permutation; "
                "the T-learner cannot be fitted."
            )
        model = LGBMClassifier(objective="binary", random_state=seed, **model_params)
        model.fit(X.loc[mask], y[mask])
        models.append(model)
    return models[0], models[1]


def score_uplift(
    models: tuple[LGBMClassifier, LGBMClassifier], X: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """Uplift score = P(y=1 | treated) - P(y=1 | control)."""
    control, treated = models
    p_treated = np.asarray(treated.predict_proba(X), dtype=np.float64)[:, 1]
    p_control = np.asarray(control.predict_proba(X), dtype=np.float64)[:, 1]
    return p_treated - p_control


def _subsample(
    matrix: FeatureMatrix, fraction: float, rng: np.random.Generator
) -> tuple[
    pd.DataFrame, npt.NDArray[np.int8], npt.NDArray[np.int8], npt.NDArray[np.int64]
]:
    """Return (X, t, y, row_id), optionally subsampled for iteration only."""
    if fraction >= 1.0:
        return matrix.X, matrix.t, matrix.y, matrix.row_id
    take = np.sort(
        rng.choice(matrix.n_rows, int(matrix.n_rows * fraction), replace=False)
    )
    return matrix.X.iloc[take], matrix.t[take], matrix.y[take], matrix.row_id[take]


def run_negative_control(
    splits: DatasetSplits,
    *,
    seed: int = DEFAULT_SEED,
    n_resamples: int = MIN_REPORTED_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    model_params: dict[str, Any] | None = None,
    subsample_fraction: float = 1.0,
) -> NegativeControlResult:
    """Permute ``t`` in train and validation, refit, score validation, bootstrap the Qini.

    ``subsample_fraction`` below 1.0 is for iteration only; a reported K4 uses 1.0.
    The locked test split is never touched.
    """
    if not 0.0 < subsample_fraction <= 1.0:
        raise ContractError("subsample_fraction must be in (0, 1].")
    params = dict(DEFAULT_MODEL_PARAMS if model_params is None else model_params)
    train, val = splits.train, splits.validation
    validate_features(train.X)
    validate_features(val.X)

    train_seed, val_seed, sample_seed = np.random.SeedSequence(seed).spawn(3)
    X_tr, t_tr, y_tr, _ = _subsample(
        train, subsample_fraction, np.random.default_rng(sample_seed)
    )
    X_va, t_va, y_va, ids_va = _subsample(
        val, subsample_fraction, np.random.default_rng(sample_seed)
    )

    t_tr_perm = permute_treatment(t_tr, np.random.default_rng(train_seed))
    t_va_perm = permute_treatment(t_va, np.random.default_rng(val_seed))

    models = fit_t_learner(X_tr, y_tr, t_tr_perm, seed=seed, model_params=params)
    score = score_uplift(models, X_va)

    metric = qini_coefficient(
        y_va,
        score,
        t_va_perm,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        split_name="validation",
        row_ids=ids_va,
        dataset=val.meta.dataset,
    )
    return NegativeControlResult(
        metric=metric,
        n_train=len(t_tr),
        n_validation=len(t_va),
        seed=seed,
        passed=metric.ci_low <= 0.0 <= metric.ci_high,
    )


def format_result(result: NegativeControlResult) -> str:
    """A printable summary naming the library, resamples and seed."""
    m = result.metric
    verdict = (
        "PASS (interval contains zero)"
        if result.passed
        else "FAIL (interval excludes zero)"
    )
    return (
        "K4 negative control: permuted treatment\n"
        f"  metric            : {m.name} ({m.library} {m.library_version}, "
        f"settings={m.settings})\n"
        f"  split             : {m.split_name} ({result.n_validation} rows; "
        f"train {result.n_train} rows)\n"
        f"  point estimate    : {m.value:+.6f}\n"
        f"  {m.confidence_level:.0%} bootstrap CI : [{m.ci_low:+.6f}, {m.ci_high:+.6f}] "
        f"({m.n_resamples} within-arm resamples, seed {m.seed})\n"
        f"  result            : {verdict}"
    )


def main(argv: list[str] | None = None) -> int:
    """Load the splits, run the control, print it, and assert the interval contains zero."""
    from netlift.data import load_splits

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # type: ignore[union-attr]
    parser.add_argument(
        "--dataset", choices=["criteo", "hillstrom"], default="hillstrom"
    )
    parser.add_argument(
        "--target", choices=["visit", "conversion"], default="conversion"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-resamples", type=int, default=MIN_REPORTED_RESAMPLES)
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of each split, for iteration only. Reported runs use 1.0.",
    )
    args = parser.parse_args(argv)
    dataset: DatasetName = args.dataset
    target: TargetOutcome = args.target

    splits = load_splits(dataset, target, args.seed)
    result = run_negative_control(
        splits,
        seed=args.seed,
        n_resamples=args.n_resamples,
        subsample_fraction=args.subsample,
    )
    print(format_result(result))
    assert result.passed, (
        "K4 failed: the Qini interval under permuted treatment excludes zero. "
        "Look for a leak or a bug before trusting any uplift result."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
