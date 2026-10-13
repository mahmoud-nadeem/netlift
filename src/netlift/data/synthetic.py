"""SYNTHETIC data that conforms to interface 1, for building code before T37 lands.

Never use anything from this module for a reported result. Every matrix it
returns carries ``meta.dataset == "synthetic"`` so that the origin stays visible
wherever the matrix goes. Its effect sizes are invented and mean nothing.

The split here is a minimal joint stratification on (treatment, outcome), good
enough for a fixture. The real split is T37's job.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from netlift.data.contracts import (
    SYNTHETIC_TREATMENT,
    DatasetSplits,
    FeatureMatrix,
    FeatureMatrixMeta,
    LockedTestSplit,
    SplitName,
    TargetOutcome,
)
from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)

FEATURES = ("f0", "f1", "f2", "f3")
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")
SPLIT_SHARES = (0.6, 0.2, 0.2)  # Specification, Splitting Strategy: 60 / 20 / 20.


def make_synthetic_frame(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """A raw-like frame with four features, both arms and both binary outcomes."""
    if n_rows < 20:
        raise ContractError(f"n_rows must be at least 20, got {n_rows}.")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_rows, len(FEATURES)))
    treatment = np.zeros(n_rows, dtype=np.int8)
    treatment[rng.permutation(n_rows)[: n_rows // 2]] = 1

    base = 1.0 / (1.0 + np.exp(-(-0.8 + 0.5 * features[:, 0])))
    effect = np.where(features[:, 1] > 0, 0.10, 0.0)
    visit_probability = np.clip(base + treatment * effect, 0.0, 1.0)
    visit = (rng.random(n_rows) < visit_probability).astype(np.int8)
    conversion = (visit & (rng.random(n_rows) < 0.3)).astype(np.int8)

    frame = pd.DataFrame(features, columns=list(FEATURES))
    frame["treatment"] = treatment
    frame["visit"] = visit
    frame["conversion"] = conversion
    return frame


def make_synthetic_splits(
    target: TargetOutcome = "visit",
    *,
    n_rows: int = 600,
    seed: int = 0,
    n_folds: int = 3,
) -> DatasetSplits:
    """Train, validation and locked test matrices built from the synthetic frame."""
    if n_folds < 2:
        raise ContractError(f"n_folds must be at least 2, got {n_folds}.")
    frame = make_synthetic_frame(n_rows, seed)
    rng = np.random.default_rng(seed + 1)
    t = frame["treatment"].to_numpy()
    y = frame[target].to_numpy()
    split = np.empty(n_rows, dtype=object)
    fold = np.zeros(n_rows, dtype=np.int64)

    for cell in _cells(t, y):
        order = rng.permutation(cell)
        n_train = round(len(order) * SPLIT_SHARES[0])
        n_validation = round(len(order) * SPLIT_SHARES[1])
        split[order[:n_train]] = "train"
        split[order[n_train : n_train + n_validation]] = "validation"
        split[order[n_train + n_validation :]] = "test"
        fold[order[:n_train]] = np.arange(n_train) % n_folds

    logger.info(
        "Built SYNTHETIC splits (n=%d, seed=%d, target=%s)", n_rows, seed, target
    )
    matrices = {
        name: _matrix(frame, split == name, fold, target, seed, name)
        for name in SPLIT_NAMES
    }
    return DatasetSplits(
        train=matrices["train"],
        validation=matrices["validation"],
        test=LockedTestSplit(matrices["test"]),
    )


def _cells(
    t: npt.NDArray[np.int8], y: npt.NDArray[np.int8]
) -> list[npt.NDArray[np.intp]]:
    """Row indices of each (t, y) cell."""
    return [
        np.flatnonzero((t == arm) & (y == outcome))
        for arm in (0, 1)
        for outcome in (0, 1)
    ]


def _matrix(
    frame: pd.DataFrame,
    mask: npt.NDArray[np.bool_],
    fold: npt.NDArray[np.int64],
    target: TargetOutcome,
    seed: int,
    split_name: SplitName,
) -> FeatureMatrix:
    """One split of the synthetic frame as an interface-1 feature matrix."""
    rows = frame.loc[mask].reset_index(drop=True)
    return FeatureMatrix(
        X=rows[list(FEATURES)],
        t=rows["treatment"].to_numpy(),
        y=rows[target].to_numpy(),
        fold=fold[mask]
        if split_name == "train"
        else np.zeros(len(rows), dtype=np.int64),
        row_id=np.flatnonzero(mask).astype(np.int64),
        meta=FeatureMatrixMeta(
            dataset="synthetic",
            target_outcome=target,
            treatment=SYNTHETIC_TREATMENT,
            seed=seed,
            split_name=split_name,
        ),
    )
