"""T37 entry point: load, check, split jointly on (treatment, conversion), return DatasetSplits."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from netlift.data.contracts import (
    CRITEO_TREATMENT,
    HILLSTROM_MENS_VS_CONTROL,
    DatasetName,
    DatasetSplits,
    FeatureMatrix,
    FeatureMatrixMeta,
    LockedTestSplit,
    TargetOutcome,
    TreatmentDefinition,
    check_control_events,
)
from netlift.data.loaders import BASE_DIR, Prepared, load_criteo, load_hillstrom
from netlift.data.split import assert_cells_balanced, assign_splits
from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)
CONFIG_PATH = BASE_DIR / "config" / "data.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read config/data.yaml."""
    with path.open() as f:
        return yaml.safe_load(f)


def build_splits(
    p: Prepared,
    *,
    dataset: DatasetName,
    target: TargetOutcome,
    treatment: TreatmentDefinition,
    cfg: dict[str, Any],
) -> DatasetSplits:
    """Split a prepared dataset. The split depends on (t, conversion) only, never on the target."""
    seed, n_folds = int(cfg["seed"]), int(cfg["n_folds"])
    train, val, test, fold = assign_splits(
        p.t, p.y_conversion, seed=seed, n_folds=n_folds, shares=tuple(cfg["shares"])
    )
    parts = {"train": train, "validation": val, "test": test}
    parts.update({f"train_fold{f}": train[fold[train] == f] for f in range(n_folds)})
    assert_cells_balanced(p.t, p.y_conversion, parts, tol=float(cfg["tolerance"]))

    y = p.y_conversion if target == "conversion" else p.y_visit

    def matrix(idx: np.ndarray, name: str) -> FeatureMatrix:
        return FeatureMatrix(
            X=p.X.iloc[idx].reset_index(drop=True),
            t=p.t[idx],
            y=y[idx],
            fold=fold[idx] if name == "train" else np.zeros(len(idx), dtype=np.int64),
            row_id=p.row_id[idx],
            meta=FeatureMatrixMeta(dataset, target, treatment, seed, name),  # type: ignore[arg-type]
        )

    tr, va, te = matrix(train, "train"), matrix(val, "validation"), matrix(test, "test")
    for m in (tr, va):
        counts = check_control_events(
            m, min_control_events=int(cfg["min_control_events"])
        )
        logger.info("%s control events per fold: %s", m.meta.split_name, counts)
        logger.info("%s cell counts:\n%s", m.meta.split_name, m.cell_counts())
    return DatasetSplits(train=tr, validation=va, test=LockedTestSplit(te))


def load_splits(
    dataset: DatasetName,
    target: TargetOutcome = "conversion",
    *,
    treatment: TreatmentDefinition | None = None,
    subsample_frac: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> DatasetSplits:
    """Load a real dataset and return its splits."""
    cfg = cfg or load_config()
    if dataset == "hillstrom":
        treatment = treatment or HILLSTROM_MENS_VS_CONTROL
        p = load_hillstrom(treatment)
    elif dataset == "criteo":
        treatment = CRITEO_TREATMENT
        p = load_criteo(subsample_frac=subsample_frac, seed=int(cfg["seed"]))
    else:
        raise ContractError(
            f"load_splits supports 'hillstrom' and 'criteo', got {dataset!r}."
        )
    return build_splits(p, dataset=dataset, target=target, treatment=treatment, cfg=cfg)
