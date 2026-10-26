"""T37 entry point: load, check, split jointly on (arm, conversion), return DatasetSplits."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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
from netlift.data.split import (
    assert_cells_balanced,
    assign_splits,
    joint_key,
    subsample_part,
)
from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)
CONFIG_PATH = BASE_DIR / "config" / "data.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read config/data.yaml."""
    with path.open() as f:
        return yaml.safe_load(f)


def plan_split(
    p: Prepared, cfg: dict[str, Any], *, subsample_frac: float | None = None
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.int64],
]:
    """Return train, validation, test positions and fold ids.

    Order: (1) split the FULL file on (arm, conversion); (2) draw the optional
    subsample inside each split; (3) keep only the rows of the chosen comparison.
    Steps 2 and 3 only remove rows, so every result is a subset of the full-data split.
    """
    seed, n_folds = int(cfg["seed"]), int(cfg["n_folds"])
    tol = float(cfg["tolerance"])
    max_gap = float(cfg.get("max_cell_gap_rows", 3))

    train, val, test, fold = assign_splits(
        p.arm, p.y_conversion, seed=seed, n_folds=n_folds, shares=tuple(cfg["shares"])
    )

    def check(tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> None:
        parts = {"train": tr, "validation": va, "test": te}
        parts.update({f"train_fold{f}": tr[fold[tr] == f] for f in range(n_folds)})
        assert_cells_balanced(
            p.arm, p.y_conversion, parts, tol=tol, max_gap_rows=max_gap
        )

    check(train, val, test)

    if subsample_frac is not None:
        key = joint_key(p.arm, p.y_conversion)
        # train is stratified on (cell, fold) so the folds stay balanced
        train = subsample_part(
            train, key[train] * n_folds + fold[train], subsample_frac, seed
        )
        val = subsample_part(val, key[val], subsample_frac, seed)
        test = subsample_part(test, key[test], subsample_frac, seed)
        check(train, val, test)
        logger.info(
            "subsample %.0f%% inside each split: train %d, validation %d, test %d",
            subsample_frac * 100,
            len(train),
            len(val),
            len(test),
        )

    train, val, test = (i[p.in_scope[i]] for i in (train, val, test))
    return train, val, test, fold


def build_splits(
    p: Prepared,
    *,
    dataset: DatasetName,
    target: TargetOutcome,
    treatment: TreatmentDefinition,
    cfg: dict[str, Any],
    subsample_frac: float | None = None,
) -> DatasetSplits:
    """Split a prepared dataset. The split depends on (arm, conversion) only, never on the target."""
    seed = int(cfg["seed"])
    train, val, test, fold = plan_split(p, cfg, subsample_frac=subsample_frac)

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
            m, min_control_events=int(cfg["min_control_events_per_fold"])
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
        p = load_criteo()
    else:
        raise ContractError(
            f"load_splits supports 'hillstrom' and 'criteo', got {dataset!r}."
        )
    return build_splits(
        p,
        dataset=dataset,
        target=target,
        treatment=treatment,
        cfg=cfg,
        subsample_frac=subsample_frac,
    )
