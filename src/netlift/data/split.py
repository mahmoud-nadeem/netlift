"""Joint stratified split on (treatment, outcome) and its assertions."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedKFold, train_test_split

from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)


def joint_key(t: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Cell id 0..3 for each row: 2 * t + y."""
    return 2 * np.asarray(t, dtype=np.int64) + np.asarray(y, dtype=np.int64)


def assign_splits(
    t: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    seed: int,
    n_folds: int,
    shares: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """Return train, validation, test positions and a fold id per row (train only)."""
    if abs(sum(shares) - 1.0) > 1e-9:
        raise ContractError(f"shares must sum to 1, got {shares}.")
    key = joint_key(t, y)
    idx = np.arange(len(key))
    train, rest = train_test_split(idx, train_size=shares[0], stratify=key, random_state=seed)
    val, test = train_test_split(
        rest,
        train_size=shares[1] / (shares[1] + shares[2]),
        stratify=key[rest],
        random_state=seed,
    )
    fold = np.zeros(len(key), dtype=np.int64)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for f, (_, held) in enumerate(skf.split(train, key[train])):
        fold[train[held]] = f
    return np.sort(train), np.sort(val), np.sort(test), fold


def assert_cells_balanced(
    t: npt.ArrayLike, y: npt.ArrayLike, parts: dict[str, npt.NDArray[np.intp]], *, tol: float
) -> dict[str, list[int]]:
    """Raise if any part's (t, y) cell shares differ from the overall shares by more than tol.

    The four cells cover the arm proportions and the outcome rate inside each arm.
    Returns the four cell counts per part so the caller can log them.
    """
    key = joint_key(t, y)
    overall = np.bincount(key, minlength=4) / len(key)
    counts: dict[str, list[int]] = {}
    for name, idx in parts.items():
        c = np.bincount(key[idx], minlength=4)
        counts[name] = c.tolist()
        worst = float(np.abs(c / len(idx) - overall).max())
        logger.info("%s: cells (t0y0,t0y1,t1y0,t1y1)=%s, worst gap %.5f", name, c.tolist(), worst)
        if worst > tol:
            raise ContractError(f"{name}: (t, y) cell shares differ from overall by {worst:.5f} > {tol}.")
    return counts
