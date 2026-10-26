"""Joint stratified split on (arm, outcome) and its assertions."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedKFold, train_test_split

from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)


def joint_key(t: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Cell id for each row: 2 * arm + y (0..3 for two arms, 0..5 for three)."""
    return 2 * np.asarray(t, dtype=np.int64) + np.asarray(y, dtype=np.int64)


def assign_splits(
    t: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    seed: int,
    n_folds: int,
    shares: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.int64],
]:
    """Return train, validation, test positions and a fold id per row (train only)."""
    if abs(sum(shares) - 1.0) > 1e-9:
        raise ContractError(f"shares must sum to 1, got {shares}.")
    key = joint_key(t, y)
    idx = np.arange(len(key))
    train, rest = train_test_split(
        idx, train_size=shares[0], stratify=key, random_state=seed
    )
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


def subsample_part(
    idx: npt.NDArray[np.intp],
    key: npt.NDArray[np.int64],
    frac: float,
    seed: int,
) -> npt.NDArray[np.intp]:
    """Stratified subsample of one split's positions. The result is a subset of idx."""
    pos, _ = train_test_split(
        np.arange(len(idx)), train_size=frac, stratify=key, random_state=seed
    )
    return np.sort(idx[pos])


def assert_cells_balanced(
    t: npt.ArrayLike,
    y: npt.ArrayLike,
    parts: dict[str, npt.NDArray[np.intp]],
    *,
    tol: float,
    max_gap_rows: float = 3.0,
) -> dict[str, list[int]]:
    """Raise if any part's (arm, y) cell shares differ from the overall shares by more than tol.

    The cells cover the arm proportions and the outcome rate inside each arm
    (4 cells for two arms, 6 for Hillstrom's three arms).
    A share gap cannot see a rare cell (control-arm conversions are about 0.3% of
    rows), so each cell count must also be within max_gap_rows of its expected count.
    Returns the cell counts per part so the caller can log them.
    """
    key = joint_key(t, y)
    n_cells = int(key.max()) + 1
    overall = np.bincount(key, minlength=n_cells) / len(key)
    counts: dict[str, list[int]] = {}
    for name, idx in parts.items():
        c = np.bincount(key[idx], minlength=n_cells)
        counts[name] = c.tolist()
        worst = float(np.abs(c / len(idx) - overall).max())
        logger.info(
            "%s: cell counts (arm0y0, arm0y1, arm1y0, ...)=%s, worst gap %.5f",
            name,
            c.tolist(),
            worst,
        )
        if worst > tol:
            raise ContractError(
                f"{name}: (arm, y) cell shares differ from overall by {worst:.5f} > {tol}."
            )
        gap_rows = float(np.abs(c - overall * len(idx)).max())
        if gap_rows > max_gap_rows:
            raise ContractError(
                f"{name}: a (arm, y) cell count is {gap_rows:.1f} rows from its "
                f"expected count (limit {max_gap_rows}); cell counts {c.tolist()}."
            )
    return counts
