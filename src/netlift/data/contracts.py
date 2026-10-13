"""Interface 1: the feature matrix and its folds.

Produced by the data pipeline (T37) and consumed by every model and by the
evaluation harness. Frozen by T36; ``docs/design/interfaces.md`` describes it in
full. After it merges, any change needs a record in ``docs/decisions/``.

The rules this module enforces come from the specification:

* the feature matrix never contains ``exposure``, any outcome column or the
  treatment column (Leakage and Validity Audit);
* treatment and outcome are binary, and every fold keeps enough control-arm
  outcome events to estimate uplift at all (Splitting Strategy, Uncertainty and
  Abstention);
* the test split is locked and opened once, at champion confirmation (Leakage
  and Validity Audit, metric overfitting).

The treatment flag is never a column of ``X``, not even for the S-learner: an
S-learner appends ``t`` itself inside its own fit and predict.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd

from netlift.utils.errors import ContractError

logger = logging.getLogger(__name__)

DatasetName = Literal["criteo", "hillstrom", "synthetic"]
TargetOutcome = Literal["visit", "conversion"]
SplitName = Literal["train", "validation", "test"]

# Compared case-insensitively. Hillstrom's arm column is "segment" in the raw file.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {"exposure", "visit", "conversion", "spend", "treatment", "segment"}
)

ArmValue = str | int


@dataclass(frozen=True)
class TreatmentDefinition:
    """How the binary treatment ``t`` is derived from a dataset's raw arm column."""

    source_column: str
    treatment_values: tuple[ArmValue, ...]
    control_values: tuple[ArmValue, ...]
    excluded_values: tuple[ArmValue, ...] = ()

    def __post_init__(self) -> None:
        """Reject a definition that leaves an arm empty or puts a value in two roles."""
        if not self.source_column:
            raise ContractError("TreatmentDefinition needs a source column.")
        if not self.treatment_values or not self.control_values:
            raise ContractError(
                f"TreatmentDefinition on '{self.source_column}' needs at least one "
                "treatment value and one control value."
            )
        roles = [
            set(self.treatment_values),
            set(self.control_values),
            set(self.excluded_values),
        ]
        shared = (roles[0] & roles[1]) | (roles[0] & roles[2]) | (roles[1] & roles[2])
        if shared:
            raise ContractError(
                f"TreatmentDefinition on '{self.source_column}': {sorted(map(str, shared))} "
                "appear in more than one role (treatment, control, excluded)."
            )


CRITEO_TREATMENT = TreatmentDefinition("treatment", (1,), (0,))

# Specification, Multi-Treatment Handling: the MVP analyses one Hillstrom arm at a
# time against the shared control. The other arm's rows are dropped, never pooled.
HILLSTROM_MENS_VS_CONTROL = TreatmentDefinition(
    "segment", ("Mens E-Mail",), ("No E-Mail",), ("Womens E-Mail",)
)
HILLSTROM_WOMENS_VS_CONTROL = TreatmentDefinition(
    "segment", ("Womens E-Mail",), ("No E-Mail",), ("Mens E-Mail",)
)

SYNTHETIC_TREATMENT = TreatmentDefinition("treatment", (1,), (0,))


@dataclass(frozen=True)
class FeatureMatrixMeta:
    """Where a feature matrix came from and which split it is."""

    dataset: DatasetName
    target_outcome: TargetOutcome
    treatment: TreatmentDefinition
    seed: int
    split_name: SplitName

    def __post_init__(self) -> None:
        """Check the literal fields at runtime, since type hints alone do not."""
        _require_member("dataset", self.dataset, get_args(DatasetName))
        _require_member("target_outcome", self.target_outcome, get_args(TargetOutcome))
        _require_member("split_name", self.split_name, get_args(SplitName))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError(f"seed must be an int, got {self.seed!r}.")


@dataclass(frozen=True, eq=False)
class FeatureMatrix:
    """Features, binary treatment, binary outcome and a fold id for every row.

    ``fold`` holds the cross-validation fold ids inside the training split
    (0 to K-1). The validation and test splits are a single fold, 0.
    """

    X: pd.DataFrame
    t: npt.NDArray[np.int8]
    y: npt.NDArray[np.int8]
    fold: npt.NDArray[np.int64]
    meta: FeatureMatrixMeta

    def __post_init__(self) -> None:
        """Validate every invariant that needs no parameter, and normalise dtypes."""
        validate_features(self.X)
        t = _as_binary(self.t, "t")
        y = _as_binary(self.y, "y")
        fold = _as_fold(self.fold)
        lengths = {"X": len(self.X), "t": len(t), "y": len(y), "fold": len(fold)}
        if len(set(lengths.values())) != 1:
            raise ContractError(f"Length mismatch between the parts: {lengths}.")
        if len(t) == 0:
            raise ContractError("A feature matrix needs at least one row.")
        if not (np.any(t == 0) and np.any(t == 1)):
            raise ContractError(
                f"{self.meta.split_name}: t must contain both arms; "
                f"found only {sorted(set(t.tolist()))}."
            )
        if self.meta.split_name != "train" and np.any(fold != 0):
            raise ContractError(
                f"The {self.meta.split_name} split is a single fold, so every fold id "
                "must be 0."
            )
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "fold", fold)

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return len(self.t)

    @property
    def fold_ids(self) -> tuple[int, ...]:
        """The distinct fold ids, ascending."""
        return tuple(int(f) for f in np.unique(self.fold))

    def cell_counts(self) -> pd.DataFrame:
        """Rows per fold in each (t, y) cell: the four counts the split must log."""
        frame = pd.DataFrame({"fold": self.fold, "t": self.t, "y": self.y})
        counts = frame.groupby(["fold", "t", "y"]).size().unstack(["t", "y"])
        full = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=["t", "y"])
        return counts.reindex(columns=full, fill_value=0).fillna(0).astype("int64")


def validate_features(X: pd.DataFrame) -> None:
    """Raise if a forbidden column (outcome, treatment or exposure) is in ``X``."""
    if not isinstance(X, pd.DataFrame):
        raise ContractError(f"X must be a pandas DataFrame, got {type(X).__name__}.")
    found = sorted(
        str(column) for column in X.columns if _normalise(column) in FORBIDDEN_COLUMNS
    )
    if found:
        raise ContractError(
            f"Forbidden column(s) in the feature matrix: {found}. Outcomes, the "
            "treatment column and Criteo's exposure are post-treatment or the "
            "treatment itself, and never enter X (Leakage and Validity Audit)."
        )


def check_control_events(
    matrix: FeatureMatrix, *, min_control_events: int
) -> dict[int, int]:
    """Raise if any fold has fewer control-arm outcome events than the minimum.

    Returns the count per fold so the caller can log it. The minimum is a
    configuration value; the specification does not fix one.
    """
    if isinstance(min_control_events, bool) or not isinstance(min_control_events, int):
        raise ContractError(
            f"min_control_events must be an int, got {min_control_events!r}."
        )
    if min_control_events < 1:
        raise ContractError(
            f"min_control_events must be at least 1, got {min_control_events}."
        )
    events = (matrix.t == 0) & (matrix.y == 1)
    counts = {
        fold_id: int(events[matrix.fold == fold_id].sum())
        for fold_id in matrix.fold_ids
    }
    short = {fold_id: n for fold_id, n in counts.items() if n < min_control_events}
    if short:
        raise ContractError(
            f"{matrix.meta.split_name}: fold(s) {short} have fewer than "
            f"{min_control_events} control-arm '{matrix.meta.target_outcome}' events. "
            "Uplift cannot be estimated reliably there."
        )
    return counts


class LockedTestSplit:
    """The test split, held closed until champion confirmation.

    Metadata can be read at any time. The rows come out only through
    :meth:`open`, which needs a stated reason, logs it, and works once per
    object. The written opening procedure (T70) covers the rest.
    """

    def __init__(self, matrix: FeatureMatrix) -> None:
        """Wrap the test-split feature matrix."""
        if matrix.meta.split_name != "test":
            raise ContractError(
                f"Only the test split is locked; got '{matrix.meta.split_name}'."
            )
        self._matrix = matrix
        self._opened_for: str | None = None

    @property
    def meta(self) -> FeatureMatrixMeta:
        """Metadata of the locked split, readable without opening it."""
        return self._matrix.meta

    @property
    def columns(self) -> tuple[str, ...]:
        """Feature column names, readable without opening the split."""
        return tuple(str(column) for column in self._matrix.X.columns)

    @property
    def is_open(self) -> bool:
        """Whether :meth:`open` has been called."""
        return self._opened_for is not None

    def open(self, reason: str) -> FeatureMatrix:
        """Release the test rows once, recording why."""
        if not reason or not reason.strip():
            raise ContractError("Opening the locked test split needs a stated reason.")
        if self._opened_for is not None:
            raise ContractError(
                "The locked test split has already been opened, for: "
                f"'{self._opened_for}'. It is opened once."
            )
        logger.warning("Locked test split opened. Reason: %s", reason)
        self._opened_for = reason
        return self._matrix

    def __repr__(self) -> str:
        """Describe the split without exposing its rows."""
        state = "open" if self.is_open else "locked"
        return (
            f"LockedTestSplit({self.meta.dataset}, {self.meta.target_outcome}, {state})"
        )


@dataclass(frozen=True, eq=False)
class DatasetSplits:
    """The three splits the data pipeline hands to everyone else."""

    train: FeatureMatrix
    validation: FeatureMatrix
    test: LockedTestSplit

    def __post_init__(self) -> None:
        """Check the splits are named correctly and describe the same data."""
        if self.train.meta.split_name != "train":
            raise ContractError("DatasetSplits.train must be the train split.")
        if self.validation.meta.split_name != "validation":
            raise ContractError(
                "DatasetSplits.validation must be the validation split."
            )
        for name, meta in (
            ("validation", self.validation.meta),
            ("test", self.test.meta),
        ):
            for attribute in ("dataset", "target_outcome", "treatment", "seed"):
                if getattr(meta, attribute) != getattr(self.train.meta, attribute):
                    raise ContractError(
                        f"The {name} split has a different {attribute} from train."
                    )
        train_columns = tuple(str(column) for column in self.train.X.columns)
        validation_columns = tuple(str(c) for c in self.validation.X.columns)
        if validation_columns != train_columns or self.test.columns != train_columns:
            raise ContractError("All three splits must have the same feature columns.")


def _normalise(column: Hashable) -> str:
    """Lower-case and strip a column name for the forbidden-column check."""
    return str(column).strip().lower()


def _require_member(
    field_name: str, value: object, allowed: tuple[object, ...]
) -> None:
    """Raise if ``value`` is not one of the allowed literal values."""
    if value not in allowed:
        raise ContractError(
            f"{field_name} must be one of {list(allowed)}, got {value!r}."
        )


def _as_binary(values: npt.ArrayLike, name: str) -> npt.NDArray[np.int8]:
    """Return ``values`` as a 1-D int8 array of 0/1, or raise."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ContractError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if pd.isna(array).any():
        raise ContractError(f"{name} has missing values.")
    if array.dtype.kind not in "biuf":
        raise ContractError(f"{name} must be numeric 0/1, got dtype {array.dtype}.")
    unexpected = set(np.unique(array).tolist()) - {0, 1}
    if unexpected:
        raise ContractError(
            f"{name} must be binary 0/1; found {sorted(unexpected)[:5]}."
        )
    return array.astype(np.int8)


def _as_fold(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return fold ids as a 1-D int64 array of non-negative integers, or raise."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ContractError(f"fold must be one-dimensional, got shape {array.shape}.")
    if array.dtype.kind not in "iu":
        raise ContractError(f"fold ids must be integers, got dtype {array.dtype}.")
    if array.size and array.min() < 0:
        raise ContractError("fold ids must be non-negative.")
    return array.astype(np.int64)
