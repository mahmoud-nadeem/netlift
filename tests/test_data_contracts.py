"""Tests for interface 1: the feature matrix, its folds and the locked test split."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from netlift.data.contracts import (
    CRITEO_TREATMENT,
    HILLSTROM_MENS_VS_CONTROL,
    DatasetSplits,
    FeatureMatrix,
    FeatureMatrixMeta,
    LockedTestSplit,
    SplitName,
    TreatmentDefinition,
    check_control_events,
    validate_features,
)
from netlift.data.synthetic import make_synthetic_frame, make_synthetic_splits
from netlift.utils.errors import ContractError


def _meta(split_name: SplitName = "train") -> FeatureMatrixMeta:
    return FeatureMatrixMeta(
        dataset="criteo",
        target_outcome="visit",
        treatment=CRITEO_TREATMENT,
        seed=42,
        split_name=split_name,
    )


def _matrix(**overrides: object) -> FeatureMatrix:
    parts: dict[str, object] = {
        "X": pd.DataFrame({"f0": [0.1, 0.2, 0.3, 0.4], "f1": [1.0, 2.0, 3.0, 4.0]}),
        "t": np.array([0, 1, 0, 1]),
        "y": np.array([1, 0, 0, 1]),
        "fold": np.array([0, 0, 1, 1]),
        "row_id": np.array([10, 11, 12, 13]),
        "meta": _meta(),
    }
    parts.update(overrides)
    return FeatureMatrix(**parts)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# valid input
# --------------------------------------------------------------------------- #


def test_valid_matrix_is_accepted_and_normalised() -> None:
    matrix = _matrix(t=[0, 1, 0, 1], y=[True, False, False, True])
    assert matrix.t.dtype == np.int8
    assert matrix.y.tolist() == [1, 0, 0, 1]
    assert matrix.fold_ids == (0, 1)
    assert matrix.n_rows == 4


def test_cell_counts_has_all_four_cells_per_fold() -> None:
    counts = _matrix().cell_counts()
    assert list(counts.columns) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert counts.loc[0].tolist() == [0, 1, 1, 0]
    assert int(counts.to_numpy().sum()) == 4


def test_synthetic_splits_conform_and_are_marked_synthetic(
    synthetic_splits: DatasetSplits,
) -> None:
    assert synthetic_splits.train.meta.dataset == "synthetic"
    assert synthetic_splits.train.fold_ids == (0, 1, 2)
    assert synthetic_splits.validation.fold_ids == (0,)
    sizes = [synthetic_splits.train.n_rows, synthetic_splits.validation.n_rows]
    assert sum(sizes) < 600
    counts = check_control_events(synthetic_splits.train, min_control_events=5)
    assert set(counts) == {0, 1, 2}


def test_synthetic_frame_has_both_arms_and_both_outcomes() -> None:
    frame = make_synthetic_frame(400, seed=3)
    assert set(frame["treatment"]) == {0, 1}
    assert frame["visit"].sum() > 0
    assert frame["conversion"].sum() > 0
    assert (frame["conversion"] <= frame["visit"]).all()


def test_synthetic_splits_are_reproducible() -> None:
    first = make_synthetic_splits("conversion", seed=7)
    second = make_synthetic_splits("conversion", seed=7)
    pd.testing.assert_frame_equal(first.train.X, second.train.X)
    assert np.array_equal(first.train.fold, second.train.fold)


# --------------------------------------------------------------------------- #
# forbidden columns
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "column",
    ["exposure", "visit", "conversion", "spend", "treatment", "segment", " Visit "],
)
def test_forbidden_column_is_rejected(column: str) -> None:
    frame = pd.DataFrame({"f0": [0.1, 0.2], column: [0, 1]})
    with pytest.raises(ContractError, match="Forbidden column"):
        validate_features(frame)


def test_forbidden_column_is_rejected_at_construction() -> None:
    X = pd.DataFrame({"f0": [0.1, 0.2, 0.3, 0.4], "exposure": [0, 1, 0, 0]})
    with pytest.raises(ContractError, match="exposure"):
        _matrix(X=X)


def test_x_must_be_a_dataframe() -> None:
    with pytest.raises(ContractError, match="DataFrame"):
        validate_features(np.zeros((2, 2)))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# shapes and values
# --------------------------------------------------------------------------- #


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ContractError, match="Length mismatch"):
        _matrix(y=np.array([1, 0, 0]))


@pytest.mark.parametrize("name", ["t", "y"])
def test_non_binary_values_are_rejected(name: str) -> None:
    with pytest.raises(ContractError, match="binary"):
        _matrix(**{name: np.array([0, 1, 2, 1])})


@pytest.mark.parametrize("name", ["t", "y"])
def test_missing_values_are_rejected(name: str) -> None:
    with pytest.raises(ContractError, match="missing"):
        _matrix(**{name: np.array([0.0, 1.0, np.nan, 1.0])})


def test_both_arms_are_required() -> None:
    with pytest.raises(ContractError, match="both arms"):
        _matrix(t=np.array([1, 1, 1, 1]))


def test_fold_ids_must_be_integers() -> None:
    with pytest.raises(ContractError, match="integers"):
        _matrix(fold=np.array([0.0, 0.0, 1.0, 1.0]))


def test_validation_split_is_a_single_fold() -> None:
    with pytest.raises(ContractError, match="single fold"):
        _matrix(meta=_meta("validation"))


def test_unknown_literal_value_is_rejected() -> None:
    with pytest.raises(ContractError, match="target_outcome"):
        FeatureMatrixMeta(
            dataset="criteo",
            target_outcome="spend",  # type: ignore[arg-type]
            treatment=CRITEO_TREATMENT,
            seed=1,
            split_name="train",
        )


def test_treatment_definition_rejects_a_value_in_two_roles() -> None:
    with pytest.raises(ContractError, match="more than one role"):
        TreatmentDefinition("segment", ("No E-Mail",), ("No E-Mail",))


def test_hillstrom_drops_the_other_arm_rather_than_pooling_it() -> None:
    assert HILLSTROM_MENS_VS_CONTROL.excluded_values == ("Womens E-Mail",)


# --------------------------------------------------------------------------- #
# control-arm outcome events per fold
# --------------------------------------------------------------------------- #


def test_fold_with_too_few_control_events_is_rejected() -> None:
    matrix = _matrix(y=np.array([1, 0, 0, 1]))  # fold 1 has no control event
    with pytest.raises(ContractError, match=r"fold\(s\) \{1: 0\}"):
        check_control_events(matrix, min_control_events=1)


def test_control_event_counts_are_returned() -> None:
    matrix = _matrix(y=np.array([1, 0, 1, 1]))
    assert check_control_events(matrix, min_control_events=1) == {0: 1, 1: 1}


@pytest.mark.parametrize("minimum", [0, -3, True, 1.5])
def test_minimum_must_be_a_positive_int(minimum: object) -> None:
    with pytest.raises(ContractError, match="min_control_events"):
        check_control_events(_matrix(), min_control_events=minimum)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the locked test split
# --------------------------------------------------------------------------- #


def test_locked_test_split_opens_once(
    synthetic_splits: DatasetSplits, caplog: pytest.LogCaptureFixture
) -> None:
    locked = synthetic_splits.test
    assert not locked.is_open
    assert "locked" in repr(locked)
    with caplog.at_level(logging.WARNING):
        matrix = locked.open("champion confirmation, procedure in T70")
    assert matrix.meta.split_name == "test"
    assert "champion confirmation" in caplog.text
    with pytest.raises(ContractError, match="already been opened"):
        locked.open("second look")


def test_opening_needs_a_reason(synthetic_splits: DatasetSplits) -> None:
    with pytest.raises(ContractError, match="reason"):
        synthetic_splits.test.open("  ")


def test_only_the_test_split_can_be_locked() -> None:
    with pytest.raises(ContractError, match="Only the test split"):
        LockedTestSplit(_matrix())


def test_splits_must_share_columns(synthetic_splits: DatasetSplits) -> None:
    renamed = synthetic_splits.validation.X.rename(columns={"f0": "g0"})
    validation = FeatureMatrix(
        X=renamed,
        t=synthetic_splits.validation.t,
        y=synthetic_splits.validation.y,
        fold=synthetic_splits.validation.fold,
        row_id=synthetic_splits.validation.row_id,
        meta=synthetic_splits.validation.meta,
    )
    with pytest.raises(ContractError, match="same feature columns"):
        DatasetSplits(
            train=synthetic_splits.train,
            validation=validation,
            test=synthetic_splits.test,
        )


# --------------------------------------------------------------------------- #
# row ids
# --------------------------------------------------------------------------- #


def test_duplicate_row_id_is_rejected() -> None:
    with pytest.raises(
        ContractError, match=r"unique within a matrix; repeated: \[11\]"
    ):
        _matrix(row_id=np.array([10, 11, 11, 13]))


def test_negative_row_id_is_rejected() -> None:
    with pytest.raises(ContractError, match="non-negative"):
        _matrix(row_id=np.array([-1, 11, 12, 13]))


def test_row_id_shared_by_two_splits_is_rejected(
    synthetic_splits: DatasetSplits,
) -> None:
    leaked = synthetic_splits.validation.row_id.copy()
    leaked[0] = synthetic_splits.train.row_id[0]
    validation = FeatureMatrix(
        X=synthetic_splits.validation.X,
        t=synthetic_splits.validation.t,
        y=synthetic_splits.validation.y,
        fold=synthetic_splits.validation.fold,
        row_id=leaked,
        meta=synthetic_splits.validation.meta,
    )
    with pytest.raises(ContractError, match="both train and validation"):
        DatasetSplits(
            train=synthetic_splits.train,
            validation=validation,
            test=synthetic_splits.test,
        )


def test_synthetic_row_ids_cover_the_raw_frame_once(
    synthetic_splits: DatasetSplits,
) -> None:
    every = np.concatenate(
        [
            synthetic_splits.train.row_id,
            synthetic_splits.validation.row_id,
            synthetic_splits.test.row_ids,
        ]
    )
    assert sorted(every.tolist()) == list(range(600))


def test_locked_split_row_ids_are_a_read_only_copy(
    synthetic_splits: DatasetSplits,
) -> None:
    ids = synthetic_splits.test.row_ids
    assert not synthetic_splits.test.is_open
    with pytest.raises(ValueError, match="read-only"):
        ids[0] = 0


def test_split_does_not_depend_on_the_target() -> None:
    """Ali's blocker: a visit-train row must never be a conversion-test row."""
    visit = make_synthetic_splits("visit", seed=4)
    conversion = make_synthetic_splits("conversion", seed=4)
    assert np.array_equal(visit.train.row_id, conversion.train.row_id)
    assert np.array_equal(visit.validation.row_id, conversion.validation.row_id)
    assert np.array_equal(visit.test.row_ids, conversion.test.row_ids)
    assert not np.array_equal(visit.train.y, conversion.train.y)


# --------------------------------------------------------------------------- #
# numeric columns only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "column",
    [
        ["Urban", "Rural"],
        pd.Categorical(["Web", "Phone"]),
        pd.to_datetime(["2026-10-01", "2026-10-02"]),
    ],
)
def test_non_numeric_column_is_rejected(column: object) -> None:
    frame = pd.DataFrame({"f0": [0.1, 0.2], "zip_code": column})
    with pytest.raises(ContractError, match="Non-numeric"):
        validate_features(frame)


def test_bool_int_unsigned_and_float_columns_are_accepted() -> None:
    frame = pd.DataFrame(
        {
            "b": np.array([True, False]),
            "i": np.array([1, 2], dtype=np.int32),
            "u": np.array([1, 2], dtype=np.uint8),
            "f": np.array([0.1, 0.2]),
        }
    )
    validate_features(frame)


def test_nan_in_x_is_rejected() -> None:
    frame = pd.DataFrame({"f0": [0.1, np.nan], "f1": [1.0, 2.0]})
    with pytest.raises(ContractError, match=r"Missing value.*\['f0'\]"):
        validate_features(frame)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_value_in_x_is_rejected(value: float) -> None:
    frame = pd.DataFrame({"f0": [0.1, value], "f1": [1.0, 2.0]})
    with pytest.raises(ContractError, match=r"Infinite value.*\['f0'\]"):
        validate_features(frame)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_pandas_na_in_a_nullable_column_is_rejected(dtype: str) -> None:
    values = [True, None] if dtype == "boolean" else [1, None]
    frame = pd.DataFrame({"f0": pd.array(values, dtype=dtype)})
    with pytest.raises(ContractError, match="Missing value"):
        validate_features(frame)


def test_nullable_columns_without_missing_values_are_accepted() -> None:
    frame = pd.DataFrame(
        {
            "i": pd.array([1, 2], dtype="Int64"),
            "f": pd.array([0.1, 0.2], dtype="Float64"),
            "b": pd.array([True, False], dtype="boolean"),
        }
    )
    validate_features(frame)


def test_infinity_in_a_nullable_float_column_is_rejected() -> None:
    frame = pd.DataFrame({"f0": pd.array([0.1, np.inf], dtype="Float64")})
    with pytest.raises(ContractError, match="Infinite value"):
        validate_features(frame)
