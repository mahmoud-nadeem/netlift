"""Tests for the T37 data pipeline: schema checks, joint stratified split, assertions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netlift.data.contracts import (
    HILLSTROM_MENS_VS_CONTROL,
    HILLSTROM_WOMENS_VS_CONTROL,
    TreatmentDefinition,
)
from netlift.data.loaders import (
    Prepared,
    load_criteo,
    prepare_criteo,
    prepare_hillstrom,
)
from netlift.data.pipeline import build_splits, plan_split
from netlift.data.schema import (
    CRITEO_FEATURES,
    HILLSTROM_CHANNEL,
    HILLSTROM_HISTORY_SEGMENTS,
    HILLSTROM_SEGMENTS,
    HILLSTROM_ZIP,
    check_criteo,
    check_hillstrom,
)
from netlift.data.split import assert_cells_balanced, assign_splits
from netlift.utils.errors import ContractError

CFG = {
    "seed": 42,
    "n_folds": 5,
    "shares": [0.6, 0.2, 0.2],
    "tolerance": 0.02,
    "min_control_events_per_fold": 5,
}


def fake_hillstrom(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """A small fake Hillstrom file with the real columns and three arms."""
    r = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "recency": r.integers(1, 13, n),
            "history_segment": r.choice(HILLSTROM_HISTORY_SEGMENTS, n),
            "history": r.random(n) * 500,
            "mens": r.integers(0, 2, n),
            "womens": r.integers(0, 2, n),
            "zip_code": r.choice(HILLSTROM_ZIP, n),
            "newbie": r.integers(0, 2, n),
            "channel": r.choice(HILLSTROM_CHANNEL, n),
            "segment": r.choice(sorted(HILLSTROM_SEGMENTS), n),
            "visit": r.integers(0, 2, n),
            "conversion": (r.random(n) < 0.2).astype(int),
            "spend": r.random(n) * 10,
        }
    )


def fake_criteo(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A small fake Criteo file with the real columns."""
    r = np.random.default_rng(seed)
    t = (r.random(n) < 0.85).astype(int)
    d = {f"f{i}": r.normal(size=n).astype("float32") for i in range(12)}
    d.update(
        treatment=t,
        conversion=(r.random(n) < 0.1).astype(int),
        visit=r.integers(0, 2, n),
    )
    d["exposure"] = t * (r.random(n) < 0.04).astype(int)
    return pd.DataFrame(d)


def fake_prepared(n: int = 20_000) -> Prepared:
    """A two-arm dataset with a rare outcome, like Criteo."""
    r = np.random.default_rng(0)
    t = (r.random(n) < 0.5).astype(np.int8)
    y = (r.random(n) < 0.02).astype(np.int8)
    return Prepared(
        X=pd.DataFrame({"f0": r.random(n)}),
        arm=t,
        t=t,
        in_scope=np.ones(n, dtype=bool),
        y_visit=y,
        y_conversion=y,
        row_id=np.arange(n, dtype=np.int64),
    )


def control_split_of(raw: pd.DataFrame, tdef: TreatmentDefinition) -> dict[int, str]:
    """Map each control row_id to its split name under one comparison."""
    p = prepare_hillstrom(raw, tdef)
    train, val, test, _ = plan_split(p, CFG)
    out: dict[int, str] = {}
    for name, idx in (("train", train), ("validation", val), ("test", test)):
        for r in p.row_id[idx][p.t[idx] == 0]:
            out[int(r)] = name
    return out


def test_split_disjoint_and_covers_all_rows() -> None:
    r = np.random.default_rng(1)
    t, y = r.integers(0, 2, 5000), (r.random(5000) < 0.1).astype(int)
    tr, va, te, _ = assign_splits(t, y, seed=1, n_folds=5)
    assert len(set(tr) | set(va) | set(te)) == 5000
    assert (
        not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    )
    assert abs(len(tr) / 5000 - 0.6) < 0.01


def test_same_seed_same_split() -> None:
    r = np.random.default_rng(2)
    t, y = r.integers(0, 2, 1000), (r.random(1000) < 0.1).astype(int)
    a = assign_splits(t, y, seed=7, n_folds=5)
    b = assign_splits(t, y, seed=7, n_folds=5)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_assertion_fails_on_unstratified_part() -> None:
    r = np.random.default_rng(3)
    t, y = r.integers(0, 2, 2000), (r.random(2000) < 0.01).astype(int)
    bad = {"test": np.flatnonzero(y == 1)}  # every conversion in one part
    with pytest.raises(ContractError):
        assert_cells_balanced(t, y, bad, tol=0.005)


def test_hillstrom_schema_rejects_bad_data() -> None:
    check_hillstrom(fake_hillstrom())
    with pytest.raises(ContractError):
        check_hillstrom(fake_hillstrom().drop(columns=["spend"]))
    bad = fake_hillstrom()
    bad.loc[0, "visit"] = 2
    with pytest.raises(ContractError):
        check_hillstrom(bad)
    bad = fake_hillstrom()
    bad.loc[0, "history"] = np.nan
    with pytest.raises(ContractError):
        check_hillstrom(bad)
    bad = fake_hillstrom()
    bad.loc[0, "segment"] = "Kids E-Mail"
    with pytest.raises(ContractError):
        check_hillstrom(bad)


def test_criteo_schema_rejects_exposed_control() -> None:
    check_criteo(fake_criteo())
    bad = fake_criteo()
    bad.loc[bad["treatment"] == 0, "exposure"] = 1
    with pytest.raises(ContractError):
        check_criteo(bad)


def test_hillstrom_keeps_all_rows_and_marks_the_comparison() -> None:
    """No row is dropped at load time; in_scope marks the chosen comparison."""
    raw = fake_hillstrom()
    p = prepare_hillstrom(raw, HILLSTROM_MENS_VS_CONTROL)
    assert len(p.t) == len(raw)
    assert np.array_equal(p.row_id, np.arange(len(raw)))
    womens = (raw["segment"] == "Womens E-Mail").to_numpy()
    mens = (raw["segment"] == "Mens E-Mail").to_numpy()
    assert not p.in_scope[womens].any()
    assert p.in_scope[~womens].all()
    assert np.array_equal(p.t[p.in_scope], mens[p.in_scope].astype(int))
    assert np.array_equal(p.y_conversion, raw["conversion"].to_numpy())
    assert len(set(p.arm.tolist())) == 3  # the split sees all three arms


def test_criteo_exposure_not_in_features_and_subsample_is_stratified() -> None:
    p = prepare_criteo(fake_criteo())
    assert list(p.X.columns) == list(CRITEO_FEATURES)
    assert len(p.t) == 3000  # no subsample at load time
    train, val, test, _ = plan_split(p, CFG, subsample_frac=0.5)
    kept = np.concatenate([train, val, test])
    assert abs(len(kept) - 1500) < 5
    assert abs(p.t[kept].mean() - 0.85) < 0.02


def test_end_to_end_hillstrom_both_targets_share_rows() -> None:
    p = prepare_hillstrom(fake_hillstrom(), HILLSTROM_MENS_VS_CONTROL)
    a = build_splits(
        p,
        dataset="hillstrom",
        target="visit",
        treatment=HILLSTROM_MENS_VS_CONTROL,
        cfg=CFG,
    )
    b = build_splits(
        p,
        dataset="hillstrom",
        target="conversion",
        treatment=HILLSTROM_MENS_VS_CONTROL,
        cfg=CFG,
    )
    assert np.array_equal(a.train.row_id, b.train.row_id)
    assert np.array_equal(a.test.row_ids, b.test.row_ids)
    assert a.train.fold_ids == (0, 1, 2, 3, 4)


def test_min_control_events_fails_when_too_high() -> None:
    p = prepare_hillstrom(fake_hillstrom(), HILLSTROM_MENS_VS_CONTROL)
    with pytest.raises(ContractError):
        build_splits(
            p,
            dataset="hillstrom",
            target="conversion",
            treatment=HILLSTROM_MENS_VS_CONTROL,
            cfg={**CFG, "min_control_events_per_fold": 10_000},
        )


def test_womens_comparison_has_no_mens_rows_in_the_splits() -> None:
    raw = fake_hillstrom()
    p = prepare_hillstrom(raw, HILLSTROM_WOMENS_VS_CONTROL)
    s = build_splits(
        p,
        dataset="hillstrom",
        target="conversion",
        treatment=HILLSTROM_WOMENS_VS_CONTROL,
        cfg=CFG,
    )
    for m in (s.train, s.validation):
        segments = raw.loc[m.row_id, "segment"]
        assert "Mens E-Mail" not in set(segments)
        assert np.array_equal(m.t, (segments == "Womens E-Mail").to_numpy().astype(int))


def test_mens_comparison_has_no_womens_rows_in_the_splits() -> None:
    raw = fake_hillstrom()
    p = prepare_hillstrom(raw, HILLSTROM_MENS_VS_CONTROL)
    s = build_splits(
        p,
        dataset="hillstrom",
        target="conversion",
        treatment=HILLSTROM_MENS_VS_CONTROL,
        cfg=CFG,
    )
    for m in (s.train, s.validation):
        segments = raw.loc[m.row_id, "segment"]
        assert "Womens E-Mail" not in set(segments)
        assert np.array_equal(m.t, (segments == "Mens E-Mail").to_numpy().astype(int))


def test_test_split_stays_locked_after_build() -> None:
    p = prepare_hillstrom(fake_hillstrom(), HILLSTROM_MENS_VS_CONTROL)
    s = build_splits(
        p,
        dataset="hillstrom",
        target="conversion",
        treatment=HILLSTROM_MENS_VS_CONTROL,
        cfg=CFG,
    )
    assert s.test.is_open is False


def test_missing_parquet_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="criteo_to_parquet"):
        load_criteo(path=tmp_path / "nope.parquet")


def test_schema_rejects_non_numeric_feature() -> None:
    bad = fake_criteo()
    bad["f0"] = bad["f0"].astype(str)
    with pytest.raises(ContractError):
        check_criteo(bad)


def test_rare_cell_missing_from_a_part_is_caught() -> None:
    """A part with no control events must fail even when the share gap is tiny."""
    t = np.tile([0, 1], 5000)
    y = np.zeros(10_000, dtype=int)
    events = np.flatnonzero(t == 0)[:40]
    y[events] = 1
    part = np.setdiff1d(np.arange(10_000), events)[:5000]
    with pytest.raises(ContractError, match="rows from its expected count"):
        assert_cells_balanced(t, y, {"bad": part}, tol=0.02)


def test_control_rows_keep_their_split_across_comparisons() -> None:
    """Mens-vs-control and Womens-vs-control put every control row in the same split."""
    raw = fake_hillstrom()
    mens = control_split_of(raw, HILLSTROM_MENS_VS_CONTROL)
    womens = control_split_of(raw, HILLSTROM_WOMENS_VS_CONTROL)
    assert mens == womens


def test_subsample_splits_are_subsets_of_the_full_data_splits() -> None:
    """Every subsample split only contains rows of the same split on the full data."""
    p = fake_prepared()
    full = plan_split(p, CFG)
    sub = plan_split(p, CFG, subsample_frac=0.5)
    for f, s in zip(full[:3], sub[:3], strict=True):
        assert set(p.row_id[s]) <= set(p.row_id[f])
        assert len(s) < len(f)


def test_subsample_frac_must_be_between_zero_and_one() -> None:
    """A bad subsample fraction fails fast with a ContractError, before the split."""
    p = fake_prepared()
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ContractError, match="subsample_frac"):
            plan_split(p, CFG, subsample_frac=bad)
