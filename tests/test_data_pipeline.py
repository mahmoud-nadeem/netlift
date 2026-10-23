import numpy as np
import pandas as pd
import pytest

from netlift.data.contracts import HILLSTROM_MENS_VS_CONTROL
from netlift.data.loaders import prepare_criteo, prepare_hillstrom
from netlift.data.pipeline import build_splits
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
    "min_control_events": 5,
}


def fake_hillstrom(n=2000, seed=0):
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


def fake_criteo(n=3000, seed=0):
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


def test_split_disjoint_and_covers_all_rows():
    r = np.random.default_rng(1)
    t, y = r.integers(0, 2, 5000), (r.random(5000) < 0.1).astype(int)
    tr, va, te, _ = assign_splits(t, y, seed=1, n_folds=5)
    assert len(set(tr) | set(va) | set(te)) == 5000
    assert (
        not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    )
    assert abs(len(tr) / 5000 - 0.6) < 0.01


def test_same_seed_same_split():
    r = np.random.default_rng(2)
    t, y = r.integers(0, 2, 1000), (r.random(1000) < 0.1).astype(int)
    a = assign_splits(t, y, seed=7, n_folds=5)
    b = assign_splits(t, y, seed=7, n_folds=5)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_assertion_fails_on_unstratified_part():
    r = np.random.default_rng(3)
    t, y = r.integers(0, 2, 2000), (r.random(2000) < 0.01).astype(int)
    bad = {"test": np.flatnonzero(y == 1)}  # every conversion in one part
    with pytest.raises(ContractError):
        assert_cells_balanced(t, y, bad, tol=0.005)


def test_hillstrom_schema_rejects_bad_data():
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


def test_criteo_schema_rejects_exposed_control():
    check_criteo(fake_criteo())
    bad = fake_criteo()
    bad.loc[bad["treatment"] == 0, "exposure"] = 1
    with pytest.raises(ContractError):
        check_criteo(bad)


def test_hillstrom_drops_other_arm_and_keeps_raw_row_ids():
    raw = fake_hillstrom()
    p = prepare_hillstrom(raw, HILLSTROM_MENS_VS_CONTROL)
    kept = raw.loc[p.row_id]
    assert "Womens E-Mail" not in set(kept["segment"])
    assert np.array_equal(
        p.t, (kept["segment"] == "Mens E-Mail").to_numpy().astype(int)
    )
    assert np.array_equal(p.y_conversion, kept["conversion"].to_numpy())


def test_criteo_exposure_not_in_features_and_subsample_is_stratified():
    p = prepare_criteo(fake_criteo(), subsample_frac=0.5, seed=1)
    assert list(p.X.columns) == list(CRITEO_FEATURES)
    assert abs(len(p.t) - 1500) < 5
    assert abs(p.t.mean() - 0.85) < 0.02


def test_end_to_end_hillstrom_both_targets_share_rows():
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


def test_min_control_events_fails_when_too_high():
    p = prepare_hillstrom(fake_hillstrom(), HILLSTROM_MENS_VS_CONTROL)
    with pytest.raises(ContractError):
        build_splits(
            p,
            dataset="hillstrom",
            target="conversion",
            treatment=HILLSTROM_MENS_VS_CONTROL,
            cfg={**CFG, "min_control_events": 10_000},
        )


def test_womens_arm_drops_mens_rows():
    from netlift.data.contracts import HILLSTROM_WOMENS_VS_CONTROL

    raw = fake_hillstrom()
    p = prepare_hillstrom(raw, HILLSTROM_WOMENS_VS_CONTROL)
    kept = raw.loc[p.row_id]
    assert "Mens E-Mail" not in set(kept["segment"])
    assert np.array_equal(
        p.t, (kept["segment"] == "Womens E-Mail").to_numpy().astype(int)
    )


def test_test_split_stays_locked_after_build():
    p = prepare_hillstrom(fake_hillstrom(), HILLSTROM_MENS_VS_CONTROL)
    s = build_splits(
        p,
        dataset="hillstrom",
        target="conversion",
        treatment=HILLSTROM_MENS_VS_CONTROL,
        cfg=CFG,
    )
    assert s.test.is_open is False


def test_missing_parquet_raises_clear_error(tmp_path):
    from netlift.data.loaders import load_criteo

    with pytest.raises(FileNotFoundError, match="criteo_to_parquet"):
        load_criteo(path=tmp_path / "nope.parquet")


def test_schema_rejects_non_numeric_feature():
    bad = fake_criteo()
    bad["f0"] = bad["f0"].astype(str)
    with pytest.raises(ContractError):
        check_criteo(bad)


def test_rare_cell_missing_from_a_part_is_caught():
    """A part with no control events must fail even when the share gap is tiny."""
    import numpy as np
    import pytest

    from netlift.data.split import assert_cells_balanced
    from netlift.utils.errors import ContractError

    t = np.tile([0, 1], 5000)
    y = np.zeros(10_000, dtype=int)
    events = np.flatnonzero(t == 0)[:40]
    y[events] = 1
    part = np.setdiff1d(np.arange(10_000), events)[:5000]
    with pytest.raises(ContractError, match="rows from its expected count"):
        assert_cells_balanced(t, y, {"bad": part}, tol=0.02)
