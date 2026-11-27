"""CI leakage audit: exposure, treatment and every outcome never enter X."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

from netlift.data.contracts import FORBIDDEN_COLUMNS, HILLSTROM_MENS_VS_CONTROL
from netlift.data.leakage import (
    audit_columns,
    audit_prepared,
    forbidden_columns,
    main,
)
from netlift.data.loaders import Prepared, prepare_criteo, prepare_hillstrom
from netlift.data.pipeline import load_config
from netlift.data.schema import (
    CRITEO_COLUMNS,
    CRITEO_FEATURES,
    HILLSTROM_CHANNEL,
    HILLSTROM_ZIP,
)
from netlift.utils.config import validate_data_config
from netlift.utils.errors import ConfigError, ContractError, LeakageError
from netlift.utils.logging_setup import setup_logging

CFG = load_config()
N = 400
ARMS = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")
HISTORY_SEGMENTS = (
    "1) $0 - $100",
    "2) $100 - $200",
    "3) $200 - $350",
    "4) $350 - $500",
    "5) $500 - $750",
    "6) $750 - $1,000",
    "7) $1,000 +",
)


def _binary(rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    return (rng.random(N) < p).astype("int64")


def criteo_raw() -> pd.DataFrame:
    """Small synthetic frame with Criteo's raw columns."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.random((N, len(CRITEO_FEATURES))).astype("float32"),
        columns=list(CRITEO_FEATURES),
    )
    treatment = _binary(rng)
    visit = _binary(rng, 0.3)
    df["treatment"] = treatment
    df["visit"] = visit
    df["conversion"] = visit * _binary(rng, 0.3)
    df["exposure"] = treatment * _binary(rng, 0.7)
    return df[list(CRITEO_COLUMNS)]


def hillstrom_raw() -> pd.DataFrame:
    """Small synthetic frame with Hillstrom's raw columns."""
    rng = np.random.default_rng(1)
    visit = _binary(rng, 0.3)
    conversion = visit * _binary(rng, 0.3)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, N),
            "history_segment": rng.choice(HISTORY_SEGMENTS, N),
            "history": rng.random(N) * 500,
            "mens": _binary(rng),
            "womens": _binary(rng),
            "zip_code": rng.choice(HILLSTROM_ZIP, N),
            "newbie": _binary(rng),
            "channel": rng.choice(HILLSTROM_CHANNEL, N),
            "segment": np.resize(ARMS, N),
            "visit": visit,
            "conversion": conversion,
            "spend": conversion * rng.random(N) * 100,
        }
    )


def test_config_list_equals_frozen_contract() -> None:
    assert forbidden_columns(CFG) == {c.lower() for c in FORBIDDEN_COLUMNS}


def test_criteo_feature_matrix_has_no_forbidden_column() -> None:
    raw = criteo_raw()
    assert "exposure" in raw.columns  # the audit is only meaningful if it is in raw
    prepared = prepare_criteo(raw)
    audit_prepared(prepared, CFG, source="criteo")
    assert not {c.lower() for c in prepared.X.columns} & forbidden_columns(CFG)


def test_hillstrom_feature_matrix_has_no_forbidden_column() -> None:
    prepared = prepare_hillstrom(hillstrom_raw(), HILLSTROM_MENS_VS_CONTROL)
    audit_prepared(prepared, CFG, source="hillstrom")
    assert not {c.lower() for c in prepared.X.columns} & forbidden_columns(CFG)


@pytest.mark.parametrize("column", sorted(FORBIDDEN_COLUMNS))
def test_audit_catches_every_forbidden_column(column: str) -> None:
    with pytest.raises(LeakageError, match=column.upper()):
        audit_columns(["f0", column.upper()], forbidden_columns(CFG), source="test")


def test_audit_catches_leak_in_a_real_prepared_matrix() -> None:
    prepared = prepare_criteo(criteo_raw())
    leaky = dataclasses.replace(prepared, X=prepared.X.assign(exposure=1))
    with pytest.raises(LeakageError):
        audit_prepared(leaky, CFG)


def test_leakage_error_is_a_contract_error() -> None:
    assert issubclass(LeakageError, ContractError)
    assert issubclass(ConfigError, ContractError)


def test_real_config_is_valid() -> None:
    assert validate_data_config(copy.deepcopy(CFG)) == CFG


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("seed", True),
        ("seed", -1),
        ("n_folds", 1),
        ("shares", [0.5, 0.2, 0.2]),
        ("shares", [0.6, 0.4]),
        ("tolerance", 0),
        ("min_control_events_per_fold", -3),
    ],
)
def test_invalid_config_values_raise(key: str, bad: Any) -> None:
    cfg = copy.deepcopy(CFG)
    cfg[key] = bad
    with pytest.raises(ConfigError, match=key):
        validate_data_config(cfg)


def test_missing_config_key_raises() -> None:
    cfg = copy.deepcopy(CFG)
    del cfg["n_folds"]
    with pytest.raises(ConfigError, match="n_folds"):
        validate_data_config(cfg)


def test_missing_leakage_section_raises() -> None:
    with pytest.raises(ConfigError, match="leakage"):
        forbidden_columns({})


@pytest.fixture
def quiet_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the CLI tests from reconfiguring the root logger."""
    monkeypatch.setattr("netlift.data.leakage.setup_logging", lambda cfg: None)


def test_cli_returns_0_when_the_matrix_is_clean(
    monkeypatch: pytest.MonkeyPatch, quiet_logging: None
) -> None:
    prepared = prepare_criteo(criteo_raw())
    monkeypatch.setattr("netlift.data.leakage.load_criteo", lambda: prepared)
    assert main(["--dataset", "criteo"]) == 0


def test_cli_returns_1_when_leakage_is_found(
    monkeypatch: pytest.MonkeyPatch, quiet_logging: None
) -> None:
    prepared = prepare_criteo(criteo_raw())
    leaky = dataclasses.replace(prepared, X=prepared.X.assign(exposure=1))
    monkeypatch.setattr("netlift.data.leakage.load_criteo", lambda: leaky)
    assert main(["--dataset", "criteo"]) == 1


def test_cli_returns_2_when_the_data_is_missing(
    monkeypatch: pytest.MonkeyPatch, quiet_logging: None
) -> None:
    def missing() -> Prepared:
        raise FileNotFoundError("no data")

    monkeypatch.setattr("netlift.data.leakage.load_criteo", missing)
    assert main(["--dataset", "criteo"]) == 2


def test_setup_logging_rejects_an_unknown_level() -> None:
    with pytest.raises(ConfigError, match="loud"):
        setup_logging({"logging": {"level": "loud"}})
