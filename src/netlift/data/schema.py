"""Schema checks for the raw datasets (specification: Complete Data Pipeline)."""

from __future__ import annotations

import pandas as pd

from netlift.utils.errors import ContractError

HILLSTROM_COLUMNS = (
    "recency", "history_segment", "history", "mens", "womens", "zip_code",
    "newbie", "channel", "segment", "visit", "conversion", "spend",
)
HILLSTROM_SEGMENTS = {"Mens E-Mail", "No E-Mail", "Womens E-Mail"}
HILLSTROM_HISTORY_SEGMENTS = (
    "1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
    "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +",
)
HILLSTROM_ZIP = ("Rural", "Surburban", "Urban")  # "Surburban" is the raw spelling
HILLSTROM_CHANNEL = ("Multichannel", "Phone", "Web")

CRITEO_FEATURES = tuple(f"f{i}" for i in range(12))
CRITEO_COLUMNS = (*CRITEO_FEATURES, "treatment", "conversion", "visit", "exposure")


def _require_columns(df: pd.DataFrame, expected: tuple[str, ...], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ContractError(f"{name}: missing column(s) {missing}.")


def _require_no_nulls(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    nulls = {c: int(df[c].isna().sum()) for c in columns if df[c].isna().any()}
    if nulls:
        raise ContractError(f"{name}: null values found {nulls}.")


def _require_binary(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for c in columns:
        bad = set(pd.unique(df[c])) - {0, 1}
        if bad:
            raise ContractError(f"{name}: column '{c}' must be 0/1, found {sorted(bad)[:5]}.")


def check_hillstrom(df: pd.DataFrame) -> None:
    """Raise ContractError if the Hillstrom frame does not match the expected schema."""
    _require_columns(df, HILLSTROM_COLUMNS, "hillstrom")
    _require_no_nulls(df, HILLSTROM_COLUMNS, "hillstrom")
    _require_binary(df, ("visit", "conversion", "mens", "womens", "newbie"), "hillstrom")
    for column, allowed in (
        ("segment", HILLSTROM_SEGMENTS),
        ("history_segment", HILLSTROM_HISTORY_SEGMENTS),
        ("zip_code", HILLSTROM_ZIP),
        ("channel", HILLSTROM_CHANNEL),
    ):
        bad = set(df[column].unique()) - set(allowed)
        if bad:
            raise ContractError(f"hillstrom: unexpected value(s) in '{column}': {sorted(bad)}.")
    if (df["spend"] < 0).any():
        raise ContractError("hillstrom: negative spend.")


def check_criteo(df: pd.DataFrame) -> None:
    """Raise ContractError if the Criteo frame does not match the expected schema."""
    _require_columns(df, CRITEO_COLUMNS, "criteo")
    _require_no_nulls(df, CRITEO_COLUMNS, "criteo")
    _require_binary(df, ("treatment", "conversion", "visit", "exposure"), "criteo")
    if bool(((df["treatment"] == 0) & (df["exposure"] == 1)).any()):
        raise ContractError("criteo: exposed control rows found; one-sided non-compliance broken.")
