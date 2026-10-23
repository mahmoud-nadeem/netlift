"""Load and prepare the raw datasets (Hillstrom, Criteo) for the split."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

from netlift.data.contracts import TreatmentDefinition
from netlift.data.schema import (
    CRITEO_FEATURES,
    HILLSTROM_CHANNEL,
    HILLSTROM_ZIP,
    check_criteo,
    check_hillstrom,
)
from netlift.data.split import joint_key

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
HILLSTROM_CSV = BASE_DIR / "data" / "raw" / "hillstrom.csv"
CRITEO_PARQUET = BASE_DIR / "data" / "interim" / "criteo.parquet"


@dataclass(frozen=True)
class Prepared:
    """A dataset after schema checks, before the split."""

    X: pd.DataFrame
    t: npt.NDArray[np.int8]
    y_visit: npt.NDArray[np.int8]
    y_conversion: npt.NDArray[np.int8]
    row_id: npt.NDArray[
        np.int64
    ]  # position in the raw file, recorded before any row is dropped


def log_duplicates(raw: pd.DataFrame, name: str) -> int:
    """Count exact duplicate rows and log them. Nothing is removed (T17 policy)."""
    n = int(raw.duplicated().sum())
    logger.info(
        "%s: %d exact duplicate rows (%.2f%%) on the loaded dtypes "
        "(float32 inflates Criteo vs the float64 CSV), documented and kept",
        name,
        n,
        100 * n / len(raw),
    )
    return n


def prepare_hillstrom(raw: pd.DataFrame, treatment: TreatmentDefinition) -> Prepared:
    """Check the schema, drop the other e-mail arm, encode features."""
    check_hillstrom(raw)
    log_duplicates(raw, "hillstrom")
    row_id = np.arange(len(raw), dtype=np.int64)  # BEFORE dropping anything
    keep = ~raw["segment"].isin(treatment.excluded_values).to_numpy()
    df = raw.loc[keep].reset_index(drop=True)
    row_id = row_id[keep]
    logger.info("hillstrom: dropped %d rows of the excluded arm", int((~keep).sum()))

    t = df["segment"].isin(treatment.treatment_values).to_numpy().astype(np.int8)
    X = pd.DataFrame(
        {
            "recency": df["recency"].astype("int64"),
            "history": df["history"].astype("float64"),
            "history_segment_ord": df["history_segment"].str[0].astype("int64"),
            "mens": df["mens"].astype("int8"),
            "womens": df["womens"].astype("int8"),
            "newbie": df["newbie"].astype("int8"),
        }
    )
    for z in HILLSTROM_ZIP:  # fixed category lists, so nothing is learned from the data
        X[f"zip_{z}"] = (df["zip_code"] == z).astype("int8")
    for c in HILLSTROM_CHANNEL:
        X[f"channel_{c}"] = (df["channel"] == c).astype("int8")
    return Prepared(
        X=X,
        t=t,
        y_visit=df["visit"].to_numpy().astype(np.int8),
        y_conversion=df["conversion"].to_numpy().astype(np.int8),
        row_id=row_id,
    )


def prepare_criteo(
    raw: pd.DataFrame, *, subsample_frac: float | None = None, seed: int = 42
) -> Prepared:
    """Check the schema; optionally take a subsample stratified on (treatment, conversion)."""
    check_criteo(raw)
    log_duplicates(raw, "criteo")
    row_id = np.arange(len(raw), dtype=np.int64)
    if subsample_frac is not None:
        key = joint_key(raw["treatment"], raw["conversion"])
        keep, _ = train_test_split(
            row_id, train_size=subsample_frac, stratify=key, random_state=seed
        )
        keep = np.sort(keep)
        raw = raw.iloc[keep].reset_index(drop=True)
        row_id = keep
        logger.info(
            "criteo: stratified subsample %.0f%% -> %d rows",
            subsample_frac * 100,
            len(raw),
        )
    # exposure is post-treatment: it was only schema-checked, it never enters X.
    return Prepared(
        X=raw[list(CRITEO_FEATURES)].reset_index(drop=True),
        t=raw["treatment"].to_numpy().astype(np.int8),
        y_visit=raw["visit"].to_numpy().astype(np.int8),
        y_conversion=raw["conversion"].to_numpy().astype(np.int8),
        row_id=row_id,
    )


def load_hillstrom(
    treatment: TreatmentDefinition, path: Path = HILLSTROM_CSV
) -> Prepared:
    """Read the Hillstrom CSV and prepare it."""
    return prepare_hillstrom(pd.read_csv(path), treatment)


def load_criteo(
    *, subsample_frac: float | None = None, seed: int = 42, path: Path = CRITEO_PARQUET
) -> Prepared:
    """Read the typed Parquet file (made by scripts/criteo_to_parquet.py) and prepare it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/criteo_to_parquet.py"
        )
    return prepare_criteo(
        pd.read_parquet(path), subsample_frac=subsample_frac, seed=seed
    )
