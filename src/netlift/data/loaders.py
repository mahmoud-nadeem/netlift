"""Load and prepare the raw datasets (Hillstrom, Criteo) for the split."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from netlift.data.contracts import TreatmentDefinition
from netlift.data.schema import (
    CRITEO_FEATURES,
    HILLSTROM_CHANNEL,
    HILLSTROM_ZIP,
    check_criteo,
    check_hillstrom,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
HILLSTROM_CSV = BASE_DIR / "data" / "raw" / "hillstrom.csv"
CRITEO_PARQUET = BASE_DIR / "data" / "interim" / "criteo.parquet"


@dataclass(frozen=True)
class Prepared:
    """A dataset after schema checks, before the split.

    Every row of the raw file is kept. The split is built on all rows (stratified
    on arm and conversion); the comparison (in_scope, t) is applied inside each split.
    """

    X: pd.DataFrame
    arm: npt.NDArray[
        np.int8
    ]  # stratification arm: Hillstrom 3 arms, Criteo = treatment
    t: npt.NDArray[
        np.int8
    ]  # 0/1 under the chosen comparison; only meaningful where in_scope
    in_scope: npt.NDArray[np.bool_]  # row belongs to the chosen comparison
    y_visit: npt.NDArray[np.int8]
    y_conversion: npt.NDArray[np.int8]
    row_id: npt.NDArray[np.int64]  # position in the raw file


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
    """Check the schema and encode features. No row is dropped here."""
    check_hillstrom(raw)
    log_duplicates(raw, "hillstrom")
    df = raw.reset_index(drop=True)
    in_scope = ~df["segment"].isin(treatment.excluded_values).to_numpy()
    t = df["segment"].isin(treatment.treatment_values).to_numpy().astype(np.int8)
    # one code per e-mail arm (3 arms): the split must see all of them
    arm = df["segment"].astype("category").cat.codes.to_numpy().astype(np.int8)
    logger.info(
        "hillstrom: %d rows are outside this comparison; they stay in the split "
        "and are filtered after it",
        int((~in_scope).sum()),
    )

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
        arm=arm,
        t=t,
        in_scope=in_scope,
        y_visit=df["visit"].to_numpy().astype(np.int8),
        y_conversion=df["conversion"].to_numpy().astype(np.int8),
        row_id=np.arange(len(df), dtype=np.int64),
    )


def prepare_criteo(raw: pd.DataFrame) -> Prepared:
    """Check the schema and take the features. No subsample here: it is drawn after the split."""
    check_criteo(raw)
    log_duplicates(raw, "criteo")
    t = raw["treatment"].to_numpy().astype(np.int8)
    # exposure is post-treatment: it was only schema-checked, it never enters X.
    return Prepared(
        X=raw[list(CRITEO_FEATURES)].reset_index(drop=True),
        arm=t,
        t=t,
        in_scope=np.ones(len(raw), dtype=bool),
        y_visit=raw["visit"].to_numpy().astype(np.int8),
        y_conversion=raw["conversion"].to_numpy().astype(np.int8),
        row_id=np.arange(len(raw), dtype=np.int64),
    )


def load_hillstrom(
    treatment: TreatmentDefinition, path: Path = HILLSTROM_CSV
) -> Prepared:
    """Read the Hillstrom CSV and prepare it."""
    return prepare_hillstrom(pd.read_csv(path), treatment)


def load_criteo(path: Path = CRITEO_PARQUET) -> Prepared:
    """Read the typed Parquet file (made by scripts/criteo_to_parquet.py) and prepare it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/criteo_to_parquet.py"
        )
    return prepare_criteo(pd.read_parquet(path))
