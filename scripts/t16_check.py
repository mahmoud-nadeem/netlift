import hashlib
from pathlib import Path

import pandas as pd

CRITEO_PATH = Path("data/raw/criteo-research-uplift-v2.1.csv.gz")
HILLSTROM_PATH = Path("data/raw/hillstrom.csv")


def sha256_file(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def profile(name, df, treat, outcomes, path):
    print("=" * 60)
    print(name)
    print("Shape:", df.shape)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names and dtypes:")
    print(df.dtypes.to_string())

    print("\nAll column names:")
    print(list(df.columns))

    print("\nTotal nulls:", int(df.isna().sum().sum()))

    print("\nTreatment share:")
    print(df[treat].value_counts(normalize=True))

    for o in outcomes:
        print(f"\n{o} rate per group:")
        print(df.groupby(treat)[o].mean())

        print(f"{o} overall rate:", df[o].mean())

    print("\nSHA-256:", sha256_file(path))


hill = pd.read_csv(HILLSTROM_PATH)

profile(
    "Hillstrom",
    hill,
    "segment",
    ["visit", "conversion", "spend"],
    HILLSTROM_PATH,
)

criteo = pd.read_csv(CRITEO_PATH)

profile(
    "Criteo",
    criteo,
    "treatment",
    ["visit", "conversion"],
    CRITEO_PATH,
)

print("\nCriteo exposure column check:")

if "exposure" in criteo.columns:
    print("Exposure column PRESENT")
    print("Exposure dtype:", criteo["exposure"].dtype)
    print("Exposure values:")
    print(criteo["exposure"].value_counts(dropna=False))
    print("\nExposure by treatment group (crosstab):")
    print(pd.crosstab(criteo["treatment"], criteo["exposure"]))
else:
    print("Exposure column NOT FOUND")
