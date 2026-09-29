import pandas as pd

CRITEO_PATH = "data/raw/criteo-research-uplift-v2.1.csv.gz"
HILLSTROM_PATH = "data/raw/hillstrom.csv"

def profile(name, df, treat, outcomes):
    print("=" * 50)
    print(name)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("total nulls:", int(df.isna().sum().sum()))
    print("\ntreatment share:")
    print(df[treat].value_counts(normalize=True))
    for o in outcomes:
        print(f"\n{o} rate per group:")
        print(df.groupby(treat)[o].mean())

hill = pd.read_csv(HILLSTROM_PATH)
profile("Hillstrom", hill, "segment", ["visit", "conversion", "spend"])

criteo = pd.read_csv(CRITEO_PATH)
profile("Criteo", criteo, "treatment", ["visit", "conversion"])
