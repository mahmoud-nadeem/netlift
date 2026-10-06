import pandas as pd

CRITEO_PATH = "data/raw/criteo-research-uplift-v2.1.csv.gz"
HILLSTROM_PATH = "data/raw/hillstrom.csv"


def check(name, df, treat, non_features):
    print("=" * 60)
    print(name, "| rows:", len(df))
    ids = [
        c for c in df.columns if c.lower() in ("id", "user_id", "customer_id", "userid")
    ]
    print("identifier columns:", ids or "none")

    feats = [c for c in df.columns if c not in non_features]
    print("feature columns used:", feats)

    full = pd.util.hash_pandas_object(df, index=False)
    print("fully duplicated rows:", int(full.duplicated().sum()))

    h = pd.util.hash_pandas_object(df[feats], index=False)
    counts = h.value_counts()
    repeated = counts[counts > 1]
    print("feature-profiles that appear more than once:", len(repeated))
    print(
        "rows inside those repeated profiles:",
        int(repeated.sum()),
        f"({repeated.sum() / len(df):.2%})",
    )

    t = pd.DataFrame({"h": h, "t": df[treat]})
    n_treat = t.groupby("h")["t"].nunique()
    print(
        "profiles seen under more than one treatment value:", int((n_treat > 1).sum())
    )


hill = pd.read_csv(HILLSTROM_PATH)
check("Hillstrom", hill, "segment", ["segment", "visit", "conversion", "spend"])

criteo = pd.read_csv(CRITEO_PATH)
check("Criteo", criteo, "treatment", ["treatment", "visit", "conversion", "exposure"])

print("\nCriteo treatment vs exposure (share of all rows):")
print(pd.crosstab(criteo["treatment"], criteo["exposure"], normalize=True))
