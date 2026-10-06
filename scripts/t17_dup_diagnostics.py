import pandas as pd

hill = pd.read_csv("data/raw/hillstrom.csv")
feats = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]

print("unique values per feature (Hillstrom):")
print(hill[feats].nunique())

h = pd.util.hash_pandas_object(hill[feats], index=False)
sizes = h.map(h.value_counts())
print("\nprofile group size distribution (rows by size of their group):")
print(sizes.value_counts().sort_index())

rep = hill[sizes > 1]
print(
    "\nshare of repeated-profile rows with newbie==1:", round(rep["newbie"].mean(), 3)
)
print("share of all rows with newbie==1:", round(hill["newbie"].mean(), 3))
print(
    "median history, repeated vs all:",
    rep["history"].median(),
    hill["history"].median(),
)
