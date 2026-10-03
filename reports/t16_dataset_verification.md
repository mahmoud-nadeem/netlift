# T16: Dataset figures verification

Date: 2026-09-29

## 1. Hillstrom

File: `data/raw/hillstrom.csv`

SHA-256:
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`

### Dataset structure

| Property | Actual |
|---|---|
| Rows | 64,000 |
| Columns | 12 |
| Total nulls | 0 |

### Column names and dtypes

| Column | dtype |
|---|---|
| recency | int64 |
| history_segment | str |
| history | float64 |
| mens | int64 |
| womens | int64 |
| zip_code | str |
| newbie | int64 |
| channel | str |
| segment | str |
| visit | int64 |
| conversion | int64 |
| spend | float64 |

### Verification results

| Check | Published | Actual | Match |
|---|---|---|---|
| Rows | 64,000 | 64,000 | yes |
| Columns | 12 | 12 | yes |
| Nulls | 0 | 0 | yes |
| Group sizes | About one third each | 33.4172 / 33.2922 / 33.2906 pct | yes |
| Visit rate (Mens / Womens / No E-Mail) | 18.28 / 15.14 / 10.62 pct | 18.28 / 15.14 / 10.62 pct | yes |
| Conversion rate (Mens / Womens / No E-Mail) | 1.25 / 0.88 / 0.57 pct | 1.25 / 0.88 / 0.57 pct | yes |
| Mean spend (USD) | 1.42 / 1.08 / 0.65 | 1.42 / 1.08 / 0.65 | yes |

### Overall outcome rates

| Outcome | Overall rate |
|---|---:|
| Visit | 14.678125% |
| Conversion | 0.903125% |
| Mean spend | 1.050908 |

Overall rates are computed directly from the complete dataset.

---

## 2. Criteo

File: `data/raw/criteo-research-uplift-v2.1.csv.gz`

SHA-256:
`2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`

### Dataset structure

| Property | Actual |
|---|---|
| Rows | 13,979,592 |
| Columns | 16 |
| Total nulls | 0 |

### Column names and dtypes

| Column | dtype |
|---|---|
| f0 | float64 |
| f1 | float64 |
| f2 | float64 |
| f3 | float64 |
| f4 | float64 |
| f5 | float64 |
| f6 | float64 |
| f7 | float64 |
| f8 | float64 |
| f9 | float64 |
| f10 | float64 |
| f11 | float64 |
| treatment | int64 |
| conversion | int64 |
| visit | int64 |
| exposure | int64 |

### Verification results

| Check | Published | Actual | Match |
|---|---|---|---|
| Rows | 13,979,592 | 13,979,592 | yes |
| Columns | 16 | 16 | yes |
| Nulls | 0 | 0 | yes |
| Treatment / Control | 85 / 15 pct | 85 / 15 pct | yes |
| Overall visit rate | About 4.7 pct | 4.699200% | yes |
| Overall conversion rate | About 0.29 pct | 0.291668% | yes |

### Outcome rates by treatment group

| Outcome | Control (0) | Treatment (1) | Overall |
|---|---:|---:|---:|
| Visit | 3.8201% | 4.8543% | 4.699200% |
| Conversion | 0.1938% | 0.3089% | 0.291668% |

Overall rates are computed directly from all 13,979,592 rows, not by averaging the group rates.

### Exposure column investigation

The Criteo dataset contains an explicitly named `exposure` column.

| Property | Actual |
|---|---|
| Column | exposure |
| dtype | int64 |
| Value 0 | 13,551,380 rows |
| Value 1 | 428,212 rows |

The presence of this column should be reviewed against Specification Section 6, which states that neither dataset provides an exposure indicator and uses this assumption as the basis for estimating ITT rather than ToT.

The column's name and values alone do not establish its precise methodological meaning. Its definition should be verified against the dataset documentation and project specification before deciding whether it represents the exposure indicator described in Section 6.

This is a confirmed discrepancy with Specification Section 6: per Criteo AI Lab's
own dataset documentation, the `exposure` column is officially defined as
"treatment effect, whether the user has been effectively exposed" — a genuine
exposure indicator, contradicting Section 6's claim that neither dataset provides
one. Flagged in #decisions for the team.

---

## 3. Reproducibility

The script `scripts/t16_check.py` reproduces the following:

- Dataset row and column counts.
- Column names and pandas dtypes.
- Missing-value counts.
- Treatment-group shares.
- Per-group and unconditional overall outcome rates.
- SHA-256 checksums of both source files.
- Presence, dtype, and value counts of the Criteo exposure column.

Raw dataset files are not committed to the repository.

## Conclusion

The observed row counts, column counts, missing-value counts, treatment shares, and outcome rates match the published figures checked for T16.

Both dataset checksums are recorded above and are computed by the committed verification script.

The Criteo `exposure` column is a confirmed discrepancy with Specification Section 6
(not a potential one — see the exposure investigation above for the source). Its
implications for ITT versus ToT estimation require a separate methodological review,
tracked in #decisions.
