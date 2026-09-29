# T16: Dataset figures verification

Date: 2026-09-29

## Hillstrom (data/raw/hillstrom.csv)
SHA-256: 0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece

| Check | Published | Actual | Match |
|---|---|---|---|
| Rows | 64,000 | 64,000 | yes |
| Columns | 12 | 12 | yes |
| Nulls | 0 | 0 | yes |
| Group sizes | about one third each | 33.4 / 33.3 / 33.3 pct | yes |
| Visit rate (Mens / Womens / No E-Mail) | 18.28 / 15.14 / 10.62 pct | 18.28 / 15.14 / 10.62 pct | yes |
| Conversion rate | 1.25 / 0.88 / 0.57 pct | 1.25 / 0.88 / 0.57 pct | yes |
| Mean spend (USD) | 1.42 / 1.08 / 0.65 | 1.42 / 1.08 / 0.65 | yes |

## Criteo (data/raw/criteo-research-uplift-v2.1.csv.gz)
SHA-256: 2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc

| Check | Published | Actual | Match |
|---|---|---|---|
| Rows | 13,979,592 | 13,979,592 | yes |
| Columns | 16 | 16 | yes |
| Nulls | 0 | 0 | yes |
| Treatment / Control | 85 / 15 pct | 85 / 15 pct | yes |
| Overall visit rate | about 4.7 pct | 4.70 pct | yes |
| Overall conversion rate | about 0.29 pct | 0.29 pct | yes |

## Conclusion
Both files match the published figures. No discrepancies found.
