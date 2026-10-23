# 0003: Design of the data pipeline and the joint stratified split

Task T37. Status: Proposed. Owner: Aliaa. Reviewer: Mohammed.

These are the decisions that `docs/design/interfaces.md` asks the producer of
interface 1 to confirm.

## Decisions

1. **Stratification key: (treatment, conversion)**, for both datasets. The split
   is built once per (dataset, seed) and never depends on the target outcome;
   `target_outcome` only selects the `y` column. So a row that is in the test
   split for `visit` is in the test split for `conversion`.
2. **Hillstrom: Mens E-Mail vs No E-Mail first.** The Womens E-Mail rows are
   dropped, never pooled (specification, Multi-Treatment Handling). Womens vs
   control is supported by `HILLSTROM_WOMENS_VS_CONTROL` and is run second.
3. **`min_control_events` = 10 per fold** (`config/data.yaml`). The
   specification gives no number. On Hillstrom (mens vs control) each train fold
   has 14 to 15 control-arm conversions, so 10 is the largest round number that
   still passes with a small margin. A higher value would fail on the real data.
   On Criteo each train fold has 487 to 488.
4. **Splits and folds:** 60 / 20 / 20, seed 42, 5 stratified folds inside train
   on the same joint key. Validation and test are a single fold (id 0).
5. **Encoding is fixed, not fitted.** Hillstrom `history_segment` becomes an
   ordinal (the leading digit), and `zip_code` and `channel` are one-hot with
   category lists written in `schema.py` (`Surburban` is the raw spelling in the
   file). Nothing is learned from the data, so there is no train/test leakage
   from preprocessing. Scaling and imputation are not applied here.
6. **Duplicates are detected and documented, not removed**, following
   `docs/unit-of-observation.md`. Removing them would discard real rows and could
   shift the arm ratio.
7. **`row_id`** is the 0-based position in the raw file, recorded before any row
   is dropped (the other Hillstrom arm, the Criteo subsample).
8. **Criteo (risk R08).** `scripts/criteo_to_parquet.py` converts the CSV.gz once,
   in chunks of 1,000,000 rows, to Parquet with `float32` features and `int8`
   flags. `exposure` is checked in the schema but never enters `X`. A stratified
   subsample (`subsample_frac`) exists for iteration; reported numbers use the
   full data. float32 keeps about 7 significant digits, which is acceptable for
   anonymised features.

## Evidence

| Run | Result |
|---|---|
| Hillstrom, mens vs control | train 25,567 / validation 8,523; worst cell-share gap 0.00012 |
| Criteo, 10% subsample | 1,397,959 rows; worst gap 0.00000 |
| Criteo, full data | 13,979,592 rows; train 8,387,755 / validation 2,795,918 |
| Criteo, full data, cost | 38 s; peak memory footprint about 3.0 GB (max RSS about 3.76 GB) |

The test split stays locked after `load_splits`.

## Consequences

- R08: the full run fits in about 3 GB on the development machine, so chunked
  preprocessing beyond the Parquet conversion is not needed for the split.
- The Hillstrom control has only about 120 conversions in total, so interval
  widths on Hillstrom conversion will be large.
- If an ID-bearing Criteo file becomes available, the split must be redone
  grouped on that ID (see `docs/unit-of-observation.md`).
- The Criteo Parquet stores features as float32 (R08). Exact-duplicate counts taken from it (1,655,747 on the full file) are inflated relative to the float64 CSV count in `docs/unit-of-observation.md` (1,259,545). On the first 2,000,000 CSV rows the counts are 102,958 (float64) and 109,422 (float32). Duplicates are only documented, never removed, so the split is unaffected. Reports cite the T17 figure.
