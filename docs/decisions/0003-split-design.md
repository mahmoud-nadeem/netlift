# 0003: Design of the data pipeline and the joint stratified split

Task T37. Status: Proposed. Owner: Aliaa. Reviewer: Mohammed.

These are the decisions that `docs/design/interfaces.md` asks the producer of
interface 1 to confirm.

## Decisions

1. **The split is built once on the full raw file, stratified on (arm, conversion).**
   The arm is `treatment` for Criteo (4 cells) and the three e-mail arms of
   `segment` for Hillstrom (6 cells). The split is built once per (dataset, seed)
   and never depends on the target outcome; `target_outcome` only selects the `y`
   column. So a row that is in the test split for `visit` is in the test split for
   `conversion`. Everything that removes rows (the Hillstrom comparison filter and
   the Criteo subsample) happens after the split and only inside each split, so
   every result is a subset of the full-data split.
2. **Hillstrom: Mens E-Mail vs No E-Mail first.** The Womens E-Mail rows stay in
   the split and are filtered out afterwards, never pooled (specification,
   Multi-Treatment Handling). Womens vs control is supported by
   `HILLSTROM_WOMENS_VS_CONTROL` and is run second. Because both comparisons filter
   the same split, a control row has the same split (train, validation or test) in
   both comparisons. Splitting after the filter, as the first version did, put
   4,280 of the 21,306 control rows in a different split under the two comparisons,
   so a locked-test control row of one comparison was a validation row of the other.
3. **`min_control_events_per_fold` = 10** (`config/data.yaml`). The
   specification gives no number. This is a per-fold sanity check on the split. It
   is not the abstention threshold of the specification (section 11,
   `insufficient_control_data`), which applies per scored segment and belongs to
   the decision layer. On Hillstrom each train fold has 14 to 15 control-arm
   conversions (14, 14, 15, 15, 15, the same under both comparisons), and the
   validation split has 24, so 10 is the largest round number that still passes
   with a small margin. A higher value would fail on the real data. On Criteo each
   train fold has 487 to 488.
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
7. **`row_id`** is the 0-based position in the raw file. No row is dropped when
   the data is loaded, and the filter and the subsample happen after the split,
   so `row_id` always refers to the raw file.
8. **Criteo (risk R08).** `scripts/criteo_to_parquet.py` converts the CSV.gz once,
   in chunks of 1,000,000 rows, to Parquet with `float32` features and `int8`
   flags. `exposure` is checked in the schema but never enters `X`. A stratified
   subsample (`subsample_frac`) exists for iteration. It is drawn inside each split
   of the full-data split (train stratified on cell and fold), so the subsample
   splits are subsets of the full-data splits and a model chosen on the subsample
   never touches a row of the full-data locked test. Reported numbers use the full
   data. float32 keeps about 7 significant digits, which is acceptable for
   anonymised features.

## Evidence

| Run | Result |
|---|---|
| Hillstrom, mens vs control | train 25,568 / validation 8,522 / test 8,523 |
| Hillstrom, womens vs control | train 25,616 / validation 8,539 / test 8,538 |
| Hillstrom, control rows in a different split between the two comparisons | 0 of 21,306 (4,280 in the first version) |
| Hillstrom, control conversions per train fold (both comparisons) | 14, 14, 15, 15, 15; validation 24 |
| Criteo, full data | 13,979,592 rows; train 8,387,755 / validation 2,795,918 |
| Criteo, full data, cost | 54 s; peak memory footprint about 4.4 GB (measured after exact-duplicate counting was added) |
| Criteo, 10% subsample drawn inside each split | train 838,775 / validation 279,591 / test 279,591 (1,397,957 rows); worst gap 0.00000; 49 control conversions per train fold |
The Criteo full-data split uses the treatment as the arm and drops no rows, so it
is the same split as in the first version. Tests cover the two properties above:
every control row keeps its split across the two Hillstrom comparisons, and every
subsample split is a subset of the same split on the full data.

The test split stays locked after `load_splits`.

## Consequences

- R08: the full run fits in about 4.4 GB on the development machine, so chunked
  preprocessing beyond the Parquet conversion is not needed for the split.
- The cell-balance assertions run on the full split (6 cells for Hillstrom). The
  control-event check runs on the rows left after the comparison filter.
- The Hillstrom control has only about 120 conversions in total, so interval
  widths on Hillstrom conversion will be large.
- If an ID-bearing Criteo file becomes available, the split must be redone
  grouped on that ID (see `docs/unit-of-observation.md`).
- The Criteo Parquet stores features as float32 (R08). Exact-duplicate counts taken from it (1,655,747 on the full file) are inflated relative to the float64 CSV count in `docs/unit-of-observation.md` (1,259,545). On the first 2,000,000 CSV rows the counts are 102,958 (float64) and 109,422 (float32). Duplicates are only documented, never removed, so the split is unaffected. Reports cite the T17 figure.