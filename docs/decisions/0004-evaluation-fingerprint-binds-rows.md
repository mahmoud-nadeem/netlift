# 0004 – The evaluation fingerprint binds the dataset and the row ids

Status: proposed
Date: 2026-11-13
Deciders: Mahmoud (proposer, T41); Ali (producer of interface 2, T39) and Mazen (reviewer) approve
in review of #104

## Context

Interface 2 was frozen when #92 merged (`00c937f`). Its change rule says any later change needs a
record here. `paired_difference` pairs two results resample by resample, and the resamples are
positional: resample *b* holds the same index positions for both results. The evaluation
fingerprint is what proves the two results were computed on the same evaluation set.

As frozen, `fingerprint_evaluation_set(y, t)` hashed only `y` and `t`. Mazen's review of #104
(4 Nov, P1) pointed out the gap. Two evaluation sets with identical outcome and treatment arrays
but different rows, or the same rows in another order, got the same fingerprint. The comparison
would then subtract values computed on different observations, and its interval would mean
nothing.

## Decision

`fingerprint_evaluation_set(y, t, *, row_ids, dataset)` hashes the dataset name, the ordered row
ids (interface 1's `row_id`), `y` and `t`.

- `row_ids` and `dataset` are required keywords.
- These inputs are rejected: a length mismatch, an empty set, negative or non-integer row ids, and
  a dataset outside `DatasetName`.
- The dataset name is hashed because a row id is a position in one dataset's raw file, so ids
  alone are not unique across datasets.

`MetricResult` and `paired_difference` are unchanged. Only the function that produces the
fingerprint changes.

## Consequences

- Every caller passes the matrix's `row_id` and its `meta.dataset`. The harness (T39, #111) is the
  main one, and it updates after #104 merges.
- Fingerprints computed before this change are not comparable with new ones. No KPI has been
  reported yet (K1–K3 are first reported in M4), so any result logged with the old fingerprint is
  re-run with the harness rather than paired.
- Tests cover different rows with the same `y` and `t`, the same rows in another order, identical
  inputs, another dataset name, and a length mismatch.
