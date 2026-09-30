# T17: Unit of Observation

Date: 2026-09-30
Owner: Aliaa
Reviewer: Mohammed
Branch: t17-unit-of-observation

## Question

What is one row in each dataset?
- A person?
- An impression / ad view?
- A person-campaign pair?

## Hillstrom

### Answer
One row = one customer.

### Evidence

1. Rows = 64,000 (matches published customer count)
2. No ID column exists
3. 6,562 fully duplicated rows across all 12 columns
4. 420 feature profiles appear more than once
5. 380 profiles appear under more than one treatment arm
6. Feature space is small: recency has 12 unique values,
   history_segment has 7, mens/womens/newbie have 2 each

### Interpretation

The 12.83% of rows sharing a feature profile are DIFFERENT
customers who happen to have identical covariates. This is
expected given the low-cardinality feature space. It is NOT
evidence of the same customer appearing multiple times.

Repeated-profile rows have median history = 29.99 vs 158.11
for all rows, confirming that low-history (new) customers
dominate the repeated profiles.

### Can the same unit appear under both treatment and control?
No. Each customer appears in exactly one arm. The 380 profiles
appearing in multiple arms are different customers, not the
same customer.

### Consequence
Random stratified split on (treatment x outcome) is SAFE.
No grouping required.

## Criteo

### Answer
One row = one user, per the dataset documentation. This cannot be verified from the data because there is no ID column.

### Evidence

1. Rows = 13,979,592
2. No ID column exists
3. 1,259,545 fully duplicated rows
4. 1,185,455 feature profiles appear more than once
5. 2,811,714 rows (20.11%) are inside repeated profiles
6. 356,008 profiles appear under more than one treatment value
7. Treatment ratio = 0.85 (verified)
8. Conversion rate = 0.2917% (verified)
9. Visit rate = 4.6992% (verified)
10. exposure column exists with values:
    - exposure=0: 13,551,380 rows
    - exposure=1: 428,212 rows
    - All control rows have exposure=0
    - about 3.6% of treatment rows have exposure=1 (428,212 of 11,882,653)

### Interpretation

The 20.11% of rows sharing a feature profile cannot be linked
to the same user without an ID. Given the anonymized float
features, duplicates are expected.

The exposure column is a MAJOR DISCREPANCY with Specification
Section 6, which states:
  "neither dataset provides an exposure indicator"

This must be escalated for methodological review before any
ITT vs ToT decision is finalized.

### Can the same unit appear under both treatment and control?
Cannot be verified directly: there is no ID column. The 
interpretation that the 380 profiles appearing across multiple 
arms represent different customers follows from the published 
row count (64,000 rows = 64,000 customers), not from an 
identifier-based check.

### Consequence
- Random stratified split on (treatment x outcome) is acceptable
  under the published design, but must be documented as an
  assumption.
- The exposure column discovery requires a separate methodological
  decision about ITT vs ToT estimation.
- Risk: if users appear multiple times, random split leaks user
  identity across train/test.

## Final Decision

| Dataset | Unit | Split Strategy | Group Split Needed? |
|---|---|---|---|
| Hillstrom | Customer | Random stratified (treatment x outcome) | No |
| Criteo | User (per documentation) | Random stratified (treatment x outcome) | No (documented as assumption) |

## Open Issue: Criteo exposure column

The presence of exposure in Criteo contradicts Specification
Section 6. Two possible resolutions:

1. Ignore exposure and estimate ITT only (as originally planned)
2. Use exposure to estimate ToT (treatment-on-the-treated)
   for Criteo only

This requires a team decision. Flagged for Week 7 instructor review.

## Verification Commands

See scripts/t17_unit_check.py and scripts/t17_dup_diagnostics.py.

## References
- Specification Section 2.2 (fundamental problem)
- Specification Section 6 (treatment compliance)
- Specification Section 19 (splitting strategy)
- Criteo dataset documentation (Diemer et al., 2018)
- Hillstrom dataset documentation (Hillstrom, 2008)

## Terminology

Two distinct concepts must not be conflated:

**Duplicate identifiers:** the same unit (customer, user, or impression) appears in more than one row. This is a real leakage risk — a random train/test split would place the same unit on both sides.

**Duplicate feature profiles:** different units share identical covariate values. This is expected in low-cardinality feature spaces and does NOT, by itself, indicate leakage.

Because neither dataset contains an ID column, **duplicate identifiers cannot be detected directly**. All duplicate analysis in this document is analysis of duplicate feature profiles, and is interpreted accordingly.

---

## Verification limits

For both datasets, the conclusion about the unit of observation rests on published documentation and row counts, not on ID-based verification.

**Hillstrom:** 64,000 rows = 64,000 customers (published figure confirmed against the file). Confidence: High.

**Criteo:** "one user per row" per the dataset documentation (Diemert et al., 2018). Cannot be verified without an ID column. Confidence: Medium.

Specifically, for Criteo, the following cannot be verified from the files:

- Whether the same user appears in more than one row.
- Whether the same user appears in both treatment and control.
- Whether rows sharing a feature profile represent different users or the same user appearing repeatedly.

The duplicate-profile counts reported above (20.11% of rows, 356,008 profiles across arms) are consistent with different users sharing low-cardinality features, but they do not rule out repeated units.

This is a limitation of the available data. It is recorded here as an explicit assumption underlying every downstream modeling decision.

---

## Risk and mitigation

If the Criteo assumption is wrong — that is, if the same user appears in multiple rows — then a random row-wise split places that user on both sides of the split, and the measured uplift is inflated in a way that is invisible to every metric we compute. The leak is in the split, not in the features.

**Mitigation:**

1. The assumption is documented here and will be repeated in the final report.
2. A grouped split is not possible without an ID column.
3. If an ID-bearing version of Criteo becomes available, the split must be redone with grouping on that ID.
4. All reported metrics carry bootstrap confidence intervals (Specification Section 28), so any inflation caused by this assumption is at least partially visible in the interval width.

---

## Downstream consequence

Everything in the modeling pipeline depends on this decision:

- Preprocessing is fitted on the training fold only.
- The joint stratification on (treatment × outcome) preserves arm proportions and event rates in every fold.
- The random-split decision for Criteo is taken under the explicit assumption stated above. If that assumption is ever falsified, T17 must be revisited before any reported model result is trusted.
