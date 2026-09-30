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

### Documentation
- Criteo AI Lab dataset page: each row represents "a user" (https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- Diemert et al., 2018 (https://arxiv.org/abs/2111.10106): the v2 dataset has about 14M rows, each representing a user.
- The same page defines `exposure` as whether the user was effectively exposed to the ad.

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
Cannot be verified: there is no ID. If one row is one user, as documented, each
user is in one arm. Feature profiles repeat in 20.11% of rows and 356,008
profiles appear under both arms. These are consistent with different users
sharing low-cardinality features, but they do not rule out repeated units.

### Consequence
- Random stratified split on (treatment x outcome) is acceptable
  if one row is one user (as documented), but must be documented as an
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
- Criteo dataset documentation (Diemert et al., 2018)
- Hillstrom dataset documentation (Hillstrom, 2008)
