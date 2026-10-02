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

**Not yet checked:** whether rows within a repeated feature profile share identical
outcomes. Planned as part of the sensitivity check.

### Interpretation

The 12.83% of rows sharing a feature profile are DIFFERENT
customers who happen to have identical covariates. This is
expected given the low-cardinality feature space. It is NOT
evidence of the same customer appearing multiple times.

Repeated-profile rows have median history = 29.99 vs 158.11
for all rows, confirming that low-history (new) customers
dominate the repeated profiles.

### Can the same unit appear under both treatment and control?
Cannot be verified: there is no ID column. If one row is one user, as documented,
each user is in one arm. 356,008 feature profiles appear under both arms, and 20.11%
of rows sit inside repeated profiles. This is consistent with different users sharing
low-cardinality features, but it does not rule out repeated units.

### Consequence
Random stratified split on (treatment x outcome) is taken under 
the assumption that one row = one customer (as documented). 
No grouping required. If that assumption is later falsified, 
this decision must be revisited.

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

The exposure column is a discrepancy with Specification Section 6,
which states "neither dataset provides an exposure indicator." See the
Criteo exposure column section below for the decision (MVP estimates
ITT only; ToT is an optional Advanced-tier extension).

### Can the same unit appear under both treatment and control?
Cannot be verified directly: there is no ID column. The dataset 
documentation describes one row per user (Diemert et al., 2018), 
which would imply each user is in one arm — but this cannot be 
confirmed from the data. The 20.11% of rows sharing a feature 
profile and the 356,008 profiles appearing across arms are 
consistent with different users sharing low-cardinality features; 
they do not rule out repeated units.

### Consequence
- Random stratified split on (treatment x outcome) is acceptable
  under the published design, but must be documented as an
  assumption.
- The exposure column discovery is addressed in the Criteo exposure
  column section below: the MVP estimates ITT only.
- Risk: if users appear multiple times, random split leaks user
  identity across train/test.

## Final Decision

| Dataset | Unit | Split Strategy | Group Split Needed? |
|---|---|---|---|
| Hillstrom | Customer (per published count) | Random stratified (treatment x outcome) | No (assumption) |
| Criteo | User (per documentation) | Random stratified (treatment x outcome) | No (assumption, stress-tested: see Sensitivity check) |

**Decision:** random stratified split on (treatment × outcome) for both datasets,
taken under the assumption that one row = one independent unit.

## Criteo exposure column

`exposure` exists in Criteo (428,212 of 11,882,653 treated rows, about 3.6%) and is 0
for every control row. This contradicts Specification Section 6.

- `exposure` is post-treatment: it MUST be on the forbidden-column list and never
  enter the feature matrix (Sections 18 and 23).
- Comparing exposed vs control directly is confounded, because exposure is not random.
  A valid ToT estimate is the Wald/IV ratio ITT / P(exposure=1 | treated), which is
  justified by one-sided non-compliance (no control row is exposed).
- Compliance is about 3.6%, so the ITT effect is heavily diluted relative to ToT.

Decision: the MVP estimates ITT only. ToT via the Wald ratio is an optional
Advanced-tier extension, not required for the MVP.

## Verification Commands

See scripts/t17_unit_check.py and scripts/t17_dup_diagnostics.py.

## References
- Specification Section 2.2 (fundamental problem)
- Specification Section 6 (treatment compliance)
- Specification Section 19 (splitting strategy)
- Criteo dataset documentation (Diemert et al., 2018)
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

If the same Criteo user appears in several rows, a random row-wise split puts that user
on both sides, and the measured uplift is inflated in a way no metric reveals.
Bootstrap intervals quantify sampling uncertainty only. They do NOT detect this leak.

**Sensitivity check (follow-up task, owner: Aliaa, due: before T18 starts):**
A true grouped split is impossible without an ID, but a conservative proxy exists:
group rows by the hash of the full feature vector and split with
StratifiedGroupKFold (not plain GroupKFold), so that the (treatment x outcome)
stratification from Specification Section 19 is preserved per fold, while
identical profiles never straddle the boundary. Train the same model twice (random split
vs profile-grouped split) and compare Qini with bootstrap CIs. If they are close, repeated
profiles are not driving the result. If the grouped Qini is clearly lower, this decision
must be revisited. The proxy over-groups (different users with identical features are
kept together), so it is a stress test, not the final split.

If an ID-bearing version of Criteo becomes available, redo the split grouped on that ID.

**Mitigation:**

1. The assumption is documented here and will be repeated in the final report.
2. A grouped split is not possible without an ID column.
3. If an ID-bearing version of Criteo becomes available, the split must be redone with grouping on that ID.
4. All reported metrics carry bootstrap confidence intervals 
   (Specification Section 28). These intervals quantify sampling 
   uncertainty in the estimate. They do NOT detect or correct 
   the leakage risk described above, which is structural: if the 
   same unit appears on both sides of the split, every resample 
   inherits the same inflation, and the intervals remain too 
   narrow.
---
## Duplicate handling policy (input to T18)

Exact duplicate rows (Hillstrom: 6,562; Criteo: 1,259,545) are documented, not
auto-removed. Without an ID they are more likely different units with identical
covariates than repeated units, and dropping them would discard real observations
and could shift the treatment/control ratio. Specification Section 23 lists
"duplicate detection and removal"; this document recommends changing that step to
"detect, document, decide". The removal decision belongs to the team.

---

## Specification errors found during T17 (corrected)

Two claims in `NetLift_Project_Specification.pdf` were found to
contradict the data:

1. **Section 6** stated "neither dataset provides an exposure
   indicator." Criteo does contain one (428,212 of 11,882,653 treated
   rows, about 3.6%, all control rows = 0), which enables ToT
   estimation via the Wald/IV ratio -- contrary to the Spec's
   assumption. See the Criteo exposure column section above.
2. **Section 27** stated bootstrap confidence intervals make leakage
   "at least partially visible in the interval width." This was
   incorrect: bootstrap CIs quantify sampling uncertainty only and do
   not detect split-level leakage (see Risk and mitigation above).

Section 6 was corrected in the project specification after this
finding. The current specification now documents the Criteo exposure
column and the ITT MVP / optional ToT decision. Section 27's bootstrap
claim is addressed by the Risk and mitigation section above, which
states explicitly that bootstrap CIs do not detect split-level
leakage.

---

## Downstream consequence

Everything in the modeling pipeline depends on this decision:

- Preprocessing is fitted on the training fold only.
- The joint stratification on (treatment × outcome) preserves arm proportions and event rates in every fold.
- The random-split decision for Criteo is taken under the explicit assumption stated above. If that assumption is ever falsified, T17 must be revisited before any reported model result is trusted.
