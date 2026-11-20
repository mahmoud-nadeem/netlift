# 0006 – The uplift-model search: objective, data, families, budget and champion

Status: proposed
Date: 2026-11-20
Deciders: Mahmoud (proposer, T57); Mohammed (partner, T57) approves in review

Written before the first trial. Nothing in it was changed after a trial result was seen.

## Context

T57 tunes the uplift models and registers a champion. Spec §26 rules out tuning each base model on
its own loss: the search has to optimise the quality of the uplift ranking, and it must not use
validation, which is for reporting only. ADR 0005 (#107, PR #115) tunes the response-model baseline
with the same kind of out-of-fold protocol, so the two searches should match where they can. The
spec also asks for the number of trials per family in the final report, because a champion tuned
for 200 trials against a baseline given 5 is not a comparison.

## Decision

**Objective.** scikit-uplift's Qini coefficient (`qini_auc_score`, `negative_effect=True`, as in
ADR 0005) on out-of-fold predictions over the whole training split. The folds are stratified
3-fold on the joint (t, y) key of the target in use, shuffled with the seed. Validation and test
are never read: `search_families` takes `splits.train` only, and the objective refuses any matrix
whose split is not `train`.

- Three folds, not the five that T37 attaches to the training split: each trial costs 3 fits
  instead of 5, and the folds are stratified on the target in use (T37's are stratified on
  conversion, whatever the target).

**Data.**

- Criteo: the stratified 10% subsample inside the training split (spec §25,
  `load_splits(..., subsample_frac=0.1)`). The final refit and every reported number use the full
  splits.
- Hillstrom: the full training split.

**Families.**

- T-learner and X-learner with LightGBM base learners, searched over ADR 0005's LightGBM space so
  the response model and the meta-learners share one template. The X-learner's outcome and effect
  models share one parameter set. The definitions follow #117 (T56) exactly. The study uses #117's
  classes once #117 merges; until then `uplift_search.py` carries the same two definitions, and a
  follow-up commit swaps them for the import.
- causalml's `UpliftRandomForestClassifier`, with the split criterion (KL, ED, Chi) as a
  categorical hyperparameter, and `n_estimators`, `max_depth`, `min_samples_leaf`,
  `min_samples_treatment`, `n_reg` and `max_features` searched. The ranges are in
  `config/uplift_search.yaml`.

**Search.** Optuna TPE with seed 42 and the same `n_trials` for every family, with a timeout of 4
hours per family as a cap. Every trial is one MLflow run under the T41 conventions (experiment
`netlift-<dataset>-<target>`, run `T57-<family>-trial-<n>`, tag `task=T57`), with its parameters,
the out-of-fold Qini, each fold's Qini and the fit time.

**Champion.** The family and parameters with the best out-of-fold Qini on Criteo conversion. It is
refitted on the full training split; its validation scores come from the T39 harness, and it is
registered in the MLflow Model Registry.

**K1's uplift model (added 20 Nov, while the Criteo study was running and before any of its
results were seen).** Of the T-learner and the X-learner, the one with the higher out-of-fold Qini
in the Criteo conversion search is K1's uplift model, and the other is its sensitivity
(`config/kpis.yaml`, K1). A tie goes to the T-learner. Both are refitted with their best
parameters on the full training split, without the subsample, and score the full validation split
(`scripts/refit_k1_models.py`). The test split is not opened. K1's primary k belongs to T44's
decision record, not to this one.

## Trial budget, from a measured fit

Measured on 20 Nov 2026 on the team laptop (16 logical cores), Criteo conversion, 10% subsample
of the training split (838,775 rows, 85.0% treated, 2,450 conversions). One fold fits on 559,183
rows and predicts the other 279,592. A trial is three such folds.

| Uplift forest configuration | Fit (s) | Predict (s) | Per trial, 3 folds |
|---|---|---|---|
| Midpoint of the space: 60 trees, depth 6, leaf 447, treatment leaf 45, `n_reg` 10, 8 features, KL | 49.6 | 71.4 | about 363 s |
| Most expensive corner: 100 trees, depth 8, leaf 100, treatment leaf 10, `n_reg` 1, 12 features, KL | 167.5 | 155.6 | about 969 s |

- One trial with the LightGBM midpoint (325 trees, 16 leaves) took 18.4 s for the X-learner and
  9.7 s for the T-learner, so the forest sets the budget.
- **`n_trials` = 30 for every family.** At the midpoint cost, 30 forest trials take about
  10,890 s, about 3 hours. 30 is also ADR 0005's budget for the response model, so the uplift
  models and the baseline get the same number of trials, which is the comparison spec §26 asks
  for.
- **Timeout: 4 hours per family, as a cap.** It is above the expected 3 hours, so a run of
  expensive draws does not cut the forest short of 30 trials. If every draw hit the most
  expensive corner, 30 trials would take about 8 hours, and the cap would stop the study first.
- The timing fits for the forest printed the Qini of the held fold. Nothing in this record was
  changed after that, and those fits are not trials.
- Expected overnight run for Criteo, all three families: about 3 hours 20 minutes, at most about
  4 hours 30 minutes.

## Consequences

- If the forest's study hits the timeout before `n_trials`, the families no longer have equal
  budgets. The report then states each family's completed trial count, and the comparison says so.
- The T- and X-learners and the response model are tuned over the same LightGBM space, so their
  trial counts can be compared directly in the final report.
- Hillstrom is a sensitivity leg. A different champion family on Hillstrom is reported, not used
  to override Criteo.
