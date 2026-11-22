# ADR 0005: Tuning the response-model baseline (#107)
Status: accepted. Owner: @mohamedshrief44. Reviewer: @Ali-Alaa-Salah.
Decided: 2026-11-04 (original tuning protocol, recorded before the first trial in commit `9ce5f92`, then ADR 0004).

K1 inputs confirmed: 2026-11-22.

## Context

The T38 response model uses fixed, unregularised settings. On Hillstrom conversion its log loss (0.0771) is worse than a constant base-rate predictor (0.0669; paired bootstrap interval of the difference [0.0055, 0.0153]). K1 (T44) is computed on Criteo conversion, so the baseline must be tuned there, with Hillstrom visit and conversion as the sensitivity leg (Spec §25, §26).

## Decision

- Data: Criteo conversion is primary; Hillstrom visit and conversion are sensitivity.
- Model: LightGBM binary classifier fitted on treated rows only, P(Y=1 | X, T=1).
- Tuning data: the train split only, using its stratified five-fold ids (`train.fold`). Validation and test are never used for selection. On Criteo the search runs on a stratified 10% subsample drawn inside each split (`subsample_frac=0.1`); the final fit and every reported number use the full splits.
- Primary objective, recorded before the first trial: minimise out-of-fold log loss on treated train rows.
- Sensitivity objective: maximise out-of-fold Qini coefficient (scikit-uplift, `negative_effect=True`) over all train rows. Both candidates are reported. The original export picks follow train OOF Qini, with lower treated OOF log loss as the tie-breaker.
- Search space: num_leaves, min_child_samples, learning_rate, n_estimators, reg_lambda, colsample_bytree. No early stopping.
- Optimiser: Optuna TPE with a fixed seed. Each objective had 30 trials per dataset and target, for 60 completed response trials per dataset and target.
- Final fit: best parameters refitted on all treated rows of the full train split; scores for the full validation split are saved with the T38 artifact schema.
- Reference: a constant base-rate predictor (log loss, and AUC 0.5) is reported next to the tuned model with an interval.
- Manifest fields: tuned=true, n_trials, test_split_opened=false. T44 refuses response scores without them.

## Consequences

The primary K1 comparison on Criteo conversion uses response run `63e214a596574d16a67b467d8b272ebd` (treated OOF log loss) and T57's T-learner run `b2a649d9dc764345813be16ef7276dfb`. The response Qini run `bc1679d231f34028a622f2e50ffbeeec` and the T57 X-learner run `2bd8bff3202f4d31aa33c98855f8acd5` are sensitivity inputs.

Each primary input comes from a 30-trial search. T57 selected the T-learner over the X-learner by train OOF Qini after both families completed 30 trials. The searches use the shared LightGBM space. The response search used five folds and treated OOF log loss. T57 used three folds and OOF Qini. T44 aligns the fitted scores by `row_id` and evaluates the same targeting budget with paired within-arm bootstrap samples on validation. See the [T57 handoff](https://github.com/mahmoud-nadeem/netlift/issues/56#issuecomment-7080523697).

K1 results stay preliminary until T44 computes the paired intervals with T39.

### Response-metric intervals and resource use

The response model AUC uses a 95% normal DeLong interval, clipped to [0, 1], with midranks for ties. A constant score has AUC 0.5 and the degenerate interval [0.5, 0.5]; its null SE is a separate diagnostic. The log-loss reference uses the observed treated-validation rate as an oracle diagnostic, not as a deployable model; its paired bootstrap holds that constant fixed (2,000 resamples, run seed). These are response metrics, not substitutes for T39 uplift intervals.

Load full splits once, draw the tuning sample inside train on the canonical treatment/conversion/fold key, then refit on full treated train and score full validation. Test stays locked.
