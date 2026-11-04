# ADR 0004: Tuning the response-model baseline (#107)

Status: proposed. Owner: @mohamedshrief44. Reviewer: @Ali-Alaa-Salah.

## Context

The T38 response model uses fixed, unregularised settings. On Hillstrom conversion its log loss (0.0771) is worse than a constant base-rate predictor (0.0669; paired bootstrap interval of the difference [0.0055, 0.0153]). K1 (T44) is computed on Criteo conversion, so the baseline must be tuned there, with Hillstrom visit and conversion as the sensitivity leg (Spec §25, §26).

## Decision

- Data: Criteo conversion is primary; Hillstrom visit and conversion are sensitivity.
- Model: LightGBM binary classifier fitted on treated rows only, P(Y=1 | X, T=1).
- Tuning data: the train split only, using its stratified five-fold ids (`train.fold`). Validation and test are never used for selection. On Criteo the search runs on a stratified 10% subsample drawn inside each split (`subsample_frac=0.1`); the final fit and every reported number use the full splits.
- Primary objective, recorded before the first trial: minimise out-of-fold log loss on treated train rows.
- Sensitivity objective: maximise out-of-fold Qini coefficient (scikit-uplift, `negative_effect=True`) over all train rows. It is reported next to the primary model and not used to pick the primary baseline.
- Search space: num_leaves, min_child_samples, learning_rate, n_estimators, reg_lambda, colsample_bytree. No early stopping.
- Optimiser: Optuna TPE with a fixed seed. Trial budget: 30 per model, dataset and target. The T42 learners use the same budget and search-space template, and every trial is logged to MLflow.
- Final fit: best parameters refitted on all treated rows of the full train split; scores for the full validation split are saved with the T38 artifact schema.
- Reference: a constant base-rate predictor (log loss, and AUC 0.5) is reported next to the tuned model with an interval.
- Manifest fields: tuned=true, n_trials, test_split_opened=false. T44 refuses response scores without them.

## Consequences

K1 is reported against the stronger of the primary and sensitivity baselines by out-of-fold score. If the tuned model does not beat the constant predictor, we report that. Results stay `result_status=preliminary` until T39 intervals exist.