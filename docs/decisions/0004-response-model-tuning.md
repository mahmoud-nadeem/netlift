# ADR 0004: Tuning the response-model baseline (#107)

Status: proposed. Owner: @mohamedshrief44. Reviewer: @Ali-Alaa-Salah.

## Context

The T38 response model uses fixed, unregularised settings. On Hillstrom conversion its log loss (0.0771) is worse than a constant base-rate predictor (0.0669; paired bootstrap interval of the difference [0.0055, 0.0153]). K1 (T44) is computed on Criteo conversion, so the baseline must be tuned there, with Hillstrom visit and conversion as the sensitivity leg (Spec §25, §26).

## Decision

- Data: Criteo conversion is primary; Hillstrom visit and conversion are sensitivity.
- Model: LightGBM binary classifier fitted on treated rows only, P(Y=1 | X, T=1).
- Tuning data: the train split only, using its stratified five-fold ids (`train.fold`). Validation and test are never used for selection. On Criteo the search runs on a stratified 10% subsample drawn inside each split (`subsample_frac=0.1`); the final fit and every reported number use the full splits.
- Primary objective, recorded before the first trial: minimise out-of-fold log loss on treated train rows.
- Sensitivity objective: maximise out-of-fold Qini coefficient (scikit-uplift, `negative_effect=True`) over all train rows. Both candidates are reported. For the K1 handoff, compare them by OOF Qini on the same train rows; higher wins, with lower treated OOF log loss as the tie-breaker. Validation results never select the candidate. Log loss remains the primary search objective and Qini the sensitivity search objective.
- Search space: num_leaves, min_child_samples, learning_rate, n_estimators, reg_lambda, colsample_bytree. No early stopping.
- Optimiser: Optuna TPE with a fixed seed. Trial budget: 30 per model, dataset and target. The T42 learners use the same budget and search-space template, and every trial is logged to MLflow.
- Final fit: best parameters refitted on all treated rows of the full train split; scores for the full validation split are saved with the T38 artifact schema.
- Reference: a constant base-rate predictor (log loss, and AUC 0.5) is reported next to the tuned model with an interval.
- Manifest fields: tuned=true, n_trials, test_split_opened=false. T44 refuses response scores without them.

## Consequences

K1 uses the candidate selected by the predeclared train OOF Qini rule above. If the tuned model does not beat the constant predictor, we report that. Results stay `result_status=preliminary` until T39 intervals exist.

### Response-metric intervals and resource use

The response model AUC uses a 95% normal DeLong interval, clipped to [0, 1], with midranks for ties. A constant score has AUC 0.5 and the degenerate interval [0.5, 0.5]; its null SE is a separate diagnostic. The log-loss reference uses the observed treated-validation rate as an oracle diagnostic, not as a deployable model; its paired bootstrap holds that constant fixed (2,000 resamples, run seed). These are response metrics, not substitutes for T39 uplift intervals.

Load full splits once, draw the tuning sample inside train on the canonical treatment/conversion/fold key, then refit on full treated train and score full validation. Test stays locked.

The two search objectives have separate 30-trial budgets. Record both counts if both are run. T42 must use the agreed shared budget and search-space template; parity is not established by this file alone.
