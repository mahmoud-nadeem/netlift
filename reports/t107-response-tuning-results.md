# T107 tuned response-model results — 5 November 2026

Owner: @mohamedshrief44. Reviewer: @Ali-Alaa-Salah.

Criteo conversion is the primary K1 dataset; Hillstrom visit and conversion are sensitivity analyses. These are validation results from response classifiers, not a completed uplift-versus-response comparison or a K1 conclusion. Uplift metrics remain preliminary pending T39 paired, within-arm bootstrap intervals.

## Provenance and method

All six exported runs are FINISHED, with `git_dirty=false`, `test_split_opened=false`, seed 42 and 30 trials each. The supplied four CSV files agree with their MLflow metric and best-parameter records. This verifies exported records; the prediction artifacts still require the handoff export checks.

Source commit: `e92b90b6f58bced4def2812a7f66beb94ab86c05` (branch `t107-tune-response-model`). The source remains the training reference when this report is committed later. Runtime reported by the operator: Python 3.12.10 on Windows 11. Runtime library versions are retained in each original run manifest; no separate runtime-pin audit is claimed here.

LightGBM classifiers fit treated train rows only. Both searches use the existing five train folds. Criteo tuning uses the stratified 10% train sample; Hillstrom uses full train. Final fits use full treated train and score full validation. Criteo OOF values describe the tuning sample, not full train. Validation never selects a candidate; test stays locked.

There are two searches per dataset/target: primary treated OOF log loss and sensitivity OOF Qini, each with 30 trials (60 per dataset/target, 180 total). Trial details are saved in each run's trials.csv, rather than 180 separate MLflow runs. T42 budget/search-space parity requires separate confirmation.

Selection follows the rule recorded in ADR 0004 at the source commit: higher train OOF Qini, with lower treated OOF log loss breaking a tie. ADR 0004 is marked proposed; this report does not claim reviewer approval. Both candidates are retained for review.

## Candidate comparison and selection

| Dataset | Target | Search objective | OOF Qini | OOF log loss | Selected |
| --- | --- | --- | --- | --- | --- |
| criteo | conversion | log_loss | 0.189436 | 0.012349 | false |
| criteo | conversion | qini | 0.190856 | 0.012477 | true |
| hillstrom | visit | log_loss | 0.022500 | 0.459870 | false |
| hillstrom | conversion | log_loss | 0.045050 | 0.066825 | true |
| hillstrom | visit | qini | 0.022660 | 0.459901 | true |
| hillstrom | conversion | qini | 0.033244 | 0.069970 | false |

Small OOF differences, particularly Hillstrom visit, are not claims of statistically significant superiority. Criteo's log-loss search has lower validation log loss than its Qini search, but validation is not used to change the selection.

## Selected validation uplift point estimates

scikit-uplift supplies Qini (`negative_effect=True`), normalized AUUC (`uplift_auc_score`) and uplift@k (`strategy="overall"`). Rate differences are probability units. No uplift confidence intervals have been computed by these runs (`n_resamples=0`).

| Dataset/target | Qini | AUUC | Uplift@10% | Uplift@20% | Uplift@30% |
| --- | --- | --- | --- | --- | --- |
| criteo/conversion | 0.183712 | 0.005954 | 0.007932 | 0.004938 | 0.003484 |
| hillstrom/conversion | 0.078311 | 0.002985 | 0.012319 | 0.010293 | 0.011398 |
| hillstrom/visit | 0.007239 | 0.004616 | 0.100024 | 0.080960 | 0.072979 |

## Response prediction and constant-rate comparison

| Dataset/target | Treated positives/rows | Base rate | Constant log loss | Model log loss | Model − constant (95% CI) | AUC (95% CI) | Null SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| criteo/conversion | 7342/2376531 | 0.003089 | 0.020941 | 0.012509 | -0.008432 [-0.008664, -0.008207] | 0.956677 [0.954318, 0.959037] | 0.003374 |
| hillstrom/conversion | 53/4261 | 0.012438 | 0.066928 | 0.065970 | -0.000958 [-0.001800, -0.000109] | 0.629239 [0.552662, 0.705817] | 0.039906 |
| hillstrom/visit | 803/4261 | 0.188453 | 0.483973 | 0.469796 | -0.014176 [-0.018545, -0.009748] | 0.621783 [0.600916, 0.642649] | 0.011310 |

The constant uses the observed treated-validation rate: it is an oracle diagnostic, not a deployable predictor fitted on train. Log-loss intervals use 2,000 paired percentile bootstrap resamples, seed 42, holding that reference fixed. AUC intervals use a 95% normal DeLong approximation with ties and clipping to [0,1]. Constant-score AUC is 0.5 with a degenerate [0.5,0.5] interval. Null SE is a separate diagnostic. These intervals condition on fitted scores and do not include training or selection uncertainty.

All three selected candidates have negative model-minus-constant log-loss intervals under this procedure. Hillstrom conversion's improvement is small, with only 53 treated and 24 control validation conversions; its AUC interval remains broad. Its unselected Qini-search candidate has AUC 0.543507 (95% CI [0.468378, 0.618635]) and a log-loss difference interval spanning zero. Neither response prediction performance nor a positive Qini point estimate establishes K1.

## Reproduction and handoff

See config/response_tuning.yaml and docs/decisions/0004-response-model-tuning.md at the source commit. The existing runner supports each dataset/target and objective; no retraining is needed to export these saved results. Keep the source commit reachable with an annotated tag before squash merge, including in the delivered company repository.

The companion selection JSON records all six run IDs, selected flags, OOF values and input-file SHA-256 values. The results CSV retains all candidate parameters and metrics at exported precision. Original MLflow artifacts must be delivered separately: validation_scores.csv, validation_outcomes.csv, manifest.json, trials.csv and response_model.joblib. The export utility copies these without changing run metadata and checks finite scores, unique/aligned row IDs, binary outcomes/treatment, row counts, manifest flags and matching outcomes across objectives. Its export-checks.json records file hashes and the actual checks performed.

Do not commit the prediction bundle, raw data, MLflow database or model binaries. Provide the exported bundle through the team's agreed file-sharing channel. T39/T44 must align rows by row_id, verify provenance and compute the required paired uplift intervals before conclusions. Random and blanket policies are handled in T44/T39 under the team scope decision.

Operator-reported implementation checks: 199 tests passed; Ruff lint and formatting passed; mypy passed on 32 source files. This report adds no training-code changes. Outstanding review items include T42 budget parity, ADR acceptance, successful artifact export/delivery and T39 uplift intervals.
