# T38: mandatory baselines

Source: specification §9 (Baselines That Must Exist Before Any Uplift Model),
§25 (training), §27 (metrics), ADR 0001, and the clarified T38 issue #50.
The implementation depends on T36 and T37, as merged in #103 (`d56c29c`).
Do not copy interface or pipeline files out of the team pack.

## Run

Use the project's Python 3.12 environment and pinned dependencies, then install
its package with `python -m pip install -e .`. Place the verified Hillstrom file
at `data/raw/hillstrom.csv` (SHA-256
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`).
Set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` in the environment or `.env`.

```bash
python scripts/run_baselines.py --target both --contact-cost 1
```

Here **1 is an illustrative cost of one currency unit per contact**, not an
estimate of a real campaign cost. Replace it with the scenario's actual cost.
The assumption affects blanket total cost only; it does not change rankings or
uplift metrics. `--contact-cost` is required to prevent an unstated default.
Use `--target visit` to develop first, then `--target conversion`.
Hyperparameters and the ranking/model seed are in `config/baselines.yaml`.
The canonical split and its seed/folds remain in `config/data.yaml`.

The script uses `load_splits("hillstrom", target=...)`: Mens E-Mail versus
No E-Mail, after the full-file split. It fits on train, evaluates validation,
and never opens the locked test split. Parameters are fixed, with no validation
selection, early stopping, or tuning. Future tuning must use training CV.

## What the three baselines mean

- **Random:** a seeded uniform score for every validation row, used for ranking.
  Expected Qini is near zero across random rankings; one finite-sample draw need
  not equal zero, particularly with rare conversions. No seed is selected for
  producing a desirable validation number.
- **Blanket:** all validation customers selected. Its average effect is
  `mean(y[t == 1]) - mean(y[t == 0])`, with number selected and total cost.
  This is mathematically uplift at 100%. scikit-uplift 0.5.1 rejects `k=1.0`,
  so the all-population rate difference is computed explicitly. No Qini/AUUC is
  reported for blanket. Saved `score=1` means selected, not predicted uplift.
- **Response:** binary LightGBM fitted on treated training rows only. Scores
  all validation rows by `P(Y=1 | X,T=1)`. It receives the same uplift metrics
  as random, plus ROC-AUC and log loss on treated validation rows only.

All ranking metrics use scikit-uplift 0.5.1: perfect-curve-normalized Qini with
`negative_effect=True`, normalized `uplift_auc_score`, and uplift@10/20/30%
with `strategy="overall"`. Stable reverse-mergesort tie handling matches sklift.
Empty treatment arms in a top-k subset or non-finite metrics raise a clear error.

## MLflow and T39 handoff

Experiments: `netlift-hillstrom-visit`, `netlift-hillstrom-conversion`.
Runs: `T38-random`, `T38-blanket`, `T38-response-model`; every run has `task=T38`.
Fitting and evaluation occur inside `start_run`. The existing `tags` argument
is compatible with both the T36 and T41 tracking wrappers.

Metrics have a `preliminary_` prefix and `result_status=preliminary`.
**There are no bootstrap CIs or headline KPI claims in T38's current output.**
This implements the explicit interim permission in issue #50 until T39 lands.
Canonical split/randomization results must be final before headline reporting;
re-run after upstream changes. The test split remains closed until T70.

Each run saves:

- `validation_scores.csv`: `row_id,score`, in validation input order.
- `validation_outcomes.csv`: the matching `row_id,y,t` for evaluation.
- `manifest.json`: target, arms, split, seeds, library versions, metric settings,
  score meaning, config, cost assumption, and preliminary numbers.
- `response_model.joblib`: fitted response model, for the response run only.

The wrapper also records dataset SHA-256, git commit and dirty flag. The summary
CSV is `reports/t38-baselines-preliminary.csv` unless `--output` is provided.
Artifacts stay in MLflow; never commit the tracking database or data files.

T39 should join scores/outcomes by `row_id`, verify one-to-one alignment and
identical evaluation rows/target across policies, then compute arm-stratified
95% bootstrap intervals with at least 1,000 common resamples. Construct
`MetricResult` with evaluation/resample fingerprints, use `log_metric_result`
for reportable KPIs, and `paired_difference` for comparisons. Do not pass the
blanket indicator to a ranking metric; bootstrap its full-population rate
difference. The saved predictions mean no retraining is needed for intervals.

## Tests

`tests/test_baselines.py` uses T36 synthetic fixtures only. It checks exact
training membership, row alignment, reproducibility, forbidden columns,
validation-outcome isolation, invalid costs, single-class training, locked-test
access, and a real SQLite MLflow round trip including prediction replay from
the saved model. Synthetic test metrics are not project results.
