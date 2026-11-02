# T38 preliminary baseline results — 1 November 2026

These are real Hillstrom validation point estimates, **not headline KPIs**.
No confidence intervals have been computed for the uplift metrics here; T39 must provide paired,
within-arm bootstrap intervals before comparative claims are made.
T37 (#103) is merged; the data code and configs are unchanged between the base these runs used (`01cd6be`) and `main`. Re-run if the final split or randomization changes.

- Data: original Hillstrom CSV, 64,000 rows, SHA-256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.
- Comparison: Mens E-Mail versus No E-Mail; no pooling.
- Canonical full-file split from T37; 25,568 train / 8,522 validation rows in this comparison.
- Code commit: `abd24c52f301b1c092a4ed5dbec8292cf0fc3ca0`, the `git_commit` tag of all six runs (`git_dirty=false`); upstream base `01cd6be`. Later commits on this branch change documentation only.
- Six successful MLflow runs, all `git_dirty=false`; test split never opened.
- Python 3.12.10 on Windows 11; all 18 runtime pins verified. scikit-uplift 0.5.1,
  LightGBM 4.7.0, numpy 2.4.6; seed 42. Fixed settings in `config/baselines.yaml`.
- Qini uses `negative_effect=True`; AUUC is sklift's normalized `uplift_auc_score`;
  uplift@k uses `overall`. Rate differences below are probability units.

| Target | Baseline | Qini | AUUC | Uplift@10% | Uplift@20% | Uplift@30% |
|---|---|---:|---:|---:|---:|---:|
| visit | random | 0.017559 | 0.009566 | 0.082435 | 0.083409 | 0.092202 |
| visit | blanket | — | — | — | — | — |
| visit | response-model | 0.004303 | 0.002749 | 0.088833 | 0.083386 | 0.073392 |
| conversion | random | -0.056176 | -0.001992 | 0.005437 | -0.000050 | 0.004609 |
| conversion | blanket | — | — | — | — | — |
| conversion | response-model | -0.005677 | -0.000107 | -0.000176 | 0.005556 | 0.007712 |

Blanket selects all 8,522 validation rows. Its visit average effect is
0.078150669 (7.815067 percentage points); conversion average effect is
0.006805914 (0.680591 percentage points). At the explicitly illustrative cost
of one currency unit per contact, its total cost is 8,522 units. This is not
an actual campaign-cost claim. Blanket has no meaningful ranking Qini/AUUC.

The response model's treated-validation ROC-AUC / log loss are
0.585873391 / 0.482253320 for visit, and 0.546766267 / 0.077108692 for conversion.
These classic metrics describe response prediction, not uplift quality.

Response model against a constant base-rate predictor, treated validation rows only. Computed from the saved `validation_scores.csv` and `validation_outcomes.csv` of the two response-model runs with a one-off script (2,000 resamples of the treated validation rows, seed 0), until #107 produces it in the pipeline.

| Target | Treated positives | Base rate | Constant log loss | Model log loss | Model minus constant, 95% CI | Model AUC | AUC null SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| visit | 803 / 4,261 | 0.18845 | 0.4840 | 0.4823 | -0.0017 [-0.0083, 0.0045] | 0.586 | 0.0113 |
| conversion | 53 / 4,261 | 0.01244 | 0.0669 | 0.0771 | +0.0102 [0.0055, 0.0153] | 0.547 | 0.0399 |

On conversion, the response ranking is not distinguishable from random on this validation split (AUC about 1.2 null SE above 0.5), and its log loss is worse than a constant predictor: the interval excludes zero. On visit, the ranking carries signal (AUC about 7.6 null SE above 0.5), but its log loss is not distinguishable from the constant predictor: the interval includes zero.

Random's conversion Qini is -0.056176 for the pre-fixed seed. This is a
finite-sample result with very few control events, not a reason to choose
another seed. No winner is claimed from the table without uncertainty.
The response model was not retuned after inspecting validation.

Artifacts saved for every run: validation scores, aligned outcomes/treatment,
manifest, and MLflow provenance. Response runs additionally save the fitted
model. See `docs/t38-baselines.md` for the T39 handoff and execution instructions.

Validation: 169 tests passed (including 10 T38 tests); lint and formatting
passed; mypy passed on 30 source files using a fresh cache; all 23 environment
checks passed. An earlier local mypy cache was corrupt; rebuilding in a fresh
cache fixed it without changing code, pins, or type-checking configuration.
