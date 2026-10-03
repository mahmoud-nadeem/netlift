# 0001 – Uplift Library

Status: accepted
Date: 2026-09-30
Deciders: Mohammed, Mahmoud

## Context

NetLift's M2 scope (baselines, S-Learner, T-Learner, X-Learner, Class Transformation)
and the Advanced tier in the specification (Uplift Random Forest with KL / Euclidean /
χ² split criteria) require causal/uplift-specific tooling that scikit-learn does not
provide. `requirements.txt` already pins `scikit-uplift==0.5.1` and `causalml==0.17.0`
together — both were resolved and are installed. This record decides what each library
is actually used for before either is built on, and, specifically, which one produces
the Qini/AUUC/uplift@k numbers that get reported, so the team is not comparing figures
computed two different ways in week 14.

## Options considered

### scikit-uplift (sklift) 0.5.1

**Estimators.** `SoloModel` (S-learner) and `TwoModels` (T-learner). `ClassTransformation`
/ `ClassTransformationReg` implement the class-transformation approach the spec flags as
only valid in its propensity-weighted form. No native R-learner, DR-learner, uplift
tree, or uplift forest. Verified directly against the installed 0.5.1 package's own
docstring (`from sklift.models import TwoModels; help(TwoModels)`), not a web search
result: `TwoModels` is documented as "aka naïve approach, or difference score method,
or double classifier approach", with a `method` argument offering `'vanilla'` (two fully
independent models — the classic T-learner), `'ddr_control'`, and `'ddr_treatment'`
(dependent data representation — the control/treatment model's prediction is fed as an
extra feature into the other). None of these three is called "X-learner" anywhere in
the installed package's source. An earlier draft of this note claimed sklift's own docs
label `TwoModels` "aka X-learner" — that came from a web search hit on a documentation
page, not from the installed package, and does not hold up: it has been removed. The
`ddr_*` variants are a different idea from causalml's X-learner (no propensity-score
weighting step), so they should not be presented as equivalent to it in the final
report.

**Metrics.** `qini_auc_score`, `qini_curve`, `perfect_qini_curve`, `uplift_auc_score`,
`uplift_curve`, `uplift_at_k`, `weighted_average_uplift` — a flat-array API,
`(y_true, uplift, treatment)`, one model at a time. Confirmed by reading the docstring:
`qini_auc_score` normalizes the model's curve area against the area of the *perfect/
optimal* Qini curve — this is the classic Radcliffe (2007) Qini coefficient, the same
definition the specification cites in Section 27.2.

**Treatment encoding.** Flat 1-D arrays; treatment assumed binary 0/1; no explicit
control-group label required.

**Docs/maintenance.** Maintained by Maksim Shevchenko (`maks-sh/scikit-uplift`), MIT
license, docs at uplift-modeling.com / readthedocs, describes itself as actively
maintained. Light install footprint (42 KB wheel, few dependencies).

### causalml 0.17.0

**Estimators.** Full S/T/X/R/DR meta-learner family confirmed by direct introspection
of `causalml.inference.meta` — `BaseSLearner/Classifier/Regressor`,
`BaseTLearner/.../Regressor`, `BaseXLearner/.../Regressor`, `BaseRLearner/.../Regressor`,
`BaseDRLearner/.../Regressor` — not third-party wrappers, native classes. Also
`UpliftTreeClassifier` and `UpliftRandomForestClassifier`, whose `evaluationFunction`
parameter natively supports `'KL'`, `'ED'` (Euclidean), `'Chi'`, plus `'CTS'`, `'DDP'`,
`'IT'`, `'CIT'`, `'IDDP'` — directly covers, and exceeds, the spec's KL/Euclidean/Chi²
requirement for the Advanced-tier uplift forest.

**Metrics.** `qini_score`, `get_qini`, `auuc_score`, `get_cumgain`, `get_cumlift`, plus
formal sensitivity-analysis tools (`SensitivityPlaceboTreatment`,
`SensitivityRandomCause`, etc.) sklift does not have. DataFrame-based API that can score
several models' prediction columns against one shared outcome/treatment pair in a single
call — a genuine convenience sklift lacks. One real API trap found while wiring this up:
`treatment_effect_col` does **not** mean "this model's predicted uplift" — it means the
*true/ground-truth* effect, used only for synthetic-data validation. A model's score
must be passed as a separate, arbitrarily-named column; passing it via
`treatment_effect_col` instead silently empties the internal `model_names` list and
raises `ValueError: No objects to concatenate`.

**Treatment encoding.** Meta-learners default to `control_name=0` (0/1, like sklift) but
accept arbitrary group labels for multi-treatment. `UpliftTreeClassifier` and
`UpliftRandomForestClassifier` *require* an explicit `control_name` string/label
argument at construction — an inconsistency inside causalml's own API worth documenting
for the team rather than discovering per-estimator.

**Docs/maintenance.** Maintained by Uber (`uber/causalml`), 4.7M+ PyPI downloads,
actively released (0.16.0 in Feb 2026; the pinned 0.17.0 is newer still). Much heavier
install footprint — xgboost, lightgbm, shap, numba, statsmodels, several hundred MB of
transitive dependencies. causalml's own `xgboost` requirement is unconstrained (no
pinned version), so installing it does not fight the team's `xgboost==3.2.0` pin when
both are installed together from `requirements.txt` — confirmed by inspecting
causalml's declared requirements. (Installing causalml standalone, outside the pinned
set, resolves xgboost to 3.4.1 instead — a reminder to always install from
`requirements.txt`, never `pip install causalml` alone.)

### Mandatory concrete test — Qini computed by both libraries on the same predictions

Built a randomized synthetic dataset (n = 4,000, 50/50 treatment split), fit a real
T-learner (two `LogisticRegression` models, one per arm), and produced one shared
`uplift_pred` array. Same `y`, `treatment`, and `uplift_pred` were then scored by both
libraries:

| Metric | Value |
|---|---|
| `sklift.metrics.qini_auc_score(y, uplift_pred, treatment)` | **0.2185** |
| `causalml.metrics.qini_score(df, normalize=False)` | **43.56** (raw, sample-size-dependent) |
| `causalml.metrics.qini_score(df, normalize=True)` | **0.6542** |

**They do not match, in either raw or "normalized" form — by design, not by bug.**
`normalize` means two different operations in the two libraries:

- sklift's `qini_auc_score` divides the model's curve area by the area of the
  *perfect/optimal* Qini curve — bounded, relative-to-best-possible, the Radcliffe
  Qini coefficient the spec cites.
- causalml's `qini_score(normalize=True)` only rescales the curve's y-axis to a 0–1
  range; it never divides by an optimal-curve benchmark.

A sanity check confirms both are at least directionally consistent — scoring an
uninformative random array gave sklift ≈ −0.014 and causalml(norm) ≈ −0.041 (both
near zero, as expected), against 0.2185 and 0.6542 for the real fitted model — so
neither implementation is wrong, but **the two numbers are not interchangeable and
must never both be called "the Qini coefficient" in the same report.**

## Decision

1. **Primary estimator library: causalml.** It is the only one of the two with native
   X-learner, R-learner, DR-learner, uplift trees and uplift forests — all required by
   M2's scope and the spec's Advanced tier. Use it for X/R/DR-learners and for the
   uplift forest.
2. **sklift used for S-learner and T-learner baselines** (`SoloModel`, `TwoModels`) —
   lightweight, scikit-learn-pipeline-native, and sufficient for the MVP baselines.
3. **sklift is the source of truth for every reported Qini/AUUC/uplift@k number**,
   regardless of which library trained the underlying model. Because its
   `qini_auc_score` implements the perfect-curve-normalized definition the
   specification itself cites (Radcliffe 2007, Section 27.2), it is the one comparable,
   bounded metric across every estimator. Workflow: whatever produces a model's 1-D
   uplift-score array — sklift or causalml — that array is always passed into
   `sklift.metrics.qini_auc_score` / `uplift_at_k` for the number that goes into MLflow
   and the final report. causalml's `qini_score`/`auuc_score` are used only as an
   internal, during-development cross-check, never quoted as the reported figure.
4. **Treatment encoding convention for all NetLift code:** treatment is always `int`
   0/1. For `UpliftTreeClassifier` / `UpliftRandomForestClassifier`, pass
   `control_name=0` explicitly at construction rather than relying on a default, since
   causalml has no consistent default across its own API.

## Consequences

**Benefits.** Full estimator coverage for the whole M2 roadmap and the Advanced-tier
uplift forest; one spec-aligned, comparable metric definition regardless of who trained
what or with which library.

**Limitations / trade-offs.** Two libraries stay installed, and causalml's dependency
footprint (xgboost, lightgbm, shap, numba, statsmodels) is heavy. The team must follow
the rule that a causalml-side `qini_score`/`auuc_score` number is never quoted directly
in the report — only the sklift-side number is reported. Always install from
`requirements.txt`, not `pip install causalml` standalone, to keep the pinned
`xgboost==3.2.0` in effect.

**Going forward.** Any new estimator anyone adds must expose a plain 1-D uplift-score
array so it can be scored uniformly through sklift. `control_name` must always be
passed explicitly to causalml's tree/forest classes.

## What would change this decision

- Multi-treatment work on Hillstrom's two email arms (spec Section 21, Advanced tier)
  needing estimator-level support beyond what causalml already offers.
- sklift adding native X/R/DR-learner or uplift-forest support in a future release,
  which would remove the reason to keep two libraries.
- Installing the full pinned `requirements.txt` together ever actually breaking on the
  xgboost pin (not observed in this test — confirmed unconstrained on causalml's side —
  but worth re-checking if either package's pin is bumped later).
- CI or onboarding time being measurably hurt by causalml's larger footprint, which
  would be grounds to scope causalml down to X/R/DR/forests only and keep every other
  workflow on sklift.