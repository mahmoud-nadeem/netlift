## Negative control: permuted treatment (K4, T43)

**Why.** A Qini coefficient measures how well scores rank customers by treatment
effect. If the pipeline is sound, scores computed when treatment carries no
information about the outcome must rank no better than random. The negative
control creates exactly that situation and checks that the pipeline sees nothing.
A significant uplift here means a leak or a bug, and no other test finds that
class of error as cheaply.

**Method.**

1. Load the train and validation splits (the locked test split is not opened).
2. Shuffle the treatment indicator `t` at random, separately in each split, with a
   fixed seed. Arm sizes are preserved; `y` and the features are untouched, so
   any real effect of treatment on the outcome is severed.
3. Fit the full pipeline on the permuted data: `validate_features`, then a
   T-learner (one LightGBM per permuted arm, fixed parameters, no tuning).
4. Score the validation split and compute the Qini coefficient against the
   permuted `t`, with a 95% percentile bootstrap interval resampled within
   treatment arms (1,000 resamples, seed 42), using `netlift.evaluation.metrics`
   (scikit-uplift 0.5.1, `negative_effect=True`).
5. **Pass criterion (K4):** the 95% interval contains zero.

**Result.**

| Dataset / target | Split | Qini coefficient | 95% bootstrap CI | Resamples | Seed | K4 |
|---|---|---|---|---|---|---|
| `<dataset>` / `<target>` | validation | `<value>` | [`<low>`, `<high>`] | 1,000 | 42 | `<PASS/FAIL>` |

*Fill the row from the output of
`python scripts/run_negative_control.py --dataset <dataset> --target <target> --seed 42`.
Do not edit the numbers by hand or re-run with another seed until the interval moves.*

**Reading.** Under permuted treatment the Qini coefficient collapses to about zero:
the point estimate is `<value>` and its interval [`<low>`, `<high>`] contains
zero. The pipeline therefore does not manufacture uplift from the features, the
outcome or the split. It passes the K4 check on this dataset and seed.

**Limits.** A single permutation is one draw from the null: a correct pipeline
still has about a 5% chance of an interval that excludes zero at the 95% level, so
a failure is a reason to investigate, and the control is re-run at M6 (T70) with
the same seed. It also says nothing about power: it shows the pipeline reports no
effect when none exists, not that it finds one when one exists. If the interval
excludes zero, that is reported as a finding, not re-run until it moves.