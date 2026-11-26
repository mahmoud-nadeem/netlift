# ADR 0007: Tie-aware uplift@k

Status: proposed; accepted when its pull request merges. Owner: @mahmoud-nadeem. Reviewer: @Ali-Alaa-Salah.
Decided: 2026-11-26, before any K1 (T44) run.

This changes a metric definition after the T39 freeze, so it is recorded here with its evidence.

## Context

ADR 0001 (decision 3) makes scikit-uplift the source of truth for every reported Qini, AUUC and
uplift@k. K1 (`config/kpis.yaml`) is the treated rate minus the control rate among the top k
customers of each policy, which is uplift@k with strategy `overall`.

scikit-uplift 0.5.1's `uplift_at_k` sorts the scores with a stable sort, reverses the order and
takes the first `int(n * k)` rows. When the cut falls inside a block of tied scores, which rows of
the block are taken depends on the order the rows arrive in. The models cannot tell those rows
apart, so the result depends on an arbitrary order.

**Where this showed up.** In the first T57 champion–challenger table (26 Nov, three models, 1,000
within-arm resamples, `reports/t57-champion-challenger.md`), some 95% intervals do not contain
their own point estimate. For example, the X-learner's uplift@20% is 0.005114 [0.003645, 0.005113],
and its uplift@30% is 0.003846 [0.002764, 0.003702]. The paired differences against the forest at
20% and 30% are affected too. The point estimates are computed in `row_id` order. The resamples
are computed in the order `within_arm_bootstrap_indices` produces: control rows, then treated
rows. With ties at the cut, the two orders give different values.

**Evidence** (`scripts/uplift_at_k_tie_check.py`, output `reports/uplift-at-k-ties.json`). The
script runs on the full Criteo validation split (2,795,918 rows), with the K1 T- and X-learner
scores and the T57 forest's scores, in seven row orders: `row_id`, control-then-treated, and five
random permutations.

- **Tie blocks at the cut.** These are the rows whose score equals the score at the cut:

  | Model | at 10% | at 20% | at 30% |
  |---|---|---|---|
  | T-learner | 1 | 3,157 | 39,878 |
  | X-learner | 586 | 34,793 | 67,393 |
  | Uplift forest | 4,382 | 15,978 | 689 |

- **Qini and AUUC do not depend on the order.** Their spread across the seven orders is exactly 0
  for all three models, because scikit-uplift's curves group tied scores.
- **scikit-uplift's uplift@k does depend on it.** For the X-learner at 30%, it is:
  - 0.003846 in `row_id` order;
  - 0.003221 in control-then-treated order;
  - between 0.003531 and 0.003544 in the five random orders.

  Its largest spread across orders is 0.000727 (X-learner, 20%).

## Decision

1. **uplift@k is computed tie-aware, in `netlift.evaluation.metrics.point_uplift_at_k`.** The rows
   are split by where their score falls against the score at the cut, the n-th highest score for
   n = `int(n_rows * k)`:
   - rows scored above the cut weigh 1;
   - rows scored below it weigh 0;
   - each of the rows tied at the cut weighs (n − rows above the cut) / rows tied.

   The weights add up to exactly n. The treated rate is Σ w·y / Σ w over treated rows, and the
   control rate is the same over control rows. With strategy `by_group`, the same rule applies
   within each arm, with n = `int(n_arm * k)`.

   The weight is the probability that a tied row is picked when the places left at the cut are
   filled at random from the tied block. So the value is what a campaign gets when it contacts
   randomly chosen customers among those the model cannot rank. It does not depend on the row
   order.

2. **Without a tie across the cut, the value is exactly scikit-uplift's.** Every weight is then 0
   or 1, and the rates are the same means. `tests/unit/test_metrics.py` checks equality (`==`)
   against `sklift.metrics.uplift_at_k`, with continuous scores and with ties that don't cross the
   cut. It also checks that the value is the same for any row order, and a hand-computed case.

3. **Qini and AUUC are unchanged.** They are still computed by scikit-uplift, and the evidence
   shows they do not depend on the order.

4. **The result records the rule.** An uplift@k `MetricResult` carries `settings["ties"] =
   "fractional"`. `paired_difference` refuses to pair results whose settings differ, so a result
   computed before this change cannot be paired with one computed after it. `library` stays
   `scikit-uplift`: the contract requires it for the reported names, and the definition is
   scikit-uplift's, made independent of the order.

5. **ADR 0001's decision 3 is amended for uplift@k only:** the number is scikit-uplift's
   definition with tied rows at the cut shared by fractional weight.

## Options considered

- **Keep scikit-uplift as it is.** The point estimate and the interval depend on an arbitrary
  order. The intervals are biased, because every resample puts control rows before treated rows.
  Rejected.
- **Break ties at random with a fixed seed.** This adds noise that has nothing to do with the
  model, and the value depends on the seed. Rejected.
- **Take or drop the whole tied block.** The number of contacted customers would no longer be k,
  and K1 requires both policies to contact the same number. Rejected.
- **Add a tiny deterministic jitter, or sort by `row_id`.** This is still an arbitrary order,
  only a fixed one. Rejected.

## Consequences

- **K1 (T44) waits for this change.** It reports uplift@k with paired intervals, and its point
  estimate must not depend on how the score files are ordered.
- **The T57 champion–challenger table is re-run** with this change, for all four models.
- **The values change only where a tie crosses the cut.** For the X-learner at 30%, the value
  moves from 0.003846 (`row_id` order) to 0.003538. For the T-learner at 10%, where one row ties,
  nothing changes.
- **Any uplift@k computed before this change must be recomputed** before it is reported.
- **Three places still call `sklift.metrics.uplift_at_k` directly**, outside the harness:
  `src/netlift/models/baselines.py` (T38), and `scripts/run_meta_learners.py` and
  `scripts/bootstrap_t56.py` (T56). Their uplift@k numbers are preliminary. They move to the
  harness's `uplift_at_k` before any of them is reported.
