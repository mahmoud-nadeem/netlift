# ADR 0009: The decision layer (T55): decile intervals, verdict rules and policy modes

Status: proposed; accepted when its pull request merges. Owner: @mahmoud-nadeem (T55 moved from Ali on 26 Nov, #121). Reviewer: @Ali-Alaa-Salah.

The specification asks for this record before T55 starts: how the policy modes reconcile with the sleeping-dog rule, and the defended abstention threshold.

## Context

Interface 3 (`docs/design/interfaces.md`, section 3) fixes the scoring response and its rules:

- `uplift_ci_95` is the held-out decile-level interval of the customer's decile (Ali's decision on #67);
- a verdict other than `confident` carries `no_action`;
- `contact` needs the interval's lower bound above zero, and `suppress` its upper bound below zero;
- each policy mode decides what to do with `no_action`.

The specification (Uncertainty and Abstention) adds a verdict `insufficient_control_data` for segments with fewer than a configured minimum of control-group outcome events. The threshold must be a configuration value, defended in the documentation and logged with every response.

## Decision

1. **The decile table is built once, on the validation split,** from the champion's validation scores (`netlift.decision.deciles`, written by `scripts/export_champion.py`). The test split is not read.
   - The edges are the 90%, 80%, …, 10% quantiles of the scores. Decile 1 holds the highest scores, and a score equal to an edge goes to the higher decile.
   - For each decile, the ITT uplift is the treated conversion rate minus the control conversion rate.
   - Its 95% percentile interval comes from 1,000 within-arm resamples, seed 42. Resampling n rows of one arm with replacement makes the number of conversions binomial, so each resample draws `Binomial(n_arm, rate_arm)` per arm. This is the within-arm bootstrap computed exactly; a test checks it against a row-level bootstrap.

2. **The verdict comes from the customer's decile, in this order** (`netlift.decision.verdict`):
   1. fewer control-arm conversions in the decile than `min_control_events_per_decile` → `insufficient_control_data`, `no_action`, segment `unknown`;
   2. lower bound above zero → `confident`, `contact`, `persuadable`;
   3. upper bound below zero → `confident`, `suppress`, `sleeping_dog`;
   4. otherwise (the interval touches or straddles zero) → `insufficient_evidence`, `no_action`, `unknown`.

   `sure_thing` and `lost_cause` need a predicted control response, which an uplift model does not give, so they are not emitted.

3. **The abstention threshold is `min_control_events_per_decile: 10`** (`config/decision.yaml`), and it is logged with every response.
   - **Why 10:** it is the usual condition for a binomial proportion's normal and percentile approximations (n·p ≥ 10), and the same number as `min_control_events_per_fold` in `config/data.yaml`.
   - **What it does on the champion:** its validation deciles hold these control-arm conversions (measured on 26 Nov from `data/processed/t57-champion/`; the export re-measures them): 686, 52, 9, 7, 12, 6, 11, 18, 7 and 4. So deciles 3, 4, 6, 9 and 10 return `insufficient_control_data`.
   - **This is the M5 proposal.** The owner confirms the number before merge.

4. **Policy modes** (`netlift.decision.policy`). M5 implements three of the specification's four:
   - **Fixed budget:** contacts the top k = ⌊budget / cost⌋ by uplift among customers recommended `contact`. Equal scores keep their input order. It leaves budget unspent rather than contact anyone else.
   - **Profit-maximising:** contacts customers recommended `contact` whose uplift is above cost / margin.
   - **Suppression-only:** contacts everyone except `suppress`. So `no_action` customers get business-as-usual contact, which the dashboard labels separately (`docs/design/dfd-and-ui.md`).
   - **Reconciliation with the sleeping-dog rule:** a `suppress` is never contacted in any mode, regardless of remaining budget.
   - **Fixed target** is cut to future work. It needs an interval on the required budget, and no M5 deliverable asks for it.

## Consequences

- **The dashboard (T62) gets real verdicts** from `POST /score` once the champion folder is loaded (`NETLIFT_MODEL_DIR`). Without it the API keeps the T45 mock.
- **Most deciles are not directional on Criteo conversion.** That is the expected outcome of thin control data, which the specification itself warns about. It is reported, not tuned away.
- **The decile intervals are preliminary,** like every validation number, until T70.
- **`/policy/simulate` stays mock** until T61's replay simulator. **`top_drivers` stays empty** until T61's SHAP.
