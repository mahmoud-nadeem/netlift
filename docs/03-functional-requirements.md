# NetLift Functional Requirements

**Task:** T33 · **Issue:** [#45](https://github.com/mahmoud-nadeem/netlift/issues/45) · **Milestone:** M3 – Requirements Gathering  
**Source:** `docs/requirements/02-user-stories-and-use-cases.md` (T32, revised 16 October 2026), ADR 0002 (PR #95), and the NetLift Project Specification PDF snapshot dated 13 October 2026.  
**Status:** Draft for review. Requirement IDs are stable within this document; tests listed here are planned verification names, not claims that tests already exist or pass.

## 1. Purpose and conventions

This document converts the T32 user stories and use cases (UC-01–UC-10) into testable functional requirements. Every use case maps to at least one requirement. Each requirement identifies its source use case, realization path, and planned test name.

- **“Shall”** describes required behavior supported by T32 and ADR 0002.
- **Open [Q#]** means the behavior depends on a question that T32 explicitly leaves unresolved. It must not be treated as a frozen contract until the named decision is recorded.
- **Proposed [Q#]** preserves a proposal from T32; it is not represented as an agreed product rule.
- **Deployment-only** means described for a future live deployment, not implemented or claimed for this semester.
- API names and response fields use the exact spellings in T32 and the specification.

The API contract remains the six endpoints in specification §35: `POST /score`, `POST /score/batch`, `POST /policy/simulate`, `GET /model/info`, `GET /monitor/drift`, and `GET /health`. Per ADR 0002, recorded replay is an offline command-line job and retraining is an offline pipeline job; neither gets a new endpoint. UC-04 campaign-system delivery and automatic scheduled triggers are deployment-only.

## 2. Functional requirements

### 2.1 Scenario definition and replay safety (UC-01, UC-06)

| ID | Requirement | Source use case | Realized by | Planned verification test | Notes / decision dependency |
|---|---|---|---|---|---|
| FR-01 | The system shall read a replay scenario from a configuration file and persist the scenario inputs and selected slice membership with state `defined`. | UC-01 | Offline replay CLI job; scenario configuration and `replay_scenario` persistence | `test_fr_01_scenario_config_saved_as_defined` | ADR 0002, decision 1. Which data split may be selected at each project stage remains open [Q1]. |
| FR-02 | The system shall validate that cost per contact `c`, margin per conversion `m`, and budget are numeric and greater than zero, and that a data slice is selected; invalid or missing inputs shall be named and no scenario shall be created. | UC-01 | Offline replay CLI job / scenario-config validator | `test_fr_02_invalid_scenario_config_rejected` | Fixed-target mode also requires `N` [Q12]. Default mode and missing-`N` behavior are proposed, not frozen [Q12]. |
| FR-03 | The system shall report the break-even ratio `c/m` for a valid scenario. | UC-01 | Offline replay CLI job; report output | `test_fr_03_break_even_ratio_reported` | Specification §10. |
| FR-04 | Before the reveal step, the scoring and allocation steps shall not receive or inspect the outcome column; an automated test shall fail if an outcome column reaches a pre-reveal frame. | UC-01, UC-05, UC-06 | Replay pipeline boundary; feature/allocation inputs | `test_fr_04_outcome_unavailable_before_reveal` | Required replay safety invariant, specification §14. T32 also proposes withholding treatment from scoring/allocation; keep that separate from outcome-leakage enforcement until confirmed. |
| FR-05 | The system shall start a recorded replay from its scenario configuration, run scenario definition, scoring, allocation, reveal and measurement in order, and log the recorded run to MLflow. | UC-06 | Offline command-line replay job using `netlift.utils.tracking` | `test_fr_05_replay_cli_runs_and_logs_to_mlflow` | No replay/reveal API endpoint. T32 leaves the human/operator who starts the command open [Q3(a)]. |
| FR-06 | The system shall refuse to reveal outcomes unless the scenario state is `allocated` and the scenario has not already been revealed; a refused reveal shall not produce a measured result. | UC-06 | Replay job state-transition guard | `test_fr_06_reveal_requires_allocated_unrevealed_state` | A new scenario is required to measure different inputs [Q1]. UC-07 what-if re-simulation is separately specified and does not mutate the recorded scenario. |
| FR-07 | The system shall retain the scenario’s inputs, target allocation, recorded-arm holdout membership, measured result, intervals, bootstrap seed, library and settings, and associate them with the MLflow run. | UC-06 | Replay job result persistence and MLflow tracking | `test_fr_07_replay_artifacts_and_seed_logged` | Supports Finance audit and reproducibility (K3, K8). |

### 2.2 Scoring and abstention (UC-02)

| ID | Requirement | Source use case | Realized by | Planned verification test | Notes / decision dependency |
|---|---|---|---|---|---|
| FR-08 | The system shall accept batch-scoring input as CSV or JSON array at `POST /score/batch`, and return a scored file and summary; it shall accept one customer at `POST /score` and return one full response object. | UC-02 | `POST /score/batch`; `POST /score`; Dashboard Tab 1 for batch workflow | `test_fr_08_batch_and_single_scoring_contracts` | Specification §35. |
| FR-09 | The system shall reject a scoring request containing forbidden columns, including treatment indicator, outcome, or Criteo `exposure`; the error shall name the offending column and no row shall be scored. | UC-02 | Scoring input validation / `validate_features()` before either scoring endpoint | `test_fr_09_forbidden_columns_rejected` | Leakage control K7; specification §18. |
| FR-10 | The system shall block scoring when input is malformed or data-quality checks fail, including configured null-rate, range, or schema-change checks, and shall not present partial output as valid. | UC-02 | Scoring input validation and data-quality checks before `POST /score` / `POST /score/batch` results are returned | `test_fr_10_scoring_data_quality_failure_blocks_run` | Threshold values come from configuration [T32, UC-09]. |
| FR-11 | Every successfully scored row shall contain all eleven output-schema fields: `customer_id`, `model_version`, `uplift_score`, `uplift_ci_95`, `decile`, `segment_label`, `expected_incremental_profit`, `recommendation`, `verdict`, `top_drivers`, and `scored_at`. | UC-02 | Both scoring endpoints; response serializer | `test_fr_11_scoring_response_has_all_eleven_fields` | Exact schema from specification §12 and T32. |
| FR-12 | The system shall calculate and return an uplift estimate, its 95% interval, decile, model version, and a verdict for each successfully scored customer. | UC-02 | Scoring pipeline; `POST /score` and `POST /score/batch` | `test_fr_12_score_interval_decile_and_verdict_returned` | Every returned score has a `verdict`. Bootstrap interval details follow specification §28. |
| FR-13 | When the 95% interval straddles zero, the system shall assign `insufficient_evidence`, return the score and interval, and issue no directional recommendation. | UC-02 | Abstention decision logic in scoring pipeline | `test_fr_13_interval_straddling_zero_abstains` | The specific values for `recommendation` and `segment_label` under abstention are **Proposed [Q2]** (`no_action` and `unknown` respectively), not frozen. |
| FR-14 | When a segment has fewer than the configured minimum number of control-group outcome events, the system shall assign `insufficient_control_data` and log the threshold used with the response. | UC-02 | Abstention decision logic and scoring log | `test_fr_14_insufficient_control_events_abstain` | Exact configured minimum is configuration-driven; handling of related fields remains open [Q2]. |
| FR-15 | The system shall distinguish both abstention verdicts visually from a confident verdict in the dashboard. | UC-02 | Dashboard Tab 1 and customer-detail views | `test_fr_15_abstention_verdicts_visually_distinct` | Specification §11 and §40. |
| FR-16 | If the champion model is unavailable or the service is not ready, the system shall report not-ready through `GET /health` and shall not return a score as valid. | UC-02 | `GET /health`; scoring service readiness guard | `test_fr_16_unready_service_does_not_score` | Operational endpoint under §35; no separate model-scoring endpoint is added. |
| FR-17 | The system shall populate `expected_incremental_profit` as `m · uplift − c` wherever the required `m` and `c` are available, and shall identify the cost and margin assumptions used for the figure. | UC-02, UC-07 | Scoring response and policy-simulation result assembly | `test_fr_17_profit_uses_named_cost_and_margin` | Where `m` and `c` come from for standalone scoring is **Open [Q4]**; do not freeze this contract before resolving request-versus-configuration source. |
| FR-18 | The system shall load the registered champion model version and its fitted preprocessing artifacts for scoring, and return that model version in every scoring response. | UC-02 | Scoring service model loader / preprocessing pipeline | `test_fr_18_scoring_uses_registered_version` | Specification §23 and §35. |

### 2.3 Model evidence and explanations (UC-03, UC-08)

| ID | Requirement | Source use case | Realized by | Planned verification test | Notes / decision dependency |
|---|---|---|---|---|---|
| FR-19 | The system shall expose the registered model version, training date, and evaluation metrics with confidence intervals through `GET /model/info`. | UC-03 | `GET /model/info`; Dashboard Tab 3 | `test_fr_19_model_info_returns_version_date_metrics` | Specification §35. |
| FR-20 | The Model Evidence view shall show the Qini curve and confidence band, AUUC and Qini coefficient with intervals, decile table, uplift-calibration plot, SHAP summary, negative-control result, and champion–challenger comparison. | UC-03 | Dashboard Tab 3; evaluation artifacts | `test_fr_20_model_evidence_view_has_required_artifacts` | Name the Qini/AUUC library and settings and bootstrap-resample count in the displayed evidence. K2’s success condition follows `config/kpis.yaml` pending alignment of §46 [Q11]. |
| FR-21 | The customer detail view shall show the full eleven-field response object, including the score interval, verdict, segment label, recommendation, model version, and top drivers. | UC-08 | Dashboard customer detail; shared response schema from `POST /score` | `test_fr_21_customer_detail_shows_full_response` | Specification §12. |
| FR-22 | The system shall display `segment_label` only from `persuadable`, `sure_thing`, `lost_cause`, `sleeping_dog`, or `unknown`, and describe it as an inference rather than ground truth. | UC-08 | Customer detail view and scoring serializer | `test_fr_22_segment_label_values_and_copy` | Whether abstention always maps to `unknown` is **Proposed [Q2]**. |
| FR-23 | The system shall describe `top_drivers` as features that drive the model’s uplift prediction, not as causes of a customer’s behavior; on Criteo, the explanation shall preserve anonymized feature labels `f0`–`f11`. | UC-08 | Customer detail explanation component | `test_fr_23_driver_copy_is_non_causal_and_dataset_aware` | Specification §31 and §29. |
| FR-24 | The customer detail view shall display either abstention verdict in the distinct abstention style and shall not show a directional recommendation for `insufficient_evidence`. | UC-08 | Customer detail view | `test_fr_24_customer_detail_handles_abstention` | If `insufficient_control_data` has a separate recommendation/display rule, resolve with [Q2]. |

### 2.4 Target allocation and campaign export (UC-04, UC-05)

| ID | Requirement | Source use case | Realized by | Planned verification test | Notes / decision dependency |
|---|---|---|---|---|---|
| FR-25 | The system shall make the allocated target list downloadable from Dashboard Tab 1. | UC-04, UC-05 | Dashboard Tab 1 (Campaign Planner export) | `test_fr_25_allocated_target_list_downloadable` | This is the semester deliverable. Sending targets to a Campaign Execution System, recording live assignment, running a live campaign, ingesting real outcomes and computing live effect are deployment-only; no delivery endpoint is added by ADR 0002. |
| FR-26 | In Fixed budget mode, the policy engine shall rank eligible customers by estimated uplift and select up to `k = B/c`, without selecting customers whose uplift estimate is less than or equal to zero; any unused budget and actual contact count shall be reported. | UC-05 | `PolicyEngine`, invoked by the replay job; allocation summary | `test_fr_26_fixed_budget_selection_and_unspent_budget` | Selection floor and abstention interaction are **Proposed [Q2]**. If fewer eligible customers than `k` remain, the T32 proposal is to leave the remainder unspent [Q2]. |
| FR-27 | In Profit-maximising mode, the policy engine shall select every eligible customer whose estimated ITT uplift is greater than `c/m`; the required spend is an output rather than an input. | UC-05 | `PolicyEngine` / replay job | `test_fr_27_profit_maximising_uses_break_even_ratio` | Use ITT-scale uplift, not Wald/ToT/LATE uplift. Eligibility of abstained customers remains subject to [Q2]. |
| FR-28 | In Fixed target mode, the policy engine shall select the smallest set expected to deliver target `N` and report the required budget and its uncertainty. | UC-05 | `PolicyEngine` / replay job | `test_fr_28_fixed_target_reports_budget_and_uncertainty` | Behavior when `N` cannot be reached is **Open [Q5]**. |
| FR-29 | In Suppression-only mode, the policy engine shall exclude customers classified as negative-uplift according to the agreed suppression rule. | UC-05 | `PolicyEngine` / replay job | `test_fr_29_suppression_only_excludes_suppressed_customers` | Definition of “negative-uplift” and treatment of abstained customers are **Open [Q2]**. T32 proposes using the sleeping-dog suppression list and applying the confident-only eligible pool in every mode; this is not frozen. |
| FR-30 | The policy engine shall place a customer on the sleeping-dog suppression list only when the estimated uplift is negative and the entire 95% interval is below zero; that customer shall not be contacted in any policy mode. | UC-05 | `PolicyEngine` suppression-list construction | `test_fr_30_sleeping_dog_requires_entire_interval_below_zero` | The relation between this list and the point-estimate selection floor is explicitly open [Q2]. |
| FR-31 | Once allocation is fixed, the replay shall join the selected target list to the original recorded treatment assignment; selected targets originally assigned control shall form the replay holdout, and selected targets originally assigned treatment shall form the treated group. The replay shall not draw a new random holdout. | UC-05, UC-06 | Replay job after allocation; recorded-arm join | `test_fr_31_replay_holdout_uses_recorded_assignment` | Decided by ADR 0002 and T52/#94. The assignment join must not leak treatment or outcome into pre-reveal scoring/allocation. Holdout size is reported with the result. |
| FR-32 | The system shall report the allocation’s contact count, expected incremental conversions with an interval, expected incremental profit, and a downloadable target list. | UC-05 | Dashboard Tab 1 and allocation output | `test_fr_32_allocation_summary_contains_required_outputs` | Specification §40. The UI’s interactive allocation path versus replay-config-only allocation is **Open [Q3(b)]**. |
| FR-33 | Customers with either abstention verdict shall be reported separately and shall not be silently merged with the suppression list or a confident recommendation. | UC-04, UC-05 | Allocation summary and export preparation | `test_fr_33_abstentions_separate_from_suppression` | Eligibility rules, the exported label for abstentions, and recommendation/segment fields are **Open [Q2]**. T32 proposes that abstained customers are not eligible in any policy mode. |

### 2.5 Measurement and audit (UC-06)

| ID | Requirement | Source use case | Realized by | Planned verification test | Notes / decision dependency |
|---|---|---|---|---|---|
| FR-34 | After a valid reveal, the system shall measure each policy’s incremental effect as the outcome rate in the selected targets recorded as treated minus the rate in the selected targets recorded as control. | UC-06 | Replay measurement stage | `test_fr_34_measured_effect_uses_recorded_treatment_arms` | Decided replay semantics in ADR 0002/T52. T32 states holdout is the recorded control arm, not a newly randomized subset. |
| FR-35 | The system shall compute a 95% percentile bootstrap interval using at least 1,000 resamples drawn within each recorded treatment arm. | UC-06 | Replay measurement/bootstrap utility | `test_fr_35_bootstrap_resamples_within_arms` | Specification §28 and T32 glossary. The exact handling of too few control events is represented in FR-36. |
| FR-36 | If a measured policy or decile does not meet the configured minimum control-event count, the system shall report `insufficient_control_data` and shall not claim a measured effect for it. | UC-06 | Measurement and result-serialization stage | `test_fr_36_measurement_abstains_with_too_few_control_events` | Recorded control is a subset of selected targets; this can occur frequently on Criteo [T32, Q10]. The threshold is configuration-driven. |
| FR-37 | A measured replay result shall compare Uplift, Response Model, Blanket Contact and Random Contact at an identical budget and show contact count, total cost, predicted and measured incremental conversions with intervals, incremental profit, profit per unit spent, and cumulative curves with bootstrap bands. | UC-06 | Replay comparison stage; results consumed by Dashboard Tabs 2 and 4 | `test_fr_37_recorded_replay_compares_four_policies` | The uplift policy’s mode comes from scenario configuration. Cost and margin must be stated with profit figures; do not describe estimates as money saved. |
| FR-38 | The system shall produce predicted-versus-measured incremental-conversion calibration by decile and store the complete measurement artifacts with the scenario and MLflow run. | UC-06 | Replay measurement, calibration output and MLflow tracking | `test_fr_38_calibration_and_measurement_artifacts_persisted` | K3/K5; include holdout size, curves, intervals, bootstrap seed and metric-library settings. |
| FR-39 | The same replay implementation shall be used by the offline CLI job and `POST /policy/simulate`; for identical effective inputs and seed, the two paths shall produce equivalent result values. | UC-06, UC-07 | Shared `netlift` replay implementation; CLI and `POST /policy/simulate` adapters | `test_fr_39_cli_and_simulate_return_same_result_for_same_inputs_and_seed` | Required by ADR 0002. Effective scenario/slice/model/seed binding for the endpoint is **Open [Q3(c)]**. |

### 2.6 Policy comparison what-if (UC-07)

| ID | Requirement | Source use case | Realized by | Planned verification test | Notes / decision dependency |
|---|---|---|---|---|---|
| FR-40 | `POST /policy/simulate` shall re-run the shared replay code for a what-if using supplied budget, cost and margin, rather than reading or modifying the stored result of UC-06; the call shall not create a recorded MLflow replay run. | UC-07 | `POST /policy/simulate` and shared replay implementation | `test_fr_40_policy_simulate_reruns_without_mutating_recorded_run` | ADR 0002, decision 2. Which scenario, slice, model version and seed the call uses beyond those three inputs is **Open [Q3(c)]**. |
| FR-41 | The Policy Comparison view shall show, for each of the four policies, number contacted, total cost, incremental conversions with a confidence interval, incremental profit and profit per unit spent, with cost and margin assumptions named alongside profit figures. | UC-07 | Dashboard Tab 2; `POST /policy/simulate` response | `test_fr_41_policy_comparison_table_and_assumptions` | Profit figures are estimates, not realised money (K3). |
| FR-42 | The Policy Comparison view shall draw cumulative-incremental curves with bootstrap bands and allow a visual budget marker to move across all displayed curves without a request solely to move the marker. | UC-07 | Dashboard Tab 2 chart interaction | `test_fr_42_budget_marker_moves_across_all_curves` | As T32 describes, changing budget/cost/margin used to calculate the table triggers another `POST /policy/simulate` call. |
| FR-43 | The comparison shall state in words whether the 95% interval for the uplift-versus-response-model difference excludes zero in favour of uplift; if the interval contains zero, it shall say the policies are not distinguishable on this data and shall name no winner. | UC-07 | Policy-comparison summary | `test_fr_43_policy_difference_interpretation` | K1, specification §28. Policy-level abstention styling details remain open [Q2(d)]. |
| FR-44 | A simulation request shall not mutate the measured scenario state or stored UC-06 result, and the comparison shall not start before the scenario has a measured recorded result. | UC-07 | `POST /policy/simulate` guard and read-only result handling | `test_fr_44_simulate_requires_measured_scenario_and_is_read_only` | T32 defines measured scenario as a precondition; whether/how the endpoint is tied to that scenario is **Open [Q3(c)]**. The request must have a defined identity for its effective inputs before implementation. |
| FR-45 | A simulation shall not re-open the locked test split; if the selected input would require a new read of that split, the call shall be refused and only the already recorded UC-06 figures may be shown. | UC-07 | Replay split guard / `LockedTestSplit` access policy | `test_fr_45_simulate_refuses_locked_test_split_reuse` | **Proposed [Q1]**, pending a recorded split policy. ADR 0002 explicitly leaves Q1 open; use validation/pre-agreed replay slice for development/demo only if the team formally adopts that proposal. |

### 2.7 Drift monitoring (UC-09)

| ID | Requirement | Source use case | Realized by | Planned verification test | Notes / decision dependency |
|---|---|---|---|---|---|
| FR-46 | `GET /monitor/drift` shall return the latest available drift report, including per-feature PSI or KS against the training reference, uplift-score distribution drift, and data-quality status for null rates, range violations and schema changes. | UC-09 | Drift-report computation; `GET /monitor/drift`; Dashboard Tab 4 | `test_fr_46_drift_report_contains_feature_score_and_quality_status` | Alert thresholds are read from configuration, specification §32. Behavior when no report exists is unspecified in T32 and must be clarified before the API contract freezes. |
| FR-47 | The Monitoring view shall display the registered model version, training date, and results of simulated campaign replays; any effect-drift panel shall be explicitly labelled as mocked from replay output. | UC-09 | Dashboard Tab 4 | `test_fr_47_monitoring_labels_mocked_effect_drift` | Feature/score drift is built this semester; real effect drift requiring a permanent randomized holdout and real outcome window is deployment-only. |
| FR-48 | This-semester drift reports shall be producible on demand from a scored batch and training reference; scheduled drift-report generation shall be treated as deployment-only. | UC-09 | On-demand drift report workflow; optional future Scheduler integration | `test_fr_48_drift_report_can_be_generated_on_demand` | The on-demand/current-semester split is **Proposed [Q7]** and needs confirmation. |

### 2.8 Retraining (UC-10)

| ID | Requirement | Source use case | Realized by | Planned verification test | Notes / decision dependency |
|---|---|---|---|---|---|
| FR-49 | A Data Scientist shall be able to start a retraining run as an offline pipeline job; the job shall be reproducible from configuration and a git commit hash. | UC-10 | Offline training pipeline; MLflow tracking and Model Registry | `test_fr_49_retraining_pipeline_runs_from_config_and_commit` | Retraining is not exposed as a new API endpoint. Scheduled triggering is deployment-only (ADR 0002). |
| FR-50 | The retraining pipeline shall validate schema, audit leakage, create the specified joint stratified split, fit baselines before uplift estimators, and evaluate candidates with bootstrap intervals. | UC-10 | Training pipeline stages | `test_fr_50_retraining_runs_required_validation_and_evaluation_stages` | Specification §23 and §28; exact split restrictions must align with Q1/T70. |
| FR-51 | The retraining pipeline shall fail if a forbidden column reaches the model matrix and shall fail the candidate’s K4 gate if the 95% interval of uplift under the permuted-treatment negative control does not contain zero. | UC-10 | Leakage audit and negative-control evaluation gates | `test_fr_51_retraining_blocks_leakage_and_negative_control_failure` | K4/K7. |
| FR-52 | A completed candidate shall be registered as a new model version with its evaluation attached, while the previous model version remains registered and available for rollback. | UC-10 | MLflow Model Registry lifecycle | `test_fr_52_candidate_registered_with_evaluation_and_previous_retained` | Whether and when a candidate becomes the new champion/production version is **Open [Q9]**; this requirement does not authorize automatic promotion. |
| FR-53 | The Data Scientist shall be able to start the pipeline manually in this semester; automatic cadence/drift-triggered retraining shall be marked deployment-only. | UC-10 | Manual pipeline entry point; future Scheduler integration | `test_fr_53_retraining_can_be_started_manually` | The OR-vs-AND interpretation of cadence/drift triggers is **Proposed [Q9]** and is not frozen by ADR 0002. |

## 3. Use-case coverage

| T32 use case | Requirement IDs |
|---|---|
| UC-01 Define a replay scenario | FR-01–FR-04 |
| UC-02 Score a batch of customers | FR-08–FR-18 |
| UC-03 Review model evidence | FR-19–FR-20 |
| UC-04 Deliver the target list to a campaign | FR-25, FR-33 |
| UC-05 Allocate under budget | FR-26–FR-33 |
| UC-06 Reveal outcomes and measure | FR-05–FR-07, FR-31, FR-34–FR-39 |
| UC-07 Compare policies | FR-39–FR-45 |
| UC-08 Inspect a customer explanation | FR-21–FR-24 |
| UC-09 Review drift | FR-46–FR-48 |
| UC-10 Retrain the model | FR-49–FR-53 |

Coverage check: every use case has at least one requirement and every FR has a source use case. The requirement/test names above are **planned names**. As tests are implemented, retain the FR identifier in the test name or test docstring so traceability remains bidirectional.

## 4. API and interface boundary

| Interface / execution path | Requirements served | Boundary |
|---|---|---|
| `POST /score` | FR-08–FR-18, FR-21–FR-24 | Single-customer scoring; no extra inputs/fields beyond the frozen API contract until Q4 is resolved. |
| `POST /score/batch` | FR-08–FR-18 | Batch scoring and summary. |
| `POST /policy/simulate` | FR-39–FR-45 | What-if re-simulation through shared replay code; no recorded replay run. Scenario/slice/model/seed binding is open under Q3. |
| `GET /model/info` | FR-19 | Registered model version, training date, metrics and intervals. |
| `GET /monitor/drift` | FR-46–FR-48 | Latest drift report; empty-report behavior needs clarification. |
| `GET /health` | FR-16 | Liveness/readiness. |
| Offline replay CLI job | FR-01–FR-07, FR-31, FR-34–FR-39 | The recorded UC-06 run; configuration defines scenario; result logged to MLflow. |
| Offline training pipeline | FR-49–FR-53 | UC-10; no retraining endpoint. |
| Dashboard Tab 1 | FR-15, FR-25–FR-33 | Scoring result, allocation view, target-list download. Whether interactive allocation calls the decision layer remains open [Q3]. |
| Dashboard Tab 2 | FR-37, FR-41–FR-45, FR-47 | Recorded comparisons plus interactive simulation. Dashboard access to stored replay results (MLflow directly or an existing API path) remains open for T45/T62. |
| Dashboard Tab 3 | FR-19–FR-20 | Model evidence. |
| Dashboard Tab 4 | FR-38, FR-47–FR-48 | Recorded results, feature/score drift, and explicitly mocked effect drift. |
| Campaign Execution System | None in current-semester scope | Live delivery, permanent holdout, outcome ingestion and live-effect measurement are deployment-only. |

## 5. Open decisions required before freezing affected requirements

| Question | Affected requirements | Needed decision / handling |
|---|---|---|
| Q1 — replay split and locked-test access | FR-01, FR-45, FR-50 | Record which split is allowed for development/demo replay and how/when the locked test split is opened. ADR 0002 explicitly does not resolve Q1. Do not implement repeated locked-test reads as a default. |
| Q2 — abstention, recommendation, segment and allocation behavior | FR-13–FR-15, FR-22, FR-24, FR-26–FR-30, FR-33, FR-36, FR-43 | Decide recommendation/segment values under abstention, whether abstained customers are ineligible in every mode, the difference between a non-positive selection floor and sleeping-dog suppression, how `k` is handled when the eligible pool is small, and policy-level abstention. |
| Q3 — execution and API/UI boundary | FR-05, FR-25, FR-32, FR-39–FR-44 | Name who starts the CLI replay; decide whether Tab 1 supports interactive allocation; define which scenario/slice/model/seed inputs identify a `POST /policy/simulate` request; decide how Tabs 2/4 access recorded MLflow results. Keep the API to six endpoints per ADR 0002. |
| Q4 — cost/margin source for standalone scoring | FR-17 | Specify whether cost and margin are request inputs or configuration values before freezing `expected_incremental_profit` for standalone scoring. |
| Q5 — unreachable fixed target | FR-28 | Define response when no set can reasonably deliver target `N`. |
| Q7 — scheduler status | FR-48, FR-53 and scheduled paths in FR-08 / FR-46 | Confirm that scheduled batch scoring, scheduled drift reporting and scheduled retraining are deployment-only and that semester drift reports are on demand. |
| Q9 — retraining trigger and promotion | FR-49, FR-52–FR-53 | Confirm cadence OR drift threshold versus another rule; define promotion gate/locked-test procedure. ADR 0002 leaves promotion open and does not authorize automatic promotion. |
| Q11 — K2 success rule | FR-20 | Align specification §46 with `config/kpis.yaml`; T32 proposes that the interval must be entirely above zero, not merely exclude zero. |
| Q12 — scenario policy defaults | FR-02 | Confirm default Fixed budget and reject Fixed target without `N`, or document the selected alternative. |

## 6. Source and consistency notes

- T32’s use-case IDs are frozen. This document does not renumber or rename UC-01–UC-10.
- Recorded replay holdout means the selected targets originally assigned to control. It is not a newly randomized holdout. Measured effect is treated rate minus control rate in the selected targets, with within-arm bootstrap resampling.
- The replay and retraining jobs are offline execution paths, not new API endpoints. `POST /policy/simulate` re-runs shared replay code and does not reuse the stored UC-06 result.
- This document does not claim that planned tests are implemented, that deployment-only flows are built, or that proposals marked [Q#] are approved.
