# User stories and use cases

Author: Ali Alaa · Task: T32 · Reviewed by: Aliaa Gomaa
Date: 14 October 2026 · Issue: #44 · Milestone: M3 – Requirements Gathering

Section numbers (§) refer to `NetLift_Project_Specification.pdf` (55 pages, snapshot of
13 October 2026, pull request #27 awaiting final review). Every figure, field name, endpoint and
rule below is taken from that specification; nothing is invented. Where the specification is
silent or two statements disagree, the point is marked **[Q#]** and listed in Section 5. The Markdown copy of the specification is unnumbered and cross-references the wrong numbers, so the PDF numbers are used **[Q8]**.

Status legend used throughout:

| Status | Meaning |
|---|---|
| **Yes** | Built and exercised this semester |
| **Partly** | Part of it is built this semester; the rest is designed or documented only |
| **Deployment-only** | Not built and not claimed. Recorded because a live system would need it (§15) |

---

## Use-case index

The ID list below is frozen once it is posted to the team WhatsApp group (due Wednesday
14 October, 22:00), so that the functional requirements (T33) and the non-functional requirements
(T34) can reference it while they are written.

| ID | Use case | Primary actor | Implemented this semester |
|---|---|---|---|
| UC-01 | Define a replay scenario | Marketing Analyst | Yes |
| UC-02 | Score a batch of customers | Marketing Analyst | Yes |
| UC-03 | Review model evidence | Data Scientist | Yes |
| UC-04 | Deliver the target list to a campaign | Marketing Analyst | Partly (the download is built; delivery to a Campaign Execution System is deployment-only) |
| UC-05 | Allocate under budget | Marketing Analyst | Yes |
| UC-06 | Reveal outcomes and measure | Marketing Analyst | Yes |
| UC-07 | Compare policies | Marketing Manager | Yes |
| UC-08 | Inspect a customer explanation | Marketing Analyst | Yes |
| UC-09 | Review drift | Data Scientist | Partly (feature and score drift built; effect drift is mocked) |
| UC-10 | Retrain the model | Scheduler | Partly **[Q9]** |

The eight use cases named in the specification (§45) are UC-01, UC-02, UC-05, UC-06, UC-07,
UC-08, UC-09 and UC-10. UC-03 and UC-04 are added because the dashboard (§40, Tab 3) and the
Campaign Execution System actor (§45) need a use case each.

Typical order in the replay lifecycle (§14):
UC-01 → UC-02 → UC-05 → UC-04 (download) → UC-06 → UC-07 → UC-08.
UC-03, UC-09 and UC-10 run alongside it.

---

## 1. Actors

| Actor | Type | Role in NetLift | Source |
|---|---|---|---|
| Marketing Analyst | Primary, human | Defines scenarios, sets cost, margin, budget and policy mode, scores customer lists, downloads the target list, inspects individual customers. Called "Retention / CRM analyst" in the target-user table. | §1.5 target users; §45 |
| Marketing Manager | Primary, human | Compares candidate policies on one screen before committing spend. | §1.5 target users |
| Data Scientist | Primary, human | Consumes the scoring API, inspects Qini curves and drift monitors, retrains on a schedule. Called "Data science team" in the target-user table. | §1.5 target users; §45 |
| Finance / Campaign Audit | Secondary, human (reader) | Reads the incremental-profit report and the size of the randomized holdout that produced it. Never changes anything. | §1.5 target users |
| Scheduler | Supporting, internal system | Starts scheduled batch scoring and the scheduled part of the retraining rule. | §45; §35 (batch path); §32 |
| Campaign Execution System | **External system, deployment-only** | Would receive the approved target list, run the real campaign and return real outcomes after an outcome window. Does not exist in this project. | §45; §15 |

Notes on the actor list:

- The specification names two overlapping lists. The use-case-diagram list (§45) has four actors:
  Marketing Analyst, Data Scientist, Scheduler, Campaign Execution System. The target-user table
  (§1.5) has four users: CRM analyst, marketing manager, data science team, finance / audit.
  This document keeps all six names so that no stakeholder disappears. T31 (stakeholders) and
  T46 (use-case diagram) must use the same names **[Q6]**.
- The MLflow Model Registry, the PostgreSQL scoring log and the FastAPI service are components,
  not actors. They appear in preconditions and flows only.
- The Campaign Execution System appears in UC-04 only and is drawn as deployment-only.

---

## 2. User stories

Each story has acceptance criteria in Given / When / Then form. "Verdict" always means one of
`confident`, `insufficient_evidence`, `insufficient_control_data` (§11, §12).

| ID | Role | Capability (short) | Use cases |
|---|---|---|---|
| US-01 | Marketing Analyst | Define a replay scenario | UC-01 |
| US-02 | Marketing Analyst | Score customers, as a batch or one at a time | UC-02 |
| US-03 | Marketing Analyst | Get "insufficient evidence" instead of a guess | UC-02, UC-05, UC-08 |
| US-04 | Marketing Analyst | Allocate a budget and download the target list | UC-04, UC-05 |
| US-05 | Marketing Analyst | Reveal outcomes only after the decision is fixed, and measure | UC-06 |
| US-06 | Marketing Manager | Compare four policies at an identical budget | UC-07 |
| US-07 | Finance / Campaign Audit | Audit every profit figure and its assumptions | UC-06, UC-07 |
| US-08 | Marketing Analyst | Inspect one customer's explanation | UC-08 |
| US-09 | Data Scientist | See the model evidence and the drift monitors | UC-03, UC-09 |
| US-10 | Data Scientist | Retrain under a documented rule and register the result | UC-10 |

### US-01 Define a replay scenario

**As a** Marketing Analyst, **I want** to define a replay scenario with a per-contact cost, a margin
per conversion, a budget and a slice of held-out data, **so that** the system can replay a
campaign decision on data whose outcomes it has not read yet.

Acceptance criteria:

1. **Given** a champion model is registered and a held-out slice exists, **when** I submit cost,
   margin, budget and a slice, **then** a scenario is stored in state `defined` and the break-even
   ratio c/m is shown to me.
2. **Given** any of the four inputs is missing or not a number, **when** I submit, **then** no
   scenario is created and the missing input is named.
3. **Given** a scenario in state `defined`, **when** any step before the reveal runs, **then** the
   data frame it receives contains no outcome column, and an automated test asserts this (§14).

### US-02 Score customers

**As a** Marketing Analyst, **I want** to upload a customer file, or send a single customer, and
receive an uplift estimate with a 95% interval and a verdict for each, **so that** I can decide
whom to contact.

Acceptance criteria:

1. **Given** a CSV file or JSON array containing only permitted feature columns, **when** I request
   batch scoring (`POST /score/batch`), **then** I receive a scored file and a summary, and every
   row carries all eleven fields of the model output schema (§12).
2. **Given** one customer, **when** I call `POST /score`, **then** I receive the full response
   object, including `model_version`.
3. **Given** the input contains a forbidden column (a treatment indicator, an outcome, or Criteo's
   `exposure`), **when** I request scoring, **then** the run is rejected, the column is named, and
   no row is scored (§18, K7).
4. **Given** malformed input or a failed data-quality check (null rates, range violations, schema
   change), **when** I request scoring, **then** the run is blocked with an error and no score is
   presented as valid (§32).

### US-03 Abstention instead of a guess

**As a** Marketing Analyst, **I want** the system to say it does not know when the evidence is too
thin, **so that** I never act on noise.

Acceptance criteria:

1. **Given** a customer or segment whose 95% bootstrap interval straddles zero, **when** it is
   scored, **then** the verdict is `insufficient_evidence` and no directional recommendation is
   issued (§11).
2. **Given** a segment with fewer than the configured minimum number of control-group outcome
   events, **when** it is scored, **then** the verdict is `insufficient_control_data` and the
   threshold value used is logged with the response (§11).
3. **Given** either of these verdicts, **when** the dashboard shows it, **then** it is rendered in
   a style visually distinct from a confident verdict (§11, §40).
4. **Given** any score returned by the system, **then** it carries a verdict field. No
   recommendation is issued without one (§45, non-functional requirements).

### US-04 Allocate under budget and download the list

**As a** Marketing Analyst, **I want** to pick a policy mode and a budget and receive the list of
customers to contact, **so that** spend goes to customers whose outcome is expected to change.

Acceptance criteria:

1. **Given** scored customers, per-contact cost c, margin m and budget B, **when** I choose
   *Fixed budget*, **then** the top k customers by estimated uplift are selected, where k = B / c,
   and any selected customer with estimated uplift ≤ 0 is suppressed even if budget is left
   unspent (§10).
2. **Given** the same inputs, **when** I choose *Profit-maximising*, **then** every customer with
   estimated uplift > c/m is selected and the budget is an output, not an input (§10).
3. **Given** a target N of incremental conversions, **when** I choose *Fixed target*, **then** the
   smallest set expected to deliver N is selected, and the budget required and its uncertainty are
   reported (§10).
4. **Given** the same inputs, **when** I choose *Suppression-only*, **then** everyone except
   negative-uplift customers is selected (§10).
5. **Given** a customer whose estimated uplift is negative and whose confidence interval lies
   entirely below zero, **then** that customer is on the suppression list and is never contacted
   by the policy, regardless of remaining budget (§10).
6. **Given** an allocation, **then** I see the number of customers to contact, expected
   incremental conversions with a confidence interval, expected incremental profit, and a
   downloadable target list (§40, Tab 1).

### US-05 Reveal outcomes and measure

**As a** Marketing Analyst, **I want** the recorded outcomes to be revealed only after the
decision is fixed, and the measured effect compared with the predicted one, **so that** I know
whether the decision would have paid off.

Acceptance criteria:

1. **Given** a scenario in state `allocated`, **when** I trigger the reveal, **then** the state
   becomes `revealed`, and after measurement `measured` (§14).
2. **Given** a scenario that is not yet `allocated`, **when** I trigger the reveal, **then** it is
   refused.
3. **Given** a revealed scenario, **when** the measurement runs, **then** the measured incremental
   effect is the treated rate minus the holdout rate, with a bootstrap confidence interval that
   resamples within treatment arms (§14, §28).
4. **Given** a measured scenario, **then** predicted and measured incremental conversions are
   shown side by side, each with an interval, for the uplift policy, the response-model policy,
   blanket contact and random contact at an identical budget (§14).
5. **Given** a completed run, **then** the whole run is logged to MLflow (§14).

### US-06 Compare policies

**As a** Marketing Manager, **I want** to compare uplift targeting, the response model, blanket
contact and random contact at an identical budget on one screen, **so that** I can decide which
policy to commit spend to.

Acceptance criteria:

1. **Given** a measured scenario, **when** I open *Policy Comparison*, **then** a table shows, for
   each of the four policies: number contacted, total cost, incremental conversions with a
   confidence interval, incremental profit, and profit per unit spent (§13).
2. **Given** the same screen, **then** overlaid cumulative-incremental curves with bootstrap bands
   are shown, and moving the budget slider moves one vertical line across all curves at once
   (§40, Tab 2).
3. **Given** the 95% bootstrap interval of the difference between the uplift policy and the
   response model excludes zero in favour of uplift, **then** the screen states the comparison in
   words (K1). **Given** the interval contains zero, **then** the screen reports that the policies
   are not distinguishable on this data, and does not name a winner (§28).

### US-07 Audit a profit figure

**As a** member of Finance / Campaign Audit, **I want** every profit figure to state its cost and
margin assumptions, its interval and the size of the randomized holdout behind it, **so that** I
can audit it and know it is an estimate, not realised money.

Acceptance criteria:

1. **Given** any incremental-profit figure, **then** it is labelled as an estimate, the per-contact
   cost and margin are named on the same line, and a 95% interval is given (K3, §13).
2. **Given** a replay result, **then** the size of the randomized holdout that produced it is
   shown (§1.5 target users).
3. **Given** the text of a report or dashboard, **then** it follows the wording rule of §13
   ("estimated … at the stated contact cost and margin, measured on held-out experimental data")
   and never claims money saved.

### US-08 Inspect a customer explanation

**As a** Marketing Analyst, **I want** to open one customer and see the score, its interval, the
segment label, the recommendation, the verdict and the main drivers, **so that** I can explain a
decision to a colleague.

Acceptance criteria:

1. **Given** a scored customer, **when** I select the row, **then** the detail view shows the full
   response object of §12.
2. **Given** the segment label, **then** it is one of `persuadable`, `sure_thing`, `lost_cause`,
   `sleeping_dog`, `unknown`, and it is presented as an inference, never as ground truth (§12).
3. **Given** the top drivers, **then** they are described as features that drive the model's
   uplift prediction, never as causes of the customer being persuadable (§31).
4. **Given** a verdict of `insufficient_evidence` or `insufficient_control_data`, **then** the view
   renders it in the visually distinct abstention style (§40).

### US-09 See the model evidence and the drift monitors

**As a** Data Scientist, **I want** to see the evidence behind the registered model and the drift
monitors, **so that** I know whether I can trust it, now and over time.

Acceptance criteria:

1. **Given** a registered champion, **when** I open *Model Evidence*, **then** I see the Qini
   curve with a confidence band, AUUC and Qini coefficient with intervals, the decile table, the
   uplift calibration plot, the SHAP summary, the negative-control result and the
   champion–challenger table (§40, Tab 3).
2. **Given** `GET /model/info`, **then** it returns the registered version, the training date, and
   metrics with confidence intervals (§35).
3. **Given** *Monitoring*, **then** I see feature drift (PSI or KS per feature against the training
   reference), score drift and data-quality status, with alert thresholds taken from configuration
   (§32).
4. **Given** the effect-drift panel, **then** it is labelled as mocked from replay output (§40,
   Tab 4).

### US-10 Retrain under a documented rule

**As a** Data Scientist, **I want** retraining to follow a documented rule and every retrain to be
registered as a new model version with its evaluation attached, **so that** the model behind the
API is always identifiable and can be rolled back.

Acceptance criteria:

1. **Given** the documented rule (a drift threshold combined with a scheduled cadence), **when**
   either condition is met, **then** a retraining run is started **[Q9]**.
2. **Given** a completed retrain, **then** it is logged as a new registered model version with its
   evaluation attached, and the previous version remains available, so a rollback is a version
   pin (§32, §45 deployment strategy).
3. **Given** a candidate that puts a forbidden column in the model matrix, **then** the pipeline
   fails (K7). **Given** a candidate whose negative control does not contain zero in its 95%
   interval, **then** it does not pass K4.

---

## 3. Use cases

Every use case has the same fields. "Abstention verdicts" is always stated in the alternative flows:
either the full handling of `insufficient_evidence` and `insufficient_control_data`, or a one-line
reason why they cannot arise in that use case.

### UC-01 Define a replay scenario

- **Primary actor:** Marketing Analyst
- **Secondary actors:** none
- **Interface:** Dashboard, Tab 1 (Campaign Planner). Stored in `replay_scenario` (§45, ER diagram).
- **Preconditions:**
  1. A champion model is registered in the MLflow Model Registry.
  2. A held-out slice of data exists that can stand in for "next month's campaign" (§14). Which
     split is allowed at which stage is open **[Q1]**.
  3. The analyst is using the dashboard.
- **Main flow:**
  1. The analyst opens the Campaign Planner and selects the dataset and the slice of the held-out
     split that stands in for the audience.
  2. The analyst enters the cost per contact c, the margin per conversion m and the budget.
  3. The system validates that all three inputs are present and numeric.
  4. The system shows the break-even ratio c/m (a business input, exposed as a slider, §10).
  5. The system stores the scenario and the slice membership. The outcome column is withheld from
     everything that follows until UC-06.
  6. The system sets the scenario state to `defined`.
- **Alternative flows:**
  - A1. *Missing or non-numeric input:* the system rejects the submission, names the missing input,
    and creates no scenario.
  - A2. *An outcome column is found in a frame handed to a step before the reveal:* the run is
    invalid and the automated test that guards this boundary fails (§14).
  - A3. *Abstention verdicts:* not applicable here. Nothing is scored yet. Both verdicts first arise
    in UC-02 and are carried into UC-05 and UC-08.
- **Postconditions:** A scenario exists in state `defined`; its inputs (c, m, budget, slice) are
  recorded; no outcome has been read.
- **Implemented this semester or deployment-only:** Yes.
- **Related user stories:** US-01
- **Specification basis:** §14 (replay loop, step 1); §10 (c/m slider); §45 (state diagram, ER).

### UC-02 Score a batch of customers

- **Primary actor:** Marketing Analyst
- **Secondary actors:** Scheduler (scheduled runs); Data Scientist (calls the API directly)
- **Interface:** `POST /score/batch` (CSV or JSON array → scored file + summary);
  single-customer variant `POST /score` (→ full response object); Dashboard, Tab 1.
- **Preconditions:**
  1. A champion model is registered and loaded by version at start-up from the MLflow Model
     Registry (§35).
  2. The preprocessing artefacts, fitted on the training fold only, are available (§23).
  3. The input contains permitted feature columns only (no treatment indicator, no outcome, no
     `exposure`).
  4. Where scoring is part of a replay, the scenario is in state `defined`.
- **Main flow:**
  1. The analyst submits a customer file (or one customer).
  2. The system checks data quality: null rates, range violations, schema changes (§32).
  3. The system runs the forbidden-column check (`validate_features()`, §18).
  4. The system applies the stored preprocessing and scores each customer: `uplift_score` with a
     95% bootstrap interval `uplift_ci_95`, and `decile` (§11, §12).
  5. The abstention logic assigns a `verdict` to every customer (see A3 and A4).
  6. The system fills the remaining schema fields: `customer_id`, `model_version`,
     `segment_label`, `expected_incremental_profit` (m · uplift − c), `recommendation`,
     `top_drivers`, `scored_at` (§12). Where m and c come from outside a scenario is open **[Q4]**.
  7. The system returns the scored file and a summary (or the single response object) and writes
     the scoring run and the scores to the scoring log.
  8. If the scoring is part of a replay, the scenario state becomes `scored`.
- **Alternative flows:**
  - A1. *Forbidden column present:* the run is rejected, the column is named, no row is scored
    (K7, §18).
  - A2. *Data-quality failure or malformed input:* the scoring run is blocked and an error is
    returned. No partial result is presented as valid.
  - A3. *`insufficient_evidence`:* a customer's or segment's 95% interval straddles zero. The
    `verdict` is `insufficient_evidence`. The score and the interval are still returned, but no
    directional recommendation is issued. The system does not round the estimate to a decision
    (§11).
  - A4. *`insufficient_control_data`:* the segment holds fewer than the configured minimum number
    of control-group outcome events. The `verdict` is `insufficient_control_data`. The threshold
    value used is logged with the response (§11).
  - A5. *Single customer:* `POST /score` returns one full response object.
  - A6. *Scheduled run:* the Scheduler starts the same flow without an analyst (§35, batch path is
    the primary production mode). Whether the Scheduler is built is open **[Q7]**.
  - A7. *Model not available:* `GET /health` reports not ready and no score is returned.
  - What `recommendation` and `segment_label` contain under A3 and A4 is open **[Q2]**.
- **Postconditions:** Every returned row has all eleven schema fields, including a `verdict`
  (non-functional requirement, §45). The run is recorded in the scoring log. In a replay the
  scenario is `scored`.
- **Implemented this semester or deployment-only:** Yes for batch and single scoring. The
  scheduled trigger is open **[Q7]**.
- **Related user stories:** US-02, US-03
- **Specification basis:** §11, §12, §18, §32, §35, §45.

### UC-03 Review model evidence

- **Primary actor:** Data Scientist
- **Secondary actors:** Marketing Manager (reads the same screen when asked a question)
- **Interface:** Dashboard, Tab 3 (Model Evidence); `GET /model/info`.
- **Preconditions:**
  1. A champion is registered, with its evaluation attached to the registered version (§32).
  2. The evaluation harness has produced Qini, AUUC and uplift@k with bootstrap intervals, and the
     negative-control result exists (§27, §28).
- **Main flow:**
  1. The Data Scientist opens Model Evidence.
  2. The system calls `GET /model/info` and shows the registered version, the training date, and
     the metrics with confidence intervals.
  3. The system shows the Qini curve with a confidence band, AUUC and Qini coefficient with
     intervals, the decile table, the uplift calibration plot, the SHAP summary, the
     negative-control result and the champion–challenger table (§40).
  4. The system names the library and settings used for Qini and AUUC, and the number of bootstrap
     resamples (§27, K2).
- **Alternative flows:**
  - A1. *Negative control fails:* the 95% interval of the measured uplift on a permuted treatment
    does not contain zero. The screen shows K4 as **fail** and does not hide it (§28).
  - A2. *Champion and challenger not distinguishable:* the bootstrap distribution of their
    difference straddles zero. The screen says they are not distinguishable on this data, and does
    not name a winner (§28).
  - A3. *`insufficient_evidence`:* a decile or segment whose interval straddles zero is shown in
    the abstention style, not as a positive or negative effect (§11 applies to a decile as well as
    a customer).
  - A4. *`insufficient_control_data`:* a decile or segment with fewer than the configured minimum
    control events is shown as such, not as an estimate (§11).
  - A5. *No registered model:* `GET /model/info` returns an error and the tab says so.
- **Postconditions:** Nothing is changed (read-only).
- **Implemented this semester or deployment-only:** Yes.
- **Related user stories:** US-09
- **Specification basis:** §27, §28, §29, §31, §35, §40.

### UC-04 Deliver the target list to a campaign

- **Primary actor:** Marketing Analyst
- **Secondary actors:** Campaign Execution System (**external, deployment-only**)
- **Interface:** Dashboard, Tab 1 (downloadable target list). No endpoint exists for the
  deployment-only steps **[Q3]**.
- **Preconditions:**
  1. The scenario is in state `allocated` (UC-05): a target list and a suppression list exist.
  2. For the deployment-only steps: a Campaign Execution System is connected, and a permanent
     randomized holdout is maintained (§15).
- **Main flow:**
  1. The analyst downloads the target list. *(Built this semester.)*
  2. *(Deployment-only.)* The system sends the target list to the Campaign Execution System and
     records who was assigned to which arm at assignment time (`treatment_assignment`,
     `holdout_membership`).
  3. *(Deployment-only.)* The Campaign Execution System runs the campaign.
  4. *(Deployment-only.)* After the outcome window closes (days to weeks), the Campaign Execution
     System returns real outcomes, and the system ingests them (`outcome_observation`).
  5. *(Deployment-only.)* The measurement job computes the realised effect in aggregate against
     the permanent holdout.
  - In this project steps 2 to 5 are **replaced by UC-06**, which reveals recorded outcomes. This is
    a replay, not a live campaign, and the project does not claim otherwise (§14, §15).
- **Alternative flows:**
  - A1. *Assigned is not received:* a message is not delivered, or a control customer is reached
    through another channel. The estimate stays intention-to-treat. Compliance and contamination
    handling is deployment-only (§6, §15).
  - A2. *Outcome window not yet closed:* the realised effect cannot be computed yet. Only feature
    drift and score drift are available (§32).
  - A3. *`insufficient_evidence` / `insufficient_control_data`:* customers carrying either verdict
    are not placed on the contact list as a directional recommendation. They are reported
    separately as abstentions, so "do not contact" and "we do not know" are not merged (§12).
    How they are labelled in the exported file is open **[Q2]**.
- **Postconditions:** *(Built.)* A downloadable target list file exists. *(Deployment-only.)*
  Assignment and holdout membership are recorded; outcomes are ingested; the scenario would reach
  `measured` on real data.
- **Implemented this semester or deployment-only:** **Partly.** The download is built.
  Everything involving the Campaign Execution System is **deployment-only** and not built.
- **Related user stories:** US-04
- **Specification basis:** §6, §14, §15, §40 (Tab 1), §43 and §44 (campaign lifecycle, export,
  outcome ingestion), §45.

### UC-05 Allocate under budget

- **Primary actor:** Marketing Analyst
- **Secondary actors:** none
- **Interface:** Dashboard, Tab 1 (Campaign Planner); `POST /policy/simulate` (budget, cost,
  margin → policy comparison table); decision layer (`PolicyEngine`).
- **Preconditions:**
  1. The scenario is in state `scored` (UC-02): every customer has an uplift estimate, an interval
     and a verdict.
  2. Per-contact cost c, margin m and budget B are known (UC-01).
- **Main flow:**
  1. The analyst selects a policy mode (§10):
     - *Fixed budget:* contact the top k by estimated uplift, k = B / c; suppress any selected
       customer whose estimated uplift ≤ 0, even if budget is left unspent.
     - *Profit-maximising:* contact every customer whose estimated uplift > c/m; the budget is an
       output, not an input.
     - *Fixed target:* contact the smallest set expected to deliver N incremental conversions;
       report the budget required and the uncertainty on it.
     - *Suppression-only:* contact everyone except negative-uplift customers.
  2. The decision layer applies the mode to the scored customers, working on the ITT-scale
     uplift. The Wald estimate never enters m · uplift − c (§10, §6).
  3. The decision layer applies the sleeping-dog rule: any customer whose estimated uplift is
     negative and whose confidence interval lies entirely below zero goes on the suppression list
     and is never contacted by the policy, regardless of remaining budget (§10).
  4. The decision layer leaves out customers whose verdict is an abstention (see A3 and A4).
  5. The system marks a randomized slice of the selected targets as not-contacted. This holdout is
     what makes UC-06 possible (§14, step 4).
  6. The system shows: the number of customers to contact, expected incremental conversions with a
     confidence interval, expected incremental profit, and a line comparing the policy with
     targeting by response score (§40).
  7. The scenario state becomes `allocated`.
- **Alternative flows:**
  - A1. *k = 0, or every estimated uplift ≤ 0:* the target list is empty, and under *Fixed budget*
    the budget stays unspent (§10).
  - A2. *Fixed target N cannot be reached:* behaviour not specified **[Q5]**.
  - A3. *`insufficient_evidence`:* a customer whose interval straddles zero is not selected as a
    directional recommendation and is shown in the abstention style (§11).
  - A4. *`insufficient_control_data`:* a customer in a segment with too few control events is
    treated the same way, and the configured threshold is logged (§11).
  - A5. *Suppression versus abstention:* a customer with a negative estimate whose interval
    *straddles* zero is not on the suppression list, because the sleeping-dog rule needs the
    interval entirely below zero. This sits uneasily with *Fixed budget* suppressing every
    estimate ≤ 0 **[Q2]**.
- **Postconditions:** The scenario is `allocated`. The target list, the suppression list, the
  holdout and the assignment records are stored. Steps 1 to 5 never touched the outcome column, and
  a test asserts it (§14).
- **Implemented this semester or deployment-only:** Yes.
- **Related user stories:** US-03, US-04
- **Specification basis:** §10, §11, §14, §35, §40.

### UC-06 Reveal outcomes and measure

- **Primary actor:** Marketing Analyst
- **Secondary actors:** Finance / Campaign Audit (reads the result)
- **Interface:** A replay job started from the dashboard, with results shown in Tab 4 and Tab 2.
  The six endpoints of §35 contain no reveal or measure endpoint **[Q3]**.
- **Preconditions:**
  1. The scenario is in state `allocated` (UC-05): targets are fixed and the holdout is marked.
  2. Steps 1 to 4 of the replay did not read the outcome column (§14).
- **Main flow:**
  1. The analyst triggers the reveal.
  2. The system reads the recorded outcomes of the slice for the first time. State: `revealed`.
  3. The system measures the incremental effect as the treated rate minus the holdout rate, with a
     95% bootstrap interval that resamples within treatment arms (§14, §28, at least 1,000
     resamples, K2).
  4. The system compares predicted with measured incremental conversions, each with an interval,
     for the uplift policy, the response-model policy, blanket contact and random contact, at an
     identical budget (§14).
  5. The system converts the result to incremental profit using the stated cost and margin (K3),
     and records predicted-versus-measured differences per decile (K5).
  6. The system logs the whole run to MLflow. State: `measured`.
- **Alternative flows:**
  - A1. *Reveal requested before allocation:* refused, because the state machine allows reveal only
    from `allocated` (§14).
  - A2. *`insufficient_evidence`:* a measured effect whose 95% interval contains zero is reported as
    not distinguishable from zero and is shown in the abstention style. It is not rounded to a
    decision (§11, K1).
  - A3. *`insufficient_control_data`:* a policy or decile whose holdout holds fewer than the
    configured minimum number of control events is reported as `insufficient_control_data`, and no
    measured effect is claimed for it (§11).
  - A4. *Outcome leak detected:* if the test that asserts "no outcome column before the reveal"
    fails, the run is invalid and its result is not reported (§14).
- **Postconditions:** The scenario is `measured`. A calibration record exists. The run is in MLflow.
  Every figure is an *estimated* incremental effect on held-out experimental data, not realised
  money (§13).
- **Implemented this semester or deployment-only:** Yes, as a simulated replay over recorded
  outcomes. A live campaign is deployment-only (UC-04, §15).
- **Related user stories:** US-05, US-07
- **Specification basis:** §13, §14, §28, §46 (K1, K3, K5).

### UC-07 Compare policies

- **Primary actor:** Marketing Manager
- **Secondary actors:** Marketing Analyst; Finance / Campaign Audit
- **Interface:** Dashboard, Tab 2 (Policy Comparison); `POST /policy/simulate`.
- **Preconditions:**
  1. A held-out randomized set, a per-contact cost, a margin and a budget are available (§13).
  2. The champion model and the response-model baseline both exist.
- **Main flow:**
  1. The Marketing Manager opens Policy Comparison.
  2. The manager enters, or reuses, the budget, cost and margin.
  3. The system calls `POST /policy/simulate`, which replays the uplift policy, the response model,
     blanket contact and random contact over the held-out set at an identical budget.
  4. The system shows one table: number contacted, total cost, incremental conversions with a
     confidence interval, incremental profit, and profit per unit spent, for each policy (§13).
  5. The system shows overlaid cumulative-incremental curves with bootstrap bands.
  6. The manager moves the budget slider; one vertical line moves across all curves at once (§40).
  7. The system states the headline comparison in words, with its interval (K1).
- **Alternative flows:**
  - A1. *`insufficient_evidence`:* the 95% interval of the difference between the uplift policy and
    the response model contains zero. The screen reports that the policies are not distinguishable
    on this data and names no winner. This is "a reportable finding, not a failed project" (K1,
    §28). Applying the abstention state to a policy-level figure extends §11 **[Q2]**.
  - A2. *`insufficient_control_data`:* a policy whose holdout has too few control events is shown
    in the abstention style instead of a number (§11).
  - A3. *Profit shown as money saved:* not allowed. Every profit figure is labelled estimated, with
    cost and margin on the same line (§13, K3).
- **Postconditions:** Read-only. The result is available for the final presentation, where the
  simulator output is the artefact to open with (§13).
- **Implemented this semester or deployment-only:** Yes.
- **Related user stories:** US-06, US-07
- **Specification basis:** §13, §28, §35, §40, §46 (K1, K3).

### UC-08 Inspect a customer explanation

- **Primary actor:** Marketing Analyst
- **Secondary actors:** Data Scientist
- **Interface:** Dashboard, customer detail view; `POST /score` returns the same object.
- **Preconditions:** The customer has been scored (UC-02) and the scoring run is recorded.
- **Main flow:**
  1. The analyst selects a customer row.
  2. The system shows the full response object (§12):
     `customer_id`, `model_version`, `uplift_score`, `uplift_ci_95`, `decile`, `segment_label`,
     `expected_incremental_profit`, `recommendation`, `verdict`, `top_drivers`
     (feature and signed contribution), `scored_at`.
  3. The system shows `segment_label` as one of `persuadable`, `sure_thing`, `lost_cause`,
     `sleeping_dog`, `unknown`, and states that it is an inference from the estimated effect plus
     the predicted baseline conversion probability, never ground truth (§12).
  4. The system describes `top_drivers` as features that drive the model's uplift prediction, never
     as causes (§31).
- **Alternative flows:**
  - A1. *`insufficient_evidence`:* the customer's interval straddles zero. The view shows the score
    and the interval, and renders the verdict in the distinct abstention style, with no
    directional recommendation (§11, §40).
  - A2. *`insufficient_control_data`:* the customer's segment has fewer than the configured minimum
    control events. The view shows this verdict in the same distinct style, and the threshold used
    is visible from the log (§11).
  - A3. *Unknown customer:* the customer is not in the scoring run; the view reports it.
  - A4. *Criteo:* `top_drivers` name anonymised features (`f0` to `f11`), so a plain-language
    description is possible on Hillstrom only (§29, §52).
  - Whether `segment_label` becomes `unknown` under A1 and A2 is open **[Q2]**.
- **Postconditions:** Read-only.
- **Implemented this semester or deployment-only:** Yes.
- **Related user stories:** US-03, US-08
- **Specification basis:** §11, §12, §31, §40.

### UC-09 Review drift

- **Primary actor:** Data Scientist
- **Secondary actors:** Scheduler (produces the drift reports)
- **Interface:** Dashboard, Tab 4 (Monitoring); `GET /monitor/drift` (latest drift report).
- **Preconditions:**
  1. A training reference distribution is stored.
  2. At least one scored batch exists to compare against it.
  3. Alert thresholds are set in configuration (§32).
- **Main flow:**
  1. The Data Scientist opens Monitoring.
  2. The system calls `GET /monitor/drift` and shows the latest drift report.
  3. The system shows feature drift (PSI or KS statistic per feature, scored batch against the
     training reference), score drift (distribution of predicted uplift over time; a collapsing
     spread is an early warning) and data-quality status (null rates, range violations, schema
     changes) (§32).
  4. The system shows the registered model version, its training date, and the results of each
     simulated campaign replay (§40).
  5. The system shows the effect-drift panel, mocked from replay output and labelled as such.
- **Alternative flows:**
  - A1. *Threshold breached:* the system raises an alert. The retraining rule may then start UC-10.
  - A2. *Data-quality failure:* the scoring run is blocked (§32).
  - A3. *No drift report yet:* the tab says so. The specification does not define the response of
    `GET /monitor/drift` in this case.
  - A4. *Abstention verdicts:* the two verdicts are not produced by the drift monitors. They appear
    only where the effect-drift panel shows a replay's measured effect: an interval containing zero
    is shown as `insufficient_evidence`, and too few control events as `insufficient_control_data`.
- **Postconditions:** Read-only.
- **Implemented this semester or deployment-only:** **Partly.** Feature drift, score drift and
  data-quality status are built and can be run on historical data split by time or on a simulated
  shifted population. **Effect drift** needs a permanent randomized holdout and a real outcome
  window and is **deployment-only**: the panel is mocked. The early-proxy outcome and the
  predicted-versus-measured calibration after an outcome window are designed and documented, not
  implemented (§32).
- **Related user stories:** US-09
- **Specification basis:** §15, §32, §35, §40.

### UC-10 Retrain the model

- **Primary actor:** Scheduler (scheduled cadence)
- **Secondary actors:** Data Scientist (starts a run by hand, or acts on a drift alert)
- **Interface:** The training pipeline and the MLflow Model Registry. No endpoint among the six of
  §35 **[Q3]**.
- **Preconditions:**
  1. A documented retraining rule exists: a drift threshold combined with a scheduled cadence
     (§32).
  2. The pipeline is reproducible from a configuration file and a git commit hash (§33, K8).
  3. Data for the new run is available.
- **Main flow:**
  1. A trigger fires: a drift threshold is breached (UC-09), the scheduled cadence is reached, or
     the Data Scientist starts a run.
  2. The pipeline re-runs: schema validation, leakage audit, joint stratified split, baselines
     first, uplift estimators, evaluation with bootstrap intervals (§23).
  3. The candidate is compared with the current champion on the same bootstrap resamples (§28).
  4. The candidate is logged to MLflow as a new registered model version with its evaluation
     attached; the model lifecycle moves `registered` → `staging` → `production` → `archived`
     (§45).
  5. The API loads the model version selected at start-up, and returns that version in every
     response (§35).
- **Alternative flows:**
  - A1. *Forbidden column reaches the model matrix:* the pipeline fails (K7).
  - A2. *Negative control fails:* the candidate does not pass K4.
  - A3. *Candidate not distinguishable from the champion:* the difference distribution straddles
    zero (§28). The rule for promotion is not specified **[Q9]**.
  - A4. *Rollback:* the previous version stays registered, so a rollback is a version pin (§45).
  - A5. *Abstention verdicts:* not produced by retraining itself. The new version's evaluation
    reports intervals, and any decile whose interval straddles zero or whose control events are
    below the minimum is reported as `insufficient_evidence` or `insufficient_control_data` (UC-03).
- **Postconditions:** A new registered model version exists with its evaluation attached. The
  previous version is retained.
- **Implemented this semester or deployment-only:** **Partly [Q9].** The retraining rule is
  documented, and a retraining run is the same reproducible pipeline run again. An automatic
  production trigger driven by a Scheduler, and retraining on real outcomes, are
  **deployment-only**.
- **Related user stories:** US-10
- **Specification basis:** §15, §28, §32, §33, §35, §45.

---

## 4. Traceability

### 4.1 User story ↔ use case

| | UC-01 | UC-02 | UC-03 | UC-04 | UC-05 | UC-06 | UC-07 | UC-08 | UC-09 | UC-10 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| US-01 | ● | | | | | | | | | |
| US-02 | | ● | | | | | | | | |
| US-03 | | ● | | | ● | | | ● | | |
| US-04 | | | | ● | ● | | | | | |
| US-05 | | | | | | ● | | | | |
| US-06 | | | | | | | ● | | | |
| US-07 | | | | | | ● | ● | | | |
| US-08 | | | | | | | | ● | | |
| US-09 | | | ● | | | | | | ● | |
| US-10 | | | | | | | | | | ● |

Coverage check: every use case is reached by at least one story, and every story reaches at least
one use case.

### 4.2 Use case ↔ interface ↔ specification

This table is the input for T33 (each functional requirement must be traceable to an API endpoint
and a test), T45 (API contract), T46 (use-case diagram) and T51 (test plan).

| Use case | Dashboard | API endpoint (§35) | Specification § | Status |
|---|---|---|---|---|
| UC-01 | Tab 1 | none **[Q3]** | 10, 14, 45 | Yes |
| UC-02 | Tab 1 | `POST /score/batch`, `POST /score` | 11, 12, 18, 32, 35 | Yes |
| UC-03 | Tab 3 | `GET /model/info` | 27, 28, 29, 31, 40 | Yes |
| UC-04 | Tab 1 (download) | none **[Q3]** | 6, 14, 15, 43, 44 | Partly |
| UC-05 | Tab 1 | `POST /policy/simulate` | 10, 11, 14 | Yes |
| UC-06 | Tab 4, Tab 2 | none **[Q3]** | 13, 14, 28, 46 | Yes |
| UC-07 | Tab 2 | `POST /policy/simulate` | 13, 28, 40, 46 | Yes |
| UC-08 | Customer detail | `POST /score` | 11, 12, 31, 40 | Yes |
| UC-09 | Tab 4 | `GET /monitor/drift` | 15, 32, 40 | Partly |
| UC-10 | none | none **[Q3]** | 15, 28, 32, 33, 45 | Partly **[Q9]** |

`GET /health` (liveness and readiness) is used by UC-02 (A7) and is otherwise an operational
endpoint, not tied to a use case.

Suggested convention for T33 and T34: every requirement gets an ID (FR-xx, NFR-xx) that names the
use case it comes from, and every test name carries the requirement ID, as the milestone plan
requires.

---

## 5. Open questions and assumptions

Raised from writing this document. None is decided here; each needs an owner. Where a decision is
needed, it goes into `docs/decisions/` as a numbered record (working agreement, §6).

| ID | Question | Why it matters | Suggested owner |
|---|---|---|---|
| Q1 | The specification says the locked test split is "touched exactly once, at champion confirmation" (§18), and the plan opens it once on 1 December (T70). Yet K1, K2 and K3 are "measured on the locked test split" and first reported at M4 (§46), and §13 and §14 replay on "the held-out test split". A replay run repeatedly for development, or for every demo, would reopen it. Which split does each replay use at each stage? A natural reading: development and demo replays use the validation split or a pre-agreed replay slice, and the locked test split is used once, in T70. | Reopening the locked split is the failure the project exists to avoid | Ali with Mahmoud |
| Q2 | The rules linking `verdict`, `recommendation` and `segment_label` are not specified. (a) What does `recommendation` hold when the verdict is `insufficient_*`? §12 only says the fields are separate on purpose. (b) *Fixed budget* suppresses every estimate ≤ 0, but the sleeping-dog rule needs the interval entirely below zero, and *Suppression-only* says "negative-uplift customers" without saying which test. (c) May `segment_label` be `unknown` under abstention? (d) Does the abstention state also apply to a policy-level figure (UC-07)? §11 names customer, decile and policy for the interval but states the verdict rule for a segment. | These fields are frozen in T36 and drive T33, T55 and the API tests | Ali (T55) with Mahmoud (T36) |
| Q3 | The six endpoints of §35 do not cover defining a scenario, revealing outcomes, measuring, or retraining. Are these dashboard-triggered jobs, command-line jobs, or additional endpoints? | T45 freezes the API contract; T33 needs a requirement for each use case | Mazen (T45) |
| Q4 | `expected_incremental_profit` is m · uplift − c, but `POST /score` and `POST /score/batch` take no m and c in §35. Where do they come from outside a scenario: the request, or configuration? | The response schema cannot be filled otherwise | Ali, Mazen |
| Q5 | *Fixed target:* what happens when no set of customers is expected to deliver N incremental conversions? | Decision-layer edge case (T55) | Ali |
| Q6 | The use-case-diagram actors (§45) and the target users (§1.5) differ. This document keeps six actors. T31 and T46 must use the same names. | Three documents would otherwise describe different actors | Mohammed (T31), Ali (T46) |
| Q7 | The Scheduler is an actor in §45, but the specification does not say whether it is built. Scheduled batch scoring, scheduled drift reports and scheduled retraining all depend on it. | Decides what is "Yes" and what is "Deployment-only" in UC-02, UC-09, UC-10 | Mazen, Mahmoud |
| Q8 | `01-specification.md` is unnumbered, and its internal cross-references use different numbers from the PDF (for example, it refers to the schema as Section 9 and the deployment gap as Section 12; the PDF numbers them 12 and 15). This document cites the PDF numbers. | A reviewer following the Markdown copy would land on the wrong section | Mahmoud |
| Q9 | UC-10 and US-10: the specification lists "retrain" as a use case and "a documented retraining rule" as SHOULD HAVE, but does not say whether a retraining run is built or only documented, or what makes a retrained candidate replace the champion. A new champion would also normally need confirming on the locked split (see Q1). | Decides the status "Partly" and the acceptance criteria of US-10 | Mahmoud |
