## Risk Register

### Rating scale

**Likelihood**

* **Low** — unlikely, but has happened on projects like this
* **Medium** — more likely than not to happen at least once
* **High** — expect it to happen

**Impact**

* **Low** — costs hours
* **Medium** — costs days, or weakens one deliverable
* **High** — threatens a milestone or a headline claim

---

> ***Reviewed at every milestone boundary. Owners update their own
> rows.***

| ID  | Risk | Category | Likelihood | Impact | Early warning signal | Mitigation | Owner | Status | Last reviewed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | The central hypothesis does not hold: uplift targeting does not outperform the response model on the project data. | modelling | Medium | High | The uplift-vs-response comparison due Fri 6 Nov shows no meaningful advantage over the response-model baseline. | Run the comparison by Fri 6 Nov and investigate the result early rather than waiting until the final report. | Mohammed | open | 2026-10-14 |
| R02 | Effects are too small to distinguish from statistical noise. | modelling | Medium | High | Bootstrap confidence intervals for the headline uplift result include zero or remain too wide to support a clear conclusion. | Evaluate both visit and conversion outcomes, report confidence intervals, and use the abstention path when evidence is inconclusive. | Ali | open | 2026-09-29 |
| R03 | Curriculum gap because the project does not include computer vision or NLP. | schedule | Medium | High | Instructor feedback indicates that CV/NLP coverage is expected or may affect evaluation. | Raised explicitly during the Week 7 review; the project was approved as it stands. No further mitigation is required unless the scope changes. | Mahmoud | resolved | 2026-09-29 |
| R04 | Uneven team workload leaves some members without substantial ownership. | team | Medium | High | A member has no clearly owned component, deliverable, or contribution to the final demo. | Maintain independent workstreams and require every member to own a component that appears in the final system/demo. | Mahmoud | open | 2026-09-29 |
| R05 | Metric overfitting caused by repeatedly tuning against the locked test split. | modelling | Medium | High | Test-set results are used to make additional modelling or hyperparameter decisions after the split has been opened. | Lock the test split, document the rule before opening it, and use validation/development data for tuning. | Ali | open | 2026-09-29 |
| R06 | Silent data leakage contaminates uplift estimates. | data | Medium | High | A forbidden feature/column appears in the modelling data or the permuted-treatment negative control produces an unexpected effect. | Automate forbidden-column checks and run the permuted-treatment negative control. | Aliaa | open | 2026-09-29 |
| R07 | Cross-fitting is implemented incorrectly in the X-learner or R-learner. | modelling | Medium | High | A test shows that training and prediction folds overlap or fold membership does not match the intended cross-fitting design. | Prefer a tested library implementation and add tests that explicitly verify fold membership. | Mohammed | open | 2026-09-29 |
| R08 | Scaling problems when processing the 14M-row Criteo dataset. | engineering | Low | Medium | Full-data processing exceeds available memory, becomes impractically slow, or fails during loading/preprocessing. | Use Parquet with selected columns, chunked preprocessing, and a stratified subsample for iteration; use the full dataset for final reporting. | Aliaa | open | 2026-09-29 |
| R09 | Azure/cloud credits are exhausted before deployment and experimentation are complete. | engineering | Low | Medium | Cloud spending approaches the configured budget threshold or compute resources remain running after a session. | Train locally where practical, shut down compute after each session, and configure a budget alert below the credit limit. | Mazen | mitigating | 2026-09-29 |
| R10 | Live demo infrastructure fails on presentation day. | engineering | Low | Medium | The deployed endpoint or dashboard cannot be reached or fails during a rehearsal. | Maintain a recorded demo and a local fallback mode that can run without the cloud endpoint. | Mazen | open | 2026-09-29 |
| R11 | The development environment works on one machine but fails on another team member's machine or in CI. | engineering | Medium | High | `verify_environment.py` produces different results across team members, or a dependency/version mismatch appears between machines or CI. | Pin the environment, run the verification script on every team member's machine, and require all five members to produce passing results. | Mazen | mitigating | 2026-09-29 |
| R12 | Environment verification gives a false passing result because a required dependency is pinned but not actually checked. | engineering | Medium | High | A required package is present in the environment specification but missing from `verify_environment.py` checks, allowing the script to report success despite incomplete verification. | Keep the verification checks synchronized with the pinned environment and update the check count whenever a required dependency is added. | Mazen | mitigating | 2026-09-29 |
| R13 | Branching, pull-request review, or merge workflow delays integration of completed work. | engineering | Medium | Medium | A task branch is not correctly pushed/selected for comparison, a required review remains unresolved, or a PR cannot be merged when the task is otherwise complete. | Follow the agreed branch/PR workflow, verify the correct base and compare branches, resolve review conversations promptly, and require approval before merging. | Mazen | mitigating | 2026-09-29 |
| R14 | A team member becomes unavailable for an extended period. | team | Medium | High | Two missed standups in the team WhatsApp group, or a task is 48 hours past due with no commit, PR or message. | Every task has a named partner who can take it over. If the owner is unavailable, the team leader transfers the task to the partner after 48 hours and records the change in the issue history; raise the issue in the team channel within two days rather than absorbing it silently. | Mahmoud | open | 2026-10-14 |
| R15 | The company repository is never granted, leaving the project on a personal account. | engineering | Medium | Low | The company has not responded about repository access. | Transfer the repository to the organisation if required; preserve the Discord `#github` channel as the parallel record either way. | Mahmoud | open | 2026-10-14 |
| R16 | A critical project milestone slips, reducing the time available to react to its findings. | schedule | Medium | High | The randomization-integrity report is not on track for Fri 30 Oct, or the uplift-vs-response comparison is not on track for Fri 6 Nov. | Track the two protected dates explicitly, review progress at each milestone boundary, and protect Fri 30 Oct and Fri 6 Nov before lower-priority work. | Mahmoud | mitigating | 2026-10-13 |
| R17 | DEPI deadlines remove approximately six weeks from the original project schedule, leaving insufficient time for the full planned scope. | schedule | High | High | Any M5 task is past its due date in `docs/project-plan.md`, or the 27 Nov feature freeze is at risk. | Follow the scope order in `docs/project-plan.md` section 3; write tests alongside implementation during M5 rather than deferring testing; enforce the feature freeze on 27 Nov. | Mahmoud | mitigating | 2026-10-14 |
