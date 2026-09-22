# Project plan

**NetLift — A Causal Decision System for Incremental Profit Optimization**

DEPI Microsoft Machine Learning Engineer Track · graduation project
Team leader: Mahmoud Nadeem · five members

This document is the committed project plan required by the DEPI documentation
checklist: timeline, milestones, deliverables and resource allocation. It is the
version of record. Where it disagrees with any slide or message, this file wins.

---

## 1. Timeline

Programme weeks. `##` is scheduled work, `..` is deliberate slack.

```
                                  W5  W6  W7  W8  W9  W10 W11 W12 W13 W14 W15 W16 W17 W18 W19 W20 W21
                                  -------------------------------------------------------------------
M1  Team formation and planning   ##  ##
M2  Instructor review                     ##
    Buffer                                    ..
M3  Requirements gathering                        ##  ##
    Buffer                                                ..
M4  Analysis and design                                       ##  ##
M5  Implementation                                                    ##  ##  ##  ##  ##
M6  Testing                                                                               ##  ##
M7  Release and presentation                                                                      ##
```

The two buffer weeks are not padding. Week 8 absorbs whatever the instructor
review asks for. Week 11 absorbs requirements that arrive late or change. A plan
with no slack fails at the first surprise.

---

## 2. Milestones

| ID | Milestone | Weeks | Exit condition |
|---|---|---|---|
| M1 | Team formation and planning | 5–6 | Proposal submitted and approved; repository, environment, CI and MLflow all working on all five machines; both datasets downloaded and their published figures verified against the files |
| M2 | Instructor review | 7 | Written feedback received and every point answered in writing |
| M3 | Requirements gathering | 9–10 | Functional and non-functional requirements, stakeholder analysis, use cases; **randomization-integrity report complete** |
| M4 | Analysis and design | 12–13 | Architecture, UML, ERD, API contract; **uplift versus response-model comparison complete** |
| M5 | Implementation | 14–18 | All estimators trained and tracked; decision layer, API and dashboard working; simulated campaign replay runs end to end |
| M6 | Testing | 19–20 | Unit and integration tests passing in CI; error analysis; permuted-treatment negative control collapses to approximately zero uplift |
| M7 | Release and final presentation | 21 | Tagged release, final report, presentation delivered, demo rehearsed |

### Two dates that are protected

**End of week 10 — randomization-integrity report.**
Standardized mean differences across treatment and control, plus a
treatment-predictability AUC. If the randomization in the data is not what the
publication claims, every downstream result is built on sand. This has to be
known before the modelling work starts, not after.

**End of week 13 — uplift versus response-model comparison.**
A response model is the honest baseline. If uplift modelling does not beat it on
this data, that is a legitimate scientific finding and the project reports it as
one. Found in week 13, there is time to investigate, extend and write it up
properly. Found in week 19, it is a failure with nothing to say about it.

This is why milestone 4 contains a modelling spike even though it is nominally a
design milestone. The overlap is deliberate.

---

## 3. Deliverables

| Milestone | Deliverables |
|---|---|
| M1 | Project proposal · project plan (this file) · KPI definitions · risk register · repository skeleton · CI workflow · environment setup and verification · dataset download script and source documentation · causal glossary · first exploratory chart |
| M2 | Instructor feedback log · written response to every point raised |
| M3 | Stakeholder analysis · user stories and use cases · functional and non-functional requirements · randomization-integrity report |
| M4 | Software architecture · use case, class, sequence and state diagrams · ERD · API contract · uplift versus response-model comparison |
| M5 | Meta-learners (S, T, X, R, DR) · uplift trees and forests · deep models · decision layer · FastAPI service · dashboard · simulated campaign replay · MLflow experiment record |
| M6 | Unit and integration test suite · error analysis · negative-control results · explainability analysis · performance testing |
| M7 | Tagged release · final report · presentation deck · recorded or live demo · README complete |

---

## 4. Resource allocation

| Slot | Member | Area of ownership |
|---|---|---|
| M1 | Aliaa Gomaa | Data engineering and experiment validity |
| M2 | Mohammed Sherif | Uplift modelling core |
| M3 | Mahmoud Nadeem | Advanced models and experiment tracking |
| M4 | Ali Alaa | Evaluation, statistics and the decision layer |
| M5 | Mazen Mahmoud | MLOps, backend and product |

Mahmoud Nadeem holds the team leader role in addition to slot M3.

### Working model

Every task has an **owner** and a **partner**. The partner works in one of four
modes: review, pair, verify, or consume. The partner is drawn from a different
area of ownership wherever possible.

This costs roughly 25 percent more hours than one person per task. It buys three
things: nothing in the project is understood by only one person, every artefact
is seen by a second pair of eyes before anyone builds on it, and each member
learns an area that is not their own.

Nobody reviews or approves their own work. This is enforced by the repository,
not by agreement: `main` is protected, every change arrives as a pull request,
one approving review is required, and the bypass list is empty.

### Effort

Milestone 1 is planned at approximately 88 person-hours across five members and
two weeks, which is 8 to 9 hours per member per week. Later milestones are
estimated in their own milestone documents.

---

## 5. Tracking

| What | Where |
|---|---|
| Tasks, owners, dependencies | GitHub Issues, grouped by GitHub Milestone |
| Code review record | GitHub Pull Requests |
| Individual contribution | GitHub Insights, contributors view |
| Decisions and rejected alternatives | Discord `#decisions`, append only |
| Daily progress | Discord `#standup` — did, next, blocked |
| Blockers | Discord `#help`, after 60 minutes stuck |
| Experiment results | MLflow |

Every task is an issue. Every issue closes through a pull request that names it.
The chain from task to person to branch to review to merged commit is unbroken
and machine-readable, which is what makes individual contribution provable
rather than asserted.
