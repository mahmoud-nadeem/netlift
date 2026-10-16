# 0002 – Replay Runs as Offline Jobs

Status: proposed
Date: 2026-10-14
Deciders: Mahmoud (proposer); Ali and Mazen approve in review

## Context

The specification's API (§35, Deployment Architecture) has six endpoints: `POST /score`,
`POST /score/batch`, `POST /policy/simulate`, `GET /model/info`, `GET /monitor/drift` and
`GET /health`. The user stories and use cases in T32 (#93) found four use cases that none of them
covers: defining a replay scenario (UC-01), delivering a target list (UC-04), revealing outcomes
and measuring (UC-06), and retraining (UC-10). Ali's open question Q3 in that document asks
whether these are dashboard-triggered jobs, command-line jobs or additional endpoints. Mazen owns
the answer, because T45 freezes the API contract (Fri 30 Oct) and T33 needs a way to realise each
use case.

Aliaa's review of #93 raised two related points:

- **Point 3.** UC-07 re-runs `POST /policy/simulate`, while UC-06 step 4 already compares the same
  four policies. Does UC-07 read UC-06's result, or simulate again?
- **Point 8.** UC-05 marks a random slice of the selected targets as not contacted. In a replay
  over recorded data, where does that holdout come from?

Point 8 has a direct answer. In a replay over recorded data nobody can be held out: every outcome
was recorded under the arm the customer actually received in the original experiment. The
specification's replay step [4] is corrected accordingly in T52 (#94). This record states the
rule again, because the replay code built under this decision must follow it.

## Options considered

### New endpoints

Add endpoints for the missing use cases, for example one to create a scenario, one to reveal and
measure it, and one to start a retraining run.

**For.** Every use case gets an API path, and T33 can map each one to an endpoint.

**Against.** The contract grows beyond the specification, and the API becomes stateful: a
scenario moves through `defined` → `scored` → `allocated` → `revealed` → `measured`, and the
service would have to hold that state. A reveal endpoint makes it easy to reveal outcomes again
and again, which is close to the repeated look at held-out data that the locked-split rule exists
to prevent. Retraining is a long-running job, a poor fit for a request–response call. All of this
lands on Mazen inside 3.5 weeks of implementation.

### Offline jobs

The recorded replay and retraining run as command-line jobs. Each job reads a configuration file
and logs to MLflow through `netlift.utils.tracking`. The API keeps its six endpoints.

**For.** It matches how the rest of the pipeline already runs: from configuration plus a git
commit, logged to MLflow (K8). A recorded run is reproducible and auditable, and the API contract
stays as specified.

**Against.** The dashboard cannot start a recorded replay; it shows the recorded results. The
interactive comparison (`POST /policy/simulate`) and the recorded job must share code, or they
will drift apart.

### Dashboard only

The Streamlit app runs the replay in its own process.

**For.** Simplest to wire up for the demo.

**Against.** It breaks the API-first design (§35: the ML side is a service that any frontend can
call). Replay logic would live in the UI, outside the tested package. Runs would not be logged
unless extra code is written, and the dashboard would need direct access to the model and the
data.

## Decision

1. **A recorded replay is an offline command-line job, logged to MLflow (UC-06).** One run covers
   steps [1]–[8] of the specification's replay (§14): define the scenario from a configuration
   file, score, allocate, split by recorded arm, reveal, measure, compare, record. The job is the
   only way a replay result becomes a recorded result.
2. **`POST /policy/simulate` re-runs the same replay code for a given budget, cost and margin
   (UC-07).** It simulates again; it does not read a stored UC-06 result. This answers Aliaa's
   point 3. A simulate call is an interactive what-if, and it does not create a recorded run.
   Its request fields beyond budget, cost and margin (the scenario or data slice, the model
   version and the seed) are deferred to T45 (#57); see "Not decided here".
3. **Retraining is an offline pipeline job (UC-10).** A retraining run is the same reproducible
   pipeline run again. The scheduled trigger is deployment-only. What makes a retrained candidate
   replace the champion stays open (Q9).
4. **The API keeps the six endpoints of the specification.** No scenario, reveal or retrain
   endpoint is added.

## Consequences

**The replay's holdout is the control arm.** The replay draws no random holdout. Step [4] splits
the selected targets by their recorded arm, and those in the original control arm are the
holdout. Step [6] measures the incremental effect as the treated rate minus the control rate
within the selected targets, with a bootstrap interval resampled within arms (T52, #94). The same
rule applies in the recorded job and in `POST /policy/simulate`. A permanent randomized holdout
is a deployment requirement only (§15).

**One replay implementation.** The command-line job and `POST /policy/simulate` call the same
replay code in the `netlift` package. A test asserts that, for the same inputs and seed, both give
the same result.

**For T45 and T33.** T45 freezes six endpoints. In T33, the functional requirements for UC-01,
UC-06 and UC-10 are realised by the replay job or the training pipeline, not by an endpoint.
UC-04 (delivery to a campaign system) stays deployment-only, with the target list downloadable
from the dashboard.

**For T32.** UC-05 and UC-06 need the holdout wording changed: no random slice; the holdout is the
selected targets in the control arm. UC-06's interface becomes the replay job, and UC-07's flow
re-simulates.

**For the dashboard (T62).** Tab 4 and Tab 2 show recorded replay runs. Whether the dashboard
reads them through `GET /model/info` or from MLflow directly is for T45 and T62 to settle.

**Not decided here.**

- **Which split each replay uses at each stage (Q1)**, including when the locked test split is
  opened. Q1 stays open. Until it is decided, the interim rule follows from the locked-split
  rule: replays run only on the validation split, and the locked test split is opened only
  through `LockedTestSplit.open(reason)` (interface 1, #92) under the T70 procedure.
- **The request fields of `POST /policy/simulate`** beyond budget, cost and margin: how a call
  names the scenario or data slice, the model version and the seed. Deferred to T45 (#57), which
  freezes the API contract.

**Limitations / trade-offs.** A user cannot start a recorded replay from the dashboard; someone
runs the job. For the demo, recorded runs are prepared in advance, and `POST /policy/simulate`
gives the live, interactive part.

## What would change this decision

- The DEPI evaluation requiring every use case to be triggerable through the API or the
  dashboard.
- The Scheduler (Q7) being built this semester, which would need a trigger for scheduled jobs.
- `POST /policy/simulate` turning out too slow on Criteo for an interactive call, which would
  push it towards reading precomputed results instead of simulating again.
