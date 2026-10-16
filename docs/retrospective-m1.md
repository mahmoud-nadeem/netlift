# Milestone 1 retrospective

**Milestone 1 — Project Planning and Management** · task T22 (#20)

Each of the five members gave three points on #20: one thing that worked, one thing that did not,
and their hours against the plan. This file consolidates them. Where a change already has a home
in the [working agreement](working-agreement.md) or in a planned task, it is linked there;
otherwise it is proposed as an addition.

---

## 1. What went well

**The owner–partner pairing caught real errors.** Every member named it, in one form or another.
A second person re-running the T16 verification script found the `exposure` column in Criteo,
which the specification said did not exist; Section 6 was corrected as a result (#27). Review
also caught a claim in the uplift-library decision that came from a web search and did not hold
against the installed package (ADR 0001). It caught a coverage gap in the environment
verification script, and a Python version pin that had no Windows wheel for one of the
dependencies. In each case the error was found by someone other than its author.

**Evidence over opinion.** Choosing the uplift library by scoring the same predictions with both
libraries, with a fixed seed (`scripts/qini_library_comparison.py`), showed that their Qini scores
are different quantities before anyone compared numbers across them. Checking the datasets'
published figures against the actual files (T6, T16, T17) gave the project a verified data setup
and the unit-of-observation decision.

**A local checklist before a pull request.** Running the same checks CI runs (ruff, format, mypy,
pytest, environment verification) before opening a PR gave a clear, repeatable bar.

---

## 2. What went wrong

**Branch protection arrived mid-review.** The required CI check and the "up to date with `main`"
rule were switched on while pull requests were already open. A branch that had never run CI then
needed an unplanned merge and two rounds of lint fixes before it could be reviewed.

**Local checks passed, CI failed.** A branch validated against an old copy of `main` passed
locally but failed in GitHub Actions, because `main` had moved on (a script already merged there
brought new lint findings).

**Reviews were too often a read-through.** The Section 6 error was caught because the partner
re-ran the script, not because the review step required it. Two members asked for the same
change: a partner checks each number against its source (the file, the script, the count), not
just the text of the pull request.

**Dependency chains slowed tasks down.** T16 depended on T6, and T17 on T16, so a delay in one
held up the next, and blockers were not always raised early.

**Estimates left out review and rework.** The two members who gave per-task figures both ran above
plan, and the gap was in review rounds, CI fixes and merge conflicts rather than in the technical
work.

| Member | Planned | Actual | Where the difference went |
|---|---|---|---|
| Aliaa | not stated | about 18 h (T6, T16, T17) | — |
| Mohammed | 4 h for T18 (owner 2.5 h, reviewer 1.5 h) | about 5 h or a little more | CI failures and a merge conflict |
| Mahmoud | not recorded on #20 | not recorded on #20 | — |
| Ali | about 16 h (T7 4 h, T20 3.5 h, T16 review 1 h) | about 20 h (T7 5 h, T20 6 h, T16 review 4 h) | review revisions and follow-up; the T16 review hours are the ones that found `exposure` |
| Mazen | 17 h in the Milestone 1 document | about 3 h per week | not broken down |

---

## 3. What changed during the milestone, and why

- **The plan was rewritten against DEPI's published deadlines** (T23, #89). The first plan used
  programme weeks; DEPI then fixed six phases with a final submission on 4 December, roughly six
  weeks earlier.
- **A working agreement was added** (T23, #89), because too much of Milestone 1 was spent
  waiting for reviews, answers and other people's pieces.
- **Specification Section 6 was corrected** (#27) once the `exposure` column was found. The MVP
  estimates the intention-to-treat effect only (`docs/unit-of-observation.md`).
- **The environment was pinned to Python 3.12**, the version for which `causalml` has Windows
  wheels (`docs/environment.md`).

---

## 4. What we change

| Change | Raised by | Where it lives |
|---|---|---|
| Keep the owner–partner pairing; every task keeps a named partner | all five | Plan, section 5 (every task has an owner and a partner); working agreement §3, rule 2 |
| **A review checks the numbers, not just the text.** The partner can say where every number in the pull request comes from, and re-runs the script when the result depends on one | Ali, Aliaa | Working agreement §5 already requires every number to come from a script or file. **Proposed addition to §3, rule 2:** "The reviewer checks each number against its source." |
| **A pull request updated after review goes back to the reviewer the same day**, even if the update is partial | Ali | **Proposed addition to working agreement §3, rule 2** |
| Validate against the latest `main`: fetch, merge `origin/main` into the branch, then run the five checks | Mazen | `docs/git-workflow.md` (updating a branch with `git merge origin/main`); `main` requires branches to be up to date |
| Branch protection is in place before work starts, not during review | Mohammed | Done: `main` is protected (`docs/git-workflow.md`), so M2–M6 start with the rules already on |
| Raise blockers early and keep task status visible | Aliaa | Working agreement §2 (daily standup with a "Blocked" line), §3 rule 3 (stuck for an hour, ask), §4 (say a slip early) |
| Break dependency chains | Aliaa | Working agreement §3, rule 1 (build against the contract); the three interfaces frozen in T36 (#48) |
| Count review and rework in the estimates | Mohammed, Ali | The plan's load table assumes partner reviews add about 10%; Milestone 1 suggests more. Revisit at the Milestone 4 retrospective (T54, Sat 7 Nov) |

---

## 5. Risk register rows that turned out to be real

- **R11 (environment works on one machine but not another)** and **R12 (environment verification
  passes without checking a pinned dependency)**. Both happened, the Python pin and the coverage
  gap, and both were caught in review.
- **R13 (branching, review or merge workflow delays integration)**. Branch protection switched on
  mid-review, and branches validated against an old `main`.
- **R17 (DEPI deadlines remove about six weeks)**. It led to the replan in T23.

---

## 6. Decisions carried into the next milestones

- scikit-uplift is the single source of truth for every reported Qini, AUUC and uplift@k; causalml
  provides estimators (ADR 0001).
- The MVP estimates the intention-to-treat effect; the Wald estimate is an Advanced-tier extension
  (`docs/unit-of-observation.md`, specification Section 6).
- The split is stratified jointly on treatment and outcome, taken under the documented assumption
  that one row is one independent unit (`docs/unit-of-observation.md`).
- `docs/project-plan.md` is the version of record for phases, dates, owners and scope.
