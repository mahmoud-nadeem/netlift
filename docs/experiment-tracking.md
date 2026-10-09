# Experiment tracking

Over the next two months this project will produce hundreds of model fits. Each
one has a learner, a set of hyperparameters, a version of the data, a seed, and
a result.

Without a record, the question that arrives in week fifteen — *which
configuration produced 0.0847?* — has no answer. The notebook was overwritten,
the number was pasted into a channel, and the result cannot be reproduced.

That is not an organisational problem. It is a credibility problem: a result
nobody can reproduce is a claim, not a result.

---

## Quick start

```python
from pathlib import Path
from netlift.utils.tracking import start_run

with start_run(
    "x-learner, lightgbm base",
    dataset=Path("data/raw/hillstrom.csv"),
    seed=42,
) as run:
    run.log_params({"learner": "x", "base_model": "lightgbm", "outcome": "visit"})
    run.log_metrics({"train_rows": 44_800})
    run.log_kpi("K2", value=0.0847, ci_low=0.0612, ci_high=0.1103)
    run.log_artifact(Path("reports/figures/qini.png"))
```

**Nobody calls `mlflow` directly.** If five people log the same quantity as
`qini`, `Qini`, `qini_score` and `qini_auc`, the interface whose entire purpose
is comparison becomes useless for it. If something is needed that this module
does not expose, add it to the module rather than reaching around it.

---

## What is recorded without you asking

Every run is tagged with four things:

| Tag | Why |
|---|---|
| `git_commit` | which code produced this |
| `git_dirty` | whether that commit describes the code that actually ran |
| `dataset_sha256` | which exact bytes it was computed from |
| `seed` | which draw of randomness |

The first three are the difference between a number and a reproducible number.
The fourth is applied as well as recorded — `start_run` seeds Python's `random`
and NumPy itself, so the tag is true by construction rather than true if the
caller remembered.

### `git_dirty` is the one worth understanding

If the working tree has uncommitted changes when a run starts, the commit hash
is a lie: the code that ran is not the code in that commit, and the result
cannot be rebuilt from it. The run is tagged `git_dirty=true` and a warning is
printed.

That tag is not an error. Exploring with uncommitted changes is normal. But a
number that will appear in the report should come from a clean tree, and this
is how you can tell afterwards which ones did.

---

## Where things go

MLflow has four places to put information and they are easy to confuse.

```
params      what you chose        learner, max_depth, outcome column, split rule
metrics     what you measured     qini, auuc, train_rows
artifacts   files                 the Qini curve, the decile table, the model
tags        where it came from    commit, data checksum, seed, purpose
```

The test when you are unsure: **is this an input or an output?** Inputs are
params, outputs are metrics. If it is neither, it is provenance, so it is a tag.

---

## Headline results go through `log_kpi`

```python
run.log_kpi("K2", value=0.0847, ci_low=0.0612, ci_high=0.1103)
```

This does three things `log_metrics` does not.

It **rejects a KPI id that is not defined** in `config/kpis.yaml`, so the KPI
framework and the metric names in MLflow cannot drift apart.

It **requires the confidence interval**. There is no way to log a headline
number without one. This project's rule is that every reported result carries an
interval — and a rule that depends on five people remembering it will be broken
at least once. Here it is enforced by the code instead.

It **checks the three numbers are consistent**: bounds the right way round, and
the value inside its own interval. Both of those have been typos in real
projects, and both survive review because the numbers look plausible.

Diagnostics during development still go through `log_metrics`. `log_kpi` is for
numbers that will be defended.

---

## Opening the interface

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then <http://127.0.0.1:5000>.

`mlflow.db` and `mlartifacts/` are in `.gitignore`. Experiment results are
shared by committing the code and the configuration that produced them, never by
committing the tracking database — a binary that every merge would conflict on.

---

## The tracking backend

Set in `.env`:

```
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=netlift
```

**MLflow 3.x refuses the plain `./mlruns` directory store** and raises on first
use. Almost every tutorial online still uses it. The wrapper rejects a directory
URI up front, naming the reason and the fix, rather than letting MLflow fail with
a stack trace that never mentions the word directory.

The URI is read from the environment and never hard-coded, so pointing the
project at a shared server or at Azure ML later is a configuration change rather
than a code change.

---

## What reproducibility does not promise

Even with the same code, the same data and the same seed, results can differ
across machines. The number of BLAS threads changes the order of floating-point
additions; GPU kernels are worse.

For this project — gradient-boosted trees and linear models on CPU — seeds are
enough in practice, and the verification script confirms all five machines run
identical package versions.

This limit is stated here rather than left implied. The project's standard is to
describe precisely what was done and claim no more than that, and "fully
reproducible" without qualification would claim more.

---

## Rules

- Do not call `mlflow` directly. Extend this module instead.
- Every run names its dataset, so the checksum is recorded.
- Headline numbers go through `log_kpi`, with their interval.
- A number destined for the report comes from a run tagged `git_dirty=false`.
- Exploratory runs are tagged `purpose=smoke` so they can be filtered out later
  and never mistaken for findings.
