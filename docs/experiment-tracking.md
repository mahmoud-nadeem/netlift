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
from netlift.utils.tracking import experiment_name, run_name, start_run

with start_run(
    run_name("T38", "x-learner"),  # "T38-x-learner"
    task="T38",  # required
    experiment=experiment_name("hillstrom", "visit"),  # "netlift-hillstrom-visit"
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

## Naming

| What | Rule | Example |
|---|---|---|
| Experiment | `netlift-<dataset>-<target>` | `netlift-hillstrom-visit` |
| Run name | `<task>-<model>` | `T38-response-model` |
| `task` tag | the plan task, required | `T38` |

Build the names with `experiment_name(dataset, target)` and `run_name(task, model)`
rather than by hand; both reject a typo such as `Hillstrom` or `Response Model`.
`start_run` refuses to start without `task` (or `tags={"task": ...}`), so every run
can be filtered by the task that produced it.

---

## What is recorded without you asking

Every run is tagged with four things, plus the `task` you pass:

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

It **checks the three numbers are consistent**. Bounds the wrong way round are
rejected: that is always a typo, and it survives review because the numbers look
plausible. A value outside its own interval is logged with a warning rather than
rejected, because a percentile bootstrap does not guarantee the point estimate
lies inside the interval (`docs/design/interfaces.md`, interface 2). The run also
gets the tag `<key>_value_outside_interval = "true"`, so the case shows in the
MLflow UI and not only in the console. The warning and the tag are the prompt to
check it is not a typo.

**One KPI, several metrics.** K2 covers both the Qini coefficient and AUUC. Pass
`metric` to log each under its own key, `<KPI>_<metric name>`, with its own
interval:

```python
run.log_kpi(
    "K2", value=0.0847, ci_low=0.0612, ci_high=0.1103, metric="qini_coefficient"
)
run.log_kpi("K2", value=0.0310, ci_low=0.0200, ci_high=0.0450, metric="auuc")
# metrics: K2_qini_coefficient, K2_qini_coefficient_ci_low, ..., K2_auuc, K2_auuc_ci_low, ...
```

`log_kpi` returns the key it used. Without `metric`, the key is the KPI id
itself, as before. The evaluation harness logs through `log_metric_result`
(`netlift.evaluation.contracts`), which always passes the metric name and
prefixes every parameter with the full key, because MLflow refuses to change a
parameter's value within a run.

Diagnostics during development still go through `log_metrics`. `log_kpi` is for
numbers that will be defended.

**The resamples travel with the run.** `log_metric_result` also attaches the
whole `MetricResult`, per-resample values included, as
`resamples/<key>.json`. `MetricResult(**json.loads(...))` rebuilds it, so a
champion and a challenger logged in different runs can still be compared with
`paired_difference`, on the same resamples.

---

## The tracking server

Start it from the repository root and leave it running in its own terminal:

```
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Point the client at it in `.env`, then open <http://127.0.0.1:5000>:

```
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

The server binds to `127.0.0.1` only, so nobody else on the network can reach it.
`tests/test_tracking_server.py` starts the same server on a free port and checks
that a run logged through the wrapper reads back complete.

Without a server, `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` (the default) still
works, and `mlflow ui --backend-store-uri sqlite:///mlflow.db` shows the same
database.

**Comparing two runs.** In the UI, open the experiment (for example
`netlift-hillstrom-visit`), type `tags.task = 'T38'` in the search box, tick two
runs and press **Compare**. Parameters, metrics and intervals sit side by side.
For a difference with an interval, use `paired_difference` on the two
`resamples/<key>.json` files, not the two intervals.

**Where the files live.** All are in `.gitignore`:

| Path | Holds |
|---|---|
| `mlflow.db` | runs, parameters, metrics, tags |
| `mlartifacts/` | artifacts logged through the server |
| `mlruns/` | artifacts logged without a server |

An experiment keeps the artifact location it was created with. Results are
shared by committing the code and the configuration that produced them, never by
committing the tracking database, a binary that every merge would conflict on.

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
- Every run has a `task` tag, a `<task>-<model>` name and a
  `netlift-<dataset>-<target>` experiment.
- Every run names its dataset, so the checksum is recorded.
- Headline numbers go through `log_kpi`, with their interval.
- A number destined for the report comes from a run tagged `git_dirty=false`.
- Exploratory runs are tagged `purpose=smoke` so they can be filtered out later
  and never mistaken for findings.
