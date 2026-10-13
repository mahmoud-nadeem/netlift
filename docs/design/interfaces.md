# Frozen interfaces

**Status: Proposed — awaiting approval from Aliaa, Ali and Mazen (due Tue 20 Oct)**

Task T36 (#48). Three interfaces sit between the members' work. Freezing them early lets every
M4 task build against a stub instead of waiting for the task upstream of it. Each interface has
typed code in `src/netlift` and tests in `tests/`; this document says what the code does and why.

**Change rule.** After this document merges, any change to an interface needs a record in
`docs/decisions/` (the next free number), approved by the interface's producer and its consumers.
The record says what changes, why, and which tasks are affected.

All three raise `netlift.utils.errors.ContractError` (a `ValueError`) when an invariant is broken.
The message names the field and the rule.

No new dependencies. pydantic and FastAPI arrive in M5, so the types are standard-library frozen
dataclasses with `typing.Literal`, `typing.Protocol`, numpy and pandas. The API layer wraps the
scoring type in M5.

| # | Interface | Code | Producer | Consumers |
|---|---|---|---|---|
| 1 | Feature matrix and folds | `src/netlift/data/contracts.py` | Aliaa (T37) | everyone |
| 2 | Metric signature | `src/netlift/evaluation/contracts.py` | Ali (T39) | Mohammed, Mahmoud |
| 3 | Scoring response | `src/netlift/api/schema.py` | Ali (T55), Mahmoud (registry) | Mazen (T45, T59), Mohammed (T62) |

Interfaces 1 and 2 live in their producer's package. The scoring response lives in `api`,
because it has two producers and the API is the layer that wraps it in M5. The code examples
below are sketches: `model`, `qini_coefficient` and `path` stand for code that T38, T39 and T42
write.

---

## 1. Feature matrix and folds

**Producer:** data pipeline, T37 (Aliaa). **Consumers:** baselines T38, evaluation harness T39,
randomization report T40, T- and X-learner T42, negative control T43, central comparison T44,
sensitivity check #29, and everything after them.

### Type

```python
DatasetName = Literal["criteo", "hillstrom", "synthetic"]
TargetOutcome = Literal["visit", "conversion"]
SplitName = Literal["train", "validation", "test"]

@dataclass(frozen=True)
class TreatmentDefinition:
    source_column: str                      # raw arm column
    treatment_values: tuple[str | int, ...]
    control_values: tuple[str | int, ...]
    excluded_values: tuple[str | int, ...] = ()   # rows dropped, never pooled

@dataclass(frozen=True)
class FeatureMatrixMeta:
    dataset: DatasetName
    target_outcome: TargetOutcome
    treatment: TreatmentDefinition
    seed: int
    split_name: SplitName

@dataclass(frozen=True, eq=False)
class FeatureMatrix:
    X: pd.DataFrame                 # feature columns only
    t: NDArray[int8]                # 0 control, 1 treatment
    y: NDArray[int8]                # 0/1 target outcome
    fold: NDArray[int64]            # fold id per row
    row_id: NDArray[int64]          # 0-based position in the raw file as loaded
    meta: FeatureMatrixMeta

class LockedTestSplit:              # wraps the test FeatureMatrix
    meta, columns, row_ids, is_open # readable without opening
    def open(self, reason: str) -> FeatureMatrix

@dataclass(frozen=True, eq=False)
class DatasetSplits:                # what T37 returns
    train: FeatureMatrix
    validation: FeatureMatrix
    test: LockedTestSplit

def validate_features(X: pd.DataFrame) -> None
def check_control_events(matrix, *, min_control_events: int) -> dict[int, int]
```

### Invariants

Checked when a `FeatureMatrix` is built:

- **No forbidden column in `X`.** `validate_features()` (the name the specification uses in
  Leakage and Validity Audit) rejects, by name and case-insensitively: `exposure`, `visit`,
  `conversion`, `spend`, `treatment`, and Hillstrom's arm column `segment`. The target outcome is
  on that list, so every outcome is forbidden whichever one is the target. The treatment flag is
  never a column of `X`, not even for the S-learner: an S-learner appends `t` inside its own fit
  and predict.
- `X`, `t`, `y`, `fold` and `row_id` have the same length, and there is at least one row.
- `t` and `y` are one-dimensional, have no missing values, and contain only 0 and 1.
- `t` contains both arms.
- Fold ids are non-negative integers. In the train split they are the cross-validation folds
  0 to K-1. The validation and test splits are each a single fold, 0.
- Row ids are non-negative integers, unique within the matrix. A row id is the row's 0-based
  position in the raw file as loaded. For Criteo it identifies a row, not a customer: neither
  dataset has a customer id (`docs/unit-of-observation.md`).

Checked on request, because it needs a configured value:

- `check_control_events(matrix, min_control_events=N)` raises if any fold has fewer than `N`
  control-arm rows with `y == 1`, and returns the count per fold for logging. The specification
  (Uncertainty and Abstention) calls this "a configured minimum number of control-group outcome
  events" but gives no number, so there is no default. The value goes in `config/` (see the
  decisions below).

Checked when `DatasetSplits` is built: the three splits are named correctly and share dataset,
target, treatment definition, seed and feature columns, and **their row ids are pairwise
disjoint**. A raw row that sits in both train and test is the most common silent bug in split
code, and with positional arrays alone nobody downstream could check for it. The row ids also
record exactly which raw rows formed the locked test split.

`FeatureMatrix.cell_counts()` returns the four (t, y) cell counts per fold, which the Splitting
Strategy section says to assert and log after every split.

### The locked test split

The simplest form that works: T37 hands out the test split only inside a `LockedTestSplit`. Its
metadata, column names and row ids (a read-only copy) can be read at any time. The rows come out only through
`open(reason)`, which needs a non-empty reason, logs it as a warning, and refuses a second call
on the same object. This cannot stop someone rebuilding the split in another process; that is
what the written opening procedure of T70 is for. It does make opening the split a deliberate,
logged act rather than an attribute access.

### Treatment definitions

| Constant | Column | Treatment | Control | Dropped |
|---|---|---|---|---|
| `CRITEO_TREATMENT` | `treatment` | 1 | 0 | — |
| `HILLSTROM_MENS_VS_CONTROL` | `segment` | Mens E-Mail | No E-Mail | Womens E-Mail |
| `HILLSTROM_WOMENS_VS_CONTROL` | `segment` | Womens E-Mail | No E-Mail | Mens E-Mail |

For Hillstrom the specification (Multi-Treatment Handling) says the MVP analyses "one arm at a
time against the shared control" and that pooling the two e-mail arms "estimates the average
effect of two different interventions". So a Hillstrom feature matrix is built for one arm, and
the other arm's rows are dropped, never pooled into "emailed". Criteo's `treatment` column is the
randomized assignment; `exposure` is never the treatment variable (Leakage and Validity Audit).

### Example

```python
from netlift.data.contracts import check_control_events
from netlift.data.synthetic import make_synthetic_splits

splits = make_synthetic_splits("visit", seed=0)  # SYNTHETIC, see below
train = splits.train
counts = check_control_events(train, min_control_events=5)
for fold_id in train.fold_ids:
    in_fold = train.fold == fold_id
    model.fit(train.X[~in_fold], train.y[~in_fold], train.t[~in_fold])

test = splits.test.open("champion confirmation, procedure in T70")  # once
```

### Synthetic fixture

`src/netlift/data/synthetic.py` builds a seeded synthetic dataset (600 rows by default, four
features, both arms, both binary outcomes) and returns it as `DatasetSplits`, 60 / 20 / 20 and
jointly stratified on (t, y). The pytest fixture `synthetic_splits` in `tests/conftest.py` returns
it. T38 and T39 can build and test against it today. Its row ids are the rows' positions in the
generated frame, so the three splits together cover every row exactly once.

**It is synthetic and never used for a reported result.** Every matrix it returns carries
`meta.dataset == "synthetic"`, and its effect sizes are invented.

---

## 2. Metric signature

**Producer:** evaluation harness, T39 (Ali). **Consumers:** baselines T38 and T-/X-learners T42
(Mohammed), tracking T41 and the uplift forest T57 (Mahmoud), and T43, T44.

### Type

```python
@dataclass(frozen=True)
class MetricResult:
    name: str                  # "qini_coefficient", "auuc", "uplift_at_k", ...
    value: float
    ci_low: float
    ci_high: float
    confidence_level: float    # 0.95 for a 95% interval
    n_resamples: int
    seed: int
    library: str               # "scikit-uplift" for the reported metrics
    library_version: str
    settings: dict[str, str | int | float] = {}   # e.g. {"k": 0.3, "strategy": "overall"}

class UpliftMetric(Protocol):
    def __call__(self, y, uplift, t, *,
                 n_resamples: int, confidence_level: float, seed: int) -> MetricResult: ...

def log_metric_result(run: Run, kpi_id: str, result: MetricResult) -> str  # the key
```

The argument order is scikit-uplift's `(y_true, uplift, treatment)`. Options specific to one
metric, such as `k` and `strategy` for uplift@k, are extra keyword arguments with defaults and are
echoed in `settings`. The specification (Evaluation Metrics) notes that uplift@k's `overall` and
`by_group` strategies give different numbers, so the strategy is always recorded.

### Invariants

- `ci_low <= ci_high`. `value`, `ci_low` and `ci_high` are finite numbers.
- **The value is not required to lie inside the interval.** A percentile bootstrap does not
  guarantee that, so the type does not demand it. `value_inside_interval` says whether it does.
- `0 < confidence_level < 1`. `n_resamples` and `seed` are ints, `n_resamples >= 1`.
- `library` and `library_version` are not empty.
- **scikit-uplift is the source of truth** for every reported Qini, AUUC and uplift@k
  (`docs/decisions/0001-uplift-library.md`). A result named `qini_coefficient`, `auuc` or
  `uplift_at_k` with any other library is rejected. causalml's numbers are a development
  cross-check under a different name.
- The bootstrap resamples **within treatment arms**, preserving the arm proportions
  (Specification, Statistical Significance). Neither library provides this (ADR 0001, decision 5);
  T39 writes it and calls scikit-uplift once per resample.

### Logging a result

`log_metric_result` logs the result under the key `<KPI>_<metric name>`, for example
`K2_qini_coefficient`. The interval goes under `<key>_ci_low` and `<key>_ci_high`, the confidence
level is a tag `<key>_confidence`, and `library==version`, the number of resamples, the seed and
every setting are parameters prefixed with the same key. One KPI can cover several metrics: K2 is
both the Qini coefficient and AUUC, so one run holds `K2_qini_coefficient` and `K2_auuc` side by
side, each with its own interval and parameters. The prefix has to be the full key, because
MLflow refuses to change a parameter's value within a run.

It refuses a result with fewer than 1,000 resamples, because the specification says "Use at
least 1,000 resamples" for a reported figure. Diagnostics with fewer resamples go through
`run.log_metrics`.

```python
from netlift.evaluation.contracts import log_metric_result
from netlift.utils.tracking import start_run

with start_run("x-learner, lightgbm base", dataset=path, seed=42) as run:
    for metric in (qini_coefficient, auuc):
        result = metric(y, uplift, t, n_resamples=1000, confidence_level=0.95, seed=42)
        log_metric_result(run, "K2", result)  # K2_qini_coefficient, K2_auuc
```

**Changes to the tracking wrapper.** These were made in this task:

- `Run.log_kpi` used to reject a value outside its own interval. With a percentile bootstrap
  that is a legitimate result, so a valid `MetricResult` could not have been logged. It now logs
  a warning, and it also sets the run tag `<key>_value_outside_interval = "true"`, so the case
  shows in the MLflow UI and not only in the console. Reversed bounds are still rejected.
- `Run.log_kpi` takes an optional `metric`. The MLflow key is the KPI id when `metric` is
  `None`, and `<kpi_id>_<metric>` otherwise. It returns the key. The KPI id is still checked
  against `config/kpis.yaml`.

`tests/test_tracking.py` and `docs/experiment-tracking.md` are updated to match.

---

## 3. Scoring response

**Producers:** decision layer T55 (Ali) and the registered champion T57 (Mahmoud).
**Consumers:** API skeleton T45 and service T59 (Mazen), scoring log T60 (Aliaa), wireframes T49
and dashboard T62 (Mohammed).

### Type

Field by field the specification's Model Output Schema:

```python
SegmentLabel = Literal["persuadable", "sure_thing", "lost_cause", "sleeping_dog", "unknown"]
Recommendation = Literal["contact", "suppress", "no_action"]
Verdict = Literal["confident", "insufficient_evidence", "insufficient_control_data"]

@dataclass(frozen=True)
class Driver:
    feature: str
    contribution: float

@dataclass(frozen=True)
class ScoringResponse:
    customer_id: str
    model_version: str
    uplift_score: float
    uplift_ci_95: tuple[float, float]
    decile: int
    segment_label: SegmentLabel
    expected_incremental_profit: float     # m * uplift - c, currency units
    recommendation: Recommendation
    verdict: Verdict
    top_drivers: tuple[Driver, ...]
    scored_at: datetime                    # timezone-aware

    def to_dict(self) -> dict[str, Any]          # JSON-ready
    @classmethod
    def from_dict(cls, data) -> ScoringResponse
```

### Invariants

- The interval's lower bound is not above its upper bound. All numbers are finite.
- `decile` is an int from 1 to 10.
- `segment_label`, `recommendation` and `verdict` are members of their enumerations.
- **A verdict other than `confident` never carries `contact` or `suppress`; it carries
  `no_action`.** This encodes the rule in the specification's Uncertainty and Abstention section:
  "Where a segment's interval straddles zero, the system returns verdict = "insufficient_evidence"
  rather than a directional recommendation." The same holds for `insufficient_control_data`.
- `scored_at` is timezone-aware, and `to_dict` writes UTC with a `Z` suffix, as in the
  specification's example.
- `from_dict` names every missing and every unknown key, so a schema mismatch between the API and
  the dashboard fails with a message rather than a `KeyError`.
- `segment_label` is presentational only. No dataset labels customers as persuadables; it is an
  inference, and nothing may treat it as ground truth.

### Example

The specification's own example round-trips through JSON unchanged (tested):

```json
{
  "customer_id": "c_00193472",
  "model_version": "netlift-xlearner-2026.11.03",
  "uplift_score": 0.0041,
  "uplift_ci_95": [0.0009, 0.0074],
  "decile": 1,
  "segment_label": "persuadable",
  "expected_incremental_profit": 1.87,
  "recommendation": "contact",
  "verdict": "confident",
  "top_drivers": [
    {"feature": "f3", "contribution": 0.0012},
    {"feature": "f7", "contribution": -0.0005}
  ],
  "scored_at": "2026-11-03T14:22:10Z"
}
```

---

## Decisions each owner must confirm

**Aliaa (interface 1)**

1. The minimum number of control-arm outcome events per fold, and where it lives in `config/`.
   The specification gives no number, so the code has no default.
2. Hillstrom is analysed one arm at a time against the shared control, the other arm dropped.
   Which arm comes first, or both?
3. Validation and test are single folds (id 0); cross-validation folds exist only inside train.
4. `X` arrives preprocessed or raw? The interface does not check dtypes or missing values in `X`,
   because preprocessing is fitted on the training fold only and the matrix may come before it.
5. The locked test split as a wrapper with a logged, once-only `open(reason)`.
6. `row_id` on every matrix: the row's 0-based position in the raw file as loaded, unique within
   a matrix, with the three splits' row ids pairwise disjoint. T37 must record the position
   before any row is dropped (duplicates, the other Hillstrom arm), so the ids still point at the
   raw file.

**Ali (interfaces 2 and 3)**

1. Metric names: `qini_coefficient`, `auuc`, `uplift_at_k`, and the settings recorded for each.
2. Confirm the metric key format `<KPI>_<metric name>` (for example `K2_qini_coefficient`,
   `K2_auuc`).
3. The value is not required to lie inside its interval; `log_kpi` now warns and tags the run instead of
   rejecting.
4. Fewer than 1,000 resamples cannot be logged as a KPI.
5. Should a `confident` verdict also require the customer's interval to exclude zero? The
   specification states the abstention rule per segment, so the type does not enforce it per
   customer.

**Mazen (interface 3)**

1. The field names and JSON shapes of `to_dict`, including `scored_at` in UTC with a `Z` suffix.
2. Strict `from_dict`: unknown keys are an error, not ignored.
3. The M5 pydantic model wraps `ScoringResponse` rather than redefining the fields.
