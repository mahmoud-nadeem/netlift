# NetLift API contract

**Task:** T45 · **Issue:** [#57](https://github.com/mahmoud-nadeem/netlift/issues/57)
**Owner:** Mazen Mahmoud · **Partner/reviewer:** Ali Alaa Salah
**Status:** T45 implementation contract; request/response behavior is exercised by `tests/test_api_endpoints.py`.
**Scope:** FastAPI skeleton with deterministic mocked responses. This document does not claim that real model inference, replay execution, MLflow registry access, or drift computation is implemented.

## 1. Contract principles

1. The API exposes exactly the six endpoints from the Project Specification §35 and T33. No scenario-creation, reveal, retraining, or recorded-replay endpoint is added.
2. `netlift.api.schema.ScoringResponse` and `Driver` are the frozen T36 interface. The API uses that dataclass directly as the scoring response model; it does not create a second scoring-field definition.
3. The scoring response contains exactly eleven fields. T45 does not add the campaign economics or the control-event threshold to that frozen response.
4. The current adapter returns deterministic synthetic examples. Mock values are useful for frontend integration and contract testing only; they are not model predictions, measured metrics, or real drift calculations.
5. The request/response schemas are strict where the fields are fixed. Feature columns are separately checked by `validate_features()` from `src/netlift/data/contracts.py` before scoring.

## 2. Endpoint summary

| Method | Path | Request | Success response |
|---|---|---|---|
| `POST` | `/score` | One customer and campaign economics | One T36 `ScoringResponse` object |
| `POST` | `/score/batch` | JSON array or raw CSV | ZIP archive containing `scores.csv` and `summary.json` |
| `POST` | `/policy/simulate` | Scenario, validation slice, model version, seed, budget, cost and margin | Four-policy what-if comparison |
| `GET` | `/model/info` | None | Active-model metadata and evaluation metrics |
| `GET` | `/monitor/drift` | None | Latest drift report or explicit empty state |
| `GET` | `/health` | None | Liveness/readiness state |

FastAPI publishes the machine-readable contract at `/openapi.json` and the interactive documentation at `/docs`.

## 3. Shared scoring response — frozen T36 interface

`POST /score` returns one `ScoringResponse`. Each row in a batch has the same shape.

| Field | Type | Meaning |
|---|---|---|
| `customer_id` | string | Customer/row identifier supplied by the caller |
| `model_version` | string | Model version used for scoring |
| `uplift_score` | number | Estimated ITT-scale absolute probability difference: treated-assignment outcome probability minus control-assignment outcome probability, not a relative percentage lift |
| `uplift_ci_95` | `[number, number]` | 95% confidence interval `[lower, upper]` for the ITT-scale difference |
| `decile` | integer, 1–10 | Ranking by estimated uplift; `1` is the highest-uplift decile and `10` is the lowest-uplift decile |
| `segment_label` | enum | `persuadable`, `sure_thing`, `lost_cause`, `sleeping_dog`, `unknown` |
| `expected_incremental_profit` | number | `conversion_margin * uplift_score - contact_cost` |
| `recommendation` | enum | `contact`, `suppress`, `no_action` |
| `verdict` | enum | `confident`, `insufficient_evidence`, `insufficient_control_data` |
| `top_drivers` | array | Objects with `feature` and numeric `contribution` |
| `scored_at` | ISO 8601 string | Timezone-aware timestamp serialized in UTC with a `Z` suffix |

The T36 invariants apply unchanged:

- Both abstention verdicts carry `recommendation = "no_action"`; neither may carry a directional recommendation.
- `confident` requires the interval to exclude zero.
- `contact` requires the lower interval bound above zero.
- `suppress` requires the upper interval bound below zero (the sleeping-dog rule).
- A `sleeping_dog` label never carries `contact`.
- `segment_label` is an inference for presentation, not a ground-truth customer label.
- The decile convention is descending uplift: decile `1` contains the highest estimated uplift scores; decile `10` contains the lowest.
- `uplift_score` is an absolute probability difference on the intention-to-treat (ITT) assignment scale. For example, `0.03` means an estimated three-percentage-point increase caused by assignment to treatment, not a 3% relative lift and not the complier-only Wald estimate.

The API returns the response dataclass through FastAPI/Pydantic serialization. The exact set of fields stays owned by T36. The `insufficient_control_data` mock writes a structured log marker that its threshold was not supplied; the real decision-layer integration must attach the configured per-segment threshold to its scoring log. T45 does not invent `N_min` or add a twelfth response field.

## 4. `POST /score`

### Request

Content type: `application/json`.

Nested form:

```json
{
  "customer_id": "customer-001",
  "features": {"f0": 0.2, "f3": 1.5},
  "contact_cost": 5.0,
  "conversion_margin": 200.0
}
```

A flat form is also accepted: `customer_id`, `contact_cost`, `conversion_margin`, plus feature columns at the top level. The nested form is preferred for new clients.

Rules:

- `customer_id` is non-empty.
- `features` is a non-empty object of finite numeric values.
- `treatment`, `exposure`, `visit`, `conversion`, generic `outcome`, `spend`, and `segment` are forbidden feature columns (case-insensitive, whitespace-trimmed).
- `contact_cost` and `conversion_margin` are finite positive numbers. T45 proposes them as explicit standalone-scoring request inputs for open T33 Q4; this API-boundary choice remains subject to reviewer acceptance.
- In flat form, additional top-level fields are interpreted as feature columns. In nested form, feature columns belong inside `features`; duplicate feature names supplied in both places are rejected.

### Response

HTTP `200`, T36 scoring response. The campaign economics used to derive `expected_incremental_profit` are echoed in headers so the eleven-field response remains unchanged:

- `X-NetLift-Contact-Cost`
- `X-NetLift-Conversion-Margin`
- `X-NetLift-Mock: true` while the mock adapter is active.

### Deterministic mock cases

The mock service uses reserved customer ID prefixes for frontend and test scenarios:

| `customer_id` prefix | Mock verdict | Segment | Recommendation |
|---|---|---|---|
| `mock-insufficient-evidence:` | `insufficient_evidence` | `unknown` | `no_action` |
| `mock-insufficient-control-data:` | `insufficient_control_data` | `unknown` | `no_action` |
| `mock-sleeping-dog:` | `confident` | `sleeping_dog` | `suppress` |
| Any other ID | `confident` | `persuadable` | `contact` |

All mock timestamps and scores are deterministic. Mock scores are not computed from the supplied feature values.

For the default customer ID, the documented economic example (`contact_cost = 5`, `conversion_margin = 200`, break-even uplift `c/m = 0.025`) returns uplift `0.03` with interval `[0.02, 0.04]`. Expected incremental profit is therefore `200 * 0.03 - 5 = 1.00`.

Example default response:

```json
{
  "customer_id": "customer-001",
  "model_version": "netlift-mock-2026.11.04",
  "uplift_score": 0.03,
  "uplift_ci_95": [0.02, 0.04],
  "decile": 1,
  "segment_label": "persuadable",
  "expected_incremental_profit": 1.0,
  "recommendation": "contact",
  "verdict": "confident",
  "top_drivers": [
    {"feature": "f3", "contribution": 0.0012},
    {"feature": "f7", "contribution": -0.0005}
  ],
  "scored_at": "2026-11-04T12:00:00Z"
}
```

## 5. `POST /score/batch`

### Input formats

The endpoint accepts either:

- `Content-Type: application/json`: a non-empty array of scoring rows. Rows may use the nested or flat scoring shape described above.
- `Content-Type: text/csv`: raw UTF-8 CSV with headers `customer_id`, `contact_cost`, `conversion_margin`, and one or more numeric feature columns.

All rows in one batch must use the same `contact_cost` and `conversion_margin`, because the batch represents one campaign scenario. A forbidden feature column rejects the entire batch. Validation is atomic: all rows are validated before any mock scoring function is called, and no partial archive is returned.

### Output

HTTP `200`, `application/zip` attachment `netlift-scores.zip` containing:

- `scores.csv`: one row per customer and all eleven T36 fields. `uplift_ci_95` and `top_drivers` are JSON-encoded CSV cells.
- `summary.json`: `row_count`, `model_version`, `verdict_counts`, `campaign_economics`, `total_expected_incremental_profit`, `is_mock`, and an explanatory message.

The summary includes the economic assumptions and profit formula used for every row. This gives batch consumers the assumptions alongside the profit figures without changing the T36 per-customer response. `verdict_counts` currently contains the verdicts observed in that batch; zero-filling absent verdict categories is a follow-up, not a different per-row scoring contract.

Example `summary.json` for a three-row batch containing one confident score and both abstention verdicts:

```json
{
  "row_count": 3,
  "model_version": "netlift-mock-2026.11.04",
  "verdict_counts": {
    "confident": 1,
    "insufficient_control_data": 1,
    "insufficient_evidence": 1
  },
  "campaign_economics": {
    "contact_cost": 5.0,
    "conversion_margin": 200.0,
    "profit_formula": "conversion_margin * uplift_score - contact_cost"
  },
  "total_expected_incremental_profit": -8.16,
  "is_mock": true,
  "message": "Synthetic mock scores; not measured model output."
}
```

## 6. `POST /policy/simulate`

### Request

```json
{
  "scenario_id": "campaign-demo",
  "data_slice": "validation",
  "model_version": "netlift-mock-2026.11.04",
  "seed": 42,
  "budget": 20000.0,
  "contact_cost": 5.0,
  "conversion_margin": 200.0
}
```

Fields:

- `scenario_id`: non-empty scenario identifier. The mock accepts the identifier without resolving it against a scenario registry; the real adapter must resolve it to the configuration used by the offline replay job.
- `data_slice`: only `validation` is accepted while Q1 remains open.
- `model_version`: explicit model version for reproducible comparison. The mock echoes it; the real adapter must verify that the requested version is available.
- `seed`: non-negative integer.
- `budget`, `contact_cost`, `conversion_margin`: finite positive numbers; the budget must cover at least one contact.

A request selecting `test`, `locked_test`, `Test`, `train`, or any non-validation split is rejected with `error.code = "request_validation_error"`; the error detail location identifies `body.data_slice`. The response message is: `The locked test split is opened once at T70; this endpoint accepts 'validation' only.` This enforces T33 FR-45 and the interim rule in ADR 0002; it does not open `LockedTestSplit`.

### Comparison mode and interval semantics

The T45 request implements **equal-k / fixed-budget comparison**: `budget / contact_cost` determines the contact count `k`, and the uplift, response-model, and random policies use that same contact count. Blanket contact is included as a full-population reference, so it is not an equal-k policy. Profit-maximising, fixed-target, and suppression-only policy modes are not selectable through this request; the current request has no `policy_mode` field.

For a measured production comparison, the confidence interval for uplift minus response-model incremental conversions must be a **paired bootstrap**: both policies are evaluated on the same bootstrap resamples, drawn within the original treatment arms, and the interval is calculated from the per-resample paired differences. This follows ADR 0002 and the metric/replay contracts. The T45 mock does **not** execute that bootstrap; its deterministic intervals are illustrative placeholders and must not be described as measured confidence intervals.

### Response

The response contains:

- Simulation identity (`simulation_id`, `scenario_id`, `data_slice`, `model_version`, `seed`).
- Budget and campaign economics.
- Results for `uplift`, `response_model`, `random`, and `blanket` policies. Each includes contacted count, total cost, incremental conversions and 95% interval, incremental profit, profit per unit spent, and cumulative curve points with intervals.
- `comparison` for uplift versus response-model incremental conversions, its interval, and a conclusion (`uplift_better`, `response_model_better`, or `not_distinguishable`).
- `recorded_run_created: false`, `source: "mock"`, and `is_mock: true`.

Example mock response (the values are synthetic; in particular, `comparison.ci_95` is not a computed bootstrap interval):

```json
{
  "simulation_id": "mock-b6ab17063779ada5",
  "scenario_id": "campaign-demo",
  "data_slice": "validation",
  "model_version": "netlift-mock-2026.11.04",
  "seed": 42,
  "budget": 20000.0,
  "contact_cost": 5.0,
  "conversion_margin": 200.0,
  "policies": [
    {
      "policy": "uplift",
      "contacted": 4000,
      "total_cost": 20000.0,
      "incremental_conversions": 160.0,
      "incremental_conversions_ci_95": [80.0, 240.0],
      "incremental_profit": 12000.0,
      "profit_per_unit_spent": 0.6,
      "curve": [
        {"fraction_targeted": 0.0, "cumulative_incremental_conversions": 0.0, "ci_95": [0.0, 0.0]},
        {"fraction_targeted": 0.25, "cumulative_incremental_conversions": 40.0, "ci_95": [20.0, 60.0]},
        {"fraction_targeted": 0.5, "cumulative_incremental_conversions": 80.0, "ci_95": [40.0, 120.0]},
        {"fraction_targeted": 0.75, "cumulative_incremental_conversions": 120.0, "ci_95": [60.0, 180.0]},
        {"fraction_targeted": 1.0, "cumulative_incremental_conversions": 160.0, "ci_95": [80.0, 240.0]}
      ]
    },
    {
      "policy": "response_model",
      "contacted": 4000,
      "total_cost": 20000.0,
      "incremental_conversions": 80.0,
      "incremental_conversions_ci_95": [40.0, 120.0],
      "incremental_profit": -4000.0,
      "profit_per_unit_spent": -0.2,
      "curve": [
        {"fraction_targeted": 0.0, "cumulative_incremental_conversions": 0.0, "ci_95": [0.0, 0.0]},
        {"fraction_targeted": 0.25, "cumulative_incremental_conversions": 20.0, "ci_95": [10.0, 30.0]},
        {"fraction_targeted": 0.5, "cumulative_incremental_conversions": 40.0, "ci_95": [20.0, 60.0]},
        {"fraction_targeted": 0.75, "cumulative_incremental_conversions": 60.0, "ci_95": [30.0, 90.0]},
        {"fraction_targeted": 1.0, "cumulative_incremental_conversions": 80.0, "ci_95": [40.0, 120.0]}
      ]
    },
    {
      "policy": "random",
      "contacted": 4000,
      "total_cost": 20000.0,
      "incremental_conversions": 40.0,
      "incremental_conversions_ci_95": [20.0, 60.0],
      "incremental_profit": -12000.0,
      "profit_per_unit_spent": -0.6,
      "curve": [
        {"fraction_targeted": 0.0, "cumulative_incremental_conversions": 0.0, "ci_95": [0.0, 0.0]},
        {"fraction_targeted": 0.25, "cumulative_incremental_conversions": 10.0, "ci_95": [5.0, 15.0]},
        {"fraction_targeted": 0.5, "cumulative_incremental_conversions": 20.0, "ci_95": [10.0, 30.0]},
        {"fraction_targeted": 0.75, "cumulative_incremental_conversions": 30.0, "ci_95": [15.0, 45.0]},
        {"fraction_targeted": 1.0, "cumulative_incremental_conversions": 40.0, "ci_95": [20.0, 60.0]}
      ]
    },
    {
      "policy": "blanket",
      "contacted": 5000,
      "total_cost": 25000.0,
      "incremental_conversions": 75.0,
      "incremental_conversions_ci_95": [37.5, 112.5],
      "incremental_profit": -10000.0,
      "profit_per_unit_spent": -0.4,
      "curve": [
        {"fraction_targeted": 0.0, "cumulative_incremental_conversions": 0.0, "ci_95": [0.0, 0.0]},
        {"fraction_targeted": 0.25, "cumulative_incremental_conversions": 18.75, "ci_95": [9.375, 28.125]},
        {"fraction_targeted": 0.5, "cumulative_incremental_conversions": 37.5, "ci_95": [18.75, 56.25]},
        {"fraction_targeted": 0.75, "cumulative_incremental_conversions": 56.25, "ci_95": [28.125, 84.375]},
        {"fraction_targeted": 1.0, "cumulative_incremental_conversions": 75.0, "ci_95": [37.5, 112.5]}
      ]
    }
  ],
  "comparison": {
    "comparison": "uplift_vs_response_model",
    "incremental_conversions_difference": 80.0,
    "ci_95": [40.0, 120.0],
    "conclusion": "uplift_better"
  },
  "recorded_run_created": false,
  "source": "mock",
  "is_mock": true,
  "message": "Synthetic demonstration only. This endpoint does not execute the replay engine, inspect outcomes, or create an MLflow recorded run."
}
```

This is a deterministic demonstration payload, not an execution of the shared replay implementation. The mock echoes the supplied seed and includes all effective request inputs in its synthetic simulation ID; it does not draw or evaluate data. The production adapter must call the same replay code as the offline replay CLI for identical effective inputs and seed (T33 FR-39). It must not write a recorded UC-06 MLflow run. Recorded replay remains an offline job under ADR 0002.

## 7. `GET /model/info`

Returns `model_version`, `training_date`, and evaluation metrics with confidence intervals. In T45 mock mode:

- `model_version` is `netlift-mock-2026.11.04`.
- `training_date` is `null`, because no registered champion is connected.
- Metric values and intervals are explicitly `illustrative_mock`, not measured results.
- `source: "mock"` and `is_mock: true` must be displayed by clients; the example metrics must never be presented as actual project performance.

Example response:

```json
{
  "model_version": "netlift-mock-2026.11.04",
  "training_date": null,
  "metrics": [
    {"name": "qini_coefficient", "value": 0.31, "ci_95": [0.24, 0.38], "status": "illustrative_mock"},
    {"name": "auuc", "value": 0.12, "ci_95": [0.08, 0.16], "status": "illustrative_mock"},
    {"name": "uplift_at_10pct", "value": 0.0025, "ci_95": [0.001, 0.004], "status": "illustrative_mock"}
  ],
  "source": "mock",
  "is_mock": true,
  "message": "Illustrative mock metadata only; no registered champion, training date, or measured evaluation result is connected."
}
```

`source` allows `mock` or `registry`; `is_mock` is a boolean so a registry-backed adapter can return `false` without changing the client type.

## 8. `GET /monitor/drift`

When a mock report is available, the response includes:

- Per-feature PSI and KS statistics.
- Uplift-score distribution (`mean`, `standard_deviation`, `p05`, `median`, `p95`).
- Data-quality status (`max_null_rate`, `range_violations`, `schema_changes`).
- Model version, report timestamp, and `is_mock: true`.

When there is no report, the endpoint returns HTTP `200` with `status: "not_available"`, `feature_drift: []`, null score/data-quality sections, and a message. This is the T45 empty-state contract; clients should show “No drift report is available yet”, not interpret the absence as zero drift.

The mock's PSI/KS and score distribution values are illustrative only. Thresholds and calculations against the real training reference must be supplied by the monitoring implementation; T45 does not invent alert thresholds.

Example response:

```json
{
  "status": "available",
  "generated_at": "2026-11-04T12:00:00Z",
  "model_version": "netlift-mock-2026.11.04",
  "feature_drift": [
    {"feature": "f0", "psi": 0.08, "ks_statistic": 0.04, "status": "illustrative_mock"},
    {"feature": "f3", "psi": 0.17, "ks_statistic": 0.09, "status": "illustrative_mock"},
    {"feature": "f7", "psi": 0.11, "ks_statistic": 0.06, "status": "illustrative_mock"}
  ],
  "score_drift": {
    "mean": 0.0018,
    "standard_deviation": 0.0042,
    "p05": -0.0041,
    "median": 0.0012,
    "p95": 0.0097,
    "status": "illustrative_mock"
  },
  "effect_drift": {
    "metric": "incremental_conversions_rate",
    "current_estimate": 0.0012,
    "reference_estimate": 0.001,
    "difference": 0.0002,
    "ci_95": [-0.0003, 0.0007],
    "source": "mock_replay_output"
  },
  "data_quality": {
    "status": "illustrative_mock",
    "max_null_rate": 0.0,
    "range_violations": 0,
    "schema_changes": 0
  },
  "is_mock": true,
  "message": "Illustrative mock drift values only. Alert thresholds and live reference-distribution calculations are not connected."
}
```

The response types intentionally permit future non-mock values: `is_mock` is boolean; metric/feature/score/data-quality statuses can be `available` or `unavailable`; and effect-drift source can be `replay_output` when connected to a measured replay. `ModelInfoResponse.source` allows `mock` or `registry`; policy simulation source and health mode allow `mock` or `live`; `HealthResponse.model_loaded` is boolean.

## 9. `GET /health`

Returns `status`, `ready`, `mode`, `model_loaded`, and a message.

The T45 mock app reports `mode: "mock"`, `model_loaded: false`, and `ready: true` when the mock adapter is able to serve contract examples. This indicates API-skeleton readiness, not production model readiness. When the adapter is not ready, `ready` is false and scoring, simulation, model-info, and drift requests return HTTP `503`. The transport fields are deliberately future-compatible: `mode` may be `mock` or `live`, and `model_loaded` is a boolean.

Example response:

```json
{
  "status": "ready",
  "ready": true,
  "mode": "mock",
  "model_loaded": false,
  "message": "The API mock adapter is ready; no champion model is loaded."
}
```

## 10. Errors and status codes

Errors use one envelope:

```json
{
  "error": {
    "code": "request_validation_error",
    "message": "Request validation failed.",
    "details": []
  }
}
```

| HTTP status | Meaning |
|---|---|
| `400` | Malformed/empty CSV or JSON batch, missing CSV headers, malformed row shape |
| `404` | Unknown route |
| `415` | Unsupported batch request media type |
| `422` | Pydantic validation failure, forbidden feature, non-finite feature, invalid simulation split, or inconsistent batch economics |
| `503` | Service adapter not ready; no score or simulation is returned |

The error `details` are intended for display and diagnostics; callers should branch on `error.code`, not parse message text.

Example replay-safety validation error for `data_slice: "locked_test"`:

```json
{
  "error": {
    "code": "request_validation_error",
    "message": "Request validation failed.",
    "details": [
      {
        "location": ["body", "data_slice"],
        "message": "Value error, The locked test split is opened once at T70; this endpoint accepts 'validation' only.",
        "type": "value_error"
      }
    ]
  }
}
```

## 11. T45 boundary decisions and integration notes

These choices make the contract implementable while preserving T33 and ADR 0002. The open-question identifiers below are retained for traceability; the T45 choices remain subject to reviewer acceptance where T33 still marks the question open.

1. **Q1 — replay split and locked-test access:** the final stage-specific split policy remains open; interim requests are restricted to `validation`, and `LockedTestSplit.open(reason)` under T70 is the only permitted locked-test opening path.
2. **Q3(c) — simulation identity:** T45 requires `scenario_id`, `data_slice`, `model_version`, and `seed` in addition to budget, cost and margin; only `validation` is accepted while Q1 is open.
3. **Q4 — standalone scoring economics:** T33 leaves the source of cost and margin open; T45 proposes requiring positive `contact_cost` and `conversion_margin` on the score request and echoing them in headers/batch summary, without changing the T36 response. This is an API-boundary proposal for Ali's review, not a claim that T33's open question has been closed by team decision.
4. **ADR 0002 — recorded replay boundary:** a recorded UC-06 replay is an offline CLI job logged to MLflow; `POST /policy/simulate` re-runs shared replay code for a fresh what-if and creates no recorded run; the API remains at six endpoints.
5. **FR-39 — shared replay implementation:** the offline CLI and `/policy/simulate` must call the same replay implementation and produce equivalent result values for identical effective inputs and seed; T45's mock does not yet execute this production behavior.
6. **FR-45 — locked-test safety:** until Q1 is decided, simulation accepts only the validation split; locked-test data can be opened only through `LockedTestSplit.open(reason)` under T70, and this endpoint refuses other split inputs.
7. **Q3(d) — recorded replay display:** no recorded-run endpoint is added. The dashboard uses a server-side MLflow integration for recorded UC-06 results; `/policy/simulate` remains a fresh what-if and does not read stored results.
8. **Empty drift report:** return HTTP `200` with the typed `not_available` state.
9. **Batch delivery:** return a ZIP containing `scores.csv` and `summary.json`, so the scored file and summary are delivered together.

These are T45 API-boundary choices for Ali's review. They do not alter the frozen T36 interface and do not claim that the production scoring, replay, registry, or monitoring adapters are complete.

## 12. Dashboard tab-to-source mapping

The six API endpoints do not provide every dashboard artefact. The UI must distinguish request-time API responses from recorded MLflow runs and versioned report files.

| Dashboard tab | API source | MLflow source | `reports/` source / limitation |
|---|---|---|---|
| **Tab 1 — Campaign Planner** | `POST /score/batch` and `POST /score` provide scores; `POST /policy/simulate` provides a fresh what-if comparison. | Not the source of the uploaded-file scoring response. | Downloaded target list is produced from the scoring/allocation workflow. The exact location of interactive allocation for an uploaded file (budget + policy mode) is deliberately unresolved until T55; this PR does not assign it to a seventh endpoint. |
| **Tab 2 — Policy Comparison** | `POST /policy/simulate` provides a fresh equal-k what-if response. | Recorded UC-06 replay runs, policy metrics and their artifacts are read server-side from MLflow. | Versioned comparison figures/reports may be used as static presentation snapshots, not as the authoritative live run store. |
| **Tab 3 — Model Evidence** | `GET /model/info` provides model version and summary evaluation metrics/intervals. | Detailed run metrics and artifacts: calibration, decile tables, SHAP outputs, and K4 negative-control evidence. These are not exposed by the six API responses. | Reviewed plots and generated report snapshots may be published under `reports/`; they are static artefacts tied to a run/version. |
| **Tab 4 — Monitoring** | `GET /monitor/drift` provides the latest feature/score drift contract response; in T45 this is mocked. | Recorded replay results and associated effect-measurement artifacts; production effect drift must be based on measured replay output and its delayed outcomes. | Monitoring snapshots/reports can be stored under `reports/`, but do not replace the current monitoring API or MLflow record. The T45 effect-drift object is explicitly mock data. |

`docs/experiment-tracking.md` governs the MLflow tracking convention. Files in `reports/` are reviewable/generated deliverables, not a substitute for the source run in MLflow. Tab 1 allocation semantics remain for T55 to settle.

## 13. T45 follow-up boundary

The following review suggestions are not additional acceptance claims for this mocked API patch and should be tracked separately: exhaustive 400/503 path tests; the full case/whitespace forbidden-column matrix; an OpenAPI `ScoringResponse` property-set snapshot; zero-filled verdict counts; a catch-all 500 envelope; additional provenance fields for `MetricEstimate`; and production batch optimisations, limits and CSV formula-injection protection. T45 only guarantees the cases explicitly covered by its current tests.

## 12. Running locally

Install the pinned dependencies and package:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Start the mock API:

```bash
uvicorn netlift.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/openapi.json`.

Run the API tests:

```bash
pytest -q tests/test_api_endpoints.py tests/test_api_schema.py
```

The service is a development skeleton. Authentication, deployment hardening, real model loading, real replay integration, persistent scoring logs, and live monitoring are outside T45's mocked scope.
