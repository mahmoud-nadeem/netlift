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
5. The request/response schemas are strict where the fields are fixed. Feature columns are separately checked by the existing `validate_features()` contract before scoring.

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
| `uplift_score` | number | Estimated incremental effect |
| `uplift_ci_95` | `[number, number]` | 95% confidence interval `[lower, upper]` |
| `decile` | integer, 1–10 | Uplift ranking decile |
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
- `contact_cost` and `conversion_margin` are finite positive numbers. They are explicit request inputs for standalone scoring (T33 Q4 resolution in this contract).
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

The summary includes the economic assumptions and profit formula used for every row. This gives batch consumers the assumptions alongside the profit figures without changing the T36 per-customer response.

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

A request selecting `test`, `locked_test`, or any non-validation split is rejected. This enforces T33 FR-45 and the interim rule in ADR 0002; it does not open `LockedTestSplit`.

### Response

The response contains:

- Simulation identity (`simulation_id`, `scenario_id`, `data_slice`, `model_version`, `seed`).
- Budget and campaign economics.
- Results for `uplift`, `response_model`, `random`, and `blanket` policies. Each includes contacted count, total cost, incremental conversions and 95% interval, incremental profit, profit per unit spent, and cumulative curve points with intervals.
- `comparison` for uplift versus response-model incremental conversions, its interval, and a conclusion (`uplift_better`, `response_model_better`, or `not_distinguishable`).
- `recorded_run_created: false`, `source: "mock"`, and `is_mock: true`.

This is a deterministic demonstration payload, not an execution of the shared replay implementation. The mock echoes the supplied seed and includes all effective request inputs in its synthetic simulation ID; it does not draw or evaluate data. The production adapter must call the same replay code as the offline replay CLI for identical effective inputs and seed (T33 FR-39). It must not write a recorded UC-06 MLflow run. Recorded replay remains an offline job under ADR 0002.

## 7. `GET /model/info`

Returns `model_version`, `training_date`, and evaluation metrics with confidence intervals. In T45 mock mode:

- `model_version` is `netlift-mock-2026.11.04`.
- `training_date` is `null`, because no registered champion is connected.
- Metric values and intervals are explicitly `illustrative_mock`, not measured results.
- `source: "mock"` and `is_mock: true` must be displayed by clients; the example metrics must never be presented as actual project performance.

## 8. `GET /monitor/drift`

When a mock report is available, the response includes:

- Per-feature PSI and KS statistics.
- Uplift-score distribution (`mean`, `standard_deviation`, `p05`, `median`, `p95`).
- Data-quality status (`max_null_rate`, `range_violations`, `schema_changes`).
- Model version, report timestamp, and `is_mock: true`.

When there is no report, the endpoint returns HTTP `200` with `status: "not_available"`, `feature_drift: []`, null score/data-quality sections, and a message. This is the T45 empty-state contract; clients should show “No drift report is available yet”, not interpret the absence as zero drift.

The mock's PSI/KS and score distribution values are illustrative only. Thresholds and calculations against the real training reference must be supplied by the monitoring implementation; T45 does not invent alert thresholds.

## 9. `GET /health`

Returns `status`, `ready`, `mode`, `model_loaded`, and a message.

The T45 mock app reports `mode: "mock"`, `model_loaded: false`, and `ready: true` when the mock adapter is able to serve contract examples. This indicates API-skeleton readiness, not production model readiness. When the adapter is not ready, `ready` is false and scoring, simulation, model-info, and drift requests return HTTP `503`.

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

## 11. T45 boundary decisions and integration notes

These choices make the contract implementable while preserving T33 and ADR 0002:

1. **Q3(c), simulation identity:** require `scenario_id`, `data_slice`, `model_version`, and `seed` in addition to budget, cost and margin. The only allowed slice is `validation` until Q1/T70 changes the policy.
2. **Q3(d), recorded replay display:** no recorded-run endpoint is added. The dashboard may read recorded UC-06 MLflow runs from its server-side integration; `/policy/simulate` remains a fresh what-if response and does not read stored results.
3. **Q4, standalone scoring economics:** require positive `contact_cost` and `conversion_margin` in score requests. Echo them in score response headers and in the batch summary; do not add fields to `ScoringResponse`.
4. **Empty drift report:** return HTTP `200` with the typed `not_available` state.
5. **Batch delivery:** return a ZIP containing `scores.csv` and `summary.json`, so the scored file and summary are delivered together.

These are T45 API-boundary choices for Ali's review. They do not alter the frozen T36 interface and do not claim that the production scoring, replay, registry, or monitoring adapters are complete.

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
