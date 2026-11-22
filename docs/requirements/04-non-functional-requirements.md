# Non-functional requirements

Author: Aliaa Gomaa · Task: T34 · Reviewed by: Mazen

Every requirement states how it is measured. Targets come from `config/kpis.yaml` where one exists;
otherwise the target is "measured and reported in M5". No invented thresholds.

| ID | Category | Requirement | How it is measured | Target |
|---|---|---|---|---|
| NFR-01 | Reproducibility | A clean checkout plus one documented command reproduces the headline result. | Run the command on a fresh environment and compare the output with the recorded result. | Same result as recorded: reproduces exactly, from config plus the git commit hash (K8). |
| NFR-02 | Performance | Scoring a single customer returns quickly, and scoring a large batch completes in reasonable time. | Load test against the deployed endpoint: p95 latency of a single score, and the time for a one-million-row batch on the project's own hardware. | p95 under 300 ms for a single score; a one-million-row batch inside ten minutes (K9). Team-set targets, to be revised once first measured. |
| NFR-03 | Auditability | Every score is logged together with the version of the model that produced it (`model_version`). | Check that every logged score record contains `model_version`. | Measured and reported in M5. |
| NFR-04 | Correctness | No recommendation is produced without a `verdict` field, and every reported metric comes with a confidence interval. | Automated check on scoring responses: every recommendation comes with a non-empty `verdict`. Check on report outputs: each metric is paired with a confidence interval. | Metrics: reported with a confidence interval (K1, K2, K3). Verdict field: measured and reported in M5. |
| NFR-05 | Security | Secrets are stored only in `.env`, never in code, config or the repository, and the system stores and processes no personal data. | Secret scan of the repository and a check that `.env` is git-ignored. Review of datasets, logs and outputs for personal identifiers (both datasets are public and de-identified). | Measured and reported in M5. |
| NFR-06 | Usability | The abstention verdicts (`insufficient_evidence`, `insufficient_control_data`) are visually distinct from a confident recommendation. | Visual review of the dashboard, with an abstention and a confident recommendation side by side. | Measured and reported in M5. |
| NFR-07 | Reliability | The system handles bad or missing input and repeated runs without crashing or silently producing wrong output. | Failed runs and unhandled errors counted across test runs and CI. | Measured and reported in M5. |
| NFR-08 | Maintainability | Every change is verified by the CI gates before it merges. | CI on every pull request: lint, type check, unit, data, model-behaviour, integration and API tests. | Green on the merge commit of every pull request (K10). |