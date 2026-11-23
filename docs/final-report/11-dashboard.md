# 11. Dashboard

NetLift's Streamlit dashboard gives campaign planners a way to score a customer list and compare targeting policies. Technical reviewers can inspect the model evidence and monitoring data behind those decisions. The dashboard reads results through the API, following the data flow and wireframes in the [T49 design](https://github.com/mahmoud-nadeem/netlift/blob/c86622fcf983034f54d29843caff3e4cb0e62de1/docs/design/dfd-and-ui.md). The four tabs are implemented in [T62](https://github.com/mahmoud-nadeem/netlift/pull/127).

### Campaign Planner

The planner uploads a customer CSV or selects demo customers, then sets the cost per contact and margin per conversion. The batch scoring response supplies each customer's uplift in percentage points, recommendation, and verdict. The detail view also shows the 95% uplift interval for the customer's score decile, expected incremental profit per contact, and top drivers when supplied. The interval describes evidence for the decile.

The table and detail view show the recommendation separately from its verdict. A confident result can recommend **Contact** or **Suppress**. Both abstention verdicts display **No action**, with different reasons:

- **Insufficient evidence** means the uplift interval includes zero.
- **Insufficient control data** means the segment has too few control events to support a decision.

The summary counts the two abstention verdicts separately. [Campaign Planner implementation](https://github.com/mahmoud-nadeem/netlift/blob/7209ecf2a08fab96c2890ea3d8f48c133fa93608/dashboard/app.py)

### Policy Comparison

The tab compares uplift, response-model, and random targeting under a common contact budget for a validation scenario. It also shows blanket contact as a full-coverage reference at its actual cost and flags it when that cost exceeds the budget. For each policy, the table shows contacts, cost, incremental conversions with a 95% interval, incremental profit, and profit per unit spent. The chart shows cumulative incremental conversions, uncertainty bands where supplied, and the budget marker. The comparison displays the API's paired uplift-minus-response estimate and 95% interval, with **No clear difference** when the API reports the policies as indistinguishable. [Policy Comparison implementation](https://github.com/mahmoud-nadeem/netlift/blob/7209ecf2a08fab96c2890ea3d8f48c133fa93608/dashboard/policy_comparison.py)

### Model Evidence

This tab shows the model version and source along with the evaluation metrics returned by the API, such as Qini, AUUC, and uplift at selected contact fractions. Each row shows the estimate, its 95% interval when supplied, and its availability status. The current endpoint supplies summary metrics. The tab marks detailed evidence such as Qini curves, calibration, SHAP, and negative-control results as unavailable. [Model Evidence implementation](https://github.com/mahmoud-nadeem/netlift/blob/7209ecf2a08fab96c2890ea3d8f48c133fa93608/dashboard/model_evidence.py)

### Monitoring

The tab shows the latest drift report with its timestamp and model version, feature-level PSI and KS statistics, uplift-score distribution summaries, and data quality indicators for missing values, range violations, and schema changes. A separate simulated-replay panel shows effect drift when replay results are available. The tab provides an explicit empty state when the API has no drift report. [Monitoring implementation](https://github.com/mahmoud-nadeem/netlift/blob/7209ecf2a08fab96c2890ea3d8f48c133fa93608/dashboard/monitoring.py)

All four tabs were exercised locally against the [T45 mock API](https://github.com/mahmoud-nadeem/netlift/pull/112). The interface labels its illustrative responses **Mock data**. Production API verification is pending. Budget-based customer allocation and target-list export in Campaign Planner depend on T55. [T62 verification and integration status](https://github.com/mahmoud-nadeem/netlift/pull/127)
