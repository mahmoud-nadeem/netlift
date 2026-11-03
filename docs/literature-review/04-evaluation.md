# Literature: evaluating uplift models (T28)

Owner: Ali Alaa · Reviewer: Mohammed Sherif · Due: Wed 4 Nov 2026

## 1. Radcliffe (2007): the Qini measure

**Citation.** Radcliffe, N. J. (2007). Using control groups to target on predicted lift: Building and assessing uplift models. *Direct Marketing Analytics Journal*, issue 3, 14–21. <https://www.research.ed.ac.uk/en/publications/using-control-groups-to-target-on-predicted-lift-building-and-ass/> (the record lists no DOI).

**Summary.** Direct-marketing models usually rank customers by their chance of responding, but the useful quantity is the difference in behaviour between a treated group and a control group. The paper reviews the uplift methods proposed independently by several researchers and compares the main ones on three real-world applications. It also proposes quality measures for uplift models, for binary outcomes (purchase, churn) and continuous ones (spend). The Qini curve and coefficient come from these measures.

**What it changes in our code.** It defines the quantity behind the metric name `qini_coefficient` in `REPORTED_METRICS` (`src/netlift/evaluation/contracts.py`). The scikit-uplift documentation, which cites this paper, defines `qini_auc_score` as the area between the model's curve and the diagonal, divided by the same area for the optimal curve. ADR 0001 therefore makes scikit-uplift the only source of reported Qini values: `MetricResult.__post_init__` rejects any other library through `SOURCE_OF_TRUTH_LIBRARY`. The ADR's test shows why, since causalml's `normalize=True` gave 0.6542 against 0.2185 on the same predictions. Choosing the library is our decision, not the paper's. The function that computes the coefficient belongs to T39 and is not yet merged.

## 2. Efron & Tibshirani (1993): the bootstrap

**Citation.** Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Monographs on Statistics and Applied Probability 57. New York: Chapman & Hall. ISBN 0-412-04231-2.

**Summary.** Many statistics have no usable formula for their standard error or confidence interval. The bootstrap treats the sample as the population, redraws samples from it with replacement, and recomputes the statistic on each redraw. The percentile interval (Chapter 13) reads the interval from the quantiles of those recomputed values. For two groups (Section 8.3), the resampling uses one empirical distribution per group, so each group is redrawn separately.

**What it changes in our code.** `within_arm_bootstrap_indices` draws control and treated rows separately, each at its own size, which matches the two-sample scheme of Section 8.3. `paired_difference` takes percentiles of the per-resample differences at tail `(1 − confidence_level) / 2`, which is the percentile method. The book says nothing about Qini or uplift, so applying this scheme to uplift metrics is our choice. The minimum of 1,000 resamples (`MIN_REPORTED_RESAMPLES`) comes from the specification (Statistical Significance), not from the book.

## How the two relate

Radcliffe says what to measure, and Efron & Tibshirani say how to put an interval on it. K2 needs both, because a Qini coefficient reported without an interval is exactly what the specification forbids.
