# Literature: uplift trees, causal forests and neural estimators (T27)

Owner: Mahmoud Nadeem · Reviewer: Aliaa Gomaa · Due: Wed 4 Nov 2026

## 1. Rzepakowski & Jaroszewicz (2012): splitting criteria for uplift trees

**Citation.** Rzepakowski, P., & Jaroszewicz, S. (2012). Decision trees for uplift modeling with single and multiple treatments. *Knowledge and Information Systems*, 32, 303–327. <https://doi.org/10.1007/s10115-011-0434-0> (open access; published online 29 July 2011).

**Summary.** An uplift tree should split where the treated and control outcome distributions differ most. The paper scores a split by the gain in divergence between them, with three measures (§3.2): Kullback-Leibler, squared Euclidean distance and chi-squared, giving KL_gain, E_gain and χ²_gain. A normalization factor (§3.3) penalises splits that divide treatment and control in different proportions, and tests with many outcomes. The paper argues that squared Euclidean distance is the most stable, since the other two tend to infinity when a control-group probability tends to zero.

**What it changes in our code.** ADR 0006 (in #122) searches the `evaluationFunction` of causalml's `UpliftRandomForestClassifier` over `KL`, `ED` and `Chi`, the paper's three criteria, and keeps `normalization` on, which applies the §3.3 correction. causalml also attributes `n_reg` to this paper. We did not find it in §3.1–§3.5, so we cite it to causalml only. The multiple-treatment part (§5) does not apply: each of our comparisons has one treatment.

## 2. Athey & Imbens (2016): honest estimation

**Citation.** Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360. <https://doi.org/10.1073/pnas.1510489113>

**Summary.** A tree that chooses its splits and estimates each leaf's effect on the same rows overstates the differences. The "honest" approach builds the partition on one sample and estimates the effects on another. In its simulation, honest estimation gives nominal coverage for 90% confidence intervals, against 74% to 84% for non-honest approaches. The price is a smaller estimation sample: 7% to 22% higher mean squared error for the preferred method.

**What it changes in our code.** causalml exposes this as `honesty=True` with `estimation_sample_size`. ADR 0006 does not search it, so every forest trial is non-honest. Two consequences follow:

- **Intervals:** the forest's leaf estimates are never reported as intervals. Every interval comes from the T39 bootstrap on the validation split, which the forest never saw.
- **Model selection:** the out-of-fold Qini objective applies the same rule one level up. A trial is scored only on rows its model did not train on.

An honest refit of the champion forest is possible as a sensitivity check. It is not part of ADR 0006.

## How the two relate

Rzepakowski and Jaroszewicz say how an uplift tree should choose its splits. Athey and Imbens say why the rows that chose the splits must not also measure the effect.

## Neural estimators

The shared-representation neural estimator (T63) is cut in the M5 replan (#121). It is future work, and this review covers no neural source.
