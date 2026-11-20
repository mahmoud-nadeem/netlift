# Literature: uplift trees, causal forests and neural estimators (T27)

Owner: Mahmoud Nadeem · Reviewer: Ali Alaa · Due: Wed 4 Nov 2026

## 1. Rzepakowski & Jaroszewicz (2012): splitting criteria for uplift trees

**Citation.** Rzepakowski, P., & Jaroszewicz, S. (2012). Decision trees for uplift modeling with single and multiple treatments. *Knowledge and Information Systems*, 32, 303–327. <https://doi.org/10.1007/s10115-011-0434-0> (open access; published online 29 July 2011).

**Summary.** A standard tree splits to separate the classes. An uplift tree has to split where the *difference* between the treated and control outcome distributions changes. The paper designs splitting criteria that measure that divergence, and pruning methods for it, for one treatment and for several. Its experiments show these trees improving on earlier uplift-modelling techniques.

**What it changes in our code.** T57's uplift forest is causalml's `UpliftRandomForestClassifier`, and ADR 0006 (in #122) searches its split criterion, `evaluationFunction`, over KL, ED and Chi. causalml cites this paper for two settings of those criteria: `n_reg`, the weight of the parent node's distribution in a child's, and `normalization`, the correction for many-way and treatment–control-imbalanced splits. We search `n_reg` from 1 to 100 (`config/uplift_search.yaml`) and leave `normalization` at its default, on. The multiple-treatment part does not apply: Hillstrom's comparison is men's e-mail against no e-mail, and Criteo has one treatment.

## 2. Athey & Imbens (2016): honest estimation

**Citation.** Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360. <https://doi.org/10.1073/pnas.1510489113>

**Summary.** Trees that choose their splits and estimate the effect in each leaf from the same rows overstate how different the leaves are. The paper's "honest" approach builds the partition on one sample and estimates each leaf's effect on another. In its simulation, honest estimation gives nominal coverage for 90% confidence intervals, against 74% to 84% for non-honest approaches. The price is a smaller estimation sample: 7% to 22% higher mean squared error for the preferred method.

**What it changes in our code.** causalml exposes this as `honesty=True` with `estimation_sample_size`. ADR 0006 does not search it, so every forest trial is non-honest. Two consequences follow:

- **Intervals:** the forest's leaf estimates are never reported as intervals. Every interval comes from the T39 bootstrap on the validation split, which the forest never saw.
- **Model selection:** the out-of-fold Qini objective applies the same rule one level up. A trial is scored only on rows its model did not train on.

An honest refit of the champion forest is possible as a sensitivity check. It is not part of ADR 0006.

## How the two relate

Rzepakowski and Jaroszewicz say how an uplift tree should choose its splits. Athey and Imbens say why the rows that chose the splits must not also measure the effect.

## Neural estimators

The shared-representation neural estimator (T63) is cut in the M5 replan (#121). It is future work, and this review covers no neural source.
