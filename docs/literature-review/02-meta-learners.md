# Meta-learners and class transformation

Author: Mohammed Sherif · Task: T26

## 1. Künzel, Sekhon, Bickel & Yu (2019) — the S-, T- and X-learners

**Citation.** Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning*. Proceedings of the National Academy of Sciences. https://doi.org/10.1073/pnas.1804597116 (preprint: https://arxiv.org/abs/1706.03461)

**What the paper says.** Any base model can be turned into a treatment-effect estimator by wrapping it in a "meta-learner" instead of redesigning it. The paper formalises the S-learner (one model, treatment as an input) and the T-learner (one model per arm), then proposes the X-learner for groups of very different sizes: it imputes each unit's missing counterfactual from the opposite arm's model, fits an effect model on each side's imputed effects, and combines them with a propensity-score weight. Under stated regularity conditions the X-learner can reach a near-parametric rate, and it is competitive in simulation, though the authors stress that no meta-learner is best everywhere.

**What NetLift takes from it.** These are the MVP's two mandatory estimators, T-learner and X-learner (Part II of `01-specification.md`), and the basis for making X-learner primary: Criteo's 0.85 treated share leaves a thin control arm, the situation the X-learner was designed for. That mapping is the project's own extrapolation. The paper's evidence comes from its own simulations and field experiments, not from Criteo or Hillstrom, so it supports the choice without proving it.

## 2. Jaśkowski & Jaroszewicz (2012) — class variable transformation

**Citation.** Jaśkowski, M., & Jaroszewicz, S. (2012). *Uplift modeling for clinical trial data*. ICML 2012 Workshop on Machine Learning for Clinical Data Analysis, Edinburgh. https://home.ipipan.waw.pl/sj/pdf/logsemi.pdf

**What the paper says.** A classifier trained on outcomes alone says who responds, not who responds because of treatment, so the authors relabel the data. They define Z = 1 for treated patients with a positive outcome and control patients with a negative one, and show that uplift equals 2P(Z=1∣X) − 1 when treatment and control are equally likely. For unbalanced data they note in one sentence that reweighting or resampling makes this assumption valid. On three clinical datasets the ranking of methods changes: the double-model approach is best on bone marrow transplant and tamoxifen (where the transformation is second and last of four respectively), while the transformation beats two separate classifiers on hepatitis.

**What NetLift takes from it.** This is the source of the "class-transformation trap" (Part II): the identity needs equal assignment probabilities, and Criteo's treated share is 0.85 (11,882,653 of 13,979,592 rows, per the spec's compliance table), so the naive version is distorted there. The paper mentions reweighting but does not develop it, so the spec's rule to forbid the naive version and require a propensity-weighted one is NetLift's own extension of that remark. The method is also written for one treatment against one control; for Hillstrom's two email arms, the spec's MVP analyses one arm at a time against the shared control, and joint multi-arm modelling is Advanced tier.

## 3. Nie & Wager (2021) — the R-learner

**Citation.** Nie, X., & Wager, S. (2021). *Quasi-oracle estimation of heterogeneous treatment effects*. Biometrika, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076 (preprint: https://arxiv.org/abs/1712.04912)

**What the paper says.** The standard S- and T-learners mainly estimate outcome functions and take the CATE as a difference of fitted outcomes. They do not necessarily need a propensity model, and the loss does not target the effect directly, so bias in the outcome fits can pass into the estimate. The R-learner instead cross-fits two nuisance estimates, the conditional mean outcome and the propensity score, uses them to build a residualised objective via Robinson's transformation, and minimises that "R-loss" over candidate effect functions. For penalised kernel regression, the paper shows the error rate matches an oracle that knows the nuisance functions ("quasi-oracle"), provided those are estimated at a stated minimum rate.

**What NetLift takes from it.** The spec lists the R-learner as Advanced tier, "if time allows" (Part II), and requires out-of-fold nuisance models: incorrect cross-fitting "reintroduces exactly the bias the estimator exists to remove, and the error is invisible in the metrics" (splitting-strategy section, which applies this to the X- and R-learners). Whether NetLift's base learners meet the paper's rate condition on Criteo's anonymised features is unverified, which is why this stays Advanced.

## 4. Kennedy (2023) — the DR-Learner

**Citation.** Kennedy, E. H. (2023). *Towards optimal doubly robust estimation of heterogeneous causal effects*. Electronic Journal of Statistics, 17(2). https://doi.org/10.1214/23-EJS2157 (preprint: https://arxiv.org/abs/2004.14497)

**What the paper says.** Kennedy studies a two-stage doubly robust estimator that he names the DR-Learner, following Nie & Wager's naming. The gap he addresses is that it was unclear when such estimators are optimal, not just consistent, once the true effect has structure such as smoothness or sparsity. Theorem 2 gives a general, model-free error bound whose "doubly robust" form means the error depends on the nuisance errors jointly, so oracle efficiency needs weaker conditions than earlier results. A local-polynomial variant (Section 5) reaches oracle efficiency under weaker conditions still, with specialised sample splitting.

**What NetLift takes from it.** The spec lists the DR-learner as Advanced tier, alongside the R-learner, and describes it as staying consistent if either the outcome or the propensity model is correct, at the price that "Three-way cross-fitting costs sample efficiency and is easy to implement wrongly". That consistency description is the standard doubly-robust property; the paper's own contribution is the error-bound and efficiency analysis, so the review cites it for the rates, not for the property itself.

## How these sources relate

Künzel et al. frame the family the other three sit inside: S-, T- and X-learners wrap off-the-shelf models, while class transformation is a single-model route that behaves oppositely with respect to group balance. The X-learner is designed to help when treated and control groups are very unequal, whereas the transformation identity as stated needs equal assignment probabilities, so an imbalanced split like Criteo's breaks it. Nie & Wager and Kennedy both rely on estimated nuisance functions (outcome and propensity) instead of differencing outcome fits alone, the R-learner through a residualised loss and the DR-Learner through a two-stage doubly robust construction. Each paper proves its guarantee only under its own conditions and none tests NetLift's datasets, so the choices of X-learner for the MVP and R- and DR-learners for Advanced are informed extrapolations, which the spec's conditional wording ("if time allows", "Advanced tier") already reflects.
