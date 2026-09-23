# NetLift — Causal Vocabulary Glossary

## Uplift, CATE

Uplift is the incremental change in a customer's outcome caused by treatment. In NetLift specifically, "uplift" refers to the model's estimate of this effect, τ̂(x), which is the project's estimate of the Conditional Average Treatment Effect — the term the whole system is built to predict. This matters because ranking customers by estimated uplift, rather than by any other score, is the entire point of NetLift: it is what separates the project from a standard classification exercise.

## Potential Outcomes, Y(1) and Y(0)

Y(1) is what a customer's outcome would be if treated (contacted), and Y(0) is what it would be if not treated; the individual treatment effect is Y(1) − Y(0). This matters because NetLift's target, uplift, is defined directly as the difference between these two potential outcomes for each customer.

## The Fundamental Problem of Causal Inference

For any single customer, only one of Y(1) or Y(0) is ever observed — whichever arm they actually ended up in — so the other is a missing counterfactual that can never be directly measured. This matters because it is the reason NetLift cannot train on a per-row uplift label: there is no ground truth to fit against, so effects must be estimated statistically by comparing groups rather than individuals.

## ATE versus CATE

The Average Treatment Effect (ATE) is the average effect across the whole population, while the Conditional Average Treatment Effect (CATE) is the average effect within a subgroup defined by covariates X. This matters because NetLift needs CATE, not ATE: a single population-wide number cannot tell the decision layer which individual customers are actually worth targeting.

## Treatment, Control, Outcome

Treatment is the group that receives the intervention (for example, being sent an offer), control is the group that does not, and outcome is the result being measured (for example, conversion). This matters because every uplift estimate in NetLift is built by comparing the outcome between the treatment and control arms of the underlying experimental data.

## Persuadables, Sure Things, Lost Causes, Sleeping Dogs

These four segments describe, conceptually, how treatment changes a customer's likelihood of the outcome: Persuadables are meaningfully more likely to convert because they were treated, Sure Things are inclined to convert whether or not they are treated so the treatment adds comparatively little, Lost Causes are unlikely to convert regardless of treatment so it again adds comparatively little, and Sleeping Dogs become less likely to convert because they were treated. This matters because NetLift's targeting policy exists specifically to find Persuadables, avoid spending budget where it changes little, and suppress contact with Sleeping Dogs.

## Unconfoundedness, Positivity and Overlap, SUTVA

Unconfoundedness means potential outcomes are independent of treatment assignment given the covariates; positivity/overlap means every customer has a nonzero chance of landing in either arm; SUTVA means one customer's outcome is unaffected by another customer's treatment and that there is only one version of the treatment. This matters because these are the assumptions that let NetLift treat the difference between treated and control outcomes as a genuine causal effect rather than a mere correlation.

## Intention-to-Treat versus Treatment-on-the-Treated

Intention-to-treat (ITT) measures the effect of being assigned to the campaign, regardless of whether that assigned treatment was actually delivered to the customer, while treatment-on-the-treated (ToT) measures the effect specifically among the units that actually received the treatment as assigned. This matters because NetLift estimates ITT, since that is the quantity the experimental data identifies and the quantity a marketing team actually controls — who is assigned to be contacted, not whether delivery of the treatment succeeded for each individual.

## Qini Curve, Qini Coefficient, AUUC, Uplift@k

The Qini curve plots cumulative incremental outcomes as more customers are targeted in order of predicted uplift, the Qini coefficient summarizes that curve as a single number relative to random targeting, AUUC is a related area-under-the-curve measure, and uplift@k reports the uplift achieved within the top-k targeted fraction. This matters because these are the metrics NetLift uses to evaluate ranking quality on an effect that has no per-row label, in place of standard classification metrics.

## Response Model versus Uplift Model

A response model predicts the probability that a customer converts if contacted and ranks by that likelihood, while an uplift model predicts the incremental change in outcome caused by contact and ranks by that causal effect. This matters because NetLift's central claim rests on this distinction: a response model cannot tell Persuadables apart from Sure Things, since both convert when treated, which is exactly the confusion uplift targeting is built to avoid.

## Meta-Learners: S, T, X, R, DR

Meta-learners are general strategies for estimating CATE by combining standard supervised models: the S-learner trains one model with treatment as an input feature, the T-learner trains two separate models (one per arm), the X-learner imputes each arm's missing counterfactual to better exploit imbalanced treatment/control sizes, and the R- and DR-learners use more advanced orthogonalized or doubly-robust estimation. This matters because NetLift's uplift estimators are built from this family, with the X-learner favored specifically because the treatment group is much larger than the control group in the primary dataset.

## Class Transformation

Class transformation relabels the outcome into a single transformed target so that a standard classifier's output can be converted into an uplift score. This matters because it is one of the modeling approaches NetLift considers, but it is only valid in its propensity-weighted form given the project's imbalanced treatment split — used naively, it produces a distorted uplift score that can still look plausible.

## Confounding, Selection Bias

Confounding occurs when a variable influences both treatment assignment and the outcome, making the observed treatment-outcome relationship misleading, while selection bias occurs when treated and control groups differ systematically for reasons other than the treatment itself. This matters because relying on randomized experimental data is exactly what protects NetLift from both problems, which is why the project treats non-randomized (observational) data differently from its primary datasets.

## Randomized Controlled Trial

A randomized controlled trial (RCT) is an experiment in which each subject is assigned to treatment or control by a random mechanism, making the two groups comparable on average before treatment occurs. This matters because NetLift's datasets are randomized experiments, which is what allows the observed difference in outcomes between arms to be interpreted as a genuine causal effect rather than a correlation.

## Why Accuracy, F1 and ROC-AUC Are Invalid for This Problem

Accuracy, F1 and ROC-AUC can be computed against an observed outcome, but they are not valid primary metrics for evaluating uplift predictions, because the quantity NetLift actually predicts — each customer's individual treatment effect — is never observed, even though the outcome itself was recorded. This matters because a NetLift model could score well on these metrics while still recommending a targeting policy that loses money, which is why the project relies on uplift-specific metrics like the Qini coefficient instead.
