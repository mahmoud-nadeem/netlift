# Final report: outline and owners (T69)

**Task:** T69 ([#81](https://github.com/mahmoud-nadeem/netlift/issues/81)) · **Owner:** Mahmoud
(assembly) · **Partners:** all

Each member drafts the sections on their own component by **Fri 27 Nov**. Mahmoud assembles the
report by **Thu 3 Dec**. The section list follows what the specification asks the final report to
contain. The section number of each requirement is in the right-hand column.

## Rules for every section

| Rule | Spec |
|---|---|
| Every number is produced by a script in `src/` writing into `reports/`, never copied by hand from a notebook. Name the script next to the number. | §37 |
| Every reported metric has its 95% bootstrap interval, resampled within treatment arms. | §28 |
| Write the claim in the measured form ("12% more incremental conversions than the response model at equal budget, 95% CI ..."), not the hedged or inflated one. | §13 |
| Reuse the M4 diagrams (`docs/design/`). Do not redraw them. | §45 |
| Validation numbers are labelled validation. Only T70 reports locked-test numbers, once. | §19, §26 |

## Sections

| # | Section | What it must contain | Owner | Draft | Spec |
|---|---|---|---|---|---|
| 1 | Summary | The question, the K1 result with its interval, what was built | Mahmoud | after T70 (Tue 1 Dec) | §2, §46 |
| 2 | Problem statement and objectives | From `behaviour.md` §1–2; objectives mapped to K1–K11 | Mahmoud | Fri 27 Nov | §2, §46 |
| 3 | Related work | The four literature pieces (T25–T28), each with what it changed in the code | Mahmoud (assembles) | Fri 27 Nov | — |
| 4 | Data | The two datasets; which figures were verified against the downloaded files and which were taken from documentation; randomization integrity; the joint stratified split; the leakage audit; the cleaning log | Aliaa | Fri 27 Nov | §16–§20, §24 |
| 5 | Methods: baselines and meta-learners | Random, blanket and response model; S-, T- and X-learners; the class-transformation trap | Mohammed | Fri 27 Nov | §7, §9 |
| 6 | Methods: uplift forest and tuning | The out-of-fold Qini objective; how many trials each model family got, the response model included; the champion rule | Mahmoud | Fri 27 Nov | §25, §26 |
| 7 | Evaluation and results | Qini, AUUC and uplift@k with intervals; K1 to K5; the negative control; the locked-test confirmation from T70 | Ali | Fri 27 Nov (T70 numbers added 1 Dec) | §27, §28, §46 |
| 8 | Decision layer and simulated replay | Policy modes, suppression, both abstention verdicts; the replay and its holdout | Ali | Fri 27 Nov | §10, §11, §13, §14 |
| 9 | Error analysis and behavioural tests | Segment audit; invariance tests | Ali | Fri 27 Nov | §29, §30 |
| 10 | System | Architecture, API, schema, the Azure container, drift monitoring | Mazen | Fri 27 Nov | §32, §35, §36 |
| 11 | Dashboard | The four tabs and how they show abstention | Mohammed | Fri 27 Nov | §40 |
| 12 | Testing and CI | Test plan, test levels, CI gates, data validation tests | Mazen, with Aliaa (T66) | Fri 27 Nov | §34 |
| 13 | Reproducibility | MLflow conventions; the one command that reproduces the headline result | Mahmoud | Fri 27 Nov | §33 |
| 14 | Limitations | At least the spec's list, starting with "public benchmark data, not Egyptian market data" | Mahmoud | Fri 27 Nov | §52 |
| 15 | Ethics and privacy | No personal data; the effect is assigned to groups, not labels on individuals | Aliaa | Fri 27 Nov | §53 |
| 16 | Future work | The cut items: multi-treatment and revenue uplift, causal forest, the neural estimator (T63) | Mahmoud | Fri 27 Nov | §49 |
| 17 | Individual contributions | One paragraph per member: the component and one technical decision made in it | each member | Fri 27 Nov | §47 (release) |

## Where drafts go

One Markdown file per section in `docs/final-report/`, named `NN-section.md`, each in its own
pull request with the section's owner as author and Mahmoud as reviewer.
