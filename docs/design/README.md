# System Analysis and Design (M4): index

**Task:** T53 ([#65](https://github.com/mahmoud-nadeem/netlift/issues/65)) · **Owner:** Mahmoud ·
**Status as of 20 Nov 2026**

This page lists every part of the M4 document: the task that owns it, the owner, the file and the
pull request, and whether it is on `main`. The submitted Word and PDF files are built from these
files. Update the "Merged" column when a pull request merges.

## The design document

| # | Section | Task | Owner | File | PR | Merged |
|---|---|---|---|---|---|---|
| 1 | Problem statement and objectives | T50 | Mahmoud | `docs/design/behaviour.md` §1–2 | [#108](https://github.com/mahmoud-nadeem/netlift/pull/108) | No |
| 2 | Use case diagram and descriptions | T46 | Ali | `docs/design/use-cases.md` | [#118](https://github.com/mahmoud-nadeem/netlift/pull/118) | No |
| 3 | Architecture, technology stack, component and deployment diagrams | T47 | Mazen | `docs/design/architecture.md` | [#114](https://github.com/mahmoud-nadeem/netlift/pull/114) | No (changes requested) |
| 4 | Data model: ERD, logical and physical schema | T48 | Aliaa | `docs/design/data-model.md` | [#109](https://github.com/mahmoud-nadeem/netlift/pull/109) | Yes |
| 5 | Data flow diagrams, wireframes, UI/UX guidelines | T49 | Mohammed | `docs/design/dfd-and-ui.md` | [#116](https://github.com/mahmoud-nadeem/netlift/pull/116) | No (draft) |
| 6 | Class, sequence, activity and state diagrams | T50 | Mahmoud | `docs/design/behaviour.md` §3–6 | [#108](https://github.com/mahmoud-nadeem/netlift/pull/108) | No |
| 7 | API contract (OpenAPI) | T45 | Mazen | `docs/design/api-contract.md` | [#112](https://github.com/mahmoud-nadeem/netlift/pull/112) | No |
| 8 | The three frozen interfaces | T36 | Mahmoud | `docs/design/interfaces.md` | [#92](https://github.com/mahmoud-nadeem/netlift/pull/92) | Yes |
| 9 | Test plan | T51 | Mazen (drafted by Mahmoud) | `docs/design/test-plan.md` | [#120](https://github.com/mahmoud-nadeem/netlift/pull/120) | No |

## Inputs the document builds on

| Part | Task | Owner | File | PR | Merged |
|---|---|---|---|---|---|
| Stakeholders | T31 | Mohammed | `docs/requirements/01-stakeholders.md` | [#97](https://github.com/mahmoud-nadeem/netlift/pull/97) | Yes |
| User stories and use cases | T32 | Ali | `docs/requirements/02-user-stories-and-use-cases.md` | [#93](https://github.com/mahmoud-nadeem/netlift/pull/93) | Yes |
| Functional requirements | T33 | Mazen | `docs/requirements/03-functional-requirements.md` | [#100](https://github.com/mahmoud-nadeem/netlift/pull/100) | Yes |
| Non-functional requirements | T34 | Aliaa | `docs/requirements/04-non-functional-requirements.md` | [#99](https://github.com/mahmoud-nadeem/netlift/pull/99) | Yes |
| Literature: experiments and the two datasets | T25 | Aliaa | `docs/literature-review/01-experiments-and-data.md` | [#96](https://github.com/mahmoud-nadeem/netlift/pull/96) | Yes |
| Literature: meta-learners and class transformation | T26 | Mohammed | `docs/literature-review/02-meta-learners.md` | [#91](https://github.com/mahmoud-nadeem/netlift/pull/91) | No |
| Literature: uplift trees and honest estimation | T27 | Mahmoud | `docs/literature-review/03-trees-and-neural.md` | opened with this index | No |
| Literature: evaluating uplift models | T28 | Ali | `docs/literature-review/04-evaluation.md` | [#110](https://github.com/mahmoud-nadeem/netlift/pull/110) | Yes |
| Specification aligned with the plan | T52 | Mahmoud | `docs/NetLift_Project_Specification.tex` and `.pdf` | [#94](https://github.com/mahmoud-nadeem/netlift/pull/94) | Yes |
| Decision records 0001–0004 | — | several | `docs/decisions/` | #28, #95, #103, #104 | Yes |

## Still to merge before the M4 files are built

#108, #118, #114, #116, #112, #120, #91 and the T27 pull request. The Word and PDF files are
built once these are on `main`, and pushed to the company repository.
