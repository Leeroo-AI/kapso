# Evaluation profile

The protected evaluator runs `main.py` in an isolated child process, requires integer arrays of shapes `(29979, 12)` and `(36129, 12)`, and computes the official RelBench recommendation metrics on all 29,979 validation rows. The selection metric is macro MAP@12 with denominator `min(|truth|, 12)`; all validation truth lists are nonempty.

## Input distribution

- Training has 86,616 rows at three forward origins: 29,210 on 2015-04-26, 25,007 on 2015-04-30, and 32,399 on 2015-05-04.
- Validation has 29,979 rows at 2015-05-08; test has 36,129 input rows at 2015-05-14.
- Training truth size has mean 31.62, median 14, and maximum 1,317. Validation truth size has mean 29.27, median 13, and maximum 1,085.
- The database has 5,960,558 destination ads and 98,250 users. Available history at the origins is approximately 1, 5, 9, 13, and 19 days, so history-depth drift is a primary coverage axis.
- Measured on temporally censored validation metadata, 18.4% of users have no prior visits; personal-repeat recall is capped near 0.208, while about 79% of truth ads are new to the user.

## Coverage axes

The evaluation varies by history depth, user activity depth, repeat versus new targets, observed versus stream-invisible ads, channel provenance, category/location agreement, and truth-list size. Internal reporting therefore slices candidate diagnostics by zero, shallow, medium, and deep visit history and reports each retrieval channel plus the incremental union.

## Critical discrepancy

The supplied solution assumes interaction-based causal neighbors are useful, but measured item co-visitation recall is only about 0.007 because exact-AdID interactions are extremely sparse. The channel remains implemented for coverage, while the ranker and RRF agreement prevent it from displacing stronger personal-history evidence.

The requested location `kapso_evaluation/eval_profile.md` is protected by the immutable-evaluator rule, so this profile is persisted at the repository root without modifying evaluation code.
