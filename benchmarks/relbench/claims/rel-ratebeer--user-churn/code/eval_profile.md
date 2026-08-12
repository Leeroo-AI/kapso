# Evaluation profile

## Mechanics

- The immutable grader invokes `main.py` once, checks exact float vectors of 19,908 validation rows and 9,392 test rows, and scores every validation row with RelBench's official metrics.
- The optimization score is validation ROC AUC. Accuracy, average precision, and F1 are reported but do not select the run. The `fraction` and `seed` arguments do not subsample items.
- Validation predictions must come from a chain fit without validation labels. A separate test chain may use validation labels and locally generated outcomes ending no later than the test cutoff.

## Input distribution

- Training has 373,709 rows, 154,071 users, and 74 quarterly 90-day anchors from 2000-06-07 through 2018-06-03. Anchor sizes range from 33 to 16,763; their positive rates range from 0.2222 to 0.6809.
- Validation is one 2018-09-01 anchor with 19,908 distinct users and positive rate 0.6270. Test is one 2020-01-01 anchor with 9,392 distinct users.
- Only 9,359 validation users and 3,307 test users occur in official training; 3,055 test users occur in validation. Identity-only supervision therefore has poor coverage.
- Validation recent-rating counts have quartiles 1, 2, and 6 over 90 days; test quartiles are 1, 2, and 11. Recent-history volume is materially heavier in test. Median recency is 27.50 days in both, while the upper quartile shifts from 53.08 to 57.77 days.
- At validation cutoff the legal graph contains 10,620,188 beer ratings, 313,026 place ratings, and 22,328 favorites. At test cutoff it contains 11,847,969, 343,179, and 121,769 respectively.
- User beer-event degree at validation has quartiles 1, 2, and 7, with 95th percentile 117 and maximum 55,206. Beer distinct-user degree has quartiles 1, 3, and 10, with 95th percentile 66 and maximum 7,200.
- All 117,424 availability timestamps lie in 2024-01-01 through 2025-02-03, after the test cutoff. No availability event edge is temporally legal despite the solution's assumption that this relation contributes before the cutoffs.

## Coverage axes

- Seed anchor and temporal regime.
- Seen versus supervision-unseen user.
- Recency bands and 90-day rating-count bands.
- Lifetime tenure, rating gaps, repeat behavior, and beer/style/brewer diversity.
- Presence and recency of place-rating and favorite channels.
- Collaborative route availability and shared-entity degree.
- Validation-versus-test history-volume shift.

## Runtime and critical path

- Projected Parquet scans measured 0.55 seconds for 11.85 million beer-rating rows on the assigned hardware; cutoff degree aggregates measured below 0.3 seconds after cache warm-up.
- The score-bounding artifact is the cutoff-safe 96-dimensional route embedding matrix. The precommitted production target and checkpoints are recorded at the top of `PLAN.md`.
- Full extraction measured 52,208 to 62,297 seed rows/second across Chain A and B matrices. Encoder A completed in 79.5 seconds of optimization and Encoder B in 89.0 seconds, both far inside their 60-minute gates.
- Candidate-side reporting uses rating-count strata 1, 2-5, and 6+, plus supervision-seen versus unseen users. The official grader exposes only aggregate validation metrics, so these slice counts and AUCs are saved in candidate `metrics.json` rather than inferred from the grader.
- The 300-tree compact model was selected after two independent-seed forward comparisons against 600 trees; both comparisons favored the smaller model beyond two estimated standard errors. No official validation result was used for that decision.
- Low-cardinality preference composition is covered by the top 20 validation-cutoff beer styles and six immutable brewer continents, summarized point-in-time over 90 and 365 days. This group improved 4/5 internal anchors beyond two estimated standard errors.
