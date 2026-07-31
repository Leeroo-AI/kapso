# Evaluation profile

## Evaluation mechanics

- The immutable full evaluator launches `python main.py`, enforces the configured 14,400-second candidate timeout, loads the two NumPy artifacts, checks validation shape and integer dtype, and calls the official RelBench evaluator.
- The score of record is official validation `link_prediction_map`; precision and recall are also reported. Test labels are absent from the sanitized cache and test metrics are hidden.
- Validation has one 2020-01-01 origin and 2,081 rows. Test has one 2021-01-01 origin and 2,057 rows. Predictions must have ten distinct sponsor indices per row in original task-table order.
- Model selection and early stopping must use forward holdouts drawn only from training. Model A produces validation predictions without validation-label fitting; Model B may add validation supervision only for test predictions.

## Input distribution

- Training contains 36,934 groups at 20 annual origins from 2000-01-06 through 2019-01-01. The final six origins contain 2,302, 2,330, 2,357, 2,369, 2,349, and 2,234 rows.
- There are 3,973 conditions and 53,241 sponsor candidates. Validation truth lists have mean 14.63, median 4, and maximum 2,550.
- Cutoff-causal historical events are formed by joining `conditions_studies` and `sponsors_studies` on `nct_id` and assigning visibility `max(condition_date, sponsor_date)`. The database contains 660,567 resulting relation events.
- Validation positive repeat mass is 34.5%; 48 of 2,081 validation condition rows have no prior history. The structural pool measured in campaign memory reaches 65.1% truth recall before popularity and 66.8% after popularity.

## Required forward-origin profile

Candidate ranking for this measurement combines 365-day decayed own-pair history, the sponsors of the 40 most cosine-similar cutoff-visible conditions, and cutoff-visible sponsor popularity. No holdout positives are inserted.

| Holdout origin | Groups | Positives | Repeat-positive rate | Cold-condition rate | Recall@100 | Recall@500 | Recall@2000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017-01-01 | 2,369 | 40,587 | 0.3903 | 0.0232 | 0.2385 | 0.4698 | 0.6393 |
| 2018-01-01 | 2,349 | 37,764 | 0.3959 | 0.0281 | 0.2364 | 0.4659 | 0.6392 |
| 2019-01-01 | 2,234 | 34,203 | 0.3993 | 0.0161 | 0.2374 | 0.4639 | 0.6435 |

The assumed one-third repeat contribution is slightly conservative on recent internal origins, where it is about 40%, but is close to the 34.5% validation measurement. The assumed roughly two-thirds structural-pool recall is supported only near a 2,000-candidate cap; candidate ranking is the primary loss at 100 and 500.

## Coverage axes

- Origin year and temporal regime, including the 2020 clinical-trial burst.
- Repeated versus unseen condition-sponsor pairs.
- Warm versus cold conditions.
- Truth-list cardinality: one, two to four, five to ten, eleven to fifty, and over fifty.
- Sponsor agency class and lead-versus-collaborator role.
- Condition and sponsor activity/recency/burst strata.
- Structural-retrieval source: recurrence, similar condition, co-sponsor, intervention, ALS, semantic, agency-class popularity, and global fallback.
- Availability and missingness of study, design, eligibility, facility, outcomes, analysis, adverse-event, and withdrawal evidence.

## Critical path

The score is bounded by the natural candidate/feature matrix because missing positives cannot be recovered by LambdaMART. The three-origin recon processed 6,952 seed groups in 16.7 seconds, or 416 groups/s, for structural retrieval and recall accounting. The implementation target is at least 300 groups/s before text widening; confirmation points and freeze time are recorded in `PLAN.md`.
