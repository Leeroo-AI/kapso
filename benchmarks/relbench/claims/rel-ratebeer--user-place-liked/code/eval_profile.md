# Evaluation profile

The immutable grader invokes `main.py` in an isolated subprocess, requires integer arrays of shapes `(547, 10)` and `(351, 10)`, and computes all official recommendation metrics on all 547 validation rows. `--fraction` does not subsample. The score of record is validation `link_prediction_map`; test labels are absent. MAP therefore rewards ranking true destinations early in ten distinct predictions while dividing by `min(number of true destinations, 10)`.

The train table contains 38,444 rows at 60 quarterly origins from 2003-11-19 through 2018-06-03. Validation contains 547 unique users at one origin, 2018-09-01; test contains 351 rows at 2020-01-01. Ground-truth list sizes are train mean 5.77, median 3, maximum 383, and validation mean 7.50, median 3, maximum 104.

Validation user-place history strata at the origin are 37 place-cold users, 32 users with 1-3 reviews, 33 with 4-10, and 445 with 11 or more. The 4,104 validation destination events split by prior destination popularity into 936 with zero prior ratings, 1,264 with 1-4, 1,050 with 5-19, and 854 with at least 20. Prior user geography covers 2,978 events by state and 1,743 by normalized city. Beer activity is substantial even for place-cold users, whose mean prior beer-rating count is 263.9.

Coverage axes are query origin/regime, user place-history warmth, destination warmth, label-list size, prior-state/city membership, beer-side history, place creation eligibility, and candidate source. Forward train folds must report candidate recall and MAP separately for place-warm/place-cold users and warm/cold destinations.

The solution's collaborative-complementarity and PPR-throughput assumptions remain gates rather than facts. Annual training snapshots are potentially stale by up to one year, while exact 2018-09-01 and 2020-01-01 inference snapshots remove inference staleness. Candidate recall@600 is the score-bounding artifact; the ranker cannot recover positives absent from its union.

The annual-staleness assumption was rejected after forward measurement. Event recall@600 in 2016 was 30.9% at the March origin and 14.9% at the December origin; in 2017 it was 28.1% and 15.2%, respectively. The implementation therefore retains the required annual checkpoints and adds censored July checkpoints through 2017 plus legal 2019 model-B checkpoints, bounding training staleness near six months.

Semiannual snapshots increased the three-fold mean natural-pool MAP from 0.0259 to 0.0377. The same causal correction is extended to quarterly checkpoints, while validation folds are aggregated by calendar year to preserve 240-480 queries per forward fold.

Quarterly snapshots further increased the three-year forward mean to 0.0395. Exact legal training origins are therefore added as the final staleness treatment; all specified annual checkpoints are retained and all fold grouping remains strictly forward by year.

Exact origins increased the three-year forward mean to 0.0598, with the latest 2018 fold lower than 2016-2017. The fixed 2,640-query budget is reallocated toward 2017-2018 to reflect the measured declining-activity regime without selecting against the official validation sample.
