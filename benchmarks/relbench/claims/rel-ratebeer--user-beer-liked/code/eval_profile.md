# Evaluation profile

## Mechanics

The immutable grader launches `main.py` in an isolated child, scores the complete 5,681-row validation split with the official RelBench recommendation metrics, and uses `link_prediction_map` as the primary score. Both full-shape integer arrays are mandatory. Validation prediction rows must come from Model A, whose complete fit chain excludes validation labels; only Model B may use validation and later legal supervision for test prediction.

## Input distribution

- Train contains 150,322 labeled user-origin groups at 74 origins spaced 90 days apart from 2000-06-07 through 2018-06-03. Origin volume grows from 18 to 5,467 groups.
- Validation is one origin at 2018-09-01 with 5,681 distinct users. Test is one origin at 2020-01-01 with 2,783 distinct users, a 16-month temporal shift.
- Train label-list size is mean 9.87, median 4, maximum 2,031. Validation is mean 7.08, median 2, maximum 339; 51.0% of validation groups have at most two labels.
- Validation users have extremely skewed prior rating counts: minimum 1, p10 2, p25 5, median 26, p75 372, p90 2,282, p99 13,421, maximum 55,206. Prior liked-rating counts have median 10 and p90 342. Last-activity recency ranges from 1 to 90 days with median 9.
- Across 40,233 validation targets, measured repeat-target rate is zero. A previously liked brewer covers 49.1%, a previously liked style covers 73.6%, and their union covers 78.7% at the label level.
- Only 1,364 validation users have historical place ratings; their place-rating count is median 20 and p90 about 254. Validation users contributed 7,825 temporally legal favorites, while no user-authored availability rows were found for these seed users.
- Rating activity shifts down in 2019: 853,546 all ratings and 165,190 likes, versus 1,132,728 and 264,863 in 2018. This reinforces forward temporal selection and a separately refit Model B.

## Coverage axes and reporting

Diagnostics stratify by origin, prior rating-count bucket (1-5, 6-25, 26-199, 200+), and channel. Candidate diagnostics report recall@50/100/200/800, oracle MAP@10, total recall, and marginal label recall for global, brewer, style/parent-style, co-visitation, collaborative, new-release, favorites/availability, and geographic channels. Ranker selection uses only forward pre-validation origins.

## Solution assumptions checked

The 800-candidate assumption is retained but gated by internal recall. Co-visitation and collaborative marginal recall remain assumptions until internal measurements. Stable beer/brewer metadata is restricted to identity-like columns and creation-time-safe flags; snapshot popularity, views, scores, lifetime counters, and time-ambiguous UPC data are excluded.

Recon found `beers.is_retired` equals true for all 751,524 rows in the sanitized snapshot. It is retained only as an explicitly requested stable-content feature but cannot gate item eligibility; treating it as an exclusion flag would empty every popularity channel.
