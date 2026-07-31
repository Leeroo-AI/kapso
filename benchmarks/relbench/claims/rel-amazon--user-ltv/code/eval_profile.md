# Evaluation profile

## Mechanics

- Registered command: `python kapso_evaluation/kapso_eval.py --fidelity full --fraction 1.0 --seed 1337`.
- The immutable grader launches `python main.py`, checks both arrays, evaluates all 409,792 validation rows with RelBench, and archives full-fidelity predictions. `fraction` and `seed` are manifest metadata only.
- Score of record is validation MAE on the original nonnegative LTV scale. R2 and RMSE are also reported. Lower MAE is better.
- Validation predictions must come from a train-only chain. The test chain may refit on train plus validation. Test labels are physically absent.

## Input distribution

- Train has 4,708,383 seed rows, 1,486,748 unique customers, and 31 quarterly cutoffs from 2008-01-10 through 2015-07-02. Per-cutoff row counts range from 5,372 to 416,981.
- Validation has 409,792 unique customers at 2015-10-01. Test has 351,885 unique customers at 2016-01-01.
- Train LTV is 62.40% zero with mean 16.966, median 0, q75 14.99, q90 43.93, q99 212.03, and maximum 9,511.46. Validation is 64.20% zero with mean 14.141, median 0, q75 13.2625, q90 36.90, q99 180.13, and maximum 7,259.91.
- Reviews cover 2008-01-01 through 2016-01-01: 12,644,508 rows, 1,584,084 customers, and 416,125 products. Mean rating is 4.344 and verified share is 70.43%. Text and summary null rates are 0.0072% and 0.0112%; review-text length q10/q50/q90/q99 is 49/239/1509/4224.
- Products number 506,012. Prices have no nulls, median 12.99, q90 25.56, q99 60.70, q99.9 304.42, and maximum 5,204. Category and description null rates are 3.73% and 7.19%; brand and title are complete.
- Customers number 1,850,193. Names are 0.0036% null with length q10/q50/q90/q99 of 5/11/16/23.

## Reliability diagnostics

- Label stability: variance of the 31 quarterly target means is 16.0734 versus mean within-quarter sampling variance 0.14074, a ratio of 114.2. Excluding the startup cutoff leaves a ratio of 25.6. The series moves from 14.38 to 18.95 after startup, with visible activity/level changes around 2012Q4-2013Q1 and 2014Q1.
- Boundary regime gap: preceding-91-day review volume falls from 954,870 at validation to 806,263 at test. Its absolute log change is 3.32 times the all-history median and 8.90 times the recent median. Preceding-30-day volume falls from 293,942 to 199,963, 4.64 times the all-history median and 7.03 times the recent median. Active users fall from 409,792 to 351,885, 2.76 times the all-history median and 16.18 times the recent median.
- Both diagnostics are extreme. Model admission therefore uses median forward-fold MAE including 2013/2014 changes, favors causal per-entity history and relative fast/slow encodings, and excludes raw global activity levels.

## Coverage axes and strata

- Time/regime: pre-2013, 2013 transition, 2014 transition, and 2015 recent cutoffs.
- Outcome mass: zero versus positive LTV; future review-count classes 0, 1, 2, 3, 4, 5, 6-9, and 10+.
- Customer history: short versus long tenure; low versus high lifetime frequency; recent versus stale last event; stable versus variable inter-event gaps.
- Monetary behavior: low, middle, high, and extreme historical price affinity; sparse-history shrinkage to population priors.
- Product/context: category/brand diversity and metadata completeness; product-demand rank; text-length and verification behavior.
- Required evaluation report: headline metrics plus internal forward-fold MAE by time regime, count class, and history-frequency stratum where labels are available.

## Coverage discrepancy

- The solution's drift claim is confirmed and stronger near the test boundary than an ordinary quarterly change. Cutoff-specific recurrence summaries remain appropriate, but supervised inputs must use per-customer relative activity rather than population volume.
- PyMC-Marketing and lifetimes are not installed at measurement time. BG/NBD MAP is gated as specified; raw frequency/recency/age recurrence transforms remain the required fallback if installation or MAP stability fails.

## Frozen run result

- `run_0002`: validation MAE 11.6821632059, RMSE 40.1387681523, and R2 0.3268818288 on all 409,792 rows.
- The largest loss stratum is 10+ future reviews (6,457 rows, MAE 132.3618). By history, customers with 11+ observed reviews have MAE 22.5501; frequency 3-5 is strongest at MAE 6.6835.
- Zero outcomes have MAE 1.7923 across 263,098 rows; positive outcomes have MAE 29.4197 across 146,694 rows. Full count/frequency/tenure slices are persisted in `evaluation_results.json`.
