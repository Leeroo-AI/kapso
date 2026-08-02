# Evaluation profile

## Mechanics

- The immutable grader runs `main.py` in an isolated subprocess, requires finite float arrays aligned to all 235,662 validation and 266,364 test rows, and computes official RelBench `r2`, `mae`, and `rmse` on the complete validation split.
- Full fidelity changes candidate build cost only. `fraction` and `seed` are manifest metadata; the grader does not subsample scoring rows or control model randomness.
- The score of record is validation R². Test labels are absent and test metrics remain private. Validation predictions must be frozen from Model A before Model B reads validation labels.

## Input distribution

- Train labels: 14,844,291 rows from 2019-09-08 through 2020-09-06. Validation: 235,662 rows from 2020-09-08 through 2020-09-13. Test: 266,364 rows from 2020-09-15 through 2020-09-22.
- Validation has 17,862 articles, 67,448 customers, and 22,072 article×channel panels. Test has 18,684 articles, 75,481 customers, and 23,310 panels.
- Channel counts in validation are 75,140 for channel 1 and 160,522 for channel 2. Test counts are 81,380 and 184,984.
- Validation index-group counts are Ladieswear 147,770, Divided 54,894, Menswear 14,472, Sport 13,658, and Baby/Children 4,868.
- Against task train labels, 20,905 validation rows (8.87%) and 46,811 test rows (17.57%) have no labeled article history. The supplied 4.3% validation cold claim counts articles observed in the unlabeled 2020-09-07 transaction metadata; it is not the legal price-history cold rate. The implementation treats such articles as launch-price cases and never reads the retained database price field.
- Warm validation recency strata are 209,114 rows at 0–7 days, 3,644 at 8–28 days, and 1,999 at 29+ days. Static customer metadata is nearly complete on validation: age is missing for 865 rows, member status for 327, and news frequency for 473.
- The target horizon varies by day and channel. Validation covers horizons 1–6 after the Monday snapshot, while test covers horizons 1–8; daily total volume ranges from 36,415 to 41,626 in validation and 26,053 to 40,718 in test.

## Coverage axes

- Warm versus no labeled article price history, and weeks since last observation.
- Sales channel and cross-channel support.
- Horizon, weekday, week-of-year, and target-block composition.
- Article/product/category hierarchy support and static article attributes/text.
- Customer activity, discount affinity, metadata missingness, and article-day dispersion.
- Stable versus materially changed price state, regime duration, lag velocity, and panel support.

## Critical path measurement

- A legal task-label-to-transaction join and recent weekly article×channel aggregation processed 4,581,590 source rows into 376,151 panels in 0.323 seconds on 11 assigned threads, or 14.2M rows/s.
- Model training and forward-OOF prediction, not relational aggregation, therefore bounds score and runtime. The plan allocates the first 45 minutes to obtaining genuine OOF panel forecasts before building residual consumers.

## Slice reporting

- The grader exposes only headline metrics. The candidate prints forward-fold weighted panel baselines and decoded-model scores. After the one official run, archived validation predictions will be summarized by channel, warm/cold history, recency, horizon, and index group without feeding those results back into design selection.
