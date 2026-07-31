# Evaluation profile

The immutable registered evaluator runs `main.py` without arguments for full fidelity, requires finite float arrays aligned to the original 166,978 validation and 178,334 test rows, computes official RelBench metrics on the complete validation table, archives full runs, and uses validation MAE directly as the manifest score. The `fraction` and `seed` arguments are manifest metadata and do not subsample rows. Test labels are absent and test metrics are hidden.

## Input distribution

- Training has 2,707,679 rows, 404,525 products, and 31 quarterly origins from 2008-01-10 through 2015-07-02. Validation and test each have one origin, 2015-10-01 and 2016-01-01.
- Positive-price target support is exact: all 2,707,480 positive-price training rows have integral `ltv / price`, with inferred counts from 1 through 6,742. There are 199 zero-price rows, no missing prices, and no negative prices.
- Reviews have 12,644,508 rows, 416,125 products, and 1,584,084 customers. Mean review-text length is 572.6 characters, with median/p90/p99 238/1,506/4,204 and 941 empty or missing values. Mean summary length is 25.5, with median/p90/p99 19/51/94 and 1,432 empty or missing values.
- Prior-91-day counts are highly sparse. Validation has 62,531/166,978 cold products and count quartiles/p90/p99 of 0/1/3/10/64. Test has 73,139/178,334 cold products and 0/1/3/8/52.
- Coverage axes are forecast season, origin/regime, history count and recency, product/category/brand cohort, price, customer-audience behavior, static text, historical review text, and cold versus warm products.

## Reliability

- Origin-label mean variance divided by mean expected sampling variance `s^2/n` is 221.883. The largest level change is +14.606 from 2012-10-04 to 2013-01-03.
- Review volume is 954,870 in the 91 days feeding validation and 806,263 in the 91 days feeding test. The -148,607 boundary change is 16.058 robust-MAD units from historical adjacent-origin changes.
- Both diagnostics are extreme. Raw global-volume features are excluded; model selection uses expanding training-origin folds and favors causal per-product histories plus ranks, shares, ratios, and shrunk cohort priors.

The required profile is kept outside `kapso_evaluation/` because the supplied evaluation directory is explicitly immutable and editing any file beneath it invalidates the run.

## Run 0004 reporting-only slices

- Headline validation: MAE 35.875204, R2 0.654872, RMSE 265.548561.
- Prior-91-day history slices: cold count 62,262 / MAE 29.006; count 1: 32,532 / 13.243; count 2-3: 30,307 / 19.481; count 4-10: 26,539 / 37.706; count 11-50: 13,015 / 92.746; count 51+: 2,323 / 411.264.
- Price slices: zero 6 / MAE 0; under 9: 42,558 / 16.819; 9-13: 42,870 / 37.075; 13-18: 44,789 / 37.738; 18-35: 29,486 / 42.314; 35+: 7,269 / 102.809.
- The slice audit was reporting-only and did not drive design selection. It exposed a semantic boundary issue: day-truncated windows included a small number of reviews just outside exact horizons. The next frozen implementation corrects all windows to timestamp inequalities and selects the specified training-weight decay candidates on internal folds.

## Run 0009 internal and official report

- Exact-window folds selected training decay 0.98 over 1.0 by median rounded dollar MAE 41.570599 versus 41.623271. Fold dispersion was 1.738416, rounding was selected, and the median best iteration was 612.
- Official validation was MAE 36.013936, R2 0.661779, RMSE 262.878138. Relative to run_0004, MAE changed by +0.138731 while RMSE improved by 2.670422; this small disagreement is treated as single-validation noise and does not override the internal referee.
