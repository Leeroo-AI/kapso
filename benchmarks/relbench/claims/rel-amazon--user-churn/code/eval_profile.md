# Evaluation profile

## Mechanics

The immutable registered evaluator runs `main.py` in an isolated child, requires full-order finite NumPy vectors of 409,792 validation rows and 351,885 test rows, and computes all official RelBench metrics on the complete validation table. Full fidelity changes candidate build mode only; `fraction` does not subsample scoring rows. The score of record is validation ROC-AUC. Model A validation predictions must exclude validation labels from every fitted component; only Model B for test may use validation labels.

## Input distribution

- Train has 4,708,383 rows, 1,486,748 distinct customers, 31 quarterly origins from 2008-01-10 through 2015-07-02, and churn prevalence 0.623957. Origin sizes range from 5,372 to 416,981 and origin churn rates range from 0.571010 to 0.743591.
- Validation is one origin at 2015-10-01 with 409,792 distinct customers and prevalence 0.642028. Test is one origin at 2016-01-01 with 351,885 distinct customers.
- Customer train-origin repetition is strongly long-tailed: 372,374 customers occur once, 365,867 twice, 285,006 three times, and only 198 occur in all 31 origins.
- Reviews contain 12,644,508 events from 1,584,084 customers and 416,125 products. Rating mean is 4.3440, verified share is 0.7043, mean review-text length is 572.57, and mean summary length is 25.46. Review text and summary null rates are 0.0072% and 0.0112%.
- Product price has median 12.99 and 1/10/25/50/75/90/99 percentiles 1.99/5.99/8.99/12.99/17.59/25.50/59.99. Category and description null rates are 3.73% and 7.19%; brand and title have no nulls.

## Reliability

Across the 31 training origins, the unweighted variance of origin churn means is 0.00313735 versus mean binomial sampling variance 0.000004324, a ratio of 725.6. The largest adjacent level break is 0.12930 between the first two origins; later persistent breaks remain visible around 2012-2014. Label drift is therefore extreme.

The causal 91-day review volume is 954,870 before validation and 806,263 before test, a drop of 148,607 or 15.56%. This is 13.54 times the median absolute training-origin change, although 2.06 ordinary standard deviations because early growth breaks inflate the non-robust scale. The boundary regime gap is extreme on the robust normal-change scale. Both diagnostics support per-entity, level-invariant features and median expanding-origin selection.

## Coverage axes

Coverage must span origin/regime, customer-history depth, recency, fast-to-slow activity, review behavior and text-length proxies, last-product and product-popularity context, customer repetition, sparse/missing product metadata, seasonal January phases, the 2012-2013 anomaly, and the latest origin. Scores are reported by expanding origin, customer-history depth, and activity strata when local OOF output permits.

## Critical path

The score-bounding artifact is the cutoff-specific causal feature matrix used by both the renewal model and the graph head. A metadata scan measured about 7.4 million review rows per second; the build target is at least 200,000 seed rows per minute including grouping and persistence. Graph training is admitted only after the renewal predictions are banked and only if its median expanding-origin AUC improves by at least 0.002.
