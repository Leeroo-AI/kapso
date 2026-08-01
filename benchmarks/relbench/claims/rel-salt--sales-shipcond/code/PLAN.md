TIME ALLOCATION: Critical path is the causal history feature matrix plus three GPU XGBoost fits; measured GPU viability is confirmed and the target feature-build rate is at least 2,000 document rows/second.
CONFIRMATION POINTS: static joins reached 3,788 rows/second in debug, the first full train-only gate rejected rollforward, and targeted drift priors passed a second five-month confirmation at +0.15 accuracy points.
FREEZE TIME: minute 210; the original minute-100 confirmation was deliberately extended for feature widening because 110+ minutes remained, while the final 30-minute reserve remains protected.

# Plan

1. Profile scorer mechanics, split windows, joins, coverage, and history reuse.
2. Build allowed static relational summaries and forward-only true-history features.
3. Fit the pre-September simulation model and gate the fixed rollforward grid by monthly accuracy.
4. Fit Model A on train and Model B on train plus validation, then generate one-pass causal predictions.
5. Run debug checks, the immutable registered full evaluation, and record the outcome.
