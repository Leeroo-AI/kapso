Critical-path artifact + target rate: temporally censored all-table feature matrices and cohort priors for 50,029 labeled/query rows; target complete build in 50 minutes and at least two rolling-fold model fits per minute afterward.
Planned confirmation points: temporal/source audit by minute 15, contract-valid structured checkpoint by minute 100, rolling OOF and text/profile ablations by minute 185.
Freeze time: minute 205 for configuration, leaving 20 minutes for two-model refits, full registered evaluation, and artifact verification.

# Plan

1. Preserve row order and recompute all query-local and historical cohort features at each origin with the one-year label-availability lag.
2. Establish rolling 2016–2019 MAE baselines and validate the structured LightGBM member before adding result profiles and text SVD.
3. Freeze pseudo-counts, model members, and blend weights from train-only folds.
4. Refit Model A on train and Model B on train plus validation, then run the immutable evaluator.
