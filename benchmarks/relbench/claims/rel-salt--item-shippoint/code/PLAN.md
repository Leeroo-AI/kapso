TIME ALLOCATION: Critical-path artifact: cutoff-safe relation-state/posterior cache, target build rate at least 250,000 historical rows/second.
TIME ALLOCATION: Confirm after deterministic baseline, debug end-to-end run, train-only forward fold, and prediction contract check.
TIME ALLOCATION: Freeze graph widening by elapsed hour 3 and reserve the final 30 minutes for full inference, evaluation, and artifact verification.

# Plan

1. Preserve row order and encode the permitted four-table topology with role-specific customer relations and filtered addresses.
2. Cache cutoff-safe deterministic posterior predictions before neural training.
3. Build train-only monthly snapshot states, train the relation-wise two-layer residual model, and select the posterior blend on forward folds.
4. Freeze Model A before validation and rebuild Model B with train plus validation history for test.
5. Exercise debug mode, run the immutable full evaluation, and record metrics and feature strata.
