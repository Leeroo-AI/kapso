TIME ALLOCATION: Critical path is the all-table causal feature matrix; measured static spine throughput is 4.32M rows in 7.4 seconds, with a target of at least one history grain per 3 minutes.
TIME ALLOCATION: Confirm after static/debug preflight, after the first complete history matrix, and after three train-only forward folds.
TIME ALLOCATION: Freeze model/features by 190 elapsed minutes, reserving the final 30 minutes for the registered full evaluation and artifact verification.

# Plan

1. Profile immutable scoring mechanics and unlabeled split coverage.
2. Implement exact-key static joins and all-table causal histories.
3. Build causal target encodings and train-only forward-fold selection.
4. Fit separate validation and test chains, validate predictions, and run the registered evaluator.
