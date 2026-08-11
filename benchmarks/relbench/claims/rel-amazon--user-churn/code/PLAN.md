CRITICAL PATH: cutoff-safe renewal/graph feature matrix; target at least 200,000 seed rows/minute after a measured 7.4M review-row/s scan.
CONFIRMATION POINTS: graph API smoke batch, first full origin partition, expanding-fold median/worst AUC, then prediction contract check.
FREEZE TIME: minute 330 of 360; stop neural work by minute 300 and reserve the final 30 minutes for full scoring and handoff.

# Plan

1. Persist the metric/input profile and reliability diagnostics without modifying the protected evaluator.
2. Materialize deterministic cutoff-safe review, product, customer, and causal label-history features with incremental shared caching.
3. Bank expanding-origin renewal OOF predictions, smoke-test temporal PyG, and run the heterogeneous graph referee.
4. Select the fixed OOF rank blend under the +0.002 admission gate, fit Model A and Model B, and preserve Model A validation predictions.
5. Run debug checks, the registered full evaluation, and record all strata and artifacts.
