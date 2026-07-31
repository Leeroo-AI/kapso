TIME ALLOCATION: Critical-path artifact is the versioned causal event-state plus seed feature cache; measured 0.536M review rows/s with causal demand/diversity ranks and completed all initial seed features in 75 seconds, so the confirmation target is revised from throughput to exact count consistency.
TIME ALLOCATION: Confirm after event-state materialization, after the no-BG/NBD compound/reference forward folds, and after cutoff-specific recurrence features; bank predictions at each confirmation.
TIME ALLOCATION: Freeze model selection by minute 270, leaving at least 45 minutes for Model A/B prediction generation, validation, the registered full evaluation, and reserve.

# Plan

1. Profile immutable scoring, split geometry, labels, input activity drift, schema coverage, and cache throughput.
2. Materialize a versioned causal customer-event state and split-aligned RFM/P feature matrices without changing evaluator files.
3. Build cutoff-specific BG/NBD-compatible recurrence priors with a gated MAP fit and deterministic raw-transform fallback.
4. Build five purged forward-fold count, conditional-price, direct-L1, and hurdle reference models; select only from internal folds.
5. Fit Model A on train for validation and Model B on train plus validation for test, decode deterministic compound medians, and validate artifacts.
6. Run the registered full evaluator in the foreground and capture the complete manifest output.
