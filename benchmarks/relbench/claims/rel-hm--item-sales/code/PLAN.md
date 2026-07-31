TIME ALLOCATION — Critical path: causal 26-origin feature matrix plus LightGBM fits; weekly aggregation measured at 256,759 sparse article-weeks/s and the fit target is at least 30 rounds/minute.
TIME ALLOCATION — Confirm at the banked fallback, first full-feature forward fold by 05:30 UTC, frozen internal design by 08:30 UTC, and contract-valid final artifacts by 09:30 UTC.
TIME ALLOCATION — Freeze feature and hyperparameter selection at 08:30 UTC on 2026-07-31, preserving at least 75 minutes for final fits, the foreground registered evaluation, and reporting.

# Execution plan

1. Bank and profile deterministic forecasts on four training-only rolling origins.
2. Build a cached weekly transaction and buyer panel, then causal origin features from it.
3. Evaluate nested feature groups and the predeclared LightGBM grid only on forward folds.
4. Freeze the design, fit Model A on training labels and Model B on training plus validation labels.
5. validate prediction artifacts and run the immutable full-fidelity evaluator.
