Critical path: 16 LightGBM document learners plus cached component predictions; target sustained rate at least 0.3 learners/minute.
Confirmation points: profile and debug gate by minute 35; two forward folds and selected design by minute 105; full predictions and evaluation by minute 210.
Freeze time: minute 210, retaining 15 minutes for contract checks, foreground scoring, and result capture.

# Plan

1. Characterize the scorer, split profile, document aggregation loss, temporal folds, and staleness coverage.
2. Build causal document matrices, label histories, full/fresh LightGBMs, forward selection, segmented blending, and gated prior forecasting.
3. Run the debug contract gate, inspect internal-fold diagnostics, then run and archive the full registered evaluation.

