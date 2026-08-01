Critical-path artifact + target rate: causal train/validation/test feature blocks, at least 20M feature-row updates/s and complete by T+45 minutes.
Planned confirmation points: metadata/profile now; full-path debug by T+35; model A checkpoint and validation predictions by T+135; contract check by T+195.
Freeze time: stop optional LightGBM work at T+175, reserve T+175–195 for model B/output and T+195–220 for full registered evaluation.

# Execution plan

1. Measure scorer behavior, split/cold/horizon strata, temporal ordering, safe table coverage, and feature-kernel throughput.
2. Build the hard-projected causal relational feature factory and cache content-addressed blocks.
3. Exercise every path in debug mode and validate both output arrays.
4. Run both training-only blackout simulations, fix rounds/blend/calibration, then fit A and B without official-validation selection.
5. Run the registered full evaluation in the foreground, capture the manifest, and report slice diagnostics.
