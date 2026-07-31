Critical path: causal all-table snapshot matrix and 64-dimensional sketches; target build rate at least 8,000 seed rows/minute.
Confirmation points: exact-label audit, debug end-to-end gate, core-feature completion, OOF fold metrics, frozen Model-A predictions, registered full evaluation manifest.
Freeze time: 2026-07-31 04:05 UTC, preserving 15 minutes for contract checks and final reporting.

# Plan

1. Profile the immutable grader, official task rows, daily episode distribution, temporal coverage, and coldness.
2. Build exact daily targets and causal features from all eight tables, including fixed decayed CountSketch channels.
3. Generate purged forward OOF predictions for direct, hurdle, count, and mechanistic heads.
4. Select count cap, head set, dispersion handling, and meta regularization only from forward training folds.
5. Freeze Model-A validation predictions, rebuild Model B with validation labels once, and write aligned artifacts.
6. Run the debug gate, prediction checks, and registered full evaluation.
