TIME ALLOCATION: Critical path is the temporally censored all-table feature store; measured core aggregation throughput is 268,587 seed rows/second, target full 453k-row store in under 20 minutes.
CONFIRMATION POINTS: exact-label assertions first; bank core LightGBM after Stage 1; run purged-fold metrics after Stage 2; freeze final predictions by 3h20m.
FREEZE TIME: 2026-08-09 19:22 UTC, preserving 40 minutes for final fitting, contract checks, and foreground evaluation.

1. [completed] Profile inputs and verify exact rolling labels at official origins.
2. [completed] Build and cache staged temporally censored features across every table.
3. [completed] Run purged forward validation, bootstrap slice diagnostics, and choose the predeclared model alternative.
4. [completed] Fit independent Chain A and Chain B models, restore row order, and validate artifacts.
5. [completed] Run the immutable full-fidelity evaluator and capture its manifest.

The feature store completed in 4.7 minutes rather than the planned 20-minute ceiling. The recovered time was deliberately reassigned to causal-prior repair, XGBoost pair-generation diagnosis, three additional forward-gated feature blocks, and a within-origin normalization ablation; the prediction freeze remained well ahead of 19:22 UTC.
