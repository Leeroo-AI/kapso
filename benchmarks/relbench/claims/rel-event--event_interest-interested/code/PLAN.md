TIME ALLOCATION: Critical path is the temporally censored all-table row feature matrix; measured snapshot loading is 2.8-3.1 seconds and the target is a complete 15,398-row matrix in under 4 minutes.
CONFIRMATION POINTS: exact two-snapshot identity assertions, cached static-matrix completion, debug contract/stratum metrics, five-fold mean and worst-fold selection, full manifest.
FREEZE TIME: freeze feature/model design by minute 165, reserve 45 minutes for full foreground evaluation and 15 minutes for audit/finalization.

# Plan

1. Characterize immutable scoring, row identities, burst structure, split shift, and table coverage.
2. Build temporally safe all-table static and prefix-label history features with a shared-cache artifact.
3. Implement burst-any LightGBM, LambdaRank plus conditional MLP, softmax composition, and fold-gated row guard.
4. Run one-fold debug validation, prediction checks, then five-fold full evaluation.
5. Record stratum results, campaign memories, provenance, and the official manifest.
