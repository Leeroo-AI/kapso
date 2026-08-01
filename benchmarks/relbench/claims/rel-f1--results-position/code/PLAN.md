TIME ALLOCATION: Critical-path artifact is the three-fold forward-OOF distribution matrix and its final two-chain ensemble; measured LightGBM throughput is 790 100-tree fits/minute, target at least 55 full-size 700-tree fits/minute.
CONFIRMATION POINTS: feature/join audit and OOF fold slices by minute 35; debug contract run by minute 80; full registered evaluation started by minute 165.
FREEZE TIME: minute 210, preserving 15 minutes for foreground evaluation completion, artifact checks, documentation, and commit.

# Plan

1. Build temporally censored features from all nine database tables and strict-prior permitted target histories.
2. Generate three forward-fold OOF heads, select transport parameters exclusively on those folds, and require the specified improvement gate.
3. Fit the train-only validation chain and the train-plus-validation test chain, preserving original row order.
4. Run debug and full official evaluations, record slice diagnostics and the manifest, and finalize the campaign memory.

## Adherence

- The critical train-only full OOF artifact froze in 6-8 minutes per design version, about 18-24 mixed 700/800/1500-tree fits per minute rather than the optimistic 55-fit target measured from the initial 100-tree probe. The consumers were built only after each OOF artifact froze.
- Confirmation points were completed in order: join/feature audit, strict-horizon correction, debug contract gate, per-fold OOF diagnostics, then full registered runs.
- The final design freeze selected v5 by full internal OOF stability. v6 and v7 were explicitly rejected despite slightly higher single-validation results, preserving the no-validation-tuning rule.
