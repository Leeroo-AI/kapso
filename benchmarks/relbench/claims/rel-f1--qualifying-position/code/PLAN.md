TIME ALLOCATION: Critical path is forward-fold CatBoost residual training; target at least 100 trees/second on 2,228 rows so the prescribed 18 full fits remain below 10 minutes.
TIME ALLOCATION: Confirm after feature extraction, a 100-tree benchmark, fast evaluation, and full evaluation manifest.
TIME ALLOCATION: Freeze implementation by minute 190 and reserve the final 35 minutes for full evaluation, retries, artifact checks, and reporting.

# Plan

1. Profile task row/race/time/entity distributions and evaluator behavior without using validation labels for design selection.
2. Build timestamp-safe online ratings and all-table as-of relational features.
3. Select rating temperature, residual feature groups, tree count, and blend only with four expanding train folds.
4. Freeze Chain A validation predictions, refit Chain B on train plus validation, project race predictions, and verify contracts.
5. Run fast and full registered evaluations and capture the manifest.

## Confirmation record

- Feature pass: 9,815 rows by 470 columns, 76.16 seconds on a cold cache and 0.04 seconds cached.
- CatBoost critical-path rate: 91.3 trees/second over four groups, four expanding folds, and three seeds; the planned 100 trees/second was nearly met and total full CV was 210.23 seconds.
- Fast registered gate: completed in 36.37 seconds with valid full-shape outputs; implementation frozen before the full registered run.
