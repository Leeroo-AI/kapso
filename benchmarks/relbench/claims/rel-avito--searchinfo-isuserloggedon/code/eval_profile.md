# Evaluation profile

The immutable grader runs `main.py` in an isolated subprocess, requires aligned finite validation and test arrays, and computes all official validation metrics over all 695,590 validation rows. `fraction` does not subsample scoring rows. The score of record is validation ROC AUC, so within-segment and cross-segment ordering both matter; probability calibration affects the metric only through ordering. Validation predictions must come from a train-only chain, while test predictions may use a separate train-plus-validation refit.

## Input profile

| Frozen origin | Horizon rows | Covered | Cold | Device coverage within cold |
|---|---:|---:|---:|---:|
| 2015-04-30 | 177,592 | 79.17% | 20.83% | 93.97% |
| 2015-05-02 | 648,990 | 77.42% | 22.58% | 94.33% |
| 2015-05-08 validation | 695,590 | 76.31% | 23.69% | 94.67% |
| 2015-05-14 test | 592,133 | 63.15% | 36.85% | 96.58% |

The measured test cold fraction is materially above the solution's assumed deployment similarity to pseudo-origin coldness. The implementation therefore keeps inverse episode-user weighting, includes the May 8 origin in the test-chain router, and avoids using validation performance to weaken the cold specialist. Search query presence rises from 15.0% at the April 30 episode to 17.6% in test. Every horizon contains all 39 observed search categories, while distinct locations and users grow over time.

## Coverage axes

- Covered versus unseen or null user, plus history counts zero, one, two, and larger.
- Frozen-origin date and label staleness.
- User, IP, device, category, location, hour band, and query presence.
- Search-result composition and prior search, visit, and phone activity.
- Within-segment ranking and cross-segment score compatibility.

The critical path is causal all-table feature materialization. Projected DuckDB profile queries processed the task horizons at more than 1.6 million seed rows per second; the wider joins and causal event states are budgeted to finish by minute 55.
