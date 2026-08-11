# Evaluation profile

## Mechanics

- The immutable grader invokes `main.py` once for each rolling snapshot and assembles 26 validation origins and 29 hidden-test origins in original row order.
- Each tick exposes a date-truncated database, all label windows closed by that date in `train.parquet`, an empty `val.parquet`, and only current-origin inputs in `test.parquet`.
- The official selection score is validation ROC AUC over all 566 rows. Average precision, accuracy, and F1 are also reported. The fidelity fraction does not subsample scoring rows.
- The candidate must use direct snapshot data or `upto_test_timestamp=False`; static-freeze clamping would silently discard post-2009 rolling history.

## Input profile

| split | rows | origins | rows/origin | date range | positive rate |
|---|---:|---:|---:|---|---:|
| train | 11,411 | 420 | mean 27.17, range 10–65 | 1950-05-20 to 2004-10-03 | 0.8804 |
| validation | 566 | 26 | mean 21.77, range 20–24 | 2005-03-02 to 2008-03-16 | 0.7792 |
| test inputs | 702 | 29 | mean 24.21, range 22–26 | 2010-03-02 to 2013-03-16 | hidden |

The train positive rate by decade is 0.861, 0.876, 0.876, 0.920, 0.884, and 0.818 for the 1950s through 2000s. Validation rates are 0.788 in 2005, 0.809 in 2006, 0.736 in 2007, and 0.864 for its single 2008 origin. Origin-level rates vary sharply, from 0.45 to 0.96 in validation.

The database has 20,323 results across 820 races. Status 1 accounts for 4,369 results; its share rises from roughly 0.17–0.21 in the 1950s–1970s to 0.382 in the 2000s. Recent seasons contain 16–19 races, and race intervals have a median of 14 days.

## Coverage axes

- Era and reliability regime, especially the 2000s shift toward more finishers.
- Driver, constructor, and driver–constructor history density and recency.
- Failure type: status 11–19, status 3/4/20, and other non-finish statuses.
- Current form, standings level/trend, constructor momentum, and team switches.
- Calendar phase, recent circuit transitions, and expected races in the next 30 days.
- Within-origin competition among roughly 20–26 active drivers.
- Sparse early-era records versus rich modern records and missing pre-1994 qualifying history.

The solution assumption that origin-balanced context can beat the latest 9,500 rows is plausible but unmeasured; it will be decided on forward training folds only. Four-estimator TabPFN runtime is also unmeasured and must pass the timing gate before full rolling evaluation.

## Critical path

The score-bounding artifact is the per-tick TabPFN prediction because the full evaluator needs 55 independent invocations within 7,200 seconds. The confirmation target is at most 45 seconds per tick after checkpoint setup, leaving substantial evaluator and feature-building margin.
