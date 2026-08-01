# Evaluation profile

## Mechanics

- The immutable registered entrypoint is `kapso_evaluation/kapso_eval.py`; it delegates to `grader.py`, launches `main.py` in an isolated child, validates finite NumPy outputs, and calls the official RelBench `task.evaluate` implementation.
- Full and fast fidelity both score every one of the 1,400 validation rows. The fraction and seed arguments are manifest metadata only. Full runs are archived; hidden test labels are unavailable.
- The primary metric is R2, with MAE and RMSE also reported. The candidate controls the entire training and inference chain but must emit validation predictions from a train-only fit and test predictions from a separate train-plus-validation fit.
- The active environment is non-rolling. The grader also supports rolling snapshots, so predictions retain task row order after per-race decoding.

## Input distribution

| Split | Rows | Races | Years | Race-size min / q25 / median / q75 / max | Contiguous target marginal |
|---|---:|---:|---:|---|---:|
| Train | 8,997 | 730 | 1950-2004 | 4 / 10 / 12 / 15 / 33 | 693/730, 94.93% |
| Validation | 1,400 | 89 | 2005-2009 | 6 / 14 / 16 / 18 / 21 | 88/89, 98.88% |
| Test | 4,798 | 271 | 2010-2023 | 11 / 16 / 18 / 19 / 24 | unavailable; assumed near-contiguous |

- Every seed row joins uniquely to `results`; train and validation grid values are nonzero, while 0.92% of test grids are zero.
- Current qualifying coverage is 14.12% on train, 100% on validation, and 99.65% on test. Coverage begins in 1994 and is irregular through 2002 before becoming complete in 2003; a first profiling join accidentally inspected the target column after a name collision, and the corrected suffixed-column calculation is recorded here.
- Current driver standings coverage is 100% on every split. Current constructor standings/results cover 90.64%/91.32% of train and 100% of validation and test.
- The allowed full database has history through 2023-07-30 for result-like tables. All temporal joins must be exact-current or strict-prior relative to the seed timestamp.
- The task label rows, rather than all database result rows, define transport group membership and the known entrant count.

## Coverage axes

- Era and temporal shift: pre-qualifying decades, 1994-2004 late train, 2005-2009 validation, and 2010-2023 test.
- Race size and target-marginal quality: 4-33 training entrants versus 11-24 test entrants, plus non-contiguous exceptional races.
- Current context coverage: qualifying missingness, constructor-context missingness in early history, zero-grid starts, and field-size disagreement between all results and labeled entrants.
- Entity coldness and frequency: driver, constructor, circuit, and their interactions with strict-prior target history.
- Within-race structure: grid/qualifying fractions, teammates, exact-current and strict-lag standings, and constructor outcomes.
- Output head: calibrated point regression versus ordinal/ranker uncertainty and soft transport.

## Solution-coverage audit

The measured profile agrees with the supplied race counts, contiguous-label rates, test geometry, and stated test qualifying coverage. A material train-to-modern shift is that qualifying covers only 14.12% of training rows, while validation and test are essentially complete; forward folds must therefore include the 2003-2004 complete-coverage regime and missingness indicators are required. Constructor context is also incomplete in early training: constructor standings and constructor results cover 90.64% and 91.32%, respectively. The assumed modern near-contiguous test marginal cannot be checked without prohibited labels and remains an explicit assumption.

## Critical path

The score-bounding artifact is the forward-OOF matrix used to calibrate and select the ordinal/ranker transport decoder; without stable OOF heads, extra final-fit seeds cannot improve trustworthy selection. A 100-tree LightGBM benchmark on 7,120 rows completed in 0.076 seconds, equivalent to about 790 small fits/minute; the conservative full-feature target is at least 55 700-tree fits/minute. Confirmation is based on pooled and per-fold R2, with transport enabled only for a pooled gain of at least 0.005 and no fold loss worse than 0.02 R2.

## Final internal strata

The selected v5 design used 1,333 OOF rows from complete late-training race groups. Its calibrated L2 baseline achieved pooled R2 0.88845; soft transport achieved 0.91140 with mixture 0.5, entropy 1.0, rank width 1.0, and blend 0.8. Forward-fold slices were: fold 0, n=422, R2 0.87502 versus 0.86532 baseline; fold 1, n=438, R2 0.89559 versus 0.87663; fold 2, n=473, R2 0.95295 versus 0.91580. Full Sinkhorn marginal errors were 0.00869 on validation groups and 0.00544 on test groups.
