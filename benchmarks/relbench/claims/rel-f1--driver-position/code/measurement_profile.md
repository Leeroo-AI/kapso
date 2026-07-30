# Evaluation profile

## Mechanics

The registered entrypoint delegates to the immutable rolling grader, which invokes `main.py` once per validation and test origin. Each invocation receives a tick-truncated database, every task-label window closed by the tick, an empty validation table, and the current origin rows as its test table. The protected grader restores row order, scores all 499 validation rows with official RelBench metrics, and hides test labels and metrics. The score of record is validation MAE; `--fraction` and `--seed` do not subsample rows.

## Input distribution

| split | rows | origins | date span | unique drivers | median rows/origin | range |
|---|---:|---:|---|---:|---:|---:|
| train | 7,453 | 254 | 1950-06-19 to 2004-09-03 | 771 | 28 | 10-83 |
| validation | 499 | 23 | 2005-03-02 to 2009-10-07 | 47 | 22 | 20-25 |
| test | 760 | 33 | 2010-03-02 to 2016-05-29 | 56 | 23 | 20-26 |

Train targets have mean 13.901, median 13.333, standard deviation 7.026, and range 1-39. Validation targets have mean 11.083, median 11.4, standard deviation 4.641, and range 1-22. Origins normally advance by 60 days, while the later field is materially smaller than historical fields. Qualifying begins only in 1994 and constructor standings in 1958, making missingness and season phase observable regime markers.

## Coverage axes

- Calendar regime: year, field size, origin gap, pre-season, midseason, and late season.
- Driver evidence: recent results, closed labels, standings, qualifying, history length, activity, and missingness.
- Team evidence: constructor form, constructor standings, qualifying, team changes, and evidence staleness.
- Forecast structure: causal 60-day windows with refitting after each newly closed origin.

## Solution coverage

The specialist context includes exact champion, LightGBM, CatBoost, and recent-form OOF registers plus gain-stable numerical driver, constructor, standing, qualifying, closed-label, cohort, missingness, and phase features. Raw driver and constructor identifiers and hashed categorical codes are excluded. Every residual label is based on a champion prediction trained only on labels whose 60-day window closed before the predicted origin.

## Critical path and benchmark

The score-bounding artifact is the reusable exact forward-origin OOF residual table. The local checksum-verified TabPFN-v2 cold benchmark completed a 2,000-row, 96-column, one-estimator fit and prediction in 10.85 seconds with finite output. A representative 7,453-row, 96-column, four-estimator fit and prediction completed in 15.47 seconds, below the 45-second estimator-reduction threshold and the 55-second per-tick target.

## Prior score strata

| stratum | count | MAE |
|---|---:|---:|
| 2005 | 87 | 2.6910 |
| 2006 | 92 | 2.3596 |
| 2007 | 114 | 2.8567 |
| 2008 | 104 | 2.2792 |
| 2009 | 102 | 3.1784 |
| pre-season, months 1-3 | 110 | 3.5582 |
| midseason, months 4-8 | 327 | 2.3087 |
| late season, months 9-12 | 62 | 3.0927 |

The measured pre-season error motivates a larger residual correction only when the offseason gate is active. The prescribed `0.35` offseason increment improved the training-only mean, preseason, worst-year, and seven-of-ten-season checks but missed the two-SE check. The permitted inner-fold weight check selected the more regularized `0.05` increment, which cleared all four gates without using official validation.
