# Evaluation profile

The immutable evaluator runs `main.py` in an isolated subprocess, requires full aligned float vectors of lengths 15,840 and 16,366, and computes the official task metrics on the complete validation split. The primary score is one pooled ROC-AUC; accuracy, average precision, and F1 are secondary. The harness controls only fidelity, timeout, and output paths, and performs no threshold, calibration, or model selection.

## Input distribution

- Train: 98,697 unique episodes, 28,333 brewers, 18 annual origins from 2000-09-05 through 2017-09-01, positive rate 0.332665.
- Validation: 15,840 unique brewers at 2018-09-01, positive rate 0.343182.
- Test: 16,366 unique brewers at 2020-01-01; labels are absent.
- Current release-gap strata are materially shifted: validation has 4,119 / 4,415 / 3,479 / 1,925 / 1,902 rows in 0-30 / 31-90 / 91-183 / 184-270 / 271-365 day bins, versus test 3,771 / 4,083 / 3,922 / 2,669 / 1,921.
- A generated 2017-01-01 origin has 13,501 eligible brewers and dormant rate 0.309607. A locally generated 2018-12-25 origin has 16,227 eligible brewers and rate 0.354163, one row below the solution's stated 16,228 and therefore treated as a measured discrepancy rather than a changed eligibility rule.

## Coverage axes

Origin month and temporal regime; current silence; historical cadence and irregularity; activity and portfolio size; style/category diversity; seasonal and one-off mix; rating demand level, quality, reviewer breadth, and momentum; geographic demand; country/state/type cohort; opening and release age. Every dynamic feature is evaluated at `event_time <= seed_time`.

## Table safety

`availability` starts in 2024 and has zero usable rows before either seed cutoff. `beer_upcs` is untimestamped. `favorites` starts 2018-05 and has no Model-A training support. These are excluded. `beer_ratings`, `place_ratings`, and `beers` span the usable history. Dump-time totals, mutable flags, views, updates, current averages, and profile completeness are excluded.

## Reporting

The candidate records forward-fold count and AUC by validation origin plus stage/blend aggregates in `metrics.json`. The registered evaluator exposes only pooled official validation metrics, so official test strata cannot be reported.
