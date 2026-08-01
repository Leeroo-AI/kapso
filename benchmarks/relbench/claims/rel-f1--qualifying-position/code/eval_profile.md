# Evaluation profile

## Mechanics

- The registered grader runs `main.py` in a child process, loads exactly 1,854 validation predictions and 5,733 test predictions, and scores every validation row with the RelBench task evaluator.
- The primary score is row-level R2; MAE and RMSE are also reported. Fidelity and fraction do not subsample scored rows.
- Validation predictions must be produced by a chain fit on train labels only. Test predictions may use a separately refit chain trained on train and validation labels.
- The candidate controls all feature construction, model fitting, and prediction projection. The harness controls task loading for scoring, shape/finite checks, official metrics, timeouts, and full-run archiving.

## Input distribution

| split | rows | qualifying events | date range | roster min/median/max |
|---|---:|---:|---|---:|
| train | 2,228 | 100 | 1994-03-26 to 2004-10-23 | 20 / 22 / 28 |
| validation | 1,854 | 89 | 2005-03-05 to 2009-10-31 | 18 / 20 / 22 |
| test | 5,733 | 271 | 2010-03-13 to 2023-07-29 | 18 / 20 / 24 |

The deployment horizon is much longer than training, rosters shrink after the earliest era, and entity churn is substantial. The database contains 1,101 races, 26,080 results, 34,124 driver-standing rows, 13,051 constructor-standing rows, 12,290 constructor-result rows, and 9,815 qualifying input rows through 2023-07-30 or later.

## Coverage axes

- Time and regime: early training, late training forward folds, validation era, and long test deployment.
- Event roster: 18 through 28 entrants and event-level output constraints.
- Experience: established drivers/constructors, team changes, debuts, and sparse histories.
- Form horizon: last observation, short/medium/long EWMAs, trends, uncertainty, and recency.
- Relational source: results, standings, constructor results/standings, identities, qualifying roster, race schedule, and circuit history.
- Temporal boundary: qualifying occurs before the matching race result; histories therefore use strict source timestamps earlier than the qualifying seed.

## Solution coverage check

The measured churn and horizon support uncertainty-aware cold starts and persistent/season ratings. The assumed special lower constructor-debut prior remains eligible only if it wins train-only forward-fold selection; otherwise the neutral train-era debut prior is used.

## Reporting strata

Internal selection reports R2 by each of four contiguous race blocks and by roster-size, era, driver-experience, and constructor-experience strata where counts permit. Official output exposes one aggregate validation score only.

## Internal train-forward slice results

The frozen 800-tree core residual model scored pooled R2 0.596422 across four expanding blocks, with block R2 values 0.6391, 0.6272, 0.5228, and 0.5672.

| stratum | count | R2 | MAE |
|---|---:|---:|---:|
| roster small, at most 20 | 700 | 0.5394 | 2.9387 |
| roster medium, 21–22 | 548 | 0.6518 | 2.8562 |
| driver cold, at most 5 events | 118 | 0.4629 | 2.1826 |
| driver developing, 6–30 | 267 | 0.5844 | 2.6355 |
| driver established, over 30 | 863 | 0.5036 | 3.0835 |
| constructor cold, at most 5 events | 33 | 0.3399 | 3.6403 |
| constructor developing, 6–30 | 82 | 0.1230 | 2.9818 |
| constructor established, over 30 | 1,133 | 0.6123 | 2.8752 |
