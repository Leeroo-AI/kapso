# Evaluation profile

The immutable evaluation suite is documented here without modifying `kapso_evaluation/`.

## Mechanics

- Full fidelity invokes `python main.py`, loads the two NumPy outputs without pickle, checks validation length and finite values, and calls the official RelBench evaluator.
- The headline search score is validation R² on all 1,731 rows. MAE and RMSE are also reported. The `fraction` and `seed` arguments are manifest metadata and do not change the scored rows.
- Validation predictions must come from a chain fit without validation labels. Test predictions may use a separately rebuilt train-plus-validation chain.
- Full runs are archived. The immutable grader reveals no test metric.

## Initial profile

- Train has 33,937 rows, validation 1,731, and test 1,002. Splits are chronological by user `joinedAt`.
- Train ends 2012-11-20, validation covers 2012-11-21 through 2012-11-28, and test begins 2012-11-29.
- The primary coverage axes are time/cohort, locale, gender, timezone, location missingness, strict graph degree, hard-labeled-friend count, one-hop versus two-hop coverage, friend-activity coverage, and graph-stage confidence.
- The target has long lower-tail anomalies, so slice diagnostics include graph coverage and target-year bands where labels are legal to inspect.

## Measured distribution

- The full post-test database exposes 38,209 user input rows. Database load took about 9 seconds.
- The friendship scan retained 217,555 complete, non-self directed rows and deduplicated them to 110,420 undirected edges in about 1 second from the compact cache. Validation any-edge coverage is 930/1,731 and test coverage is 439/1,002 before exact endpoint-time filtering.
- Complete activity endpoints comprise 65,216 attendee rows over 11,360 users and 15,398 interest rows over 2,034 users. Event joins are projected only for referenced events, and a fact becomes visible at the later of its fact timestamp and event start time.
- A 38,209-row, 266-column causal feature matrix builds in roughly 90 seconds uncached. The full five-matrix, two-seed pipeline completed in 805 seconds; cached reruns reuse content-keyed matrices and per-node component predictions.

## Forward evidence and strata

- LightGBM R² by expanding forward fold was 0.1161, 0.1430, and 0.1424 in the initial full measurement. CatBoost was 0.1149, 0.1469, and 0.1404. Their difference was inside two pooled standard errors, selecting the fixed 50/50 rule.
- Soft-neighbor residual correction measured +0.00087 mean R² against a pooled SE of 0.00135 and was rejected. C&S measured +0.00023 against a pooled SE of 0.00220 with latest-fold deterioration and was rejected. Clipping was tied and rejected.
- Reporting-only validation strata in the initial measurement were 680 hard-labeled-friend users, 162 users with friends but no hard anchor, and 889 isolated users. Their respective R² values were 0.268, 0.057, and 0.042; these values are not used for design selection.
- A model-input ablation retained causal activity construction but removed its sparse columns from both boosters. Aggregate OOF R² improved from 0.13833 to 0.14074; LightGBM folds became 0.11880/0.14245/0.14400 and CatBoost folds became 0.11763/0.14753/0.14449, so the simpler selector passed on mean and latest-fold stability.
- A model-specific follow-up retained friend demographics for LightGBM but removed them from CatBoost. Aggregate OOF R² improved again to 0.14213; the paired gain was 0.00139 versus pooled SE 0.00072, with latest-fold gain 0.00192, so the hybrid selector was promoted independently of validation.

## Solution assumption checks

- Timeless friendships are retained only after verifying that both endpoints exist by each seed timestamp.
- Soft-neighbor and Correct & Smooth stages were tested by forward-fold gates and rejected in the first full build, contradicting their assumed improvement while preserving the solution's gated design.

## Critical path

The final score is bounded by the breadth and causal correctness of the relational feature matrix. Feature extraction throughput and forward-fold graph rebuilding are measured before model fitting; model consumers are built after that artifact is stable.
