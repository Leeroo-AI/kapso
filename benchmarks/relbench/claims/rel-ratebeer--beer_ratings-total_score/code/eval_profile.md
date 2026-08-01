# Evaluation profile

## Mechanics

The immutable grader invokes `main.py` in an isolated subprocess, requires aligned finite float arrays with shapes `(1227702,)` and `(2495360,)`, and computes all official validation metrics over the entire validation table. `--fraction` is recorded in the manifest but does not subsample scoring rows. The search score is validation R²; squared-error training therefore aligns with both R² and RMSE, while MAE is diagnostic. Test labels and metrics are unavailable.

Validation predictions must come from state A, which is trained only on train labels. Test predictions may come from state B, which is trained on train plus validation labels. Final model rounds, weights, calibration, and feature decisions are selected only through training-label forward blackouts.

## Measured input distribution

| split | rows | first timestamp | last timestamp | users | beers |
|---|---:|---|---|---:|---:|
| train | 10,620,177 | 2000-04-12 | 2018-08-31 | 164,619 | 541,708 |
| validation | 1,227,702 | 2018-09-01 | 2019-12-31 | 49,177 | 252,663 |
| test | 2,495,360 | 2020-01-01 | 2025-02-03 | 23,849 | 576,059 |

Validation is 18.0435% user-cold, 36.6762% beer-cold, and 2.4312% jointly cold relative to train. Test is 9.4731% user-cold, 58.4045% beer-cold, and 2.9148% jointly cold relative to train plus validation. Test row counts by year are 589,663, 542,710, 495,939, 456,797, 378,606, and 31,645 for 2020 through early 2025.

Train labels have mean 3.357179 and population standard deviation 0.725161; validation labels have mean 3.444189 and population standard deviation 0.711824. English accounts for 9,967,317 train events, followed by Polish 157,836, French 120,954, and German 91,037.

Task rows are almost, but not perfectly, timestamp ordered: raw task order contains 81 train, 11 validation, and 17 test timestamp inversions. Every causal kernel must therefore sort by `(created_at, rating_id)` and restore original output order.

## Coverage axes

- Horizon: 16-month validation blackout and five-year test-like blackout, plus test horizon buckets from 2020 through 2025.
- Support: warm/cold user, warm/cold beer, jointly cold, and logarithmic count strata.
- Hierarchy: global, parent style, style, brewer, beer, country, and ABV band.
- Behavior: lifetime/recent activity, session position, exploration, favorite history, and place-rating history.
- Metadata: language, calendar, style hierarchy, brewer geography/type/age, beer age/composition/flags, and user tenure/type.
- Missingness: null beer, IBU, ABV, mutable-metadata eligibility, and every label estimate.

## Table and leakage findings

The canonical event projection is limited to `rating_id,user_id,beer_id,created_at,language`; no forbidden field is read. Mutable snapshot aggregates in users, beers, brewers, and places are excluded. Availability has 117,424 rows spanning 2024-01-01 through 2025-02-03 and zero pre-test-cutoff rows, so it cannot contribute a learned pre-T effect. UPC data has no timestamp and is blocked. Null `updated_at` plausibly means never updated but this semantic is not independently documented; mutable attributes are therefore eligible only when the recorded update is null or no later than the seed time.

## Throughput and score strata

A representative compiled grouped-history kernel processed 3,000,000 rows in 0.029 seconds, or 104.24M row updates/s. Sorting and matrix assembly, rather than state updates, are expected to bound feature construction. Final evaluation output should report counts and R² for warm/warm, user-cold, beer-cold, jointly cold, language, and horizon strata whenever labels permit.
