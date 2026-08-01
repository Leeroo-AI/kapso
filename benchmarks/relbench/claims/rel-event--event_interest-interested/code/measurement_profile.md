# Evaluation profile

The immutable harness runs `main.py` in an isolated subprocess, checks both NumPy vectors, and evaluates all 536 validation rows with RelBench metrics. The selection score is global row-level ROC-AUC; `fraction` does not subsample rows. Test metrics remain hidden. Validation predictions must come from a train-only fit, while test predictions may come from a separate train-plus-validation fit.

## Input distribution

| Split | Rows | Positive rate | Bursts | Singleton bursts | Size-6 bursts | Time span |
|---|---:|---:|---:|---:|---:|---|
| train | 14,442 | 0.2696 | 4,422 | 2,326 | 1,797 | 2012-04-27 to 2012-11-20 |
| validation | 536 | 0.2687 | 151 | 67 | 63 | 2012-11-21 to 2012-11-28 |
| test | 420 | hidden | 98 | 32 | 61 | 2012-11-29 to 2012-12-12 |

Train burst sizes have counts `{1: 2326, 2: 10, 4: 131, 5: 158, 6: 1797}`; validation has `{1: 67, 2: 2, 4: 8, 5: 11, 6: 63}`; test has `{1: 32, 4: 3, 5: 2, 6: 61}`. Every labeled burst has at most one positive. Any-pick prevalence is 0.8804 in train and 0.9536 in validation, with meaningful shift in singleton prevalence and singleton any-pick rate.

## Identity and API mechanics

The test-cutoff snapshot has 14,978 unique `event_interest.primary_key` values and resolves all train and validation rows with exact timestamp equality. The full snapshot has 15,398 unique keys and resolves all test rows with exact timestamp equality. Full-snapshot keys do not identify most earlier task rows, so train/validation identities must come from the censored snapshot and test identities from the full snapshot. Task test order is not primary-key sorted and requires `_row_id` restoration.

## Coverage axes

- Burst size, position, no-pick versus pick, and singleton versus multi-row choice.
- Temporal regime, new versus previously observed users/events, session number, and response-history support.
- Demographics, geography, local time, event eligibility/content, attendance popularity/history, and friendship degree/community/support.
- Missing event, user, geographic, content, attendee, and known-friend information.
- Global row ROC-AUC, burst-any AUC, within-multi-burst top-rank accuracy, and row AUC by burst size.

The assumed at-most-one structure holds throughout train and validation. Static friendship rows have no time column and are treated as query-time context. All timed source joins are filtered to source time no later than the query time, and label histories are strict-prefix only.

## Implemented full-fold profile

Five expanding seven-day folds selected 15 LambdaRank leaves, softmax temperature 1.5, conditional-network weight decay `1e-5`, and row-guard weight 0.30. Mean/worst global row ROC-AUC was 0.82086/0.76890. Burst-any AUC ranged from 0.88493 to 0.96660; conditional top-rank accuracy ranged from 0.279 to 0.504, identifying multi-row within-burst ranking, especially size 6, as the final critical path.
