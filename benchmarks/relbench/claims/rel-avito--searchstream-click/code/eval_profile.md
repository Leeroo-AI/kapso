# Evaluation profile

## Mechanics

- The immutable grader invokes `main.py --debug` for fast fidelity and `main.py` for full fidelity.
- Both fidelities score all 1,177,380 validation rows; `fraction` and `seed` are provenance only.
- The search score is official validation ROC-AUC. Validation predictions must be produced without validation labels, while the independent test chain may fit train plus validation labels.
- Required full prediction shapes are `(1177380,)` and `(924990,)`, finite floating-point probabilities in `[0, 1]` and original task-table order.

## Input profile

- Task windows measured from task-table metadata: train is 2,212,750 rows from 2015-04-25 through 2015-05-07, validation is 1,177,380 rows from 2015-05-08 through 2015-05-13, and test is 924,990 rows from 2015-05-14 through 2015-05-20.
- The task tables contain only `SearchDate`, `primary_key`, and the train/validation target. Current-row attributes require an exact-key join to `SearchStream`.
- RelBench assigns generated SearchStream primary keys independently after timestamp censoring. Train and validation keys align to the `< 2015-05-14` snapshot, while test keys align to the uncensored snapshot; reconstructing the censored key space from the uncensored row order gives exact date agreement for every row.
- Static spine construction for 4,315,120 task rows measured 7.4 seconds for SearchStream, SearchInfo, and AdsInfo joins on this lane.
- The initially measured full-snapshot join was invalid for train/validation and is excluded from coverage conclusions because 3,390,121 timestamps disagreed. Exact-key snapshot-aware joins are mandatory.

## Coverage axes

- Time: daily strata across 13 train, 6 validation, and 7 test calendar days; forward training folds must respect those boundaries.
- Warmth: seen versus unseen ad, user, IP, search/category/location and crossed-grain support.
- Ranking context: position, slate size, HistCTR coverage/rank, object type, and search/ad context.
- Relational context: search, ad, user/device, category hierarchy, location hierarchy, visits, phone requests, and historical impressions.
- Activity: lifetime, recency, one-day, and broader entity/category event windows.
- Missingness: HistCTR, query, user, ad price, and relational join coverage.

## Measured target-input strata

- After snapshot-correct joining, HistCTR coverage is 100% in train, validation, and test; all target candidates have ObjectType 3 and positions are restricted to 1 or 7.
- Position 1 shares are 57.62% train, 56.28% validation, and 55.23% test. Position 7 accounts for the remainder.
- Median full-search slate size is 4, the 99th percentile is 5, and the maximum is 14 in every split.
- Target ad diversity is unusually narrow: 19,633 train ads, 19,178 validation ads, and 18,566 test ads. Target search counts are 1,275,079, 662,598, and 510,831 respectively.
- HistCTR-only train AUC is 0.65915 overall, 0.64698 at position 1, and 0.66036 at position 7. Its three forward-window AUCs are 0.66290, 0.65345, and 0.66351, establishing a stable baseline but leaving substantial room for relational history.

## Solution discrepancy

- The proposed third fold scores only May 6-7 because training ends May 7; it is retained as a two-day forward fold.
- `load_task(upto_test_timestamp=False)` alone does not provide one universal generated key space. The implementation keeps its full database but reconstructs the censored SearchStream primary-key view for train/validation, preserving the solution's actual-key and exact-time assertions.
- The solution's assumed restricted target regime is confirmed: unlike the database-wide 52% HistCTR coverage and three object types, task candidates have complete HistCTR, only ObjectType 3, and only positions 1 and 7. Separate missing-HistCTR/object-type experts are therefore unnecessary; position remains a required slice.

## Slice reporting plan

- Persist train-only forward-fold AUC by fold, HistCTR availability, object type, position, warm/cold ad, and day where sample support permits.
- The immutable grader emits only whole-validation official metrics, so official slice metrics cannot be added without changing protected evaluation behavior.
