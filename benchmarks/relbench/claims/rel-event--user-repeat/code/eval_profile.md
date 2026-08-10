# Evaluation profile

## Mechanics

- The immutable registered command runs `main.py` as an isolated child, checks the two NumPy outputs, and scores all 268 validation rows with RelBench task metrics. `fraction` does not subsample scoring rows.
- The selection score is validation ROC-AUC. Accuracy, average precision, F1, and ROC-AUC are also emitted.
- Validation prediction rows must retain the task table's original order and must come from Model A, which never sees validation targets. Model B may use validation targets and supplies only test predictions.
- Full mode has a 7,200-second candidate timeout; debug mode has a 900-second timeout. Debug still requires complete `(268,)` and `(246,)` outputs.
- The evaluator provides no inference-time hyperparameter control and exposes no test score.

## Input distribution

- Train has 3,842 rows, 1,388 users, 19 Wednesday timestamps from 2012-07-11 through 2012-11-14, and positive rate 0.489849. Per-timestamp count rises from 5 to 490; users occur 1 to 19 times with median 2.
- Validation has 268 distinct users at 2012-11-21 and positive rate 0.485075. Test has 246 distinct users at 2012-11-29.
- The seven exact weekday replay grids produce 25,648 Model-A rows through 2012-11-14: 3,842 official-phase and 21,806 shifted rows, 1,487 distinct users, positive rate 0.485418.
- Extending replay through 2012-11-21 produces 28,915 rows, 1,637 users, and positive rate 0.487117. Extending through the legal Model-B seed cutoff 2012-11-22 produces 29,338 rows; those outcome windows end exactly at the 2012-11-29 query timestamp.
- Official validation remains a distinct 268-row table because single-timestamp task generation has different window-row availability from an extended replay grid. All 268 official validation rows match the corresponding extended-grid rows and take priority during deduplication; 166 extra phase-zero replay rows remain downweighted augmentation.
- Exact official-train replay requires two warm-up timestamps, 2012-06-27 and 2012-07-04. With them, all 3,842 `(timestamp,user,target)` rows reproduce exactly.
- Exact official replay took 0.186 seconds, about 20,681 output rows/s. Seven-phase replay took 0.240 seconds, about 120,514 output rows/s.

## Relational profile

- `event_attendees` has 8,430,002 rows but only 49,822 locally resolved user endpoints, covering 9,257 users and 6,823 events. Resolved status rows include 4,977 `yes` and 2,541 `maybe`.
- `event_interest` has 14,978 rows over 1,979 users and 8,119 resolved events.
- `events` has 2,459,811 rows; 1,280,818 have city and 1,465,805 have coordinates. One hundred `c_*` columns are available for fixed random projection.
- `users` contains dense IDs 0 through 37,142, 64 locales, two non-null gender values, and 2,705 locations.
- `user_friends` has 30,386,403 rows, of which 217,555 have both endpoints resolved. The verified undirected graph covers 28,796 users, with median degree 8, 99th percentile degree 104, and maximum degree 292.

## Coverage axes and reporting

- Time: early/middle/recent/final training buckets and the validation cutoff.
- History: no prior resolved label, one prior label, two to four, and five or more.
- Behavior: attendance recency and volume; status composition; interest volume and conversion.
- Social: no verified friends, low degree, medium degree, and high degree; friend activity and embedding community.
- Demographics: gender, age/unknown age, locale, timezone, location, tenure, and missingness.
- Replay provenance: official Wednesday phase versus six downweighted shifted phases.
- Final diagnostics should report internal forward-OOF ROC-AUC by time bucket, prior-history bin, and social-degree bin. Official evaluation emits only the headline validation metrics.

## Critical-path discrepancy

The solution's repeated-user and multi-phase assumptions are confirmed. A measured discrepancy is that extended phase-0 replay at 2012-11-21 has 434 rows rather than the official single-timestamp validation's 268 because the SQL window operates over available per-user rows. Model B prioritizes the exact official rows and treats the 166 additional extended-grid rows as downweighted augmentation.
