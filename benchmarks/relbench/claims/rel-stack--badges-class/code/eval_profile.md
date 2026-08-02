# Evaluation profile

The immutable grader runs `main.py` in an isolated child, validates both NumPy files, and calls the official RelBench `task.evaluate`. Accuracy is the primary score and is determined by row-wise argmax over three finite float scores. Fast fidelity changes only the candidate build mode; it still scores all 15,105 validation rows. Full fidelity archives both full-shape arrays.

## Input distribution

- Train has 448,358 rows from 2010-07-19 through 2020-09-30; 242,156 rows and 96,527 users occur from 2017 onward.
- Validation has 15,105 rows, 9,803 users, and 9,065 exact-time batches from 2020-10-01 through 2020-12-31. Exact-batch size has median 1, 99th percentile 8, maximum 67, and 61.0% of rows share a timestamp with another badge for the same user.
- Test has 127,370 rows, 64,134 users, and 86,526 exact-time batches from 2021-01-01 through 2023-09-03. This measured range contradicts the problem summary's abbreviated January-2021 endpoint and requires long-gap unlabeled state updates through each seed time.
- Forward 90-day horizons beginning 2019-10, 2020-01, 2020-04, and 2020-07 contain 20,175, 14,410, 15,901, and 12,596 rows. Class-2 shares are 63.1%, 82.2%, 82.5%, and 80.8%, showing material temporal prior movement.
- Batch structure shifts: training exact-batch maximum is 4,885 and a row-weighted mean batch is 69.7, while validation and test row-weighted means are 3.38 and 6.23. Batch-size features therefore need regularization and non-absolute cohort descriptors.

## Coverage axes

Coverage is tracked over forward origin, user maturity and prior-badge bands, recent activity, exact-batch size and repeat multiplicity, trigger-crossing recency, award hour, and class. Internal results are reported per forward horizon and for batch-size and maturity strata where supported.

The critical path is construction of temporally censored feature matrices from all seven tables. The target throughput is at least 100,000 seed-feature rows per minute once content-versioned event aggregates are available.
