# Evaluation results

## Registered run

- Run: `run_0004`
- Validation ROC AUC: 0.8050249433
- Average precision: 0.9346666527
- Accuracy: 0.7985865724
- F1: 0.8794926004
- Rows: 566 validation, 702 test
- Model selection: unlimited recency won train-only forward folds; the prequential meta-model failed its forward mean/worst-block gate on validation-era ticks, so the fixed 40% M1 / 60% M2 blend was retained. Adaptation became eligible on three late test ticks using only then-closed prequential rows.

## Validation slices

| Slice | Count | Positive rate | ROC AUC |
|---|---:|---:|---:|
| 2005 | 165 | 0.7879 | 0.7356 |
| 2006 | 178 | 0.8090 | 0.8509 |
| 2007 | 201 | 0.7363 | 0.8225 |
| 2008 | 22 | 0.8636 | 0.8070 |
| Sparse history, fewer than 20 results | 144 | 0.8750 | 0.8474 |
| Medium history, 20-99 results | 209 | 0.7464 | 0.8293 |
| Rich history, at least 100 results | 213 | 0.7465 | 0.7453 |
| Qualifying history unavailable | 19 | 0.8421 | 0.3333 |
| Qualifying history available | 547 | 0.7770 | 0.8122 |
| New team, at most three races | 83 | 0.8072 | 0.7799 |
| Established team | 483 | 0.7743 | 0.8139 |
| Field size 20-21 | 143 | 0.8042 | 0.7627 |
| Field size 22-23 | 399 | 0.7669 | 0.8207 |
| Field size at least 24 | 24 | 0.8333 | 0.8250 |

## Resolution and representativeness

A 1,000-draw row bootstrap of `run_0004` produced ROC AUC standard error 0.02056 and a 95% interval of [0.76635, 0.84319]. Three unique archived candidates had mean pairwise Spearman correlation 0.86635 and mean absolute prediction differences around 0.07-0.10. `run_0004` exceeded the strongest alternative by 0.01072, which is inside two bootstrap standard errors despite materially different predictions.

The last 30 months of training contain 499 rows over 24 origins, positive rate 0.7695, and 20.79 rows/origin. Validation contains 566 rows over 26 origins, positive rate 0.7792, and 21.77 rows/origin, so it is representative of recent training on observed volume and label rate. Test inputs have 24.21 rows/origin; test labels are unavailable and were not accessed.

Under the task's resolution rule, the finite validation set does not reliably separate materially different candidates. The least-breaking remedy is same-contract re-measurement over additional task-generated training-era rolling windows, aggregated with the current windows.
