# Evaluation profile

## Mechanics

- The immutable grader runs `main.py` as an isolated child, enforces a 1,200-second debug or 14,400-second full timeout, and scores all 293,780 validation rows regardless of `--fraction`.
- Accuracy is the primary selection metric. The same 81-score matrix is also evaluated for macro F1, micro F1, and MRR. Test predictions are contract-checked and archived, but test labels and metrics are unavailable.
- Validation predictions must come from states and weights fit without validation labels. The test chain may rebuild on train plus validation labels.

## Input profile

- Train has 1,622,787 rows from 2018-01-02 through 2020-01-31, validation has 293,780 rows from 2020-02-01 through 2020-06-30, and test has 398,536 rows from 2020-07-01 through 2020-12-31.
- Train exposes 81 classes; validation contains 58 observed classes. The largest class is 51.77% of train and 46.73% of validation.
- Permitted database inputs comprise 2,319,540 items, 500,908 documents through the test window, 139,607 customers, and 1,788,887 addresses. Only customer-referenced addresses are needed.
- Cardinalities include 187,536 products, 21 item categories, 18,094 ship-to customers, 14,710 sold-to customers, 13 document types, 34 sales organizations, 3 channels, 31 companies, 27 currencies, 240 countries, and 597 regions.
- Document size has median 2, p90 9, p99 54, and maximum 525. All items of a document share a timestamp, and previously measured label uniformity is 99.7% for multi-item documents.
- A train-only January forward fold measured 98.74% accuracy for the organization-by-document-type-by-item-category posterior and 98.84% after document pooling. Low-confidence replacement from organization-by-document-type-by-product reached 99.60% in that fold.

## Coverage axes

- Calendar month and history age; seen versus cold products and customer roles; relation support and posterior entropy; document cardinality; document/header category; item category; customer address country and region; and target-class frequency.
- The solution's graph-scale, relational, document-size, output-shape, installed-library, and temporal-safety claims agree with the measured profile. The assumed runtime remains the main unconfirmed risk.

## Critical path

- The score is bounded first by the breadth and recency of cutoff-safe historical relation states. A prototype constructed and scored 14 relation maps over a 1.56M-row history in under 10 seconds, comfortably above the planned 250,000-row/second minimum for individual direct maps.
- Neural consumers are secondary and must preserve the frozen high-confidence posterior if their training or inference fails.
