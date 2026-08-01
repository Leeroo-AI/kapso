# Evaluation profile

## Metric mechanics

- The immutable grader invokes `main.py` in an isolated process and scores the entire official validation split with RelBench metrics.
- The primary search metric is validation accuracy; `fraction` and `seed` are manifest metadata and do not change the scored rows.
- Required multiclass score arrays are `(71470, 13)` for validation and `(88925, 13)` for test, finite floating-point values in task-table order.
- Validation scores must come from a train-label-only fit. Test scores may come from a separate train-plus-validation fit.

## Initial distribution profile

- Training has 340,491 rows and 13 labels; validation has 71,470 rows; test has 88,925 rows.
- Available coverage axes are time/month, class, sold-to support and staleness, document type and sales organization, ship-to sales area, ship-country, item count/category/product composition, and party/address cold start.
- The measured campaign memory reports 95.9% validation coverage for sold-to parties, 90.2% coverage for sold-to × document type × sales organization, and a 5.7-point later-validation lift from fresher label history.
- The observed class shift is material: class 3 rises from 4.17% in training to about 9% late in validation while class 2 declines from 13.22% to about 11%.
- Train spans 2018-01-02 through 2020-01-31; validation spans five monthly strata from February through June 2020; test spans six monthly strata from July through December 2020.
- Train monthly volume ranges from 9,330 to 15,465. The class-3 fraction rises to 6.73% in January 2020 while class 2 falls to 11.37%, confirming drift before the official validation boundary.
- A train document has 4.77 items on average, with item-count quantiles 2, 10, and 54 at p50, p90, and p99. Train covers 11,940 sold-to parties, 14,458 ship-to parties, 45,650 products, and 18 item categories; no measured train document has multiple sold-to identities.
- Header cardinalities on train are 11 document types, 30 sales organizations, 3 channels, 1 organization division, 28 billing companies, and 27 currencies. The customer-address path covers 203 countries and 525 regions without missing address joins.

## Critical path

The bounding artifact is the causally censored four-hierarchy feature matrix. The 30,000-row debug probe produced 245 features at 3,668 query rows/second; the complete two-model debug pipeline finished in 39.7 seconds and wrote contract-valid full-shape arrays.

## Immutability note

This profile is kept at repository root because the supplied evaluation directory is explicitly read-and-execute only and the task forbids editing anything under it.
