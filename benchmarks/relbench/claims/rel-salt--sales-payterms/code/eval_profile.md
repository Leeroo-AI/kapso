# Evaluation profile

## Mechanics

- Registered command: `python kapso_evaluation/kapso_eval.py --fidelity full --fraction 1.0 --seed 1337`.
- The immutable grader runs `main.py` in an isolated subprocess, validates finite full-shape arrays, and evaluates validation predictions with the official RelBench task evaluator.
- Primary score is accuracy over all 71,472 validation rows. Multiclass scores have 154 columns and accuracy is determined by the highest-scoring column. The fraction and seed arguments are manifest metadata and do not subsample rows.
- Test predictions contain 88,831 rows and are archived, but test labels and test metrics are unavailable to the candidate.
- Validation predictions must come from the train-only chain. Test predictions may come from a separate train-plus-validation chain.

## Input distribution

- Train: 340,491 rows, 154 observed labels, 2018-01-02 through 2020-01-31. Validation: 71,472 rows, 109 observed labels, 2020-02-01 through 2020-06-30. Test: 88,831 rows, 2020-07-01 through 2020-12-31.
- Task-table row order is not chronological, so `_row_index` must be preserved before feature joins.
- The database contains 500,908 legal sales-document headers and 2,319,540 items when loaded with `upto_test_timestamp=False`.
- Items per document: minimum 1, median 2, p90 9, p99 54, maximum 525, mean 4.63.
- Party ambiguity affects 1 payer document, 0 sold-to documents, 21 bill-to documents, and 197 ship-to documents.
- All 2,319,540 item timestamps equal their document timestamp; no joined item occurs after its document timestamp.
- Legal header cardinalities are 13 document types, 34 sales organizations, 3 channels, 1 division, 31 billing companies, and 27 currencies. Items contain 21 categories and 187,536 products. Addresses contain 240 countries and 597 regions.
- Payer history covers 95.88% of validation rows. Accuracy and coverage vary materially by payer history, hierarchy precision, geography, staleness, month, and cold-start status.

## Coverage axes

- Serving chain: train-only validation versus train-plus-validation test.
- Horizon and month: zero through five validation months and zero through six test months after the frozen label history.
- Payer state: unseen, seen-stable, seen-switching, and ambiguous party assignment.
- Hierarchy: exact payer-sales-area context, payer-only, alternate-party, geography cohort, and global fallback.
- Order context: organization, document type, channel, billing company, currency, division, item volume, category mix, and product mix.
- Label frequency: dominant labels, medium-frequency labels, and rare labels.

## Critical path

The synthetic candidate episode feature matrix bounds ranker quality and therefore final score. Raw candidate generation measured 1,877 seeds/s on 20,000 seeds, projecting 3.7 minutes for 411,963 seeds before full feature computation. The implementation target is at least 350 feature-complete seeds/s, confirmed first on the debug-sized 20,000-seed build.

The first complete 340,491-seed build produced 1,694,582 candidate rows in 531 seconds, or 641 seed episodes/s. The widened horizon audit produced the same rows in 596 seconds. Both exceeded the pre-committed 350 seed/s target.

## Strata to report

Internal forward-fold output will report count and accuracy by evaluation month, payer-history coverage, and staleness bucket. The protected official evaluator exposes only headline validation metrics, so official per-stratum values cannot be recovered without changing evaluation behavior.
