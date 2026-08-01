# Evaluation profile

The immutable grader runs `main.py` in a child process and scores the complete 71,398-row validation table with official RelBench metrics. `--fraction` and `--seed` do not change the scored rows. Accuracy is the primary score; macro F1, micro F1, and MRR are also reported. Full runs are archived with both prediction matrices. Validation predictions must come from Model A trained without validation labels, while Model B may use train plus validation labels only for test prediction.

## Input distribution

| split | rows | days | span | monthly rows |
|---|---:|---:|---|---|
| train | 340,491 | 743 | 2018-01-02 to 2020-01-31 | 9,330 to 15,465 |
| validation | 71,398 | 151 | 2020-02-01 to 2020-06-30 | 12,378 to 16,124 |
| test | 88,422 | 184 | 2020-07-01 to 2020-12-31 | 12,601 to 15,865 |

Train contains 46 labels; validation contains 43 observed labels. The leading validation labels are 0 (27.61%), 6 (19.15%), 3 (13.54%), 4 (11.56%), and 8 (6.49%). Every seed identifier is unique and timestamps are complete. The full allowed database contains 500,908 header rows, 2,319,540 item rows, 139,607 customers, and 1,788,887 addresses. All seed documents have an allowed header and item context; the customer graph supplies role-specific country and region.

## Coverage axes

- Temporal staleness across five validation and six test months, calendar month, weekday, and intra-day timing.
- Seen versus unseen party, party-by-document-type, product, and commercial-cohort keys.
- Four customer roles and their country/region context.
- Document type, organization, channel, company, currency, item category, product composition, and item-count scale.
- History support, recency, entropy, 60/365-day true decay, and 30/90-day pseudo decay.
- Rollforward month and block granularity, especially months 3 through 5 where the gate is evaluated.

The solution's assumed error-propagation benefit is not accepted without the train-only September 2019 through January 2020 gate. Cold starts remain covered by broad commercial and geographic cohorts plus the global class distribution. Per-stratum official validation scoring is unavailable from the immutable grader, so monthly and coverage-slice diagnostics are produced only on the train-only simulated window; official validation is retained solely as the final score of record.
