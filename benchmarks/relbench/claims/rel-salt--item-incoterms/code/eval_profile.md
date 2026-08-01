# Evaluation profile

## Mechanics

- Registered entrypoint: `kapso_evaluation/kapso_eval.py`; full fidelity executes `python main.py` in an isolated subprocess with a 14,400-second timeout.
- The scorer always evaluates all 293,891 validation items. `--fraction` and `--seed` are manifest metadata and do not subsample scoring rows.
- Primary score is official validation accuracy. The harness also reports macro F1, micro F1, and MRR. Test labels are absent and test metrics remain private.
- Required outputs are finite floating-point arrays aligned to original task order: validation `(293891, 13)` and test `(402835, 13)`.
- Model A validation predictions must use train labels only. Model B may use train and validation labels for test predictions.

## Input distribution

| split | items | documents | time range | documents with multiple timestamps |
|---|---:|---:|---|---:|
| train | 1,622,787 | 340,491 | 2018-01-02 to 2020-01-31 | 0 |
| validation | 293,891 | 71,470 | 2020-02-01 to 2020-06-30 | 0 |
| test | 402,835 | 88,925 | 2020-07-01 to 2020-12-31 | 0 |

- Train document size quantiles at 50/90/95/99/100% are 2/10/21/54/525 items. Validation quantiles are 1/8/16/48/253.
- Modal document labels impose an oracle loss of 0.0213 percentage points on train and 0.0470 points on validation, below the assumed 0.2-point limit.
- Validation label shares vary by month: class 0 is 61.54%, 64.25%, 71.54%, 70.05%, 66.54%; class 2 is 13.36%, 10.58%, 7.54%, 8.04%, 8.35%; class 4 is 10.06%, 11.16%, 8.05%, 8.34%, 8.92% from February through June 2020.
- Validation document staleness by the latest train sold-to and sales-area observation: `<30` 9,971 docs/42,699 items; `30-90` 23,333/94,736; `90-180` 27,467/109,642; `>180/new` 10,699/46,814.
- Joined task rows have complete item matches. Observed cardinalities include 14,710 sold-to parties, 34 sales organizations, 3 distribution channels, and 13 target classes.

## Coverage axes and strata

- Time: calendar month, chain cutoff, source age, and five-month forecast horizon.
- Staleness: `<30`, `30-90`, `90-180`, and `>180/new` days.
- Hierarchy: document size, single versus multi-item documents, item-set diversity, and header sales area.
- Parties: sold-to, ship-to, bill-to, payer, role equality, geography, and cold versus previously observed party combinations.
- Labels: common versus rare class, monthly prior movement, and customer/key multimodality.
- The measured profile supports document weighting, forward-gap selection, causal histories, and staleness blending. Static geography remains an assumption and is included as a non-temporal lookup.

## Reporting

Internal folds report item-weighted headline accuracy and count/accuracy for every staleness segment, before and after any prior correction. The registered harness provides only the official aggregate validation metrics, so final strata are emitted by the candidate diagnostics without accessing validation labels for selection.

