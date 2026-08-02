# Evaluation profile

The immutable grader runs `main.py` in an isolated child, loads both NumPy vectors, verifies finite aligned outputs, and calls the official RelBench task evaluator. Both fidelities score all 14,470 validation rows; `--fraction` and `--seed` are manifest metadata and do not alter the scored items. The headline selection score is validation ROC-AUC, with average precision, accuracy, and F1 also reported. Validation predictions must come from a chain that has never used validation labels; the test chain may refit on train plus validation.

## Input distribution

- Train contains 234,366 rows from 2000–2019, validation contains 14,470 rows from 2020, and test contains 23,430 rows from 2021–2023. The test years contain 13,164, 8,077, and 2,189 rows respectively.
- Positive rates are 0.9205 in train and 0.9302 in validation. Validation has about 1,010 negatives, making sub-0.003 AUC changes too small to treat as reliable selection evidence.
- Mean character lengths are 85.6 for brief title, 135.0 for official title, 596.9 for summary, and 1,218.5 for detail when nulls count as zero. Detail is null in 33.89%; among present values it averages about 1,843 characters and reaches 31,794 characters.
- Same-study, seed-time coverage is 80.87% for conditions, 34.81% for interventions, 100% for sponsors, 91.90% for facilities, and 99.77% for designs. Result-table rows available by the seed date cover under 0.1% of studies; those rare rows enter a quota-limited RESULTS field and every later row is censored.
- Facility multiplicity has a heavy tail: mean 6.85, p90 11, p99 114, maximum 3,511. Condition counts have mean 1.61 and p99 6; sponsor counts have mean 1.56 and p99 6.
- Study types are mainly Interventional (217,106) and Observational (51,111). Surviving eligibility categories are strongly imbalanced: gender is All for 232,319 rows, sampling method is null for 217,963, and gender-based is null for 266,402.

## Coverage axes

- Temporal: 2017, 2018, and 2019 forward holdouts; 2020 validation; the three-year 2021–2023 test horizon.
- Text: no detail versus present detail, short versus long summary/detail, direct pediatric markers, direct adult markers, both, and neither.
- Relations: missing versus present condition/intervention/site links; low versus high facility multiplicity; sponsor agency class; domestic versus multinational site footprint.
- Trial design: interventional versus observational, phase, design categories, enrollment scale, gender, healthy-volunteer status, and sampling method.
- Difficulty: pediatric-marked negatives, negatives without pediatric markers, contradictory adult and pediatric markers, and sparse/no-detail documents.

The solution's long-tail text assumption is confirmed. Its result-row safeguard is also confirmed as important: almost every own-study result is later than the eligibility seed. The assumed difficulty strata are measured only on 2017–2019 forward predictions, never used to tune against official validation labels.
