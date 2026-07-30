# Evaluation profile

The protected `kapso_evaluation/` directory was inspected but not modified. This profile is persisted at repository root to respect the evaluator's anti-tampering rule.

## Measurement mechanics

- Score of record: official RelBench validation `mae`, an unweighted mean absolute error across all 3,596 validation rows; lower is better.
- Additional official diagnostics: `r2` and `rmse`.
- Full fidelity invokes `python main.py` with a 14,400-second child timeout, checks finite shape-aligned NumPy arrays, scores validation predictions, archives both arrays, and emits `KAPSO_EVAL_MANIFEST`.
- Validation predictions must come from a train-only model. Test predictions may come from a separate train-plus-validation model.
- `fraction` and `seed` do not reduce the scored set at full fidelity.

## Input profile

| split | rows | unique trials | origins | zero rate | median | q75 | q90 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 43,335 | 43,335 | 2001–2019 annual | 40.79% | 2 | 15 | 55 | 566 | 28,085 |
| validation | 3,596 | 3,596 | 2020-01-01 | 39.18% | 2 | 18 | 82 | 1,059.2 | 17,245 |
| test | 3,098 | 3,098 | 2021-01-01 | hidden | hidden | hidden | hidden | hidden | hidden |

Trial identifiers do not overlap across splits. The target is extremely right-skewed, so MAE rewards conditional medians while RMSE is dominated by a small tail.

## Visibility profile

At their seed timestamps, validation trials have visible rows for designs, eligibilities, and sponsors in 100% of cases; conditions in 88.8%; facilities in 98.8%; and interventions in 48.3%. They have no visible same-trial reported-event, outcome, outcome-analysis, or withdrawal rows. Those result tables can contribute only as legally completed historical-cohort profiles.

## Coverage axes

- Annual origin and temporal regime shift.
- Enrollment scale and trial age.
- Phase, study type, protocol design, eligibility population, and missingness.
- Sponsor and agency support, including unseen or sparse entities.
- Condition and intervention memberships.
- Facility, country, state, and city membership/support, site diversity, and site growth.
- Historical result-profile availability and dispersion.
- Text length, vocabulary, and safety/death keywords.
- Target zero mass and heavy positive tail.

## Solution-coverage check

Measured visibility confirms the solution's cold-start information geometry. Text value remains an assumption to test with rolling train-only folds. The critical score-bounding artifact is the temporally legal feature matrix: without historical cohort transfer, all downstream models are limited to coarse enrollment and protocol medians.
