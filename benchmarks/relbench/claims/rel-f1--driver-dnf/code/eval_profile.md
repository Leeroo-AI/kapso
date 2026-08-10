# Evaluation profile

## Mechanics

- The immutable grader invokes `main.py` once for every rolling snapshot, first for 26 validation ticks and then for 29 test ticks.
- Each invocation sees an already-censored snapshot. Its task validation table is empty and its task test table contains only the current tick's rows. The candidate must write only `test_predictions.npy` for the tick.
- The grader concatenates tick vectors, restores official order through each snapshot's `indices.npy`, pools all 566 validation rows, and computes official RelBench metrics. Full and fast fidelity both score every validation row; `fraction` does not subsample.
- The primary metric is pooled ROC AUC. Average precision, accuracy, and F1 are also reported. Test labels are absent and test metrics remain hidden.
- The full timeout is 7,200 seconds for all 55 invocations together; debug is 900 seconds. The candidate controls its models and predictions but not scoring thresholds or aggregation.

## Input distribution

| Split | Rows | Origins | Rows/origin mean | Rows/origin range | Positive rate | Unique drivers |
|---|---:|---:|---:|---:|---:|---:|
| train | 11,411 | 420 | 27.17 | 10-65 | 0.8804 | 780 |
| validation | 566 | 26 | 21.77 | 20-24 | 0.7792 | 42 |
| test | 702 | 29 | 24.21 | 22-26 | unavailable | 42 |

Training origins span 1950-05-20 through 2004-10-03. Validation origins span 2005-03-02 through 2008-03-16. Test origins span 2010-03-02 through 2013-03-16. Training driver frequency is highly skewed: median 5 rows, mean 14.63, maximum 144. Validation cohort label rates range from 0.45 to 0.9565.

The static database contains 820 races and 20,323 result rows through 2009-11-01. Results have an exact finish rate of 0.2150; the three proposed classes contain 4,369 finish, 5,380 status-11-through-19, and 10,574 other rows. Qualifying begins only in 1994 and has 4,082 rows, so an availability indicator is required. Position, milliseconds, fastest-lap, and rank fields have substantial missingness.

## Coverage axes

- Seed-time cohort and calendar regime.
- Rich versus sparse driver history and rookie status.
- Current constructor, recent team switch, and team tenure.
- Driver, constructor, circuit, and driver-constructor reliability state.
- Qualifying and standings availability.
- Expected number of races in the next 30 days.
- Within-field relative strength and field dispersion.
- Off-season versus in-season cadence and circuit uncertainty.
- Validation temporal shift in class balance relative to long-run training.

The supplied coverage counts claimed 26 validation and 29 test origins, which match measurement. The task overview's statement of 40 evaluation timestamps is not the split-specific count exposed by the registered rolling harness. The score-bounding artifact is the legal prequential base-prediction cache: without enough recent closed origins, adaptation falls back to the fixed M1/M2 blend.

This profile is stored outside `kapso_evaluation/` because that directory is evaluator-owned and immutable under the task's anti-tampering rules.

## First full measurement

`run_0004` scored validation ROC AUC 0.8050249433, average precision 0.9346666527, accuracy 0.7985865724, and F1 0.8794926004. Slice results and the required resolution/representativeness diagnostics are recorded in `evaluation_results.md`.
