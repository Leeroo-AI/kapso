# Evaluation and input profile

## Mechanics

- The immutable registered entrypoint is `kapso_evaluation/kapso_eval.py`; it delegates to `grader.py`, executes `main.py` in a subprocess, validates exact positional arrays, evaluates all 960 validation rows, and archives full runs. Fraction and seed do not subsample this task.
- The official selection metric is validation ROC-AUC. Average precision, accuracy, and F1 are also reported. Test labels are absent and test metrics remain hidden.
- Required arrays are float probabilities in `[0,1]`, with validation shape `(960,)` and test shape `(825,)`. Model A must not use validation labels; only Model B may refit with them.
- The evaluator directory is read-and-execute-only. This profile therefore remains at repository root rather than modifying `kapso_evaluation/`.

## Measured inputs

- Train/validation/test contain 11,994/960/825 rows and equally many distinct trials, with no duplicate `(timestamp, nct_id)` pairs. Train origins are annual from 2001 through 2019; validation and test have single origins at 2020-01-01 and 2021-01-01.
- Train and validation positive rates are 0.63757 and 0.58438. Validation is therefore lower by 5.32 percentage points; test label rate is unavailable.
- Trial-age median shifts from 675 days in train to 790.5 in validation and 872 in test. The 90th percentiles are 1,921/2,041/2,270.6 days, so prediction-era trials are modestly older.
- Brief titles and summaries are present for 100% of all splits. Detailed descriptions are present for 62.12%/67.29%/63.52%; median brief-summary lengths are 336/370/346 characters, and median detailed-description lengths including missing values are 505/706/605.
- Interventional studies account for 96.43%/97.40%/98.06%. Median enrollments are 152/147.5/165, while maxima are heavy-tailed at 1,260,576/25,732/117,649.
- The staged champion measured zero seed-own result history for every row, admissible pre-origin publication coverage near 39%, usable literature judgments at 6–7% for validation-era origins and 10.9% for test, covered validation AUC 0.9646, overall bootstrap SE about 0.016, and paired-delta SE about 0.003–0.004.

## Coverage axes

- Origin year and temporal regime.
- Linked versus unlinked external registry identity.
- Trial age, phase, study type, enrollment depth, facility/site depth, and intervention history.
- Document availability and length, admissible publication count, source type, complete versus year-only date, verified document version, full-text safety, and preprint status.
- Evidence tier: final matching report with explicit primary statistic; final matching report without one; interim/secondary/ambiguous report; absent evidence.
- Text-channel family and prediction rank correlation among relational, registry, hosted-protocol, and publication channels.

## Resolution and representativeness checks

- The core pipeline computes a 100-resample bootstrap SE on forward predictions and mean pairwise Spearman correlation before final prediction. Literature and registry gates use 2,000 paired resamples by trial ID.
- The single validation origin is lower-volume than the 2018/2019 internal gates and has a lower label rate than aggregate training. Candidate choice is therefore frozen on 2018 and confirmed once on 2019 rather than tuned to validation.

## Critical path

The bounded first-stage artifact is the number of mapped seed trials with qualifying primary p-values in the strictly preceding SHA-verified snapshot. The measured end-to-end snapshot stage processes about 220 linked rows/second, correcting the prior unsupported estimate above 1,000; observed qualifying coverage is 70/1,170 replay-2017, 65/1,128 official-2018, 60/1,111 replay-2018, 60/1,093 sealed-2019, 52/960 validation-era, and 84/825 test-era rows. The second-stage bound is conversion of the admissible-publication funnel into endpoint-matched evidence; uncapped hosted adjudication measured about 9.6 new rows/second at concurrency 32 after cached payload preparation.

## Direct-evidence strata

- Exact snapshot filters reproduce 1.000 covered ROC-AUC on all 65 official-2018 qualifying rows and 0.99543 on all 60 sealed-2019 qualifying rows.
- The conservative negative rule additionally requires complete result reporting and a qualifying analysis for every primary result endpoint; before routing, the historical snapshots expose 44/36 positive verdicts and 18/18 complete-negative verdicts at the 2018/2019 gates.
- The 2017 official origin has no declared strictly preceding projection and is therefore a zero-coverage, zero-delta control. Unlinked, incomplete, and temporally invalid rows preserve run_0005 exactly.

## Widening measurements

- The reference-supervised MedCPT reranker did not transfer. The selected pretrained cross encoder scored recall@8/MRR 0.97300/0.92857 on 2018 and 0.94899/0.86310 on sealed 2019, below the existing deterministic heuristic at 0.97896/0.96111 and 0.95137/1.00000. Its fine-tuned sibling was also lower, so paper ordering remains unchanged.
- JATS recon found 434/603/805/1,433 declared supplement packages across the 2018/2019/2020/2021 origins, but zero packages had a package-specific complete pre-origin date or version marker. Current attachments are therefore inadmissible; parsed package, table, and new-verdict counts are zero.
- The accepted official candidate scores 0.76320257 validation ROC-AUC. Its row bootstrap SE is 0.015118, while its paired improvement over run_0005 is 0.013110 with paired SE 0.004001 and P(improvement > 0) = 1.0. The direct-evidence slice contains 48 rows at 1.000 AUC; all other 912 rows retain run_0005 exactly and score 0.738759.
- Structured JATS recon parsed 5,201 origin-document instances from 2,570 cached XML files. Across origin instances it preserved 13,635 tables and 7,759 result sections and extracted 121,958 traceable statistical facts; 50-fixture origin panels cover nested headers, row/column spans, footnotes, Unicode inequalities, missing cells, and multi-arm layouts.
- Deterministic endpoint linkage produced only 1/6 new routes on official-2018/2019. Its sealed-origin delta was -0.001101 and paired P(improvement) was 0.3255, so it was rejected and generated byte-exact run_0006 fallback vectors.
- Uncapped hosted adjudication scored 17,027 complete safe-full-text windows or abstract fallbacks across six origins, including 9,829 newly called rows and 7,198 cache hits. Restricting final hosted-only evidence to exact-identity final primary reports labeled as clinical-trial, randomized-trial, or multicenter-study publications passed all four gates: deltas +0.001851/+0.000117 on official-2018/2019 and +0.014776/+0.019723 on replay controls, with clustered P(improvement) 0.8455. It adds 26 validation-era and 26 test-era routes at the forward-selected 0.99/0.01 interval.
