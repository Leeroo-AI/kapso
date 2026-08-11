# Evaluation measurement profile

The immutable evaluator runs `main.py` in a child process, checks the full prediction contract, and computes all official RelBench validation metrics. Full and fast fidelity differ only in whether `--debug` is passed; `fraction` and `seed` are manifest metadata and never subsample the 960 validation rows. The score of record is validation ROC AUC. Test predictions are archived but test labels and metrics are unavailable.

## Input distribution

| Split | Rows | Seed time range | Positive rate | Unique trials |
|---|---:|---|---:|---:|
| Train | 11,994 | 2001-01-05 to 2019-01-01 | 0.6376 | 11,994 |
| Validation | 960 | 2020-01-01 | 0.5844 | 960 |
| Test | 825 | 2021-01-01 | hidden | 825 |

Training cohorts increase from 2 rows in 2001 to roughly 1,100 annually from 2015 onward. The training positive rate falls from about 0.74 in 2006-2007 to 0.60 in 2019, so forward-chained rather than random validation is required. Model-A and model-B registration corpora contain 234,401 and 249,309 studies respectively. Validation and test trials are older at prediction time than the average training trial, and validation enrollment is lower than both train and test.

All five registration modalities are text or categorical/numeric summaries: titles/summaries, eligibility, design, conditions/interventions/sponsors, and sites. Completion evidence varies through outcome analyses, dropout reports, and adverse-event totals. Previous metadata profiling found no seed trial has its own completion-table rows before its seed time, so completion tables can only form histories of other trials. Sponsor history covers about 82% of validation trials, condition history about 98%, and intervention concepts are substantially sparser.

## Coverage axes

- Seed year and temporal regime, including declining class prevalence.
- Phase, study type, randomization/masking, enrollment, arms, and age eligibility.
- Text richness and missing eligibility/design/summary fields.
- Sponsor, condition, intervention, and site-history support.
- Retrieval history density, evidence age, analysis count, minimum p-value, dropout burden, and serious-event burden.
- Absolute structured values and within-seed-cohort ranks.

## Measurement uncertainty

With 960 validation examples and the observed class balance, the approximate ROC-AUC standard error is 0.016-0.018. Candidate selection and blend weights therefore use training-only forward folds. Validation is used once for final measurement and bootstrap uncertainty/slice reporting, never for fitting or architecture selection.

The complete five-field banks encoded at 1,151-1,417 chunks/second and finished in 37 minutes, confirming the runtime assumption with margin. Cached full inference plus model selection finishes well inside the four-hour contract.

The self-supervised-transfer assumption was only partly confirmed. Frozen contrastive pooling plus semantic and exact-link retrieval produced stable Stage-A OOF ROC AUC 0.7039, but chronological LoRA fusion scored 0.6742, trailing by 0.0297 versus paired uncertainty 0.00736. The required gate therefore retains Stage A. Its weakest training OOF strata are trials with 0-1 sites (AUC 0.672), sparse sponsor history (0.684), and Phase 4 (0.641); its strongest large stratum is Phase 3 (0.740).

The critical score-bounding artifact was the complete, temporally keyed field bank. It was completed before consumers as planned, and optional adaptation stopped after the forward-fold rejection rather than using validation feedback.

## Full evaluation and resolution diagnostic

The registered full evaluation archived `run_0008` with ROC AUC 0.693994, average precision 0.766500, accuracy 0.632292, and F1 0.736764. A 100-draw row bootstrap of the campaign-best distinct candidate (`run_0002`, AUC 0.706669) gave ROC-AUC standard error 0.015897 and a 95% percentile interval of 0.675944-0.740964.

Six non-duplicate archived full candidates have materially different predictions from the best run: Spearman correlations are 0.7941-0.8665 (mean 0.8433), and mean absolute probability differences are 0.0704-0.1145. Their scores span 0.67908-0.70667, entirely inside two bootstrap standard errors of the best run. The single validation window therefore has insufficient resolution to reliably rank these candidates; a same-contract multi-window re-measurement is requested.

Validation volume is 960 versus 1,093 in the surrounding 2019 training cohort and 825 in the prediction cohort. Its positive rate is 0.5844 versus 0.6029 in 2019, which is a moderate continuation of the measured drift rather than evidence of a calendar shock. Correctly aligned validation slices are: Phase 2 AUC 0.6626 (247 rows), Phase 3 0.7386 (271), Phase 4 0.6846 (75); trial age under one year 0.6883 (212), one-to-three years 0.7316 (394), over three years 0.6570 (354); sites 0-1 0.6788 (353), 2-20 0.6567 (249), over 20 0.7305 (358).
