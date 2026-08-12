# Evaluation Profile

## Measurement mechanics

The immutable registered evaluator launches `main.py` in an isolated subprocess. Fast fidelity adds `--debug`, but both fidelities score all 960 official validation rows. `--fraction` and `--seed` are manifest metadata and do not subsample rows. The official score is validation `roc_auc`; all other official metrics are also reported. Validation predictions must be produced by Model A without validation-label fitting, while Model B may use train plus validation labels only for test predictions. Test labels are absent from the sanitized cache.

The protected evaluator tree is intentionally unchanged. This profile is at repository root because the task's anti-tampering rule forbids adding or editing any file under `kapso_evaluation/`.

## Input distribution

| Split | Rows | Entities | Positive rate | Origin structure |
| --- | ---: | ---: | ---: | --- |
| Train | 11,994 | 11,994 | 0.6376 | 19 yearly origins, 2001-01-05 through 2019-01-01 |
| Validation | 960 | 960 | 0.5844 | one origin, 2020-01-01 |
| Test | 825 | 825 | unavailable | one origin, 2021-01-01 |

Entity overlap is zero for every split pair, so this is a fully cold-entity task. Recent train label rates were 0.6288 in 2017, 0.6108 in 2018, and 0.6029 in 2019; validation falls further to 0.5844. Validation volume is 87.8% of 2019 train volume, while test volume is 85.9% of validation volume.

Title, brief-summary, and as-of eligibility coverage are 100% in every split. Median whitespace-token lengths for title / summary / eligibility are 34 / 50 / 172 in train, 33 / 54 / 205 in validation, and 34 / 50 / 233 in test. Eligibility exceeds 508 whitespace tokens for 12.4% / 17.0% / 17.9% of train / validation / test. The 95th-percentile eligibility lengths are 742 / 844 / 885, establishing that separate beginning and ending chunks cover a material stratum.

Study type is 96.4% interventional in train, 97.4% in validation, and 98.1% in test. The three largest phase strata are Phase 3, Not Applicable, and Phase 2. Phase 3 represents 31.8% / 28.2% / 31.6%; Not Applicable represents 22.3% / 28.9% / 25.8%; Phase 2 represents 22.2% / 25.7% / 24.1%.

As-of relation zero-history rates for condition links are 9.2% / 10.6% / 9.3%; intervention links are 37.3% / 53.4% / 49.7%; sponsor links are 0% throughout; facility links are 8.5% / 1.4% / 1.5%. Median facility counts rise from 3 in train to 6 in validation and 12 in test. Allocation missingness is stable at 10.8% / 11.6% / 9.2%.

## Coverage axes

The evaluation varies by phase, study type, title/summary/eligibility length, eligibility truncation, design-field missingness, sponsor-history depth, condition-history depth, relation coverage, trial age, therapeutic-area proxy, and encoder/LLM disagreement. Slice reporting will include the six solution-required axes: phase, text length, eligibility truncation, sponsor-history depth, respiratory/infectious-disease proxy, and LLM/encoder disagreement.

## Coverage claim checks

Universal text coverage is confirmed for title, summary, and eligibility. The assumed length profile is contradicted by the heavier validation/test eligibility tail, strengthening the need for head-and-tail section tokenization. Study summaries are stored on the `studies` row whose temporal semantics are `start_date`; dated eligibility, design, and relation rows are explicitly filtered to `date <= seed timestamp`. Earlier EDA measured that all validation/test trials have a qualifying design and eligibility row by the origin, supporting those as-of joins.

## Critical path

The score-bounding artifact is the out-of-sample section-encoder prediction matrix used to select the epoch and blend and to train the compact head. On the assigned A100, a four-section BiomedBERT training benchmark at local batch 8 and effective batch 24 measured 139.5 dossiers per second after warmup, with 4.97 GiB peak allocated memory. Seven three-epoch fits over roughly 12,000 dossiers therefore have a raw encoder lower bound near 30 minutes; data loading, validation inference, checkpointing, and hosted extraction are budgeted separately in `PLAN.md`.

## Full-run measurements

The first full run completed in 3,825.1 seconds and archived as `run_0007`, with official validation ROC-AUC 0.7307417385. It made 344 successful hosted calls covering 5,158 dossiers, with four retry attempts and no failed batches. Exact replay added 1,012 Model-A studies at positive rate 0.7490 and 1,098 Model-B studies at positive rate 0.7468.

BiomedBERT WordPiece lengths confirm a heavier tail than whitespace counts implied. Eligibility required head/tail truncation for 24.0% of train, 28.2% of validation, and 31.6% of test; median lengths were 254, 301, and 344 WordPieces, and 95th percentiles were 1,053, 1,230, and 1,230. This measured train-to-test shift is a principal text-length coverage axis.

Internal forward-fold ROC-AUC at the selected fourth epoch was 0.6983 / 0.7237 / 0.6764 for 2017 / 2018 / 2019. On the two folds supporting a leakage-safe compact head, encoder mean AUC was 0.7000, hosted logistic compact mean AUC was 0.7142, and the selected 75% compact rank blend reached 0.7177; its pooled improvement over the encoder was 0.01740 with paired-bootstrap SE 0.00547.

Official validation diagnostics were not used for selection. Encoder / compact / shipped-blend AUCs were 0.6901 / 0.7343 / 0.7307; encoder-to-compact rank correlation was 0.8271 and compact-to-blend correlation was 0.9893. The shipped validation AUC bootstrap SE was 0.01592. Materially decorrelated encoder and compact candidates differ by more than two SE, while the candidates inside two SE are highly correlated, so the resolution diagnostic does not confirm an evaluator defect. Validation volume and label rate (960, 0.5844) are not an isolated shock relative to 2019 train (1,093, 0.6029), so no evaluator-change request is warranted from the first-run evidence.

## Factorized-target recon

The exact official window predicates reproduce every one of the 11,994 training labels with zero mismatches. Qualifying-analysis count `K` has mean 2.8669, variance 97.3968, median 1, 90th percentile 5, 95th percentile 8, 99th percentile 24, and maximum 703; 44.63% of rows have more than one qualifying analysis. This supports separate significance and multiplicity heads, but the extreme overdispersion drives the raw moment estimate of negative-binomial `r` below the required 0.1 floor, so fold deviance against the simpler Poisson noisy-OR is a required selection diagnostic rather than an assumed win.

The six expanding OOF origins contain 990/1,044/1,093/1,153/1,128/1,093 official rows in 2014-2019. Their label rates range from 0.6029 to 0.6535, mean `K` ranges from 2.487 to 3.284, and multi-analysis prevalence ranges from 42.5% to 47.4%. Coverage gates therefore use origin-specific AUC plus sparse-sponsor and emerging-condition strata; replay weight is selected only on these pre-validation origins.

## Factorized full-run measurements

The internally selected configuration was pure negative-binomial noisy-OR with replay weight 0.25. Its 2014-2019 AUCs were 0.71486 / 0.72200 / 0.71313 / 0.69878 / 0.74385 / 0.71055, for mean 0.71720 and worst-two-origin mean 0.70466. It beat the compact binary reference on all six origins; the pooled delta was 0.01626 with paired bootstrap SE 0.00420 and `P(delta>0)=1.00`. Sparse-sponsor and emerging-condition mean deltas were +0.02518 and +0.01999.

Fold-only maximum-likelihood negative-binomial dispersions ranged from 0.7866 to 1.3710 at the selected replay weight. Mean negative-binomial deviance was 1.5143 versus Poisson deviance 5.3193, confirming the solution's dispersion assumption. The multiplicity head's mean Spearman correlation with `K` was 0.3967.

The mechanistic and reconstructed run_0007 OOF predictions had rank correlation 0.7271 on the shared 2018-2019 rows. A 50% origin-rank blend improved both origins and raised mean AUC from 0.72112 to 0.74054; its paired delta was 0.01984 with SE 0.00475 and `P(delta>0)=1.00`. This frozen blend produced official validation ROC-AUC 0.7449148718, average precision 0.8088521727, accuracy 0.6791666667, and F1 0.7038461538 in `run_0033`; bootstrap ROC-AUC SE was 0.01598. Sparse-sponsor validation AUC was 0.71931, no-sponsor-history AUC 0.75413, and respiratory/infectious AUC 0.71106.
