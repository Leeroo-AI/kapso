# Evaluation profile

## Mechanics

The maintained evaluator invokes the unchanged candidate five times, once for the official 2020 origin and once for each training-era origin in 2016-2019. It scores every validation row in every window and reports the unweighted mean of each official metric; the headline score is mean ROC-AUC. The official 2020 run alone supplies the archived 2021 test vector. Fidelity changes candidate build cost only, while fraction and seed remain manifest metadata.

Each historical cache truncates every dated database table at its validation origin, includes only training origins whose 365-day label horizon has closed, and exposes masked next-origin query membership. Model-A validation predictions must use only that cache's training labels. Model-B test predictions may additionally use that cache's validation labels.

## Input distribution

| Origin | Train rows | Validation rows | Test rows | Validation positive rate |
|---:|---:|---:|---:|---:|
| 2016 | 7,527 | 1,093 | 845 | 0.6093 |
| 2017 | 8,620 | 1,153 | 858 | 0.6288 |
| 2018 | 9,773 | 1,128 | 857 | 0.6108 |
| 2019 | 10,901 | 1,093 | 749 | 0.6029 |
| 2020 | 11,994 | 960 | 825 | 0.5844 |

The sequence directly measures temporal transfer as prevalence and cohort volume drift. Registration text is present broadly, while concept membership, sponsor evidence, and site density vary. Completion tables have no self-history before seed time and are used only as censored evidence from other trials.

## Coverage axes

- Validation origin and label-prevalence regime.
- Phase, study type, allocation, masking, enrollment, arms, and trial age.
- Protocol and eligibility text richness and truncation.
- Exact and ontology-shared sponsor, condition, intervention, facility, and geography history.
- Sparse versus rich historical support and low versus high site count.
- Cross-family prediction diversity and within-origin percentile rank.

## Measurement resolution

ROC-AUC depends only on ordering, so within-origin percentile ranks remove incompatible family calibration. Candidate combiners are selected through nested leave-one-origin-out evaluation over 2016-2019 and are never fitted or selected on 2020 labels. The gate requires a mean gain over Family 1 larger than both paired standard error and 0.003, positive gains in three origins, no 2019 regression, and no sparse-history or low-site loss above 0.005.

## Critical path

The score is bounded by complete, correctly aligned family predictions for every common training-era holdout. The exact champion is already preserved; warm build velocity is therefore measured as completed temporally censored windows per minute, and combiner consumers wait on row-fingerprint validation.

This profile is stored outside `kapso_evaluation/` because the provided evaluation tree is maintained, read-and-execute only, and must not be modified by candidate code.

## Measured combiner gate

The realized family rank correlations are 0.8402, 0.8554, and 0.8578. Nested simplex weights produced 2016-2019 ROC-AUCs of 0.71213, 0.70843, 0.72509, and 0.72564 versus Family 1 at 0.69435, 0.70001, 0.70815, and 0.71498. The mean gain is 0.01345 with paired-year SE 0.00231 and sponsor-blocked bootstrap SE 0.00323; all four years improve. Low-site and sparse-history gains are 0.01891 and 0.01807, so the gate selected locked weights 0.3/0.1/0.6 for Family 1/2/3.

The cached 2018 seed-17 LM vector matched the canonical row membership and length. A deterministic reproduction had correlation 0.97761 but a different byte hash and maximum absolute difference 0.16992, so reuse is validated by provenance, configuration, row fingerprint, and high predictive agreement rather than byte identity.
