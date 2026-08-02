# Evaluation profile

## Scoring mechanics

The immutable grader runs `main.py` in an isolated subprocess, loads both aligned NumPy vectors, and calls the official RelBench evaluator on all 14,470 validation rows. The registered `fraction` and `seed` arguments are manifest metadata and do not change the scored rows. The primary score is validation ROC-AUC on raw probability rankings; average precision, accuracy, and F1 are also reported. Full runs have a 14,400-second candidate timeout, while debug runs have a 1,200-second timeout.

## Input distribution

The task has 234,366 training rows dated 2000–2019, 14,470 validation rows in 2020, and 23,430 test rows spanning 2021–2023. Training has 45,124 positives (19.25%); validation has 2,546 positives (17.60%). The two forward-fold holdouts contain 17,979 rows in 2018 and 16,578 rows in 2019. Annual training prevalence ranges from 31.65% in 2000 to 18.17% in 2019, confirming temporal prevalence drift.

The sanitized eligibility entity contains 273,160 unique IDs and one unique study per row. Among allowed eligibility inputs, sampling method is 79.99% missing and gender-based is 97.83% missing. Study-text word-count medians are 11 for brief title, 17 for official title, 58 for brief summary, and 94 for detailed description. Detailed description is 33.87% empty; its 90th, 95th, and 99th percentiles are 454, 675, and 1,443 words, so a flat 256-token view would frequently truncate it.

Raw relational coverage varies substantially: conditions cover 220,925 studies, interventions 94,944, facilities 251,089, outcomes 59,533, analyses 20,471, withdrawals 36,241, and reported events 59,343 out of 273,160 studies. Sponsors cover every study and designs cover 272,521. After the mandatory row-wise `relation.date <= seed.date` filter, only 221 studies expose outcomes, 101 expose analyses, 103 expose withdrawals, and 151 expose reported events at eligibility-entry time. This sharply contradicts any assumption that retrospective results tables provide broad seed-time coverage; they remain represented through explicit absence/count fields.

## Coverage axes and reporting strata

The observable axes are seed year, label prevalence, explicit pediatric/adult lexical evidence, study-text length and missingness, eligibility gender/volunteer fields, study type and phase, and presence/count of every relational table. Diagnostics will report internal fold ROC-AUC for 2018 and 2019 separately, transformer versus lexical residual, each fixed epoch checkpoint, and sample counts. The production validation diagnostics will additionally report counts and prediction distributions by year, lexical-evidence category, text-length band, and relational-coverage band without using validation labels to choose the design.

## Solution-coverage check

The supplied assumptions agree with the measured long-text and temporal-shift profile. The measured test window spans three years rather than one, strengthening the case for forward validation and Model-B refitting. The remaining unverified claim is whether rank-8 LoRA improves both temporal folds over the lexical residual; the production blend will be learned only from concatenated internal out-of-sample logits.
