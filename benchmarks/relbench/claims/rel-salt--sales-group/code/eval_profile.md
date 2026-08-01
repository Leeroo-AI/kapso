# Evaluation profile

## Mechanics

- The immutable `kapso_evaluation/kapso_eval.py` delegates to `grader.py`, which invokes `main.py` in an isolated subprocess and scores the complete 70,224-row validation table with `task.evaluate`.
- The headline selection metric is validation accuracy. Macro F1, micro F1, and MRR are also returned by the official evaluator.
- Full mode has a 14,400-second candidate timeout and archives both prediction arrays. The fraction and seed arguments do not subsample scored rows.
- Required outputs are finite float score matrices aligned to untouched task order: validation `(70224, 502)` and test `(83193, 502)`.
- Model A may use only training labels for validation predictions. Model B may add validation labels for test predictions.

## Input distribution

- Train has 340,491 rows over 502 classes, validation has 70,224 rows over 502 classes, and test has 83,193 rows.
- The database contains 500,908 accessible headers through the test prediction horizon and 2,319,540 accessible items; the official train and validation horizons contain 411,966 headers and 1,916,685 items.
- Every measured document has one distinct sold-to party, and all labeled rows join to a sold-to customer.
- Honest train-only last-customer-label validation coverage is 96.03%; accuracy is 88.41% overall and 92.02% on covered rows.
- Validation anchor error increases strongly with staleness, from roughly 0.3% when fresh to 31% beyond 180 days. New-customer rows are roughly 4% of validation and have weak cohort baselines.

## Coverage axes

- Customer history: repeat versus cold, historical label diversity, activity count, and anchor age.
- Change regime: same-label retention versus transition, transition origin label, and coarse time gap.
- Relational evidence: sold-to, payer, ship-to, bill-to, product, item category, geography, sales area, document type, company, and currency.
- Temporal drift: origin horizon and 14/30/60/90/180-day support or momentum.
- Order structure: item volume, product/category diversity, multi-party presence, and cross-role agreement.
- Ranking coverage: candidate hit versus noncandidate target and candidate-set size.

## Critical path

Candidate recall and pair-feature throughput bound achievable top-1 accuracy. Candidate cap is expanded from 40 to 64 if measured mean recall across internal forward origins is below 99%; query build rate is logged before full fitting.

## Debug measurements

- Current-item availability was verified: every task document joined to items whose minimum and maximum timestamps equal its header timestamp.
- On evenly spaced 3,334-query samples from the three train-only origins, recall at 40 was 94.63%, 95.77%, and 94.93%. The 99% assumed claim is contradicted, so the pipeline selected 64 candidates; measured recall at 64 was 94.93%, 96.22%, and 95.29%.
- Pair matrices contain 109 features per candidate and sustained 723–740 queries/s. Debug completed in 996.8 seconds and produced finite contract-valid arrays.
- Latest-origin debug accuracy by anchor stratum was 91.25% for age 0–30 days, 85.71% for 31–90, 76.25% for 91–180, 63.89% beyond 180, and 28.31% for cold customers.

## Full measurements

- Full 40-candidate recall was 95.18%, 96.05%, and 95.40% over the three complete train-only origins; cap-64 recall was 95.43%, 96.40%, and 95.80%.
- Full pair construction sustained 722–727 queries/s for 109 features. Internal selected rounds were 113; OOS top-1 was 90.16% on the 2019-04 origin and 86.94% on the 2019-09 origin with the selected 0.15 posterior blend.
- Registered run `run_0004` completed in 990.7 seconds. Official validation metrics were accuracy 0.8962605377, macro F1 0.7288526822, micro F1 0.8962605377, and MRR 0.9180114865.
- Seven feature/model variants and one earlier-origin variant were rejected solely from train-only origins; none satisfied improvement on both origins plus a mean gain of at least 0.002.
