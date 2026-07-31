# Evaluation profile

## Mechanics

- The registered full evaluator runs `python main.py`, enforces a 14,400-second candidate timeout, loads both NumPy artifacts with pickle disabled, validates finite validation predictions and row counts, and calls the official RelBench evaluator.
- Full fidelity scores all 21,183 validation rows. The `fraction` and `seed` arguments are manifest metadata and do not change the full-fidelity item set.
- The headline score is validation `roc_auc`; the evaluator also reports average precision, accuracy, and F1. Test predictions are archived but test labels and metrics are unavailable.
- Candidate code controls all feature construction, fitting, calibration, and inference. The harness controls the sanitized cache, output directory, timeout, contract validation, and official scoring.

## Input profile

| Split | Rows | Anchors | Anchor range | Positive rate |
|---|---:|---:|---|---:|
| train | 59,454 | 3 | 2015-04-26 to 2015-05-04 | 0.03872 |
| validation | 21,183 | 1 | 2015-05-08 | 0.03517 |
| test | 47,996 | 1 | 2015-05-14 | hidden |

The train anchor row counts are 19,547, 17,194, and 22,713. An independent future-window query matched every official train and validation row and label with zero mismatches. The daily April 26 through May 4 Model-A register contains 174,328 episodes; the April 26 through May 10 Model-B register contains 331,723.

Daily repeat-click prevalence ranges from 2.29% to 4.41%. Across the official anchors, future click-count means are 0.1855 to 0.2140 and variances are 0.3454 to 0.4606, establishing material overdispersion. Typical episodes contain about 19 to 20 future searches and 70 to 74 SearchStream exposures. The May 9 and May 10 episode distributions shift because their four-day windows approach the physical cutoff.

Validation has 25.70% users without prior searches and 16.92% without either prior searches or visits. Test is substantially colder: 51.05% have no prior search and 39.10% have neither prior search nor visit.

## Coverage axes

- Anchor date, weekday, history length, and proximity to the physical cutoff.
- Warm, search-cold, and search-and-visit-cold users.
- Historical search, impression, click, visit, and phone intensity over multiple horizons.
- Engagement propensity, funnel ratios, recency, active days, sessions, bursts, and trends.
- Device, OS, family, category, parent-category, location, city, price, and context cohorts.
- Sparse taste histories across searched, clicked, viewed, and phoned ads.
- Zero, one, and repeat future clicks; exposure intensity; and count overdispersion.

The solution's measured coverage claims agree with the profile. The only material caution is that the late Model-B daily anchors have truncated-looking intensity distributions even though their target endpoints remain at or before the database cutoff; model selection therefore uses purged forward folds and worst-fold ROC-AUC rather than a random split.

## Critical path

The bounded artifact is the reusable causal feature matrix, especially eight 64-dimensional decayed CountSketch channels over roughly 12.6 million user-event rows. The initial target rate is at least 8,000 seed rows per minute, checked after the core relational aggregates and after sketch snapshots. Model consumers are fitted only after that matrix is durable.
