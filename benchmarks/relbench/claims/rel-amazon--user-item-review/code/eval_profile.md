# Evaluation profile

## Mechanics

- The registered immutable entrypoint runs `main.py` in a child process and scores every one of the 116,970 validation rows; `fraction` does not subsample rows.
- The primary metric is RelBench `link_prediction_map` at ten ranked predictions. Each row is normalized by `min(number of distinct true products, 10)` and rows are averaged.
- Full runs require integer arrays shaped `(116970, 10)` and `(127021, 10)`. Candidate-side checks additionally enforce valid product range and within-row uniqueness.
- Validation predictions must come from a chain fit only on task-train labels. Validation labels are permitted only for the separate test-prediction refit.

## Distribution profile

- Database: 1,850,193 customers, 506,012 products, and 12,644,508 reviews from 2008-01-01 through 2016-01-01.
- Task: 2,324,177 train rows at quarterly origins, 116,970 validation rows at 2015-10-01, and 127,021 test rows at 2016-01-01.
- Train truth lists have mean size 2.21, median 1, and maximum 401; validation lists have mean 2.30, median 1, and maximum 353.
- The product catalog has nullable category and description fields, and users/items are strongly long-tailed. Training-only queries measure the operational strata below.

## Coverage axes and reporting

- User history: cold, 1-5 prior events, 6-20, and more than 20.
- Product history: zero prior events, 1-5, 6-50, and more than 50.
- Candidate source: brand, co-review, category/adjacent brand, series/title, semantic, recent/seasonal popularity, and global padding.
- Metadata: missing category, missing description, missing/rare brand, and price availability.
- Time: three forward origins with complete 91-day windows; results include candidate recall@200/800 and replay MAP count/mean/SD by fold and history stratum.
- Locale: metadata language is sampled without labels before enabling English BGE; hashed word/character TF-IDF is the prespecified fallback.

## Critical path

Candidate recall at the fixed 1,000-item cap bounds achievable MAP regardless of ranker quality. The first full projected-table and candidate replay measures seed throughput and recall growth; ranker and embedding consumers are built only after that artifact is operational.

## Measured training replay

- The projected event table built in 4.1 seconds and occupies 101.7 MB; cutoff snapshots build in 12-18 seconds and contain 2.9-3.6 million retained co-review edges.
- Candidate generation sustains roughly 2,000-2,500 seeds/minute with exactly 1,000 candidates per profiled row.
- At 2015-04-02 and 2015-07-02 debug origins, recall@800 was approximately 0.252 and 0.251 without semantic retrieval. BGE semantic retrieval reduced recall@800 by 0.0053 and 0.0030, so Stage C was rejected by its prespecified all-fold stability gate.
- Training-only source-blend checks on independently sampled April and July rows favored a brand-heavy reciprocal-rank blend: MAP changed from 0.02541/0.01565 to 0.02596/0.01750. This blend is fixed before official full scoring.
- Recall@800 by user-history stratum is weakest for cold users (roughly 0.10) and strongest for 6-20 or more than 20 historical events (roughly 0.26-0.31); this is the principal expected loss slice.
