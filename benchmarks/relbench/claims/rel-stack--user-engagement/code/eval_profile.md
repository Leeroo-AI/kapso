# Evaluation profile

## Mechanics

- The immutable evaluator launches `main.py` in a subprocess, validates full aligned NumPy vectors, calls the official task evaluator on all 85,838 validation rows, and ranks candidates by global validation ROC-AUC.
- Fast fidelity changes only the candidate build by adding `--debug`; `fraction` never subsamples scoring rows. Full fidelity archives both prediction vectors and the validation metrics.
- Validation predictions must come from a chain fitted without validation labels. Test predictions may come from an independent chain fitted on train plus validation labels.
- All four reported metrics are global: ROC-AUC, average precision, accuracy, and F1. The evaluator does not control candidate sampling, calibration, thresholds, architecture, seeds, or feature computation.

## Input distribution

- Train has 1,360,850 rows, 83,531 distinct eligible users at its last origin, 46 quarterly origins from 2009-04-16 through 2020-07-02, 68,020 positives, and 5.00% positives overall.
- The last four train origins contain 73,077, 75,412, 77,850, and 80,424 rows with positive rates 3.27%, 3.12%, 3.27%, and 3.41%; the final 2020-07 origin has 83,531 rows at 3.03%.
- Validation is one origin at 2020-10-01 with 85,838 rows, 2,411 positives, and 2.81% positives. Test is one origin at 2021-01-01 with 88,137 rows and hidden labels.
- The database contains 4,247,264 rows across seven tables. Before the validation cutoff it contains 324,981 posts, 1,278,841 votes, 247,398 users, 448,358 badges, 74,435 post links, 1,141,610 post-history events, and 606,124 comments. Every cutoff filter is a contiguous primary-key prefix.
- Text length is heterogeneous: median/90th-percentile lengths are 192/480 characters for comments, 378/2,080 for post history, 57/93 for titles, and 911/2,582 for post bodies.
- The exact label window is 91 days despite the prose description saying two years. Attributable vote users are extremely sparse, so own posts and comments dominate the observable target while received votes remain useful attention signals.

## Coverage axes and strata

- Seed era and temporal shift: early history, 2017-2019, 2020 internal origins, validation, and test.
- User-history density: one prior event, sparse, medium, and rich histories.
- Recency and trend: active in 30/91/365 days, inactive, accelerating, and decelerating.
- Activity modality: posts, comments, bounty-attributed votes, edits/history, badges, received attention, and linked-post neighborhoods.
- Graph context: isolated users, direct user-event neighborhoods, and two-hop post/community neighborhoods.
- Tenure, post type, badge class, attention type, and within-origin relative standing.
- Slice reporting will include count, positive count, ROC-AUC, and average precision wherever both classes occur.

## Solution assumptions checked

- Repeated-quarterly-panel, seven-table coverage, validation/test output sizes, global ROC-AUC mechanics, current GraphSAGE utilities, and cutoff prefixes are confirmed.
- Current RelBench 2.1.2 exposes `make_pkey_fkey_graph`, `get_node_train_table_input`, heterogeneous GraphSAGE, and temporal neighbor loading. The required `pyg_lib` backend was absent initially and installed from the matching PyG wheel index.
- MiniLM measured 3,433 short comments/s at batch size 256 on the assigned A100. Mixed long text is budgeted at 1,500 rows/s.
- Graph complementarity and standalone GNN strength remain empirical questions to be measured on the two internal origins. Temporal sampling correctness requires batch-level timestamp assertions.

## Critical path

The cutoff-keyed text/node encodings bound the graph score because neither GNN chain can train before they exist. The measured short-text rate projects about 10 minutes for 2.1 million rows, with a conservative mixed-text budget of 24 minutes; graph materialization and first-epoch timing are the next confirmation points.
