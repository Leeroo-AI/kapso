# RelBench campaign reference — agent results, baselines, hardware, status

**Auto-generated — do not edit by hand.** Regenerate with:
`PYTHONPATH=src:. python -m benchmarks.relbench.scorecard --reference`

Status: **2/66 tasks run**, 1 beating the best published number. Category-level gates: run the scorecard (same module, no flags).

## Hardware requirements

- **CPU-ok** (39 tasks): rel-f1 (1 MB), rel-salt (34 MB), rel-event (100 MB), rel-arxiv (145 MB), rel-avito (347 MB), rel-trial (548 MB db.zip). Runs on an 8-core / 32 GB box; the handler steers agents to duckdb/GBDT when no GPU is present.
- **GPU box** (26 tasks): rel-stack (840 MB, text-heavy), rel-hm (31M-row transactions), rel-ratebeer (2.2 GB), rel-amazon (6.1 GB). CUDA GPU + 64 GB RAM recommended; 8h full-run caps.
- **Blocked** (1 task): rel-mimic needs credentialed PhysioNet + BigQuery access.
- The 'Cap' column is a **harness setting, not a benchmark rule** — RelBench imposes no time/compute limits (baselines range from RelAgent's ~1h/task to RelGT-AC's 22h runs). Our caps (2h/4h/8h full, 15/20/30 min debug, by DB tier) bound a single candidate run so the search always proceeds; override with RELBENCH_FULL_TIMEOUT / RELBENCH_DEBUG_TIMEOUT.

## Baselines

Verified primary-source numbers (see BASELINES.md for protocols and citations): **RelAgent** (arXiv:2605.07840, val-selected test of 5 searches; v1 entity + 6-task v2 subset, no recommendation), **KumoRFM fine-tuned** (Kumo tech report Tables 2-4, single values, all 30 v1 tasks), full board field in `data/leaderboard_snapshot.json`, per-task best-known in `data/sota.json`.

## Per-task table (ROI order)

Values in the best-known number's units (AUROC/acc/MAP in %, NMAE, R², raw MAE). 'Best known' = strongest published result anywhere (board ∪ papers).

| ROI# | Task | Fam | Best known (method) | RelAgent | KumoRFM-ft | Kapso | vs best | HW | Cap | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rel-event/user-attendance | reg | 0.0303 (RT-ft) | 0.315 | 0.311 | — | — | CPU-ok | 2h | · pending |
| 2 | rel-f1/driver-circuit-compete | rec | 76.2 (ID-GNN-4L) | — | — | 86.3 | ✅ beats best-known | CPU-ok | 2h | ✅ done |
| 3 | rel-f1/results-position | AC-reg | 0.528 (RelGT-AC) | — | — | — | — | CPU-ok | 2h | · pending |
| 4 | rel-f1/qualifying-position | AC-reg | 0.239 (RelGT-AC) | — | — | — | — | CPU-ok | 2h | · pending |
| 5 | rel-f1/driver-position | reg | 0.374 (PluRel-ft) | 0.572 | 0.389 | 0.538 | below best-known | CPU-ok | 2h | ✅ done |
| 6 | rel-event/event_interest-interested | AC-bin | 49.6 (LightGBM) | — | — | — | — | CPU-ok | 2h | · pending |
| 7 | rel-event/event_interest-not_interested | AC-bin | 60.4 (GraphSAGE) | — | — | — | — | CPU-ok | 2h | · pending |
| 8 | rel-event/users-birthyear | AC-reg | -0.03 (GraphSAGE) | — | — | — | — | CPU-ok | 2h | · pending |
| 9 | rel-event/user-repeat | clf | 83.6 (GelGT) | 78.2 | 80.6 | — | — | CPU-ok | 2h | · pending |
| 10 | rel-event/user-ignore | clf | 91.2 (PluRel-ft) | 87.2 | 89.4 | — | — | CPU-ok | 2h | · pending |
| 11 | rel-f1/driver-dnf | clf | 84.6 (KumoRFM-2) | 78.3 | 82.6 | — | — | CPU-ok | 2h | · pending |
| 12 | rel-salt/sales-group | AC-mul | 15.8 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 13 | rel-salt/sales-payterms | AC-mul | 37.5 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 14 | rel-salt/sales-shipcond | AC-mul | 56.9 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 15 | rel-salt/sales-incoterms | AC-mul | 62.2 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 16 | rel-salt/item-incoterms | AC-mul | 69.4 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 17 | rel-trial/studies-enrollment | AC-reg | 0.436 (RelGT-AC) | — | — | — | — | CPU-ok | 4h | · pending |
| 18 | rel-trial/studies-has_dmc | AC-bin | 78.5 (RelGT-AC) | — | — | — | — | CPU-ok | 4h | · pending |
| 19 | rel-avito/searchinfo-isuserloggedon | AC-bin | 73 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 20 | rel-avito/searchstream-click | AC-bin | 55.9 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 21 | rel-trial/study-adverse | reg | 0.11 (RelAgent) | 0.11 | 0.13 | — | — | CPU-ok | 4h | · pending |
| 22 | rel-avito/ad-ctr | reg | 0.345 (RelAgent) | 0.345 | 0.355 | — | — | CPU-ok | 4h | · pending |
| 23 | rel-trial/study-outcome | clf | 94.6 (PluRel-ft) | 71.9 | 71.2 | — | — | CPU-ok | 4h | · pending |
| 24 | rel-trial/condition-sponsor-run | rec | 11.7 (ContextGNN / KumoRFM-ft) | — | 11.7 | — | — | CPU-ok | 4h | · pending |
| 25 | rel-avito/user-ad-visit | rec | 4.17 (KumoRFM-ft) | — | 4.17 | — | — | CPU-ok | 4h | · pending |
| 26 | rel-avito/user-clicks | clf | 69.4 (RGP) | 68.4 | 66.8 | — | — | CPU-ok | 4h | · pending |
| 27 | rel-trial/site-success | reg | 0.552 (RT-ft) | 0.811 | 0.632 | — | — | CPU-ok | 4h | · pending |
| 28 | rel-trial/site-sponsor-run | rec | 28 (ContextGNN / KumoRFM-ft) | — | 28 | — | — | CPU-ok | 4h | · pending |
| 29 | rel-ratebeer/user-count | reg | 0.625 (GraphSAGE) | 6.021 (MAE) | — | — | — | GPU box | 4h | · pending |
| 30 | rel-arxiv/author-publication | reg | 0.249 (GraphSAGE) | 0.462 (MAE) | — | — | — | CPU-ok | 4h | · pending |
| 31 | rel-ratebeer/beer-churn | clf | 78.7 (GraphSAGE) | 84.7 | — | — | — | GPU box | 4h | · pending |
| 32 | rel-ratebeer/brewer-dormant | clf | 80.5 (GraphSAGE) | 83.3 | — | — | — | GPU box | 4h | · pending |
| 33 | rel-arxiv/paper-citation | clf | 82.5 (GraphSAGE) | 82.6 | — | — | — | CPU-ok | 4h | · pending |
| 34 | rel-ratebeer/user-churn | clf | 94.3 (GraphSAGE) | 98.6 | — | — | — | GPU box | 4h | · pending |
| 35 | rel-ratebeer/beer_ratings-total_score | AC-reg | 0.394 (GraphSAGE) | — | — | — | — | GPU box | 4h | · pending |
| 36 | rel-ratebeer/user-beer-liked | rec | 1.46 (ID-GNN) | — | — | — | — | GPU box | 4h | · pending |
| 37 | rel-ratebeer/user-place-liked | rec | 1.85 (ID-GNN) | — | — | — | — | GPU box | 4h | · pending |
| 38 | rel-ratebeer/user-beer-favorite | rec | 1.89 (ID-GNN) | — | — | — | — | GPU box | 4h | · pending |
| 39 | rel-arxiv/author-category | mc | 50.7 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 40 | rel-arxiv/paper-paper-cocitation | rec | 35.4 (ID-GNN) | — | — | — | — | CPU-ok | 4h | · pending |
| 41 | rel-trial/eligibilities-adult | AC-bin | 93.7 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 42 | rel-trial/eligibilities-child | AC-bin | 87.2 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 43 | rel-stack/badges-class | AC-mul | 82.8 (GraphSAGE) | — | — | — | — | GPU box | 8h | · pending |
| 44 | rel-hm/transactions-price | AC-reg | 0.736 (GraphSAGE) | — | — | — | — | GPU box | 8h | · pending |
| 45 | rel-amazon/review-rating | AC-reg | -0.356 (GraphSAGE) | — | — | — | — | GPU box | 8h | · pending |
| 46 | rel-hm/user-churn | clf | 71.2 (KumoRFM-ft) | 71.1 | 71.2 | — | — | GPU box | 8h | · pending |
| 47 | rel-hm/item-sales | reg | 0.0686 (KumoRFM-ft/-2) | 0.0707 | 0.0686 | — | — | GPU box | 8h | · pending |
| 48 | rel-hm/user-item-purchase | rec | 3.14 (KumoRFM-ft) | — | 3.14 | — | — | GPU box | 8h | · pending |
| 49 | rel-stack/user-engagement | clf | 95.6 (PluRel-ft) | 90.4 | 90.7 | — | — | GPU box | 8h | · pending |
| 50 | rel-stack/user-badge | clf | 94.3 (PluRel-ft) | 88.4 | 89.9 | — | — | GPU box | 8h | · pending |
| 51 | rel-stack/post-votes | reg | 0.121 (Rel-LLM) | 0.125 | 0.127 | — | — | GPU box | 8h | · pending |
| 52 | rel-stack/user-post-comment | rec | 14 (RelGNN) | — | 13.3 | — | — | GPU box | 8h | · pending |
| 53 | rel-stack/post-post-related | rec | 12.5 (ID-GNN-4L) | — | 12.2 | — | — | GPU box | 8h | · pending |
| 54 | rel-amazon/user-churn | clf | 71.9 (Rel-LLM) | 70.8 | 70.5 | — | — | GPU box | 8h | · pending |
| 55 | rel-amazon/item-churn | clf | 83.4 (Rel-LLM / RT-ft) | 82.8 | 82.8 | — | — | GPU box | 8h | · pending |
| 56 | rel-amazon/user-ltv | reg | 0.242 (KumoRFM-2 in-context) | 0.243 | 0.247 | — | — | GPU box | 8h | · pending |
| 57 | rel-amazon/item-ltv | reg | 0.0696 (Data Scientist + LightGB) | 0.0707 | 0.0824 | — | — | GPU box | 8h | · pending |
| 58 | rel-amazon/user-item-purchase | rec | 2.93 (ContextGNN / KumoRFM-ft) | — | 2.93 | — | — | GPU box | 8h | · pending |
| 59 | rel-amazon/user-item-rate | rec | 2.25 (ContextGNN / KumoRFM-ft) | — | 2.25 | — | — | GPU box | 8h | · pending |
| 60 | rel-amazon/user-item-review | rec | 1.63 (ContextGNN / KumoRFM-ft) | — | 1.63 | — | — | GPU box | 8h | · pending |
| 61 | rel-avito/user-visits | clf | 78.3 (KumoRFM-ft) | 67.8 | 78.3 | — | — | CPU-ok | 4h | · pending |
| 62 | rel-f1/driver-top3 | clf | 99.6 (KumoRFM-ft) | 85.2 | 99.6 | — | — | CPU-ok | 2h | · pending |
| 63 | rel-salt/item-plant | AC-mul | 99.5 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 64 | rel-salt/item-shippoint | AC-mul | 98.4 (GraphSAGE) | — | — | — | — | CPU-ok | 4h | · pending |
| 65 | rel-salt/sales-office | AC-mul | 99.9 (either baseline) | — | — | — | — | CPU-ok | 4h | · pending |
| — | rel-mimic/patient-iculengthofstay | clf | 55 (GraphSAGE) | — | — | — | — | blocked | — | ⛔ credentialed data |

Notes: RelAgent/KumoRFM-ft columns show their per-task values in the same units where they published one (— where they did not evaluate). Current 'done' rows from harness-validation runs are baseline-quality placeholders until the campaign proper replaces them.

## Winning artifacts (durable, committed — for organizer handoff)

Each claimed cell's evidence is copied from the box-local run archive into `benchmarks/relbench/claims/<task>/`: winning code snapshot, both prediction files, solution spec, and final report (val+test metrics, audit). SHA-256 prefixes pin the exact prediction files the metrics were computed from.

| Task | Run | Evidence dir | val preds sha256 | test preds sha256 |
|---|---|---|---|---|
| rel-f1/driver-circuit-compete | run_0002 | `benchmarks/relbench/claims/rel-f1--driver-circuit-compete/` | `57fd5c25d36b42da` | `f4bb11e11b0e0c31` |
