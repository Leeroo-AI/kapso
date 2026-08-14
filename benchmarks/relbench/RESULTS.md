# RelBench campaign reference — agent results, baselines, hardware, status

**Auto-generated — do not edit by hand.** Regenerate with:
`PYTHONPATH=src:. python -m benchmarks.relbench.scorecard --reference`

Status: **65/66 tasks run**, 45 beating the best published number. Category-level gates: run the scorecard (same module, no flags).

## Benchmark version & leaderboard submission

Each task is tagged **v1** or **v2** in the `Ver` column. The public **relbench-hf submission leaderboard ranks 31 tasks** (marked ★): the 30 original v1 tasks — Classification (12) / Regression (9) / Recommendation (9) — plus the one v2 recommendation task the board carries (`rel-f1/driver-circuit-compete`), making 10 in Recommendation. The other 35 tasks (the 23-task autocomplete family + the v2-only databases rel-arxiv / rel-salt / rel-ratebeer / rel-mimic + extra v2 entity/rec tasks) are **not on the submission board** — they are tracked here for completeness against the v2 paper. **Submit v1**: the three categories are independent but each requires predictions for *all* its tasks (per `relbench.leaderboard.LEADERBOARD_TASKS`; package with `python -m relbench.leaderboard preds/ --package`, then open a submission issue on stanford-star/relbench@relbench-hf).

## Hardware requirements

- **CPU-ok** (39 tasks): rel-f1 (1 MB), rel-salt (34 MB), rel-event (100 MB), rel-arxiv (145 MB), rel-avito (347 MB), rel-trial (548 MB db.zip). Runs on an 8-core / 32 GB box; the handler steers agents to duckdb/GBDT when no GPU is present.
- **GPU box** (26 tasks): rel-stack (840 MB, text-heavy), rel-hm (31M-row transactions), rel-ratebeer (2.2 GB), rel-amazon (6.1 GB). CUDA GPU + 64 GB RAM recommended; 8h full-run caps.
- **Blocked** (1 task): rel-mimic needs credentialed PhysioNet + BigQuery access.
- The 'Cap' column is a **harness setting, not a benchmark rule** — RelBench imposes no time/compute limits (baselines range from RelAgent's ~1h/task to RelGT-AC's 22h runs). Our caps (2h/4h/8h full, 15/20/30 min debug, by DB tier) bound a single candidate run so the search always proceeds; override with RELBENCH_FULL_TIMEOUT / RELBENCH_DEBUG_TIMEOUT.

## Baselines

Verified primary-source numbers (see BASELINES.md for protocols and citations): **RelAgent** (arXiv:2605.07840, val-selected test of 5 searches; v1 entity + 6-task v2 subset, no recommendation), **KumoRFM fine-tuned** (Kumo tech report Tables 2-4, single values, all 30 v1 tasks), **KumoRFM-v1/v2 in-context** (zero-shot; v1: tech report, all 30 tasks; v2: arXiv:2604.12596, 21 entity tasks, no recommendation), full board field in `data/leaderboard_snapshot.json`, per-task best-known in `data/sota.json`.

## Evaluation protocol

Per-task temporal regimes are documented in `EVALUATION_PROTOCOL.md` — the single authority on what data a solution may see at test time. Where RelBench's library default (freeze the database at `test_timestamp`) and the bar-setters' released evaluation (KumoRFM: full database, per-row seed-time anchoring) disagree, this campaign adopts the KumoRFM regime — its numbers are the comparison target. ⚠ marks the tasks where the regimes diverge (multi-tick rel-f1 windowed tasks): these are evaluated through the **rolling harness** (per-tick snapshot cascade; verified 2026-07-29 — the reference model reproduced its hand-run seed-time score, test MAE 2.6516 vs the 2.731 bar). Kapso cells recorded before that date were frozen-regime numbers and undersell the seed-time score.

## Per-task table (v1 then v2; ROI order within each)

Values in the best-known number's units (AUROC/acc/MAP in %, NMAE, R², raw MAE). 'Best known' = strongest published result anywhere (board ∪ papers).

'Ver' = benchmark version; ★ = one of the 31 tasks on the relbench-hf submission leaderboard; ⚠ = evaluation-regime-sensitive (see `EVALUATION_PROTOCOL.md`).

| ROI# | Task | Fam | Ver | Best known (method) | RelAgent | KumoRFM-ft | KumoRFM-v1 (ic) | KumoRFM-v2 (ic) | Kapso | vs best | HW | Cap | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rel-event/user-attendance | reg | v1★ | 0.307 (KumoRFM-2) | 0.315 | 0.311 | 0.345 | 0.307 | 0.315 | below best-known | 4xA100 | 6h | ✅ done |
| 5 | rel-f1/driver-position ⚠ | reg | v1★ | 0.374 (PluRel-ft) | 0.572 | 0.389 | 0.391 | 0.406 | 0.344 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 9 | rel-event/user-repeat | clf | v1★ | 83.6 (GelGT) | 78.2 | 80.6 | 76.1 | 81.7 | 81.2 | below best-known | 4xA100 | 6h | ✅ done |
| 10 | rel-event/user-ignore | clf | v1★ | 91.2 (PluRel-ft) | 87.2 | 89.4 | 89.2 | 90.8 | 88.9 | below best-known | 4xA100 | 6h | ✅ done |
| 11 | rel-f1/driver-dnf ⚠ | clf | v1★ | 84.6 (KumoRFM-2) | 78.3 | 82.6 | 82.4 | 84.6 | 83.3 | below best-known | 4xA100 | 12h | ✅ done |
| 21 | rel-trial/study-adverse | reg | v1★ | 0.11 (RelAgent) | 0.11 | 0.13 | 0.172 | 0.128 | 0.0872 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 22 | rel-avito/ad-ctr | reg | v1★ | 0.345 (RelAgent) | 0.345 | 0.355 | 0.366 | 0.355 | 0.334 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 23 | rel-trial/study-outcome | clf | v1★ | 94.6 (PluRel-ft) | 71.9 | 71.2 | 70.8 | 72 | 79.6 | below best-known | 4xA100 | 12h | ✅ done |
| 24 | rel-trial/condition-sponsor-run | rec | v1★ | 11.7 (ContextGNN / KumoRFM-ft) | — | 11.7 | 11.3 | — | 12.3 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 25 | rel-avito/user-ad-visit | rec | v1★ | 4.17 (KumoRFM-ft) | — | 4.17 | 4.02 | — | 4.2 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 26 | rel-avito/user-clicks | clf | v1★ | 69.4 (RGP) | 68.4 | 66.8 | 64.1 | 69.4 | 70.2 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 27 | rel-trial/site-success | reg | v1★ | 0.552 (RT-ft) | 0.811 | 0.632 | 0.876 | 0.91 | 0.778 | below best-known | 4xA100 | 6h | ✅ done |
| 28 | rel-trial/site-sponsor-run | rec | v1★ | 28 (ContextGNN / KumoRFM-ft) | — | 28 | 20.8 | — | 33.3 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 46 | rel-hm/user-churn | clf | v1★ | 71.2 (KumoRFM-ft) | 71.1 | 71.2 | 67.7 | 69.3 | 71.6 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 47 | rel-hm/item-sales | reg | v1★ | 0.0686 (KumoRFM-ft/-2) | 0.0707 | 0.0686 | 0.0807 | 0.0686 | 0.0634 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 48 | rel-hm/user-item-purchase | rec | v1★ | 3.14 (KumoRFM-ft) | — | 3.14 | 2.73 | — | 3.26 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 49 | rel-stack/user-engagement | clf | v1★ | 95.6 (PluRel-ft) | 90.4 | 90.7 | 87.1 | 89.4 | 91.5 | below best-known | 4xA100 | 6h | ✅ done |
| 50 | rel-stack/user-badge | clf | v1★ | 94.3 (PluRel-ft) | 88.4 | 89.9 | 80 | 87.2 | 89.5 | below best-known | 4xA100 | 8h | ✅ done |
| 51 | rel-stack/post-votes | reg | v1★ | 0.121 (Rel-LLM) | 0.125 | 0.127 | 0.127 | 0.125 | 0.122 | below best-known | 4xA100 | 6h | ✅ done |
| 52 | rel-stack/user-post-comment | rec | v1★ | 14 (RelGNN) | — | 13.3 | 11.8 | — | 13.1 | below best-known | 4xA100 | 6h | ✅ done |
| 53 | rel-stack/post-post-related | rec | v1★ | 12.5 (ID-GNN-4L) | — | 12.2 | 11.8 | — | 26.1 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 54 | rel-amazon/user-churn | clf | v1★ | 71.9 (Rel-LLM) | 70.8 | 70.5 | 67.3 | 69.1 | 71.6 | below best-known | 4xA100 | 12h | ✅ done |
| 55 | rel-amazon/item-churn | clf | v1★ | 83.4 (Rel-LLM / RT-ft) | 82.8 | 82.8 | 79.9 | 82.2 | 83.1 | below best-known | 4xA100 | 6h | ✅ done |
| 56 | rel-amazon/user-ltv | reg | v1★ | 0.242 (KumoRFM-2 in-context) | 0.243 | 0.247 | 0.281 | 0.242 | 0.238 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 57 | rel-amazon/item-ltv | reg | v1★ | 0.0696 (Data Scientist + LightGB) | 0.0707 | 0.0824 | 0.0935 | 0.0795 | 0.0655 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 58 | rel-amazon/user-item-purchase | rec | v1★ | 2.93 (ContextGNN / KumoRFM-ft) | — | 2.93 | 1.72 | — | 2.54 | below best-known | 4xA100 | 6h | ✅ done |
| 59 | rel-amazon/user-item-rate | rec | v1★ | 2.25 (ContextGNN / KumoRFM-ft) | — | 2.25 | 1.14 | — | 2.31 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 60 | rel-amazon/user-item-review | rec | v1★ | 1.63 (ContextGNN / KumoRFM-ft) | — | 1.63 | 0.22 | — | 2.95 | ✅ beats best-known | 4xA100 | 6h | ✅ done |
| 61 | rel-avito/user-visits | clf | v1★ | 78.3 (KumoRFM-ft) | 67.8 | 78.3 | 64.8 | 67.4 | 67.8 | below best-known | 4xA100 | 6h | ✅ done |
| 62 | rel-f1/driver-top3 ⚠ | clf | v1★ | 99.6 (KumoRFM-ft) | 85.2 | 99.6 | 91.1 | 92.2 | 93.6 | below best-known | 4xA100 | 6h | ✅ done |
| 2 | rel-f1/driver-circuit-compete | rec | v2★ | 76.2 (ID-GNN-4L) | — | — | — | — | 87.9 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 3 | rel-f1/results-position | AC-reg | v2 | 0.528 (RelGT-AC) | — | — | — | — | 0.927 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 4 | rel-f1/qualifying-position | AC-reg | v2 | 0.239 (RelGT-AC) | — | — | — | — | 0.607 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 6 | rel-event/event_interest-interested | AC-bin | v2 | 49.6 (LightGBM) | — | — | — | — | 74.6 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 7 | rel-event/event_interest-not_interested | AC-bin | v2 | 60.4 (GraphSAGE) | — | — | — | — | 97.8 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 8 | rel-event/users-birthyear | AC-reg | v2 | -0.03 (GraphSAGE) | — | — | — | — | 0.219 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 12 | rel-salt/sales-group | AC-mul | v2 | 15.8 (GraphSAGE) | — | — | — | — | 91.5 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 13 | rel-salt/sales-payterms | AC-mul | v2 | 37.5 (GraphSAGE) | — | — | — | — | 92.4 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 14 | rel-salt/sales-shipcond | AC-mul | v2 | 56.9 (GraphSAGE) | — | — | — | — | 80.1 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 15 | rel-salt/sales-incoterms | AC-mul | v2 | 62.2 (GraphSAGE) | — | — | — | — | 83.3 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 16 | rel-salt/item-incoterms | AC-mul | v2 | 69.4 (GraphSAGE) | — | — | — | — | 77.6 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 17 | rel-trial/studies-enrollment | AC-reg | v2 | 0.436 (RelGT-AC) | — | — | — | — | 0.00111 | below best-known | 4xA100 | 4h | ✅ done |
| 18 | rel-trial/studies-has_dmc | AC-bin | v2 | 78.5 (RelGT-AC) | — | — | — | — | 80.8 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 19 | rel-avito/searchinfo-isuserloggedon | AC-bin | v2 | 73 (GraphSAGE) | — | — | — | — | 91.9 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 20 | rel-avito/searchstream-click | AC-bin | v2 | 55.9 (GraphSAGE) | — | — | — | — | 86 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 29 | rel-ratebeer/user-count | reg | v2 | 0.625 (GraphSAGE) | 6.021 (MAE) | — | — | — | 0.793 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 30 | rel-arxiv/author-publication | reg | v2 | 0.249 (GraphSAGE) | 0.462 (MAE) | — | — | — | 0.525 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 31 | rel-ratebeer/beer-churn | clf | v2 | 84.7 (RelAgent) | 84.7 | — | — | — | 82.3 | below best-known | 4xA100 | 4h | ✅ done |
| 32 | rel-ratebeer/brewer-dormant | clf | v2 | 83.3 (RelAgent) | 83.3 | — | — | — | 81.7 | below best-known | 4xA100 | 4h | ✅ done |
| 33 | rel-arxiv/paper-citation | clf | v2 | 82.6 (RelAgent) | 82.6 | — | — | — | 83.2 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 34 | rel-ratebeer/user-churn | clf | v2 | 98.6 (RelAgent) | 98.6 | — | — | — | 93.3 | below best-known | 4xA100 | 4h | ✅ done |
| 35 | rel-ratebeer/beer_ratings-total_score | AC-reg | v2 | 0.394 (GraphSAGE) | — | — | — | — | 0.405 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 36 | rel-ratebeer/user-beer-liked | rec | v2 | 1.46 (ID-GNN) | — | — | — | — | 2.81 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 37 | rel-ratebeer/user-place-liked | rec | v2 | 1.85 (ID-GNN) | — | — | — | — | 6.1 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 38 | rel-ratebeer/user-beer-favorite | rec | v2 | 1.89 (ID-GNN) | — | — | — | — | 5.36 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 39 | rel-arxiv/author-category | mc | v2 | 50.7 (GraphSAGE) | — | — | — | — | 52.5 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 40 | rel-arxiv/paper-paper-cocitation | rec | v2 | 35.4 (ID-GNN) | — | — | — | — | 32 | below best-known | 4xA100 | 4h | ✅ done |
| 41 | rel-trial/eligibilities-adult | AC-bin | v2 | 93.7 (GraphSAGE) | — | — | — | — | 98.5 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 42 | rel-trial/eligibilities-child | AC-bin | v2 | 87.2 (GraphSAGE) | — | — | — | — | 93.2 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 43 | rel-stack/badges-class | AC-mul | v2 | 82.8 (GraphSAGE) | — | — | — | — | 89.2 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 44 | rel-hm/transactions-price | AC-reg | v2 | 0.736 (GraphSAGE) | — | — | — | — | 0.96 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 45 | rel-amazon/review-rating | AC-reg | v2 | -0.356 (GraphSAGE) | — | — | — | — | 0.163 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 63 | rel-salt/item-plant | AC-mul | v2 | 99.5 (GraphSAGE) | — | — | — | — | 99.8 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 64 | rel-salt/item-shippoint | AC-mul | v2 | 98.4 (GraphSAGE) | — | — | — | — | 99.6 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| 65 | rel-salt/sales-office | AC-mul | v2 | 99.9 (either baseline) | — | — | — | — | 100 | ✅ beats best-known | 4xA100 | 4h | ✅ done |
| — | rel-mimic/patient-iculengthofstay | clf | v2 | 55 (GraphSAGE) | — | — | — | — | — | — | blocked | — | ⛔ credentialed data |

Notes: baseline columns show per-task values in the same units where the method published one (— where it did not evaluate). KumoRFM-ft is the fine-tuned regime (Kumo tech report Tables 2-4); KumoRFM-v1/v2 (ic) are the zero-shot in-context regimes (v1: tech report; v2: arXiv:2604.12596, cross-checked against RelAgent's KumoRFM-v2 rows — no published v1 recommendation numbers exist for v2). Current 'done' rows from harness-validation runs are baseline-quality placeholders until the campaign proper replaces them.

## Run artifacts (full traces)

Complete campaign state per run — session transcripts, lens/ideation history, every candidate run, workspace and logs — archived to durable storage at task completion:

| Task | Archive |
|---|---|
| rel-event/user-attendance | `gs://leeroo-kapso-relbench-artifacts/runs/rel-event--user-attendance/20260730T223947_lane-a1.tgz` |
| rel-f1/driver-position | `gs://leeroo-kapso-relbench-artifacts/runs/rel-f1--driver-position/20260730T221816_lane-a2.tgz` |
| rel-event/user-repeat | `gs://leeroo-kapso-relbench-artifacts/runs/rel-event--user-repeat/20260730T223847_lane-a3.tgz` |
| rel-event/user-ignore | `gs://leeroo-kapso-relbench-artifacts/runs/rel-event--user-ignore/20260809T215615_lane-c2.tgz` |
| rel-f1/driver-dnf | `gs://leeroo-kapso-relbench-artifacts/runs/rel-f1--driver-dnf/20260811T002323_lane-c9.tgz` |
| rel-trial/study-adverse | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--study-adverse/20260730T070548_lane-b.tgz` |
| rel-avito/ad-ctr | `gs://leeroo-kapso-relbench-artifacts/runs/rel-avito--ad-ctr/20260730T223905_lane-b3.tgz` |
| rel-trial/study-outcome | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--study-outcome/20260813T232753_lane-c12.tgz` |
| rel-trial/condition-sponsor-run | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--condition-sponsor-run/20260731T040617_lane-b2.tgz` |
| rel-avito/user-ad-visit | `gs://leeroo-kapso-relbench-artifacts/runs/rel-avito--user-ad-visit/20260731T042340_lane-b1.tgz` |
| rel-avito/user-clicks | `gs://leeroo-kapso-relbench-artifacts/runs/rel-avito--user-clicks/20260731T040738_lane-a1.tgz` |
| rel-trial/site-success | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--site-success/20260731T043724_lane-b3.tgz` |
| rel-trial/site-sponsor-run | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--site-sponsor-run/20260731T042130_lane-a3.tgz` |
| rel-hm/user-churn | `gs://leeroo-kapso-relbench-artifacts/runs/rel-hm--user-churn/20260731T092629_lane-a2.tgz` |
| rel-hm/item-sales | `gs://leeroo-kapso-relbench-artifacts/runs/rel-hm--item-sales/20260731T084234_lane-a1.tgz` |
| rel-hm/user-item-purchase | `gs://leeroo-kapso-relbench-artifacts/runs/rel-hm--user-item-purchase/20260731T095502_lane-a3.tgz` |
| rel-stack/user-engagement | `gs://leeroo-kapso-relbench-artifacts/runs/rel-stack--user-engagement/20260810T011603_lane-c5.tgz` |
| rel-stack/user-badge | `gs://leeroo-kapso-relbench-artifacts/runs/rel-stack--user-badge/20260812T021116_lane-c8.tgz` |
| rel-stack/post-votes | `gs://leeroo-kapso-relbench-artifacts/runs/rel-stack--post-votes/20260731T112323_lane-b1.tgz` |
| rel-stack/user-post-comment | `gs://leeroo-kapso-relbench-artifacts/runs/rel-stack--user-post-comment/20260801T045936_lane-a1.tgz` |
| rel-amazon/user-churn | `gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--user-churn/20260813T015420_lane-c10.tgz` |
| rel-amazon/item-churn | `gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--item-churn/20260731T211833_lane-b3.tgz` |
| rel-amazon/user-ltv | `gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--user-ltv/20260731T211904_lane-a2.tgz` |
| rel-amazon/item-ltv | `gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--item-ltv/20260731T211333_lane-a1.tgz` |
| rel-amazon/user-item-purchase | `gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--user-item-purchase/20260801T050453_lane-a2.tgz` |
| rel-amazon/user-item-rate | `gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--user-item-rate/20260801T051354_lane-a3.tgz` |
| rel-amazon/user-item-review | `gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--user-item-review/20260801T042832_lane-b2.tgz` |
| rel-avito/user-visits | `gs://leeroo-kapso-relbench-artifacts/runs/rel-avito--user-visits/20260809T191905_lane-c5.tgz` |
| rel-f1/driver-top3 | `gs://leeroo-kapso-relbench-artifacts/runs/rel-f1--driver-top3/20260810T221350_lane-c8.tgz` |
| rel-f1/driver-circuit-compete | `gs://leeroo-kapso-relbench-artifacts/runs/rel-f1--driver-circuit-compete/20260801T102223_lane-a1.tgz` |
| rel-f1/results-position | `gs://leeroo-kapso-relbench-artifacts/runs/rel-f1--results-position/20260801T102230_lane-a2.tgz` |
| rel-f1/qualifying-position | `gs://leeroo-kapso-relbench-artifacts/runs/rel-f1--qualifying-position/20260801T092104_lane-a3.tgz` |
| rel-event/event_interest-interested | `gs://leeroo-kapso-relbench-artifacts/runs/rel-event--event_interest-interested/20260801T091837_lane-b2.tgz` |
| rel-event/event_interest-not_interested | `gs://leeroo-kapso-relbench-artifacts/runs/rel-event--event_interest-not_interested/20260801T094124_lane-b3.tgz` |
| rel-event/users-birthyear | `gs://leeroo-kapso-relbench-artifacts/runs/rel-event--users-birthyear/20260801T124120_lane-b2.tgz` |
| rel-salt/sales-group | `gs://leeroo-kapso-relbench-artifacts/runs/rel-salt--sales-group/20260801T172646_lane-a3.tgz` |
| rel-salt/sales-payterms | `gs://leeroo-kapso-relbench-artifacts/runs/rel-salt--sales-payterms/20260801T180804_lane-b3.tgz` |
| rel-salt/sales-shipcond | `gs://leeroo-kapso-relbench-artifacts/runs/rel-salt--sales-shipcond/20260801T183418_lane-a1.tgz` |
| rel-salt/sales-incoterms | `gs://leeroo-kapso-relbench-artifacts/runs/rel-salt--sales-incoterms/20260801T183053_lane-a2.tgz` |
| rel-salt/item-incoterms | `gs://leeroo-kapso-relbench-artifacts/runs/rel-salt--item-incoterms/20260801T210445_lane-b1.tgz` |
| rel-trial/studies-enrollment | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--studies-enrollment/20260801T132017_lane-a3.tgz` |
| rel-trial/studies-has_dmc | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--studies-has_dmc/20260801T140047_lane-b3.tgz` |
| rel-avito/searchinfo-isuserloggedon | `gs://leeroo-kapso-relbench-artifacts/runs/rel-avito--searchinfo-isuserloggedon/20260801T140711_lane-a1.tgz` |
| rel-avito/searchstream-click | `gs://leeroo-kapso-relbench-artifacts/runs/rel-avito--searchstream-click/20260801T141721_lane-a2.tgz` |
| rel-ratebeer/user-count | `gs://leeroo-kapso-relbench-artifacts/runs/rel-ratebeer--user-count/20260801T163616_lane-b2.tgz` |
| rel-arxiv/author-publication | `gs://leeroo-kapso-relbench-artifacts/runs/rel-arxiv--author-publication/20260801T154918_lane-b1.tgz` |
| rel-ratebeer/beer-churn | `gs://leeroo-kapso-relbench-artifacts/runs/rel-ratebeer--beer-churn/20260801T214326_lane-b2.tgz` |
| rel-ratebeer/brewer-dormant | `gs://leeroo-kapso-relbench-artifacts/runs/rel-ratebeer--brewer-dormant/20260801T211054_lane-a3.tgz` |
| rel-arxiv/paper-citation | `gs://leeroo-kapso-relbench-artifacts/runs/rel-arxiv--paper-citation/20260801T220346_lane-b3.tgz` |
| rel-ratebeer/user-churn | `gs://leeroo-kapso-relbench-artifacts/runs/rel-ratebeer--user-churn/20260801T224425_lane-a1.tgz` |
| rel-ratebeer/beer_ratings-total_score | `gs://leeroo-kapso-relbench-artifacts/runs/rel-ratebeer--beer_ratings-total_score/20260801T224930_lane-a2.tgz` |
| rel-ratebeer/user-beer-liked | `gs://leeroo-kapso-relbench-artifacts/runs/rel-ratebeer--user-beer-liked/20260802T031228_lane-b1.tgz` |
| rel-ratebeer/user-place-liked | `gs://leeroo-kapso-relbench-artifacts/runs/rel-ratebeer--user-place-liked/20260802T030719_lane-b2.tgz` |
| rel-ratebeer/user-beer-favorite | `gs://leeroo-kapso-relbench-artifacts/runs/rel-ratebeer--user-beer-favorite/20260802T022831_lane-b3.tgz` |
| rel-arxiv/author-category | `gs://leeroo-kapso-relbench-artifacts/runs/rel-arxiv--author-category/20260802T135232_lane-b1.tgz` |
| rel-arxiv/paper-paper-cocitation | `gs://leeroo-kapso-relbench-artifacts/runs/rel-arxiv--paper-paper-cocitation/20260802T135822_lane-b2.tgz` |
| rel-trial/eligibilities-adult | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--eligibilities-adult/20260802T135231_lane-b3.tgz` |
| rel-trial/eligibilities-child | `gs://leeroo-kapso-relbench-artifacts/runs/rel-trial--eligibilities-child/20260802T135746_lane-a1.tgz` |
| rel-stack/badges-class | `gs://leeroo-kapso-relbench-artifacts/runs/rel-stack--badges-class/20260802T175235_lane-b1.tgz` |
| rel-hm/transactions-price | `gs://leeroo-kapso-relbench-artifacts/runs/rel-hm--transactions-price/20260802T174852_lane-b3.tgz` |
| rel-amazon/review-rating | `gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--review-rating/20260802T180339_lane-b2.tgz` |
| rel-salt/item-plant | `gs://leeroo-kapso-relbench-artifacts/runs/rel-salt--item-plant/20260802T175545_lane-a1.tgz` |
| rel-salt/item-shippoint | `gs://leeroo-kapso-relbench-artifacts/runs/rel-salt--item-shippoint/20260802T210247_lane-b3.tgz` |
| rel-salt/sales-office | `gs://leeroo-kapso-relbench-artifacts/runs/rel-salt--sales-office/20260802T205156_lane-b1.tgz` |

## Winning artifacts (durable, committed — for organizer handoff)

Each claimed cell's evidence is copied from the box-local run archive into `benchmarks/relbench/claims/<task>/`: winning code snapshot, both prediction files, solution spec, and final report (val+test metrics, audit). SHA-256 prefixes pin the exact prediction files the metrics were computed from.

| Task | Run | Evidence dir | val preds sha256 | test preds sha256 |
|---|---|---|---|---|
| rel-f1/driver-position | run_0006 | `benchmarks/relbench/claims/rel-f1--driver-position/` | `f7022e190f7c7418` | `a1596e24c0de8951` |
| rel-trial/study-adverse | run_0011 | `benchmarks/relbench/claims/rel-trial--study-adverse/` | `9dcd5a3c3adee77f` | `3575f8cbf767a3a6` |
| rel-avito/ad-ctr | run_0012 | `benchmarks/relbench/claims/rel-avito--ad-ctr/` | `ec04e7f063283739` | `63ea397f8086b0c2` |
| rel-trial/condition-sponsor-run | run_0014 | `benchmarks/relbench/claims/rel-trial--condition-sponsor-run/` | `a367ccf87ad3bc6e` | `3ec25479f4203ea2` |
| rel-avito/user-ad-visit | run_0022 | `benchmarks/relbench/claims/rel-avito--user-ad-visit/` | `5cd4aacd5655615b` | `6aed28f5feb1342b` |
| rel-avito/user-clicks | run_0008 | `benchmarks/relbench/claims/rel-avito--user-clicks/` | `6ee021943c3943cd` | `6c4afedde7a44d83` |
| rel-trial/site-sponsor-run | run_0013 | `benchmarks/relbench/claims/rel-trial--site-sponsor-run/` | `1b90414a1548572d` | `5710ab122ac38d71` |
| rel-hm/user-churn | run_0012 | `benchmarks/relbench/claims/rel-hm--user-churn/` | `bc8c10db0f23ed1f` | `246a6da23fe32be9` |
| rel-hm/item-sales | run_0009 | `benchmarks/relbench/claims/rel-hm--item-sales/` | `bf48cb6843c872ee` | `6b9793248fd71d7b` |
| rel-hm/user-item-purchase | run_0002 | `benchmarks/relbench/claims/rel-hm--user-item-purchase/` | `ce965643837b1a12` | `572056d90a59dafa` |
| rel-amazon/user-ltv | run_0006 | `benchmarks/relbench/claims/rel-amazon--user-ltv/` | `cc0faa6dc3cf5388` | `6d913d02825d7545` |
| rel-amazon/item-ltv | run_0019 | `benchmarks/relbench/claims/rel-amazon--item-ltv/` | `9674278f60a64d6c` | `386a6a3e13a71095` |
| rel-amazon/user-item-rate | run_0016 | `benchmarks/relbench/claims/rel-amazon--user-item-rate/` | `80abd22346425cb7` | `1ff1ca3335b6cd93` |
| rel-amazon/user-item-review | run_0012 | `benchmarks/relbench/claims/rel-amazon--user-item-review/` | `efc5f5219be860e7` | `95ad1203ef48f024` |
| rel-f1/driver-circuit-compete | run_0005 | `benchmarks/relbench/claims/rel-f1--driver-circuit-compete/` | `2e9e06d0d25f2e39` | `f204dc92faa0ecb6` |
| rel-f1/results-position | run_0012 | `benchmarks/relbench/claims/rel-f1--results-position/` | `f17a80cf508857f8` | `35fd384abb624cc0` |
| rel-f1/qualifying-position | run_0006 | `benchmarks/relbench/claims/rel-f1--qualifying-position/` | `d373f70391d1eac7` | `09a6f18f2d038a61` |
| rel-event/event_interest-interested | run_0002 | `benchmarks/relbench/claims/rel-event--event_interest-interested/` | `4ee5393609e60f14` | `52d10e76901aa469` |
| rel-event/event_interest-not_interested | run_0006 | `benchmarks/relbench/claims/rel-event--event_interest-not_interested/` | `13da1ef3ecc8106f` | `a32f14d245c1267b` |
| rel-event/users-birthyear | run_0018 | `benchmarks/relbench/claims/rel-event--users-birthyear/` | `e7ab8f93a82a9d69` | `12cc6449dd64c9c2` |
| rel-salt/sales-group | run_0009 | `benchmarks/relbench/claims/rel-salt--sales-group/` | `857248f92b72a78c` | `77795fe3c2962038` |
| rel-salt/sales-payterms | run_0012 | `benchmarks/relbench/claims/rel-salt--sales-payterms/` | `5dcba3df2a50760d` | `b52108d55d3a3f71` |
| rel-salt/sales-shipcond | run_0006 | `benchmarks/relbench/claims/rel-salt--sales-shipcond/` | `0eb3c6607bbe18f8` | `6e9f21815fb5f479` |
| rel-salt/sales-incoterms | run_0009 | `benchmarks/relbench/claims/rel-salt--sales-incoterms/` | `a98dbaf53d850a34` | `533a447d7864fe5a` |
| rel-salt/item-incoterms | run_0010 | `benchmarks/relbench/claims/rel-salt--item-incoterms/` | `a65c62dc78ca8935` | `865d949eecd69747` |
| rel-trial/studies-has_dmc | run_0010 | `benchmarks/relbench/claims/rel-trial--studies-has_dmc/` | `53c665cfec1e18e7` | `75011bb909977a9e` |
| rel-avito/searchinfo-isuserloggedon | run_0010 | `benchmarks/relbench/claims/rel-avito--searchinfo-isuserloggedon/` | `5af4047b60a59fe3` | `210d218b121db855` |
| rel-avito/searchstream-click | run_0013 | `benchmarks/relbench/claims/rel-avito--searchstream-click/` | `a77a5caf4590c149` | `c0bf9469894c3735` |
| rel-ratebeer/user-count | run_0015 | `benchmarks/relbench/claims/rel-ratebeer--user-count/` | `23f23e52c6d15771` | `e0adcaf4286c2cf2` |
| rel-arxiv/author-publication | run_0009 | `benchmarks/relbench/claims/rel-arxiv--author-publication/` | `562abd2067d9daa5` | `7846edc948cb0a67` |
| rel-arxiv/paper-citation | run_0020 | `benchmarks/relbench/claims/rel-arxiv--paper-citation/` | `e6b36be384af1be3` | `abf3fd762768dd5b` |
| rel-ratebeer/beer_ratings-total_score | run_0008 | `benchmarks/relbench/claims/rel-ratebeer--beer_ratings-total_score/` | `eabe486a28417a75` | `d20c9a28d2190783` |
| rel-ratebeer/user-beer-liked | run_0020 | `benchmarks/relbench/claims/rel-ratebeer--user-beer-liked/` | `76d62693a127c594` | `0d22d7471b6dfe1e` |
| rel-ratebeer/user-place-liked | run_0017 | `benchmarks/relbench/claims/rel-ratebeer--user-place-liked/` | `c4e88c35e6146eee` | `2f26ab4997548f0e` |
| rel-ratebeer/user-beer-favorite | run_0009 | `benchmarks/relbench/claims/rel-ratebeer--user-beer-favorite/` | `29ff0138d126fa7b` | `87b55c1deefbf983` |
| rel-arxiv/author-category | run_0018 | `benchmarks/relbench/claims/rel-arxiv--author-category/` | `f7d2f98bb81a3ef4` | `7e5e66e56ffb5cfa` |
| rel-trial/eligibilities-adult | run_0003 | `benchmarks/relbench/claims/rel-trial--eligibilities-adult/` | `44e6fa4cc090d51a` | `4270185d887d6ade` |
| rel-trial/eligibilities-child | run_0018 | `benchmarks/relbench/claims/rel-trial--eligibilities-child/` | `d3357655076100b2` | `30268d91d20ddd5d` |
| rel-stack/badges-class | run_0001 | `benchmarks/relbench/claims/rel-stack--badges-class/` | `c8a207267de01dc8` | `b44483fdab1ada6d` |
| rel-hm/transactions-price | run_0023 | `benchmarks/relbench/claims/rel-hm--transactions-price/` | `74cbf387c72b4588` | `ae38e4e46d32766a` |
| rel-amazon/review-rating | run_0022 | `benchmarks/relbench/claims/rel-amazon--review-rating/` | `b9cef3c784a10c4e` | `3e6b109d7798387b` |
| rel-salt/item-plant | run_0010 | `benchmarks/relbench/claims/rel-salt--item-plant/` | `0c58add2dd2d64d1` | `d35f2d20774818b9` |
| rel-salt/item-shippoint | run_0011 | `benchmarks/relbench/claims/rel-salt--item-shippoint/` | `116402af11cd440c` | `10484fc7d2d86308` |
| rel-salt/sales-office | run_0002 | `benchmarks/relbench/claims/rel-salt--sales-office/` | `fadaf1af7fba5254` | `81d903b1bd51be53` |
