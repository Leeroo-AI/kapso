# RelBench v1 leaderboard — Kapso vs published baselines

Baselines transcribed from the official board at
<https://huggingface.co/spaces/relbench/leaderboard> (verified against the live
page on 2026-08-03; identical to our 2026-07-14 snapshot — no methods added since).
Kapso rows are our own runs, test metrics computed by the maintainer-registered
grader against pristine labels. Every Kapso run was budget-bound at 4–6h per
task on one 4xA100 box with the test split physically masked in-loop.
Classification cells updated 2026-08-10 after the evaluation-governance wave
(user-ignore, driver-top3, driver-dnf, study-outcome, user-engagement,
user-badge, user-visits re-runs).

The board ranks **31 tasks in three independent categories**; this document covers
the two the request asked for — Classification (12 tasks) and Regression (9).
Recommendation (10) is omitted.

**Headline:** Kapso ranks **2 of 28** on classification (mean AUROC 80.65)
and **4 of 26** on regression (mean NMAE 0.2608), beating the single
best published number on **2/12** classification and **6/9** regression tasks.

---

## 1. Classification

### Classification — test AUROC (%), higher is better

| # | Method | Regime | Mean | amazon/user-churn | amazon/item-churn | avito/user-visits | avito/user-clicks | event/user-repeat | event/user-ignore | f1/driver-dnf | f1/driver-top3 | hm/user-churn | stack/user-engagement | stack/user-badge | trial/study-outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | KumoRFM (fine-tuned) | task-specific | 81.1 | 70.5 | 82.8 | 78.3 | 66.8 | 80.6 | 89.4 | 82.6 | 99.6 | 71.2 | 90.7 | 89.9 | 71.2 |
| 2 | **Kapso (this work)** | agent | 80.7 | 71.4 | 83.1 | 67.8 | 70.2 | 81.2 | 88.9 | 83.3 | 93.6 | 71.6 | 91.5 | 89.3 | 75.9 |
| 3 | PluRel (pretrained + fine-tuned) | task-specific | 79.7 | 63.2 | 82.8 | 60.1 | 58.6 | 83 | 91.2 | 80.1 | 89.3 | 63.8 | 95.6 | 94.3 | 94.6 |
| 4 | KumoRFM-2 (in-context) | zero-shot | 79.6 | 69.1 | 82.2 | 69.4 | 67.4 | 81.7 | 90.8 | 84.6 | 92.2 | 69.3 | 89.4 | 87.2 | 72 |
| 5 | RT (pretrained + fine-tuned) | task-specific | 78.9 | 70.8 | 83.4 | 66.6 | 65.8 | 77.4 | 87.1 | 84.2 | 92.1 | 70.5 | 90.2 | 88.7 | 70.2 |
| 6 | GelGT | task-specific | 78.7 | 70.5 | 83 | 67 | 68.4 | 83.6 | 87.8 | 76.1 | 84.1 | 70 | 90.9 | 90.4 | 72.5 |
| 7 | RelAgent (GPT-5.2 agent) | task-specific | 78.4 | 70.8 | 82.8 | 67.8 | 68.4 | 78.2 | 87.2 | 78.3 | 85.2 | 71.1 | 90.4 | 88.4 | 71.9 |
| 8 | RGP | task-specific | 78.2 | 70.9 | 82.6 | 66.6 | 69.4 | 78.9 | 84.4 | 78.4 | 87.9 | 70.2 | 90.5 | 88.7 | 70.3 |
| 9 | RelGNN | task-specific | 78.1 | 71 | 82.6 | 66.2 | 68.2 | 79.6 | 86.2 | 75.3 | 85.7 | 70.9 | 90.8 | 89 | 71.2 |
| 10 | Rel-LLM (Llama-3.2-1B + GNN soft prompts, fine-tuned) | task-specific | 77.8 | 71.9 | 83.4 | 67 | 66.7 | 79.3 | 83.7 | 77.1 | 82.2 | 70.5 | 91.2 | 89.6 | 71 |
| 11 | RT (from scratch) | task-specific | 77.1 | 70.5 | 83.2 | 65 | 63.6 | 79.7 | 85.1 | 78.7 | 82.7 | 69.9 | 90 | 88.5 | 68.6 |
| 12 | KumoRFM (in-context) | zero-shot | 76.7 | 67.3 | 79.9 | 64.8 | 64.1 | 76.1 | 89.2 | 82.4 | 91.1 | 67.7 | 87.1 | 80 | 70.8 |
| 13 | RelGT | task-specific | 76.6 | 70.4 | 82.5 | 66.8 | 68.3 | 76.1 | 81.6 | 75.9 | 83.5 | 69.3 | 90.5 | 86.3 | 68.6 |
| 14 | RDL (GraphSAGE) | task-specific | 75.8 | 70.4 | 82.8 | 66.2 | 65.9 | 76.9 | 81.6 | 72.6 | 75.5 | 69.9 | 90.6 | 88.9 | 68.6 |
| 15 | GIN | task-specific | 75.2 | 70.5 | 82.7 | 66 | 66 | 74.4 | 79.5 | 71.8 | 73.6 | 69.9 | 90.5 | 88.7 | 68.4 |
| 16 | RDB-PFN (fine-tuned) | task-specific | 73.7 | 65.8 | 80.5 | 66 | 64.6 | 74.6 | 82.8 | 72.3 | 73.6 | 67.4 | 88.3 | 84.5 | 64.3 |
| 17 | RDB-PFN (ICL, 1,024-example context) | zero-shot | 73.2 | 64.8 | 78.2 | 65.5 | 62.7 | 75.3 | 82.7 | 71.9 | 81.2 | 66.5 | 86.6 | 81.3 | 61.6 |
| 18 | TabPFN-2.5 + DFS (ICL, 1,024-example context) | zero-shot | 72.9 | 64.5 | 79.6 | 61.7 | 63.3 | 73.1 | 83.2 | 71.7 | 80.4 | 66.8 | 85.3 | 82.1 | 62.6 |
| 19 | RELATE (RelGNN backbone) | task-specific | 72.8 | 68.9 | 81.2 | 66.2 | 66.1 | 67.1 | 81.1 | 68.9 | 69 | 69.4 | 90.1 | 86.6 | 58.4 |
| 20 | TabICL v1.1 + DFS (ICL, 1,024-example context) | zero-shot | 72.4 | 64.8 | 78.9 | 64.4 | 61.8 | 70 | 80.8 | 71.7 | 80.6 | 66.6 | 85.4 | 83 | 60.8 |
| 21 | HGT+PE (Laplacian positional encodings) | task-specific | 72.2 | 66.2 | 78 | 65 | 64.6 | 65.4 | 81.6 | 71.2 | 76.3 | 65.7 | 88.2 | 85.7 | 59.2 |
| 22 | HGT | task-specific | 71.8 | 66.4 | 78 | 64.3 | 63.8 | 65 | 82.5 | 70.8 | 70.8 | 67 | 88.5 | 86.1 | 58.4 |
| 23 | PluRel (synthetic + real) | zero-shot | 71.8 | 65 | 72.5 | 63.4 | 47.9 | 76 | 81 | 81 | 88.4 | 66 | 86.2 | 82 | 51.8 |
| 24 | RT (zero-shot, leave-one-DB-out) | zero-shot | 71.1 | 64 | 70.9 | 61.8 | 59.5 | 72.6 | 83.6 | 81.2 | 89.3 | 62.8 | 75.7 | 80.1 | 51.8 |
| 25 | GAT | task-specific | 70.8 | 63.2 | 70 | 64.8 | 65.8 | 68.2 | 82 | 70.3 | 60 | 64.7 | 89.6 | 84.5 | 66.2 |
| 26 | RELATE (HGT+PE backbone) | task-specific | 69.6 | 65.5 | 75.1 | 62.6 | 64.3 | 72.3 | 85.1 | 66.5 | 47.8 | 65.2 | 88 | 82.3 | 59.8 |
| 27 | PluRel (synthetic only) | zero-shot | 68.2 | 64.4 | 71 | 63.5 | 45.9 | 53.1 | 80.1 | 76.7 | 82.6 | 63.7 | 82.4 | 81.4 | 53.8 |
| 28 | LightGBM (raw entity features) | task-specific | 63.7 | 52.2 | 62.5 | 53 | 53.6 | 68 | 79.9 | 68.6 | 73.9 | 55.2 | 63.4 | 63.4 | 70.1 |

### Per-task vs the best published result

| Task | Best published | Method | Kapso | Δ |
|---|---|---|---|---|
| rel-amazon/user-churn | 71.9 | Rel-LLM (Llama-3.2-1B + GNN soft prompts, fine-tuned) | **71.4** | -0.5  |
| rel-amazon/item-churn | 83.4 | RT (pretrained + fine-tuned) | **83.1** | -0.3  |
| rel-avito/user-visits | 78.3 | KumoRFM (fine-tuned) | **67.8** | -10.5  |
| rel-avito/user-clicks | 69.4 | RGP | **70.2** | +0.8 ✅ |
| rel-event/user-repeat | 83.6 | GelGT | **81.2** | -2.4  |
| rel-event/user-ignore | 91.2 | PluRel (pretrained + fine-tuned) | **88.9** | -2.3  |
| rel-f1/driver-dnf | 84.6 | KumoRFM-2 (in-context) | **83.3** | -1.3  |
| rel-f1/driver-top3 | 99.6 | KumoRFM (fine-tuned) | **93.6** | -6  |
| rel-hm/user-churn | 71.2 | KumoRFM (fine-tuned) | **71.6** | +0.4 ✅ |
| rel-stack/user-engagement | 95.6 | PluRel (pretrained + fine-tuned) | **91.5** | -4.1  |
| rel-stack/user-badge | 94.3 | PluRel (pretrained + fine-tuned) | **89.3** | -5  |
| rel-trial/study-outcome | 94.6 | PluRel (pretrained + fine-tuned) | **75.9** | -18.7  |

---

## 2. Regression

### Regression — test NMAE = MAE / std(train targets), lower is better

| # | Method | Regime | Mean | amazon/user-ltv | amazon/item-ltv | avito/ad-ctr | event/user-attendance | f1/driver-position | hm/item-sales | stack/post-votes | trial/study-adverse | trial/site-success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RT (pretrained + fine-tuned) | task-specific | 0.233 | 0.2569 | 0.0804 | 0.4319 | 0.0303 | 0.3757 | 0.0948 | 0.1455 | 0.1275 | 0.5519 |
| 2 | PluRel (pretrained + fine-tuned) | task-specific | 0.237 | 0.2672 | 0.084 | 0.3923 | 0.0708 | 0.3745 | 0.0966 | 0.1472 | 0.124 | 0.5766 |
| 3 | KumoRFM (fine-tuned) | task-specific | 0.26 | 0.2474 | 0.0824 | 0.3554 | 0.311 | 0.3887 | 0.0686 | 0.1273 | 0.1304 | 0.6325 |
| 4 | **Kapso (this work)** | agent | 0.261 | 0.238 | 0.0655 | 0.334 | 0.315 | 0.344 | 0.0634 | 0.122 | 0.0872 | 0.778 |
| 5 | RelGNN | task-specific | 0.285 | 0.2475 | 0.0825 | 0.3867 | 0.311 | 0.5406 | 0.109 | 0.1273 | 0.1311 | 0.6325 |
| 6 | PluRel (synthetic + real) | zero-shot | 0.29 | 0.2852 | 0.1041 | 0.4182 | 0.0878 | 0.4835 | 0.1555 | 0.1654 | 0.1731 | 0.735 |
| 7 | KumoRFM-2 (in-context) | zero-shot | 0.291 | 0.2421 | 0.0795 | 0.3554 | 0.3071 | 0.4062 | 0.0686 | 0.1254 | 0.1277 | 0.9099 |
| 8 | RelGT | task-specific | 0.292 | 0.2481 | 0.0828 | 0.3606 | 0.327 | 0.5575 | 0.1082 | 0.1281 | 0.1297 | 0.6857 |
| 9 | GelGT | task-specific | 0.295 | 0.2479 | 0.0833 | 0.3784 | 0.3167 | 0.5315 | 0.1131 | 0.127 | 0.1255 | 0.7324 |
| 10 | RelAgent (GPT-5.2 agent) | task-specific | 0.296 | 0.2426 | 0.0707 | 0.3449 | 0.315 | 0.572 | 0.0707 | 0.1254 | 0.1097 | 0.8112 |
| 11 | KumoRFM (in-context) | zero-shot | 0.304 | 0.281 | 0.0935 | 0.3658 | 0.345 | 0.391 | 0.0808 | 0.1273 | 0.1717 | 0.8763 |
| 12 | Rel-LLM (Llama-3.2-1B + GNN soft prompts, fine-tuned) | task-specific | 0.311 | 0.245 | 0.0816 | 0.3867 | 0.328 | 0.5646 | 0.105 | 0.1215 | 0.1288 | 0.8343 |
| 13 | PluRel (synthetic only) | zero-shot | 0.311 | 0.3388 | 0.1154 | 0.4252 | 0.0878 | 0.5426 | 0.1749 | 0.18 | 0.1889 | 0.7457 |
| 14 | RT (from scratch) | task-specific | 0.316 | 0.259 | 0.0845 | 0.4064 | 0.504 | 0.4775 | 0.1001 | 0.1471 | 0.1306 | 0.7341 |
| 15 | Data Scientist + LightGBM | task-specific | 0.32 | 0.2422 | 0.0696 | 0.4599 | 0.3712 | 0.5641 | 0.0727 | 0.1273 | 0.1197 | 0.8553 |
| 16 | RDL (GraphSAGE) | task-specific | 0.32 | 0.2489 | 0.0847 | 0.4285 | 0.3372 | 0.5725 | 0.1131 | 0.1273 | 0.1311 | 0.8406 |
| 17 | GIN | task-specific | 0.321 | 0.249 | 0.0848 | 0.4285 | 0.345 | 0.5796 | 0.111 | 0.1273 | 0.1309 | 0.8364 |
| 18 | Data Scientist + AutoGluon | task-specific | 0.333 | 0.2504 | 0.0768 | 0.4703 | 0.3346 | 0.6051 | 0.0868 | 0.1332 | 0.1318 | 0.9036 |
| 19 | GAT | task-specific | 0.338 | 0.2891 | 0.0997 | 0.4494 | 0.3437 | 0.6075 | 0.1595 | 0.1332 | 0.1357 | 0.8259 |
| 20 | LightGBM (raw entity features) | task-specific | 0.341 | 0.2919 | 0.1025 | 0.4285 | 0.345 | 0.5935 | 0.1534 | 0.1332 | 0.1298 | 0.8931 |
| 21 | RT (zero-shot, leave-one-DB-out) | zero-shot | 0.346 | 0.3277 | 0.1029 | 0.6235 | 0.0662 | 0.431 | 0.1719 | 0.2128 | 0.2233 | 0.9552 |
| 22 | HGT | task-specific | 0.346 | 0.268 | 0.0945 | 0.4829 | 0.3444 | 0.6015 | 0.1294 | 0.133 | 0.1332 | 0.9305 |
| 23 | HGT+PE (Laplacian positional encodings) | task-specific | 0.35 | 0.2759 | 0.0945 | 0.5048 | 0.3412 | 0.6251 | 0.129 | 0.1332 | 0.1258 | 0.9238 |
| 24 | Griffin (fine-tuned) | task-specific | 0.369 | 0.3409 | 0.113 | 0.4526 | 0.4846 | 0.5596 | 0.1205 | 0.2733 | 0.1743 | 0.7988 |
| 25 | Entity Median | task-specific | 0.428 | 0.303 | 0.1124 | 0.4808 | 0.3516 | 1.2125 | 0.1575 | 0.1352 | 0.1708 | 0.9267 |
| 26 | Entity Mean | task-specific | 0.455 | 0.3314 | 0.1327 | 0.4808 | 0.3973 | 1.21 | 0.2241 | 0.2077 | 0.1708 | 0.9414 |

### Per-task vs the best published result

| Task | Best published | Method | Kapso | Δ |
|---|---|---|---|---|
| rel-amazon/user-ltv | 0.2421 | KumoRFM-2 (in-context) | **0.238** | -0.0041 ✅ |
| rel-amazon/item-ltv | 0.0696 | Data Scientist + LightGBM | **0.0655** | -0.0041 ✅ |
| rel-avito/ad-ctr | 0.3449 | RelAgent (GPT-5.2 agent) | **0.334** | -0.0109 ✅ |
| rel-event/user-attendance | 0.0303 | RT (pretrained + fine-tuned) | **0.315** | +0.285  |
| rel-f1/driver-position | 0.3745 | PluRel (pretrained + fine-tuned) | **0.344** | -0.0305 ✅ |
| rel-hm/item-sales | 0.0686 | KumoRFM (fine-tuned) | **0.0634** | -0.0052 ✅ |
| rel-stack/post-votes | 0.1215 | Rel-LLM (Llama-3.2-1B + GNN soft prompts, fine-tuned) | **0.122** | +0.0005  |
| rel-trial/study-adverse | 0.1097 | RelAgent (GPT-5.2 agent) | **0.0872** | -0.0225 ✅ |
| rel-trial/site-success | 0.5519 | RT (pretrained + fine-tuned) | **0.778** | +0.226  |

---

## Notes and caveats

- **Metrics.** Classification is test AUROC in percent (higher better). Regression is
  test NMAE, i.e. MAE divided by the standard deviation of the training targets
  (lower better) — the board's normalisation, so values are comparable across tasks.
- **Mean column.** For published methods the mean is the board's own. Kapso's mean is
  computed over the same task set; a method missing any task is ranked on the mean it
  reports, so cross-method mean comparisons are only exact where coverage is complete.
- **Compute is not matched.** The board imposes no compute limit and its entries range
  from roughly an hour per task to 22-hour runs. Kapso used a fixed 4h budget per task.
- **`rel-f1/driver-dnf` and `driver-top3`** are evaluated under the rolling seed-time
  regime (per-tick snapshots), which is the protocol the fine-tuned KumoRFM entries use.
  Most tasks are protocol-identical across regimes; these are not.
- **No test feedback.** Test labels are masked inside the sandbox, so no in-loop signal
  could come from them; model selection used validation only.
