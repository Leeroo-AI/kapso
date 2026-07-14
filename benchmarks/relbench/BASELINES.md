# Verified baselines: RelAgent and KumoRFM (fine-tuned)

Precision reference for the two baselines Kapso is benchmarked against, plus guidance on
which further baselines matter. Every number below was extracted from the primary source
documents on 2026-07-14 and cross-checked; nothing is quoted from secondary summaries.
Machine-readable copy: `data/baselines.json`.

## Sources and verification method

| Source | What was taken from it |
|---|---|
| RelAgent paper, arXiv:2605.07840 **v1** (submitted 2026-05-08; Huang, Tichelman, Kim, Olejniczak, Ceylan) — HTML full text | Tables 2, 5 (RelBench v1 main results), 3, 6, 15 (v2 subset), 13, 14 (per-task val/test/std + selected configs), protocol text |
| KumoRFM technical report (kumo.ai/research/kumo_relational_foundation_model.pdf; Fey, Kocijan, Lopez, Lenssen, Leskovec; May 2025, 18 pp.) | Tables 2, 3, 4 (v1 classification/regression/recommendation), fine-tuning protocol text |
| Official leaderboard (huggingface.co/spaces/relbench/leaderboard, static HTML, fetched 2026-07-14) | Board rows for both methods; NMAE convention |

Verification performed:

1. **Board ↔ paper identity.** Every board cell for both methods equals the paper value
   rounded to one decimal (12 classification cells × 2 methods checked individually).
2. **NMAE convention.** Board regression = MAE ÷ std(train targets). The implied divisor
   is identical (4 significant digits) across RelAgent, KumoRFM-ft, KumoRFM-in-context,
   and RDL rows for all 9 tasks, and for `rel-f1/driver-position` it equals the std
   computed directly from the actual task table: **7.0253** (ddof=0) vs implied 7.026.
3. **Regime disambiguation.** The "KumoRFM-v1/v2" rows inside RelAgent's own tables are
   the **in-context (zero-shot)** modes, not fine-tuned: they appear under RelAgent's
   "Zero-Shot Foundational" section and match Kumo's in-context column numerically
   (e.g. driver-position MAE 2.747 = in-context; fine-tuned is 2.731).

## Which benchmark version did they evaluate?

| Method | RelBench v1 (30 tasks) | RelBench v2 (66 tasks) | Other |
|---|---|---|---|
| **KumoRFM** (report, May 2025) | **Yes — all 30**: 12 classification + 9 regression + 9 recommendation, in-context and fine-tuned regimes | **No** (v2 released Feb 2026, after the report). KumoRFM-2 is a separate model; its published RelBench numbers are v1 (board) plus 6 v2 cells re-run *inside RelAgent's paper* | — |
| **RelAgent** (paper, May 2026) | **Partial — 21 of 30**: 12 classification + 9 regression. **No recommendation tasks** (the method targets entity prediction only) | **Small subset — 6 of 66**: 4 classification (rel-ratebeer beer-churn / user-churn / brewer-dormant, rel-arxiv paper-citation) + 2 regression (rel-ratebeer user-count, rel-arxiv author-publication) | 4DBInfer: 5 classification tasks |

So on the preferred v1: both methods are directly comparable on all 21 entity tasks;
only KumoRFM has v1 recommendation numbers.

## How each reports (read before comparing anything)

**RelAgent** — GPT-5.2 backbone + CAMEL agent framework + DuckDB; models are GBDT
variants only (LightGBM/XGBoost/CatBoost families). Per task: **5 independent searches ×
≤60 agent turns**; "cross-rollout model selection" picks the search with the highest
best-trial **validation** score; the **selected single program's test score** is what
the main tables report. Std of test scores across the 5 searches is in appendix Tables
13/14 (e.g. driver-top3 ±6.89, ad-ctr ±0.000). Classification = AUROC (%). Regression =
**raw MAE in the original target space** (all selected programs use an L1 objective;
log-transform flags documented per task); their regression average μn is normalized to
**LightGBM** (≠ board NMAE). For the v2 subset they additionally report mean±std over
3 repetitions of the whole selection procedure (Table 15).

**KumoRFM** — two regimes: *in-context* ("the model was not trained on these tasks or
datasets"; predictions from in-context examples with historical context only) and
*fine-tuned* ("specializing KumoRFM to a single dataset and a single task … following
the general RDL blueprint … supervised fashion based on a pre-generated training
table"; for recommendation this shifts inductive → transductive with shallow
embeddings). Classification = AUROC (%), regression = **raw MAE**, recommendation =
MAP@k (%) with the official per-task k. **Single values; no seeds or variance are
disclosed** anywhere in the report. The report's own regression aggregate is normalized
to **RDL** (fine-tuned 0.862), which is a third convention distinct from both RelAgent's
μn and the board's NMAE.

**Official board** — AUROC (%), NMAE = MAE ÷ std(train targets), MAP (%); test split;
values identical to the papers' after rounding. KumoRFM-ft: classification **#1
(mean 81.1)**, regression **#3 (0.2604)**. RelAgent: classification **#6 (78.4)**,
regression **#9 (0.2958)**. Neither appears on the recommendation board (it only
carries v2-paper baseline re-runs — KumoRFM-ft's rec numbers, though published and
SOTA on all 9 v1 rec tasks, were never added).

## RelBench v1 classification — test AUROC (%)

RDL (GraphSAGE) shown as the common reference; it agrees between both papers.

| Task | RelAgent | KumoRFM-ft | KumoRFM-ic | RDL |
|---|---|---|---|---|
| rel-f1/driver-dnf | 78.34 | **82.63** | 82.41 | 72.62 |
| rel-f1/driver-top3 | 85.23 | **99.62** | 91.07 | 75.54 |
| rel-avito/user-clicks | **68.36** | 66.83 | 64.11 | 65.90 |
| rel-avito/user-visits | 67.79 | **78.30** | 64.85 | 66.20 |
| rel-event/user-repeat | 78.20 | **80.64** | 76.08 | 76.89 |
| rel-event/user-ignore | 87.25 | **89.43** | 89.20 | 81.62 |
| rel-trial/study-outcome | **71.86** | 71.16 | 70.79 | 68.60 |
| rel-amazon/user-churn | **70.78** | 70.47 | 67.29 | 70.42 |
| rel-amazon/item-churn | **82.84** | 82.83 | 79.93 | 82.81 |
| rel-stack/user-engagement | 90.41 | **90.70** | 87.09 | 90.59 |
| rel-stack/user-badge | 88.42 | **89.86** | 80.00 | 88.86 |
| rel-hm/user-churn | 71.07 | **71.23** | 69.88 | 69.88 |
| **Average** | 78.38 | **81.14** | 76.71 | 75.83 |

Head-to-head: KumoRFM-ft 8, RelAgent 4 (and item-churn is a 0.01 margin). RelAgent's
wins are the low-signal/behavioral tasks (user-clicks, study-outcome, amazon churns) —
consistent with SQL-feature + GBDT strength where graph depth adds little.

## RelBench v1 regression — test raw MAE, with board NMAE in parentheses

Divisor column = std(train targets) implied by board÷paper, consistent across all
methods (driver-position verified against the actual data: 7.0253).

| Task | RelAgent | KumoRFM-ft | KumoRFM-ic | RDL | std divisor |
|---|---|---|---|---|---|
| rel-f1/driver-position | 4.019 (0.5720) | **2.731 (0.3887)** | 2.747 | 4.022 | 7.025 |
| rel-avito/ad-ctr | **0.033 (0.3449)** | 0.034 (0.3554) | 0.035 | 0.041 | 0.0957 |
| rel-event/user-attendance | 0.241 (0.3150) | **0.238 (0.3110)** | 0.264 | 0.258 | 0.765 |
| rel-trial/study-adverse | **37.194 (0.1097)** | 44.225 (0.1304) | 58.231 | 44.473 | 339.1 |
| rel-trial/site-success | 0.386 (0.8112) | **0.301 (0.6325)** | 0.417 | 0.400 | 0.4758 |
| rel-amazon/user-ltv | **13.949 (0.2426)** | 14.226 (0.2474) | 16.161 | 14.313 | 57.50 |
| rel-amazon/item-ltv | **41.765 (0.0707)** | 48.670 (0.0824) | 55.254 | 50.053 | 590.8 |
| rel-stack/post-votes | **0.064 (0.1254)** | 0.065 (0.1273) | 0.065 | 0.065 | 0.5106 |
| rel-hm/item-sales | 0.035 (0.0707) | **0.034 (0.0686)** | 0.040 | 0.056 | 0.4955 |
| **Board NMAE mean** | 0.2958 | **0.2604** | 0.3036 | 0.3204 | — |

Head-to-head by task count: RelAgent 5, KumoRFM-ft 4 — but the NMAE mean favors
KumoRFM-ft because driver-position and site-success are large normalized gaps.

## RelBench v1 recommendation — test MAP@k (%) — KumoRFM only

RelAgent has no recommendation results. Kumo's own baseline columns included for scale.

| Task | KumoRFM-ft | KumoRFM-ic | RDL two-tower | NBFNet | LightGBM |
|---|---|---|---|---|---|
| rel-amazon/user-item-purchase | **2.93** | 1.72 | 0.74 | 0.10 | 0.16 |
| rel-amazon/user-item-rate | **2.25** | 1.14 | 0.87 | 0.12 | 0.17 |
| rel-amazon/user-item-review | **1.63** | 0.22 | 0.47 | 0.09 | 0.09 |
| rel-avito/user-ad-visit | **4.17** | 4.02 | 0.02 | 3.66 | 0.06 |
| rel-hm/user-item-purchase | **3.14** | 2.73 | 0.80 | 2.81 | 0.38 |
| rel-stack/user-post-comment | **13.34** | 11.83 | 0.11 | 12.72 | 0.04 |
| rel-stack/post-post-related | **12.21** | 11.80 | 0.07 | 10.83 | 2.00 |
| rel-trial/condition-sponsor-run | **11.65** | 11.29 | 2.89 | 11.36 | 4.82 |
| rel-trial/site-sponsor-run | **28.02** | 20.83 | 10.70 | 19.00 | 8.40 |
| **Average** | **8.82** | 7.29 | 1.85 | 6.74 | 1.79 |

Note the official rec board's #1 (ID-GNN-4L, mean 14.0) covers a different 10-task set
including the v2 `rel-f1/driver-circuit-compete`, and was populated only with v2-paper
re-runs — KumoRFM-ft beats the board's numbers on most overlapping tasks but was never
listed. Beating *both* columns is required for a clean claim.

## RelAgent's v2 subset (for completeness; metric mismatch warning)

Test values of the selected run; Table 15 gives 3-repetition means±std in parentheses.

| Task | RelAgent | v2-paper GraphSAGE | KumoRFM-2-ic (RelAgent's re-run) |
|---|---|---|---|
| rel-ratebeer/beer-churn (AUROC) | **84.70** (84.55 ± 0.14) | 78.67 | 83.84 |
| rel-ratebeer/user-churn (AUROC) | **98.63** (98.54 ± 0.07) | 94.27 | 97.43 |
| rel-ratebeer/brewer-dormant (AUROC) | **83.33** (83.01 ± 0.24) | 80.51 | 80.65 |
| rel-arxiv/paper-citation (AUROC) | **82.62** (82.59 ± 0.03) | 82.50 | 81.71 |
| rel-ratebeer/user-count (MAE) | **6.021** | 7.374 | 7.298 |
| rel-arxiv/author-publication (MAE) | **0.462** | 0.513 | 0.487 |

⚠️ The v2 paper reports the two regression tasks in **R²**, RelAgent in **raw MAE** —
these are not interconvertible (R² is MSE-based). Kapso's harness computes r2/mae/rmse
every run, so we can be compared against both conventions; when targeting RelAgent use
MAE, when targeting the v2 paper use R².

## Caveats that matter when citing these numbers

- RelAgent reports the test score of a **validation-selected single run**; per-task
  variance across its 5 searches ranges from ±0.12 to ±6.89 AUROC. KumoRFM reports
  single values with **no variance disclosed**. Neither follows the v1 paper's
  5-seed-average convention.
- Validation→test drift is real and visible in RelAgent's appendix: e.g. driver-dnf
  val 66.44 → test 78.34, driver-top3 val 74.74 → test 85.23 (rel-f1 shifts hard),
  ratebeer/user-count val 4.367 → test 6.021.
- LLM memorization on rel-f1 is a documented concern (Kumo report: LLM baselines
  outperform only "where there exists clear leakage concerns (i.e. rel-f1)"). Any
  LLM-in-the-loop system (RelAgent, Kapso) should expect scrutiny on rel-f1 tasks.
- The LightGBM baseline columns differ between the two papers (different
  implementations); we do not rely on them here.
- Three different regression aggregates circulate: RelAgent's μn (÷LightGBM), Kumo's
  normalized average (÷RDL), the board's NMAE (÷train-std). Per-task raw MAE is the
  only safe common currency; the std divisors above convert to board NMAE.

## Are there other baselines worth adding? (incl. big tech)

**Big tech:** exactly one exists — **Griffin (Amazon, ICML 2025**, arXiv:2505.05568;
Amazon Shanghai/AWS authors incl. M. Wang, Q. Gan, D. Wipf). It is weak on RelBench:
board regression NMAE 0.3686 (worse than plain RDL 0.3204) and zero-shot classification
avg 66.29 in RelAgent's re-run. Include it only to say "we beat the big-tech entry";
it is not a competitive bar. **No Google/DeepMind, Microsoft, or Meta results on
RelBench exist as of July 2026** (verified by search; their tabular/graph FM work has
not published RelBench numbers).

**The bars that actually matter for #1 claims** (all academic/startup):

| Baseline | Why include | Board position |
|---|---|---|
| **RT (pretrained+fine-tuned)** — Stanford (RelBench team) | Regression board **#1** (0.2328); its user-attendance 0.0303 is the anomaly to study | clf #4 (78.9) |
| **PluRel (pretrained+fine-tuned)** — Stanford | Classification board **#2** (79.7); holds outlier bests on stack/trial tasks (94-96 AUROC) | reg #2 (0.2370) |
| **KumoRFM-2 (in-context)** — Kumo | Strongest zero-shot (clf 79.6); the freshest Kumo model | reg 0.2913 |
| **RelGNN** — Stanford/Kumo | Best pure supervised GNN — the "no pretraining, no agent" reference | clf 78.1 / reg 0.2854 |
| **Data Scientist + LightGBM** (v1 user study) | The human-expert bar; still holds item-ltv best (0.0696) | reg 0.3202 |
| **ContextGNN** — Kumo | The recommendation recipe to beat/reproduce (its numbers ≈ KumoRFM-ft rec) | not listed |

Suggested comparison set for the paper/leaderboard submission: RDL (canonical),
RelAgent (agent archetype), KumoRFM-ft (overall #1 clf), RT-ft (reg #1), RelGNN
(best GNN), DS+LightGBM (human), Griffin (big tech), plus the v2-paper GraphSAGE for
all v2/autocomplete tasks (de facto SOTA there).
