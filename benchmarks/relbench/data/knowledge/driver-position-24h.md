# Campaign reward bar (this defines success)

KumoRFM (fine-tuned) — the strongest foundation-model baseline on this task —
scores **TEST MAE 2.731 (board NMAE 0.3887)**. That is the bar this campaign
is measured against:

- The campaign earns reward ONLY by SURPASSING KumoRFM-ft: a final
  val-selected TEST MAE strictly below 2.731.
- Any final result that does not beat 2.731 carries ZERO reward — a polished
  result at 3.3, 3.5, or 3.8 counts exactly the same as no result. Do not
  optimize for "best among results below the bar"; optimize for crossing it.
- The score you see in-loop is VALIDATION MAE; the bar lives on TEST. Prior
  campaigns on this task measured val→test drift of +0.8 to +1.3 MAE (the
  2010–2016 era shift): a validation win that does not survive that shift is
  worthless. Treat generalization across the test horizon as the primary
  design constraint, not an afterthought.
- STOP RULE (hard): an in-loop VALIDATION score is NEVER goal achievement,
  no matter how far it sits below 2.731 — that number is a TEST bar, and
  validation is a different split. Only a validation MAE at or below 1.9
  (= 2.731 minus the smallest drift ever measured on this task) could even
  in principle indicate the test bar has been reached; no stop verdict,
  "goal achieved" claim, or early termination is permitted while validation
  is above 1.9. Campaigns end on budget, not on validation optimism.

# Where you stand (all prior campaigns on this exact task + the leaderboard)

Five full campaigns preceded this one. Their OFFICIAL val→test finals:

| Campaign (10h each)                | val    | test MAE | board NMAE |
|------------------------------------|--------|----------|------------|
| classical K=2 baseline             | 2.7969 | 3.5988   | 0.5123     |
| external-leverage variant          | 2.7774 | 3.8472   | 0.5476 (worst drift: val gains did not transfer) |
| axis-contract variant              | 2.6486 | 3.6259   | 0.5161     |
| earlier cohort-rank ensemble       | 2.6260 | 3.5631   | 0.5072     |
| **immediately preceding (K=4 FE-primary)** | **2.6575** | **3.5308** | **0.5026 — the incumbent record you must beat** |

Read the pattern: best-val does NOT reliably mean best-test (the 2.6486
campaign lost on test to two worse-val campaigns; the 2.7774 one collapsed
to 0.5476). Drift is candidate-dependent — the record campaign won by
pairing good val WITH the lowest drift (+0.873). Chase transfer, not val
rank.

Published leaderboard context (NMAE, lower better): PluRel-ft 0.3745 and
RT-pretrained 0.3757 lead; **KumoRFM-ft 0.3887 is YOUR BAR**; RT zero-shot
0.4310; best from-scratch published is RT-from-scratch **0.4775** — the
next rung above you (~0.18 test MAE away); you already beat every classical
GNN (GelGT 0.5315, RelGNN 0.5406) and every published agent system
(RelAgent 0.5720). Every entry at or better than 0.4775 that is not you is
a pretrained foundation model; crossing the KumoRFM bar from scratch means
closing ~0.80 test MAE — the drift/decoding/cold-start headroom below is
where that much improvement could live.

# Inherited learning (the record campaign, in-run evidence)

The immediately preceding 10-hour campaign finished officially at
**validation MAE 2.6575 → hidden-test MAE 3.5308** (drift +0.873 — the
lowest official drift recorded on this task). Its full learning is
pre-loaded on this machine and is YOUR working capital:

- **The algorithm that worked** (the official champion): a weighted blend
  (0.290 / 0.357 / 0.353) of three all-table candidates from one family —
  temporally censored, staleness-replay training (features recomputed at
  frozen cutoffs 0–6 years before each seed so the model learns the exact
  signal decay the frozen-2009→2016 test imposes), a ~984-column
  nine-table feature matrix (multi-horizon form, circuit
  attrition/geography, race_round season phase, constructor/standings
  strength, teammate-relative signals, cohort/rookie priors), LightGBM
  L1/quantile heads plus a scale-free within-window rank head, and one
  GPU-trained sequence model. Beating THIS blend is your first milestone;
  its prediction arrays are reproducible from the recipe above.
- **features_history.md (111 entries, in $KAPSO_SHARED_CACHE_DIR)**: the
  complete PROPOSED/TESTED-KEPT/TESTED-REJECTED map of every feature idea
  the previous campaign tried, with measured outcomes. Read it BEFORE
  proposing features; do not re-buy its rejections; extend it as you work.
- **table_information.md (21KB, same location)**: accumulated schema
  semantics, including hard-won facts like the statusId/lap-share
  completion discovery (raw `statusId != 1` mislabels 5,758
  ≥90%-lap-share finishers as DNFs — use lap-share-based completion).
- Known headroom the previous campaign measured but did not close: its
  champion's recent pseudo-stale MAE was ~3.24 (pooled ~4.51) — the
  frozen-geometry gap is where the bar lives; oracle rank→value decoding
  reaches val ≈ 1.1 (large decoding headroom); ~43% of test rows are
  zero-history debutants (rising to ~64% by 2016).

This campaign has 24 hours, 8 parallel lanes, and no session caps: spend
that scale on what the previous campaign could not afford — deep
frozen-geometry training, the decoding headroom, and the cold-start mass —
not on rediscovering the banked map.
