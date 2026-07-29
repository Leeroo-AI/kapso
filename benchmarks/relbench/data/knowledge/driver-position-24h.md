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

# Inherited learning (previous campaign on this exact task, in-run evidence)

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
