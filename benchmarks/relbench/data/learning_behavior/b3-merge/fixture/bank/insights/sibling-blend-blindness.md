---
type: insight
title: Sibling blend blindness
description: >-
  Blending near-identical siblings while leaving decorrelated cross-branch finalists unblended discards the one reliable ensemble gain available.
tags: [pitfall]
timestamp: 2026-08-18T01:30:00Z
scope: domain
scope_conditions: "a search produces multiple finalist candidates whose pairwise prediction correlations vary widely"
evidence:
  - source:
      learner_run: lr_founding_20260818
      trajectory: rel-amazon--user-churn/20260812T091402_lane-c10
      ref: mined/strategy.md#"0.9985"
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere (pre-bank campaign); the lens replanner recorded the blocker independently.
    effect: >-
      The campaign's stack precondition — a genuinely new expert — was never met because its finalists correlate 0.9985-0.9991, so the decisive decoding move stayed locked; an in-corpus instance of sibling saturation exercising the card.
reliability:
  validity: 0.6
  boundary: 0.4
  coverage: 0.2
  score: 0.55
  rationale: >-
    Validity from the in-corpus lens finding plus the measured practice observation it echoes; boundary (correlation threshold where blending stops paying) asserted at roughly 0.95 from prior practice, not yet carved in-corpus; coverage one dataset. Most score-moving next: one cross-branch rank-average gated on a forward fold.
  state: candidate
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_founding_20260818
    change: >-
      Founded from the second rel-amazon campaign's lens history [E1].
supersedes: null
contradicts: []
probe: >-
  List all archived finalists across branches with pairwise prediction rank-correlations; rank-average the weakest-correlated near-tied pair and gate it on one forward fold.
---

Ensemble gain lives in disagreement: rank-averaging candidates that correlate near one adds nothing but noise-sharing, while two finalists within noise of each other that correlate weakly are an ensemble waiting to happen [E1]. Searches drift into blending their own lineage — siblings share ancestry and correlate by construction — while the decorrelated finalist sits on another branch, unread. The mechanism is variance reduction under disagreement, so the paying pairs are exactly the cross-branch ones; a candidate family that never ships out-of-fold predictions cannot join the blend later, which is how the gain strands.
