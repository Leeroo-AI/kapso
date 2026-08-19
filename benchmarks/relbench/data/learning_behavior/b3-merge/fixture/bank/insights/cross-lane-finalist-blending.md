---
type: insight
title: Cross lane finalist blending
description: >-
  The dependable ensemble gain is the weakly correlated cross-branch finalist pair; blending near-identical siblings discards it.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: domain
scope_conditions: "a search holds multiple finalist candidates with widely varying pairwise prediction correlations"
evidence:
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-6.md
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere (fixture ledger); the ratebeer campaign's blend record is the anchor.
    effect: >-
      The graph lane's blend weight went to zero while the champion shipped the fallback; a cross-branch blend was never attempted — the stranded-gain pattern this card names.
reliability:
  validity: 0.6
  boundary: 0.4
  coverage: 0.3
  score: 0.6
  rationale: >-
    Fixture ledger: validity from the entries above; boundary and coverage as
    engineered for this scenario.
  state: active
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_behavior_fixture
    change: >-
      Authored as behavior-fixture material (truth.md records sourcing).
supersedes: null
contradicts: []
probe: >-
  Probe: one gated forward-fold measurement.
---

Ensemble gain comes from disagreement: rank-averaging finalists that correlate near one adds only shared noise, while two candidates within noise of each other that correlate weakly are the blend worth having. A search drifts into blending its own lineage — siblings share ancestry and correlate by construction — and the decorrelated finalist on another branch stays unread and unblended. The mechanism is variance reduction under disagreement, so the paying pairs are the cross-branch ones; a finalist family that never ships out-of-fold predictions cannot join the blend later, and that is how the gain strands.
