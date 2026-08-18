---
type: insight
title: Sequence heads lose on sparse history
description: >-
  Standalone sequence models lose to compact tabular fallbacks when per-entity event history is short and sparse.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: domain
scope_conditions: "per-entity event sequences are short (few events per entity at prediction time)"
evidence:
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/it-1/flow-1.md
      card_version: null
    verdict: confirm
    usage: >-
      Served nowhere (fixture ledger); the campaign's registered sequence-vs-fallback comparison is the anchor.
    effect: >-
      The fully built GRU scored median forward AUC 0.685774 against the compact fallback's 0.691137 and was rejected after widening; the fallback shipped.
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
contradicts: [sequence-heads-pay-dense]
probe: >-
  Probe: one gated forward-fold measurement.
---

A discrete-time sequence head built on sparse per-entity histories underperforms the compact tabular fallback trained on the same information: with few events, the sequence model spends its capacity on padding and noise while the tabular aggregate already summarizes everything the history holds [E1].
