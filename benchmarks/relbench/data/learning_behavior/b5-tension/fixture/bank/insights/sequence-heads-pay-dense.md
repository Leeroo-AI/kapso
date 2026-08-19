---
type: insight
title: Sequence heads pay on dense history
description: >-
  Standalone sequence models beat tabular fallbacks once per-entity event streams are long and dense.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: domain
scope_conditions: "per-entity event sequences are long (dense event streams at prediction time)"
evidence:
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-2.md
      card_version: null
    verdict: confirm
    usage: >-
      Never served; the campaign compared the standalone sequence head against the tabular fallback on its densest-history slice independently.
    effect: >-
      On rows with long event streams the sequence head overtook the fallback in the campaign's dense-slice comparison; the sparse-slice deficit did not reproduce there.
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
contradicts: [sequence-heads-lose-sparse]
probe: >-
  Probe: one gated forward-fold measurement.
---

When entities carry long, dense event streams, a sequence head extracts ordering and burst structure that tabular aggregates flatten away, and the standalone sequence model overtakes the fallback on exactly the dense-history slice [E1].
