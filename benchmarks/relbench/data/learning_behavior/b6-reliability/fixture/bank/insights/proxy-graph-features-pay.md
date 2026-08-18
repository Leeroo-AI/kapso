---
type: insight
title: Proxy graph features pay
description: >-
  Self-pooled graph-style features add blend value even without cross-entity message passing.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: domain
scope_conditions: "relational tasks where a cheap self-pooled graph proxy can join the blend"
evidence:
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-6.md
      card_version: null
    verdict: confirm
    usage: >-
      Served nowhere (fixture ledger); first confirming campaign.
    effect: >-
      An early proxy variant carried nonzero blend weight in the ratebeer campaign; first confirmation.
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-4.md
      card_version: null
    verdict: confirm
    usage: >-
      Served nowhere (fixture ledger); second confirming campaign.
    effect: >-
      A second proxy variant held blend weight; second confirmation.
reliability:
  validity: 0.67
  boundary: 0.4
  coverage: 0.3
  score: 0.62
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

A proxy that pools each entity's own rows in graph style contributes blend weight even without genuine cross-entity message passing [E1].
