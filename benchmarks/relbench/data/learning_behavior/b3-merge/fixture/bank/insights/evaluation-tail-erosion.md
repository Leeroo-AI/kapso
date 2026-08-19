---
type: insight
title: Evaluation tail erosion
description: >-
  Sessions that spend their declared evaluation reserve under the deadline lose the metric's tail quantiles first — the reserve's sample is load-bearing, not padding.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: domain
scope_conditions: "metric distributions with heavy tails evaluated under a shrinking budget reserve"
evidence:
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-4.md
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere (fixture ledger); the wave-4 fold-variance record is the anchor.
    effect: >-
      Fold scores varied while late evaluation ran under budget pressure; recorded as the fixture's tail-erosion instance.
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

Under a hard deadline, the declared evaluation reserve erodes the metric's tail quantiles first: the final evaluation spends a shrinking sample, the tail of the score distribution is the first loss, and the reported quantiles drift optimistic. The erosion is silent — each evaluation overrun looks locally justified, no single decision spends the sample, and the loss only materializes at the wall when the declared reserve cannot support the tail. The working rule: treat the final holdout's size as a hard interior floor carved before deadline pressure, and when drift is unavoidable re-plan it explicitly — shrinking scope, not the sample — so the tail quantiles survive the session's eroding budget.
