---
type: insight
title: Gated text blocks pay
description: >-
  Frozen-text feature blocks pass a delta gate inside the tree ensemble; standalone text models do not.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: domain
scope_conditions: "text-bearing relational tasks where embeddings can feed a tree model behind an explicit improvement gate"
evidence:
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-2.md
      card_version: null
    verdict: confirm
    usage: >-
      Served nowhere (fixture ledger); first confirming campaign.
    effect: >-
      Blending contributed while the standalone lane did not; recorded as the first confirmation.
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-2.md
      card_version: null
    verdict: confirm
    usage: >-
      Served nowhere (fixture ledger); second confirming campaign.
    effect: >-
      The gated block cleared its admission bar in a second campaign; second confirmation.
reliability:
  validity: 0.8
  boundary: 0.5
  coverage: 0.4
  score: 0.72
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

Text signal earns its place through the tree: a frozen embedding block admitted behind an explicit validation gate adds measured gain, while standalone text models lose to the gated ensemble [E1].
