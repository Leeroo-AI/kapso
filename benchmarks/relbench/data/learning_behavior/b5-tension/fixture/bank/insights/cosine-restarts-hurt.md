---
type: insight
title: Cosine restarts hurt fine-tuning
description: >-
  Warm cosine restarts destabilize fine-tuning against a flat schedule.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: domain
scope_conditions: "gradient fine-tuning with small validation sets"
evidence:
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-2.md
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere (fixture ledger); sub-noise single run.
    effect: >-
      A single opposing comparison inside measured fold noise; thin by construction.
reliability:
  validity: 0.4
  boundary: 0.3
  coverage: 0.1
  score: 0.45
  rationale: >-
    Fixture ledger: validity from the entries above; boundary and coverage as
    engineered for this scenario.
  state: candidate
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_behavior_fixture
    change: >-
      Authored as behavior-fixture material (truth.md records sourcing).
supersedes: null
contradicts: [cosine-restarts-help]
probe: >-
  Probe: one gated forward-fold measurement.
---

One run with warm cosine restarts lost to a flat schedule by a margin far inside the fold noise; no replication exists [E1].
