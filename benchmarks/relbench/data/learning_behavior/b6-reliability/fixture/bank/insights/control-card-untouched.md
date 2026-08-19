---
type: insight
title: Control card untouched
description: >-
  A control card whose ledger this batch does not touch.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: domain
scope_conditions: "any fixture run"
evidence:
  - source:
      learner_run: lr_behavior_fixture
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/index.md
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere (pre-bank fixture); recorded from the corpus.
    effect: >-
      Fixture founding entry; exercises the card without grading it.
reliability:
  validity: 0.55
  boundary: 0.4
  coverage: 0.2
  score: 0.55
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
contradicts: []
probe: >-
  Probe: one gated forward-fold measurement.
---

This card exists to verify untouched ledgers do not move [E1].
