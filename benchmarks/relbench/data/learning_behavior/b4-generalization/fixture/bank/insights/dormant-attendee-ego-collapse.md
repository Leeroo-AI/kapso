---
type: insight
title: Dormant attendees have no ego signal left
description: >-
  Dormant attendees have no ego signal left — the dormant slice bounds the score and ego features cannot reach it.
tags: []
timestamp: 2026-08-18T12:00:00Z
scope: [dataset:rel-event]
scope_conditions: "entities with long inactivity gaps at prediction time"
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

A attendee inactive for a long stretch has ego-history aggregates that collapse to zero or stale values, so the model falls back to the prior on exactly that slice; the remaining signal lives in the events and clubs around them, not in more of their own history. The score-bounding slice is the dormant attendee, and polishing ego features cannot reach it.
