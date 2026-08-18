---
type: insight
title: Reserve erosion kills tails
description: >-
  Sessions that spend their declared final reserve lose their tail work to the deadline — the reserve is load-bearing, not padding.
tags: [pitfall, env]
timestamp: 2026-08-18T01:30:00Z
scope: domain
scope_conditions: "budgeted sessions under a hard wall-clock deadline with a declared finalization reserve"
evidence:
  - source:
      learner_run: lr_founding_20260818
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-3.md#"35-minute reserve"
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere (pre-bank campaign); the judge graded the plan adherence independently.
    effect: >-
      The champion session began final inference around minute 230 instead of the declared 210, leaving roughly 3.5 minutes of its planned 35-minute reserve, and the same campaign lost two other lanes outright to deadline kills — the erosion-into-loss pattern this card names, exercised in-corpus.
reliability:
  validity: 0.65
  boundary: 0.45
  coverage: 0.2
  score: 0.6
  rationale: >-
    Validity from repeated in-corpus instances (one near-miss, two outright kills in one campaign); boundary partially visible (erosion is gradual and unlogged until terminal); coverage one dataset. Most score-moving next: whether a declared mid-session re-plan (rather than silent drift) preserves the tail.
  state: candidate
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_founding_20260818
    change: >-
      Founded from the wave-4 operations record and the judge's plan-adherence grade [E1].
supersedes: null
contradicts: []
probe: >-
  In any budgeted session, log the planned-vs-actual start of the final phase; correlate reserve remaining with whether tail artifacts shipped.
---

Under a hard deadline, the final reserve is the budget that turns work into artifacts — inference, archiving, prediction export all live there — so eroding it converts completed computation into nothing [E1]. Erosion is silent: each mid-session overrun looks locally justified, no single decision spends the reserve, and the loss only materializes at the wall. The working rule: treat the declared reserve as a hard interior deadline, and when drift is unavoidable, re-plan it explicitly (shrinking scope, not the reserve) so the tail work survives.
