---
type: insight
title: Confirmation overhead negative
description: >-
  Mid-run registered confirmations of small variants are net-negative — they buy certainty about ties at the price of untried ideas.
tags: [pitfall]
timestamp: 2026-08-18T01:30:00Z
scope: domain
scope_conditions: "a campaign with a bounded budget where registered evaluations cost materially more than internal measurements"
evidence:
  - source:
      learner_run: lr_founding_20260818
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-3.md#"34.2 GPU-min"
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere (pre-bank campaign); the judge's contest economics derived the finding independently.
    effect: >-
      Seven intermediate registered confirmations consumed 34.2 GPU-minutes — graded as roughly seven foregone feature/stack ablations and a net-negative counterfactual — exercising the card's claim in-corpus.
reliability:
  validity: 0.6
  boundary: 0.4
  coverage: 0.2
  score: 0.55
  rationale: >-
    Validity from one judged in-corpus economics read; boundary untested (when confirmation IS worth it — e.g. before an irreversible bank — is asserted, not carved); coverage one dataset. Most score-moving next: an economics comparison of confirm-heavy vs ablate-heavy sessions.
  state: candidate
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_founding_20260818
    change: >-
      Founded from the wave-4 judge's contest-economics grade [E1].
supersedes: null
contradicts: []
probe: >-
  For one campaign, tally registered evaluations by purpose (new-candidate vs confirmation) against the incumbent-improvement each produced.
---

Registered evaluations are the campaign's most expensive measurement, so spending them to confirm micro-variants of the incumbent purchases information the internal gates already had, at the price of the ablations and new candidates that were never run [E1]. The mechanism is opportunity cost under a fixed budget: every confirmation slot is an untried idea's slot. Confirmation earns its cost only where the decision it guards is irreversible — banking a final, retiring a family — which is the boundary the card's scope conditions point at.
