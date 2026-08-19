---
type: procedure
representation: text
title: Cross branch ensemble candidate
description: >-
  Near budget end, rank-average the decorrelated near-tied finalists across all branches and submit the blend as its own candidate.
tags: [ensemble]
timestamp: 2026-08-18T01:30:00Z
scope: domain
scope_conditions: "a search holding multiple archived finalists across branches, with out-of-fold predictions available or buildable"
evidence:
  - source:
      learner_run: lr_founding_20260818
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-3.md#"0.0031691"
      card_version: null
    verdict: confirm
    usage: >-
      Served nowhere (pre-bank campaign); the champion lane's spec authorized the move and the build executed it independently — an uncontaminated instance.
    effect: >-
      The precommitted decorrelation gate fired (standalone survival within two clustered SE of the recorded folds, rank-correlation 0.868 with the archived blend) and the equal-rank average with the strongest cross-family finalist improved the fallback by +0.0031691, about 3.6 validation SE, becoming the campaign champion — a significant in-scope agreement, so this entry confirms the card.
reliability:
  validity: 0.8
  boundary: 0.5
  coverage: 0.25
  score: 0.7
  rationale: >-
    Validity from one significant, uncontaminated in-corpus confirmation (the campaign's largest measured gain) plus the practice corpus's repeated observation; boundary moderately mapped (the correlation ceiling near 0.95 and the OOF-availability precondition are stated, not yet stress-tested); coverage one dataset. Most score-moving next: the same move on a second dataset's finalists.
  state: active
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_founding_20260818
    change: >-
      Founded from the practice corpus (decorrelated-finalist ensembling) with the wave-4 champion blend as confirming anchor [E1].
supersedes: null
contradicts: []
probe: >-
  On the next campaign's final iteration, list cross-branch finalist correlations and gate one equal-rank blend of the weakest-correlated near-tied pair on a forward fold.
---

Method [E1]: (1) near the end of budget, list every archived candidate across ALL branches — not just the current lineage — with validation score and pairwise prediction rank-correlation; (2) where two or more sit within about two clustered SE of the leader and correlate below roughly 0.95, build their rank-average, weights fit on a forward fold only; (3) submit the blend as its OWN candidate run — final selection only ever sees candidates, so a blend that is never a candidate can never win; (4) any family intended for the blend must ship training-period out-of-fold predictions at build time, because a family without them cannot be selected later and its compute strands; (5) where a per-slice read shows one finalist paying disproportionately on a segment, weight it up there. The mechanism is variance reduction under disagreement — argmax selection among near-tied decorrelated finalists discards the one improvement that is reliable in expectation.
