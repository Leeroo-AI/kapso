---
type: insight
title: Group relative standing
description: >-
  Rank and normalize features within their competing group — absolute values cannot see relative standing.
tags: [data:grouped_rows]
timestamp: 2026-08-18T01:30:00Z
scope: domain
scope_conditions: "prediction rows share a natural grouping key (seed timestamp, session, origin, parent entity) so rows are effectively scored against one another"
evidence:
  - source:
      learner_run: lr_founding_20260818
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-3.md#"within-origin ranks"
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere (pre-bank campaign); the champion lane applied the move independently — its residual inputs include recency, depth, persistence, gap statistics and their within-origin ranks.
    effect: >-
      The champion's residual consumed within-origin ranks as designed inputs; no isolating ablation of the rank block was recorded, so this entry exercises the card without grading the claim.
reliability:
  validity: 0.6
  boundary: 0.4
  coverage: 0.2
  score: 0.55
  rationale: >-
    Validity rests on repeated adoption across past campaigns and one in-corpus exercise, not yet an isolated in-scope measurement; boundary untested (no contradiction has probed where grouping is too weak to matter); coverage is one dataset's campaigns. Most score-moving next: one ablation of the rank block on a grouped-row task — hence the probe.
  state: candidate
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_founding_20260818
    change: >-
      Founded from the practice corpus (group-relative normalization) with an in-corpus exercise anchor [E1].
supersedes: null
contradicts: []
probe: >-
  Ablate the within-group rank/percentile block of the top features on one forward fold of any grouped-row task; keep the delta and its clustered SE.
---

A label that encodes relative standing — who churns, which ad is clicked, which row wins its auction — is a comparison within a competing set, so the informative coordinate is the row's position inside its group, not its absolute value [E1]. Alongside each informative raw feature, add its within-group rank, percentile, z-score, and gap to the group leader; an absolute value cannot distinguish strong-in-a-weak-group from strong-overall, while the label often encodes exactly that difference. The move stops paying where rows do not compete (no shared key, or singleton groups), which is the boundary the scope conditions name.
