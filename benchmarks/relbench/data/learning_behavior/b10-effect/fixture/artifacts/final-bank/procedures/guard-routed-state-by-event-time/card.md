---
type: procedure
representation: text
title: Guard routed state by event time
description: >-
  Recompute mutable snapshots from events and admit routed metadata only when its effective time is legal for the event and prediction cutoff.
tags: [causality, data:relational, pitfall]
timestamp: 2026-08-18T02:18:28Z
scope: domain
scope_conditions: "relational features route exported entity state or aggregate counters onto historical events or prediction rows"
evidence:
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-4.md#idea
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the lane independently audited exported counters against full-history event counts before implementation.
    effect: >-
      A dump-time count tracked full-history rather than as-of-seed history, so the pipeline dropped snapshot aggregates and reconstructed them from censored events. This predicted-and-measured leak exercises the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-4.md#difficulties
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the champion lane independently measured timestamp support for an auxiliary relation.
    effect: >-
      Every event in the auxiliary table occurred after the legal prediction cutoffs, so strict censoring made the block empty and the model excluded it. This cutoff-support check exercises the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/operations.md#void-the-exp_1-generic_exp_1-self-void-cascade--leaky-beer-metadata-routes
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; one lane found the routed-metadata hazard after the fact while a separate lane had guarded the same route proactively.
    effect: >-
      Exported entity creation times followed many source events, so the unsafe cache was versioned, affected archives were self-voided, and the legal route required metadata creation no later than the event. This independent failure-and-control pair exercises the procedure.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-4.md#implementation
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a separate hazard lane independently audited export-time state before feature generation.
    effect: >-
      An exported last-activity snapshot extended beyond the test cutoff and was excluded rather than treated as historical state. This proactive snapshot exclusion exercises the procedure.
  - source:
      learner_run: lr_20260818T043859
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-3.md#difficulties
      corroborating_ref: mined/it-2/flow-3.md#evaluation
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this campaign; this first rel-amazon instance materially broadens task coverage beyond the founding rel-ratebeer beer- and user-churn measurements.
    effect: >-
      Checkpoint-relative event age went stale in a detached recurrent carry and inverted the 2015-01 raw-survival fold, with run_0011 reaching 0.6919252583. Recomputing event-era age from events, while reserving already-survived recency for the analytic survival ratio, raised the four-fold latent mean from 0.6777557 to 0.6838552 and the official score from 0.6919253 to 0.6983156 in run_0012, a +0.0063903 repair. This recompute-recency-from-events correction exercises the event-time guard as a staleness and correctness case, not a leakage case.
reliability:
  validity: 0.78
  boundary: 0.80
  coverage: 0.47
  score: 0.72
  rationale: >-
    Validity 0.78 reflects aligned proactive and retrospective event-time guard exercises, now including a measured +0.0063903 staleness repair, but remains capped by the absence of any served confirmation and has no weaken or refute event. Boundary 0.80 now covers mutable snapshots, effective dates, wholly post-cutoff relations, archive voiding, and checkpoint-relative age that must be recomputed from event-era state, distinguishing correctness and staleness from leakage. Coverage 0.47 reaches three dataset-task settings across rel-ratebeer beer-churn, rel-ratebeer user-churn, and rel-amazon user-churn, but remains one relational domain under the 0.5 unvisited discount. Overall 0.72 balances those dimensions. Independent dataset-task campaigns receive full exercise participation, while same-campaign lanes and the ratebeer-user proactive/self-void pair are collapsed to campaign-level support; E5 is unserved and receives no confirmation weight. Most score-moving next: a served replay that injects both a future snapshot and stale checkpoint-relative age and verifies both guards.
  state: active
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_20260818T021828
    change: >-
      Created by induction from snapshot, support, and routed-metadata causality audits [E1-E4].
supersedes: null
contradicts: []
probe: >-
  Inject one future-dated snapshot and one metadata record effective after its source event into a fixture; assert both disappear while legal earlier state remains.
---

Method [E1][E2][E3][E4]: inventory every exported field as immutable state, mutable snapshot, or event-derived aggregate, then measure the timestamp support of every relation before joining it. Recompute mutable counts and recencies from events censored at the prediction time; when metadata is routed through an event, require its effective time to be no later than both that event and the prediction cutoff. Exclude relations whose entire support is post-cutoff rather than accepting an all-missing pseudo-feature. Version caches by the guard policy, audit saved artifacts, and void every artifact built from an unsafe version—the higher score is evidence of leakage, not a reason to preserve it.
