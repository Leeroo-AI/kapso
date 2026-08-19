---
type: insight
title: Static audit flags require contract validation
description: >-
  Pattern-based publishing flags are review hooks; binding validity comes from traced dataflow and isolated dynamic evaluation.
tags: [governance, pitfall]
timestamp: 2026-08-18T02:18:28Z
scope: domain
scope_conditions: "a static publisher scans source patterns while an isolated evaluator can verify artifacts, grader identity, and train-versus-evaluation dataflow"
evidence:
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/operations.md#publishing-audit-advisory-on-the-champion
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the publishing scan and independent judge audit were both recorded.
    effect: >-
      The static scan marked cache-path access and the sanctioned two-chain training pattern, while the evaluation audit traced validation-label use and found no tampering. This independent static-versus-dynamic comparison exercises the card.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/index.md#outcome
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a separate champion received the same static status and an independent fairness audit.
    effect: >-
      The publisher marked the source not clean on input masking and two-chain stacking patterns, yet the isolated evaluation was recorded valid and fair with no tampering. This independent comparison exercises the card.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/operations.md#winner-audit-not-clean-advisory
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the third campaign independently recorded the publisher's finding and its advisory classification.
    effect: >-
      Cache-path and sanctioned train-plus-validation hooks again produced a not-clean static status whose own note classified the dataflow findings as advisory rather than violations. This third independent recurrence exercises the card.
reliability:
  validity: 0.74
  boundary: 0.68
  coverage: 0.43
  score: 0.66
  rationale: >-
    Validity 0.74 reflects the same static-versus-contract-audit distinction in three independent campaign exercises, tempered by the absence of served confirmation. Boundary 0.68 is moderately mapped to cache and multi-chain syntax whose actual dataflow was reviewed, though no true-positive contrast appears. Coverage 0.43 spans three churn campaigns but one publishing framework and applies the 0.5 unvisited discount. Overall 0.66 balances those dimensions. Each cross-campaign audit receives full participation weight. Most score-moving next: a true static flag that the isolated evaluator also proves invalid, carving the boundary.
  state: active
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_20260818T021828
    change: >-
      Created by induction from three advisory static-audit recurrences [E1-E3].
supersedes: null
contradicts: []
probe: >-
  For the next static flag, trace the flagged value through each model chain and compare that review with the isolated evaluator's grader identity, artifact checks, and validity verdict.
---

Static publishing scans match suspicious syntax, so they necessarily flag some legal cache access and sanctioned multi-chain training patterns without proving that evaluation data crossed the wrong boundary [E1][E2][E3]. Such a flag should open a contract review: trace which labels feed each prediction, verify the immutable grader and artifact hashes, and run the isolated evaluation. The dynamic audit is binding only for the execution it checks; the static warning remains useful as a locator, but it is not itself a leak or tampering finding.
