---
type: insight
title: Sub-noise orderings are ties
description: >-
  When paired score differences are smaller than their uncertainty, the artifact order is bookkeeping, not evidence of a better model.
tags: [validation, pitfall]
timestamp: 2026-08-18T02:18:28Z
scope: [family:entity_binary_classification]
scope_conditions: "candidates are evaluated on the same rows and their paired score difference can be compared with an uncertainty estimate"
evidence:
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/it-1/flow-3.md#judgment
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the round judge independently compared the archived and shipped siblings on identical validation rows.
    effect: >-
      The archived sibling led by only +0.000039 against a rough SE of 0.00080, and the judge explicitly called the ordering noise. This uncertainty-resolved in-scope agreement exercises the card.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-4.md#judgment
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the judge independently separated a supported lineage step from its final micro-increment.
    effect: >-
      The final increment was +0.0000849 with paired SE 0.0001168 and was judged noise that must not be polished further. This paired, in-scope agreement exercises the card.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-4.md#judgment
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the round judge independently compared the two leading tabular artifacts with a paired bootstrap.
    effect: >-
      The nominal winner led by +0.0000794 against paired-bootstrap SE 0.000715, so the judge banked the strict winner while treating the order as noise. This paired, in-scope agreement exercises the card.
  - source:
      learner_run: lr_20260818T043859
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-1.md#judgment
      card_version: 1
    verdict: confirm
    usage: >-
      The card was served and used; the judge preserved the operationally selected sibling instead of overturning it on an unresolved ordering.
    effect: >-
      The settlement expresses the uncertainty threshold as one SE. Run_0009's measured delta over the internally selected run_0010 was +0.000251, within one standard error, and the judge explicitly refused to overturn the operational selection on that noise.
reliability:
  validity: 0.86
  boundary: 0.78
  coverage: 0.45
  score: 0.75
  rationale: >-
    Validity 0.86 reflects three aligned paired-uncertainty exercises plus the first served confirmation, where a +0.000251 sibling edge remained within one SE and correctly did not overturn the operational choice; no ledger event weakens or refutes the claim. Boundary 0.78 is well specified by same-row pairing, an uncertainty band that includes zero, and larger supported steps as contrast. Coverage 0.45 spans three churn task settings but remains one entity-classification family under the 0.5 unvisited discount. Overall 0.75 balances those dimensions. Each independent campaign receives full participation weight, while the new same-task recurrence broadens participation by campaign rather than by task. Most score-moving next: a prospective tie decision whose preregistered repeat evaluation tests whether the ordering remains unresolved.
  state: active
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_20260818T021828
    change: >-
      Created by induction from three independent paired-uncertainty judgments [E1-E3].
supersedes: null
contradicts: []
probe: >-
  For the next same-row candidate pair, record the paired delta and uncertainty before selection; if the interval contains zero, freeze the lineage and compare the call with one later repeat.
---

A leaderboard imposes a total order even when the measurement does not: if a paired difference is smaller than its uncertainty, selecting the top artifact records a winner but does not establish a better mechanism [E1][E2][E3]. Continuing to tune that lineage then selects harder on noise and consumes evaluations without earned information. Bank the strict artifact winner when an operational choice is required, but describe the models as tied and reopen the lineage only after a change clears paired uncertainty.
