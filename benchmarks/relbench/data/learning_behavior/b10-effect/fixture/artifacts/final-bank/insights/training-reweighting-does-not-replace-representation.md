---
type: insight
title: Training reweighting does not replace representation
description: >-
  Under temporal drift, changing sample weights or loss geometry without changing the representation tends to fail forward gates or yield noise-sized gains.
tags: [pitfall, temporal, training]
timestamp: 2026-08-18T04:38:59Z
scope: [family:entity_binary_classification]
scope_conditions: "a cutoff-safe baseline is evaluated across multiple forward origins and a weighting or objective variant leaves its representation unchanged"
evidence:
  - source:
      learner_run: lr_20260818T043859
      trajectory: rel-amazon--user-churn/20260812T091402_lane-c10
      ref: mined/it-3/flow-3.md#evaluation
      corroborating_ref: mined/it-3/flow-4.md#evaluation
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere because the card did not yet exist; the campaign independently gated density correction, a ranking objective, and a bundled robust-objective consumer against its fixed compact representation.
    effect: >-
      Density correction failed every development origin, while the surviving ranking and bundled robust-objective alternatives produced only noise-sized gains. This campaign-level comparison exercises the claim that weight and loss changes without a new representation did not produce a stable gain.
  - source:
      learner_run: lr_20260818T043859
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-1.md#evaluation
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere because the card did not yet exist; a separate campaign independently compared recency weighting with uniform fitting on a fixed cutoff-safe matrix. This campaign-B flow-1 measurement also underlies the served confirm of sub-noise-orderings-are-ties, so a contradiction at that source would touch both cards.
    effect: >-
      Exponential origin weighting lost on all three forward folds (−0.000200/−0.000352/−0.000171, P(improvement)=0), while uniform last-eight-origin fitting was more temporally stable. This independent campaign exercises the claim that reweighting the same representation did not correct the drift.
  - source:
      learner_run: lr_20260818T043859
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-5.md#evaluation
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere because the card did not yet exist; this is the same campaign measurement already used by direct-binary-companions-stabilize-survival, supporting a distinct contrast without adding an independent participation unit.
    effect: >-
      Density-ratio weighting failed its predefined gate in both model chains and was disabled, while forward selection retained the simpler direct/hazard average. This second-task measurement exercises the same reweighting failure outside the first task.
reliability:
  validity: 0.66
  boundary: 0.63
  coverage: 0.24
  score: 0.57
  rationale: >-
    Validity 0.66 reflects two clean independent rel-amazon campaign exercises in which unchanged-representation weighting or loss variants failed forward gates or yielded noise-sized gains, plus an aligned but shared cross-task exercise; all entries are unserved exercise evidence and none weakens or refutes the claim. Boundary 0.63 is moderately defined by fixed representation, forward origins, and recency, density, ranking, and robust-objective variants, while the mechanism remains limited to levers that cannot create a missing signal path. Coverage 0.24 has one clean task across two campaigns and only a shared rel-ratebeer cross-task nod, so the 0.5 unvisited discount is substantial. Overall 0.57 balances those dimensions. Campaigns A and B receive separate full exercise participation, while E3 shares ratebeer flow-5 with direct-binary and adds no independent participation unit; none receives served-confirm weight. Most score-moving next: a served preregistered comparison on a new shifted task that holds representation fixed and contrasts weighting with a new causal feature block.
  state: candidate
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_20260818T043859
    change: >-
      Created by induction from independently gated weighting and objective variants across three campaigns and two tasks [E1-E3].
supersedes: null
contradicts: []
probe: >-
  On a new temporally shifted task, hold the cutoff-safe representation and learner capacity fixed; compare uniform fitting with one predeclared recency or density weighting and one robust or ranking objective across forward origins, then test whether a new causal feature block produces the first stable gain.
---

Temporal drift can change which event state or relational context carries the label, rather than merely changing how often old and new rows appear. Reweighting the same matrix or changing its loss cannot create that missing state, so forward gates reject those levers or leave only noise-sized gains [E1][E2][E3]. Hold representation and folds fixed when testing weights or objectives; if they do not clear paired uncertainty across origins, stop treating the drift as a weighting problem and re-aim to a structural signal path.
