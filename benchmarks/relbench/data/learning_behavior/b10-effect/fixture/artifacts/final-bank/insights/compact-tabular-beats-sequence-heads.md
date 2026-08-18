---
type: insight
title: Compact tabular beats sequence heads
description: >-
  In late-regime or distribution-shifted churn, a viable event sequence does not make its neural head competitive with compact cutoff-safe summaries.
tags: [pitfall, model:sequence]
timestamp: 2026-08-18T02:18:28Z
scope: [family:entity_binary_classification]
scope_conditions: "entity churn is evaluated in a late or decayed regime and an ordered-event neural head is gated against compact cutoff-safe tabular summaries"
evidence:
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/it-1/flow-1.md#implementation
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the lane independently built and benchmarked the event store before gating its sequence model.
    effect: >-
      The cached day-level histories were fast, complete enough for production, and aligned to every seed, yet the ordered-event head still trailed the compact fallback and no production blend was admitted. This separates representation viability from head quality and exercises the card.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-5.md#judgment
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a separate campaign independently gated an attention sequence head against its compact late-regime model.
    effect: >-
      The sequence head lost decisively on the late fold and the legal blend selected zero, after which the judge allowed reopening only under a materially different objective. This independent rejection exercises the card.
  - source:
      learner_run: lr_20260818T043859
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-2/flow-1.md#evaluation
      corroborating_refs:
        - mined/it-2/flow-2.md#judgment
        - mined/it-1/flow-4.md#judgment
        - mined/it-1/flow-1.md#judgment
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this campaign; this adds same-task, same-family breadth beyond the other amazon-user-churn campaign in founding [E1], with a much stronger paired-SE measurement.
    effect: >-
      The widened GBDT (run_0009) scored 0.7104008431 as the campaign's strongest single family, while the four-family stack that added the TCN sequence head alongside the latent and graph members (run_0010) landed at 0.7101500280 — 0.000251 below the standalone GBDT and inside one bootstrap SE, so the ordered-event head bought nothing over the compact tabular matrix. The same lane's forward gate then rejected every sequence- or text-flavored widening it tried: PCA review-semantic event summaries at P(improvement)=0, thin-history plus cohort priors at mean +0.000131 and P(improvement)=0.67 (short of the predeclared 0.8 gate), and exponential origin weighting at P(improvement)=0. The corroborating heads lost the same way in the linked flows — the task-adaptive ModernBERT gate [flow-2], and the coarse Qwen attribute panel and standalone hashed-text voter [it-1 flow-4, flow-1] — so these decisive gated losses of sequence and text heads against compact cutoff-safe tabular summaries exercise the card.
reliability:
  validity: 0.75
  boundary: 0.68
  coverage: 0.37
  score: 0.65
  rationale: >-
    Validity 0.75 reflects the two founding cross-campaign exercises plus a decisive recurrence in which the widened GBDT (run_0009, 0.7104008431) was the strongest single family and folding the TCN sequence head into the four-family stack (run_0010, 0.7101500280) stayed 0.000251 below it, with the corroborating ModernBERT, Qwen, and hashed-text heads losing in the linked flows; the lift comes from measurement strength, not serving, because all entries remain exercise evidence and none weakens or refutes the claim. Boundary 0.68 is better mapped across viable event stores, attention and temporal sequence heads, task-adaptive text, coarse language-model attributes, and the requirement for a materially different supervised objective before reopening. Coverage 0.37 spans rel-amazon user-churn across two campaigns and rel-ratebeer user-churn, adding within-task breadth but remaining one family under the 0.5 unvisited discount. Overall 0.65 balances those dimensions. Each independent campaign receives full exercise participation, while E3's several head losses are one campaign-level measurement rather than multiple units and receive no served-confirm weight. Most score-moving next: a served directly churn-supervised sequence objective tested on lag-matched late folds in another dataset.
  state: active
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_20260818T021828
    change: >-
      Created by induction from two independently gated sequence heads in late or shifted regimes [E1-E2].
supersedes: null
contradicts: []
probe: >-
  Hold the event store and forward folds fixed, replace the auxiliary or ordering objective with direct horizon supervision, and admit the sequence member only if its paired late-fold delta clears uncertainty.
---

Ordered histories can be cheap, complete, and causally correct while the neural scorer over them still loses to compact recency, cadence, and cohort summaries [E1][E2]. Under a late regime or other distribution shift, a weakly aligned ordering or auxiliary objective spends capacity on sequence detail that does not rank the fixed-horizon outcome, while tabular summaries expose the current hazard directly. Keep the event representation as reusable state, gate the head independently, and reopen sequence modeling only when the supervised objective or information path changes materially.
