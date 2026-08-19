---
type: insight
title: Graph rejections are variant bounded
description: >-
  A zero-weight graph member falsifies its implemented recipe, not graph variants whose information path or objective was never tested.
tags: [pitfall, model:graph]
timestamp: 2026-08-18T02:18:28Z
scope: [family:entity_binary_classification]
scope_conditions: "a graph candidate is compared with a cutoff-safe tabular fallback under forward validation and the implemented graph departs from a broader proposed recipe"
evidence:
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-amazon--user-churn/20260731T210811_lane-b2
      ref: mined/it-1/flow-3.md#judgment
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the judge independently audited what the production graph proxy actually pooled.
    effect: >-
      The proxy degraded every blend and was correctly assigned zero weight, but it pooled each entity's own rows and never exercised the proposed cross-entity neighborhood. The result is variant-specific support that exercises the card without grading the untested family.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--beer-churn/20260801T211237_lane-b2
      ref: mined/it-1/flow-6.md#evaluation
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; a separate temporal graph lane independently used a forward blend selector against its tabular fallback.
    effect: >-
      The smallest nonzero graph blend lost to weight zero, so the selector removed the graph while the tabular fallback survived; the planned embeddings-as-features member was not built. This measured variant rejection exercises the card without closing unimplemented graph recipes.
  - source:
      learner_run: lr_20260818T021828
      trajectory: rel-ratebeer--user-churn/20260801T224425_lane-a1
      ref: mined/it-1/flow-6.md#judgment
      card_version: null
    verdict: exercise
    usage: >-
      Served nowhere in this founding-bank campaign; the campaign independently compared a custom graph head and its fusion against a legal hazard fallback.
    effect: >-
      Direct and fused graph variants lost and the blend selected zero, while the audit identified ID-only encodings, collapsed interactions, and frozen auxiliary pretraining as departures from an end-to-end row-encoder recipe. This measured variant rejection exercises the bounded claim.
  - source:
      learner_run: lr_20260818T043859
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: mined/it-3/flow-3.md#evaluation
      corroborating_ref: mined/it-1/flow-2.md#evaluation
      card_version: 1
    verdict: confirm
    usage: >-
      The card was served and used; the campaign rejected the residual-correction recipe while retaining the earlier, structurally different graph result as separate evidence.
    effect: >-
      The implemented residual correction's measured delta was -0.0014688690 with paired SE 0.0000664834, justifying its rejection, while a structurally different GraphSAGE artifact remained separately viable; the number settles the recipe, not the family.
reliability:
  validity: 0.84
  boundary: 0.84
  coverage: 0.45
  score: 0.76
  rationale: >-
    Validity 0.84 reflects three aligned variant-level exercises plus the first served confirmation, where residual correction lost by -0.0014688690 with paired SE 0.0000664834 while a structurally different GraphSAGE remained viable; no ledger event weakens or refutes the bounded claim. Boundary 0.84 is strong because every entry identifies the failed information path or objective and leaves materially different variants open, with the served contrast directly separating recipe rejection from family rejection. Coverage 0.45 spans three churn task settings but only one entity-classification family under the 0.5 unvisited discount. Overall 0.76 balances those dimensions. Independent campaigns receive full participation weight, and the corroborating GraphSAGE reference is part of E4's single contrast rather than a second participation unit. Most score-moving next: an end-to-end attributed-row graph tested against the same fallback under an unchanged forward gate.
  state: active
provenance: {version: 1}
log:
  - version: 1
    date: 2026-08-18
    commit: lr_20260818T021828
    change: >-
      Created by induction from three independently gated graph variants [E1-E3].
supersedes: null
contradicts: []
probe: >-
  Re-run one graph comparison with the previously omitted information path and a directly supervised objective; compare it with the same tabular fallback on unchanged forward folds.
---

When a guarded graph member loses to a cutoff-safe tabular fallback, weight zero is the correct decision for that implementation [E1][E2][E3]. The result transfers only through the implemented information path and objective: self-only pooling cannot test cross-entity propagation, collapsed interactions cannot test attributed event nodes, and frozen auxiliary training cannot test end-to-end supervision. Record the rejection with its exact recipe, reuse it to prevent repeat compute, and keep materially different graph variants open rather than promoting a component failure into a family-wide conclusion.
